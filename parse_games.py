"""
Parses raw monthly chess.com JSON files with python-chess and loads them
into data/chess.db (SQLite).  Re-runs are idempotent: duplicate game_ids
are silently ignored via INSERT OR IGNORE.
"""

import io
import json
import pathlib
import sqlite3

import chess.pgn

USERNAME = "ozengnr"
RAW_DIR = pathlib.Path("data/raw")
DB_PATH = pathlib.Path("data/chess.db")

# chess.com result codes → win / draw / loss
RESULT_MAP = {
    "win": "win",
    "checkmated": "loss",
    "timeout": "loss",
    "resigned": "loss",
    "abandoned": "loss",
    "lose": "loss",
    "stalemate": "draw",
    "insufficient": "draw",
    "repetition": "draw",
    "agreed": "draw",
    "50move": "draw",
    "timevsinsufficient": "draw",
    "bughousepartnerlose": "loss",
}

DDL = """
CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT PRIMARY KEY,
    date          TEXT,
    color         TEXT,
    opponent      TEXT,
    opp_rating    INTEGER,
    my_rating     INTEGER,
    result        TEXT,
    result_detail TEXT,
    time_control  TEXT,
    time_class    TEXT,
    eco           TEXT,
    opening       TEXT,
    num_moves     INTEGER,
    termination   TEXT,
    pgn           TEXT
);
"""


def opening_from_url(eco_url: str) -> str:
    """'https://www.chess.com/openings/Bishops-Opening-2...Nc6' → 'Bishops Opening 2...Nc6'"""
    if not eco_url:
        return ""
    slug = eco_url.rstrip("/").split("/")[-1]
    # replace hyphens with spaces but preserve ellipsis-style dots
    return slug.replace("-", " ")


def count_moves(game: chess.pgn.Game) -> int:
    """Return number of full moves played."""
    node = game
    plies = 0
    while node.variations:
        node = node.variations[0]
        plies += 1
    return (plies + 1) // 2


def parse_game(raw: dict) -> dict | None:
    pgn_text = raw.get("pgn", "")
    if not pgn_text:
        return None

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None

    h = game.headers
    white_user = raw["white"]["username"].lower()

    if white_user == USERNAME.lower():
        color = "white"
        me = raw["white"]
        them = raw["black"]
    else:
        color = "black"
        me = raw["black"]
        them = raw["white"]

    result_detail = me.get("result", "")
    result = RESULT_MAP.get(result_detail, result_detail)

    date = h.get("UTCDate", h.get("Date", "")).replace(".", "-")

    eco_url = h.get("ECOUrl", "")
    opening = h.get("Opening", "") or opening_from_url(eco_url)

    game_id = raw.get("uuid") or raw.get("url", "").rstrip("/").split("/")[-1]

    return {
        "game_id": game_id,
        "date": date,
        "color": color,
        "opponent": them["username"],
        "opp_rating": them.get("rating"),
        "my_rating": me.get("rating"),
        "result": result,
        "result_detail": result_detail,
        "time_control": str(raw.get("time_control", h.get("TimeControl", ""))),
        "time_class": raw.get("time_class", ""),
        "eco": h.get("ECO", raw.get("eco", "")),
        "opening": opening,
        "num_moves": count_moves(game),
        "termination": h.get("Termination", ""),
        "pgn": pgn_text,
    }


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(DDL)
    con.commit()

    files = sorted(RAW_DIR.glob("*.json"))
    total_inserted = total_skipped = total_errors = 0

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        games = data.get("games", [])
        inserted = skipped = errors = 0

        rows = []
        for raw in games:
            try:
                row = parse_game(raw)
                if row:
                    rows.append(row)
            except Exception as exc:
                errors += 1
                print(f"  ERROR parsing game in {path.name}: {exc}")

        if rows:
            cur = con.executemany(
                """
                INSERT OR IGNORE INTO games
                    (game_id, date, color, opponent, opp_rating, my_rating,
                     result, result_detail, time_control, time_class,
                     eco, opening, num_moves, termination, pgn)
                VALUES
                    (:game_id, :date, :color, :opponent, :opp_rating, :my_rating,
                     :result, :result_detail, :time_control, :time_class,
                     :eco, :opening, :num_moves, :termination, :pgn)
                """,
                rows,
            )
            inserted = cur.rowcount
            skipped = len(rows) - inserted
            con.commit()

        print(f"{path.name}: {inserted} inserted, {skipped} skipped, {errors} errors")
        total_inserted += inserted
        total_skipped += skipped
        total_errors += errors

    con.close()
    print(f"\nTotal — inserted: {total_inserted}, skipped: {total_skipped}, errors: {total_errors}")
    print(f"DB: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
