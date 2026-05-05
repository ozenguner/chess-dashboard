"""
Analyze chess games with Stockfish and store accuracy metrics in data/chess.db.
Resumable: games where accuracy_me IS NOT NULL are skipped automatically.

Usage
-----
python analyze_games.py                      # games from 2026-05-01 onward (test default)
python analyze_games.py --since 2025-01-01  # custom cutoff
python analyze_games.py --all               # every game (~1-2 hrs at depth 16)
python analyze_games.py --depth 14          # shallower = ~3x faster, slightly less accurate
python analyze_games.py --workers 4         # override parallel workers
python analyze_games.py --stockfish "C:\\path\\to\\stockfish.exe"
"""

import argparse
import concurrent.futures
import io
import math
import os
import pathlib
import sqlite3
import time

import chess
import chess.pgn
import chess.engine

DB_PATH = pathlib.Path("data/chess.db")


# ── Accuracy formulas (chess.com methodology) ─────────────────────────────────

def _win_pct(cp: float) -> float:
    """Centipawns (white perspective) → win probability [0, 100] for that side."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def _move_accuracy(delta_wp: float) -> float:
    """Win-% lost on one move → move accuracy score [0, 100]."""
    return max(0.0, 103.1668 * math.exp(-0.04354 * max(0.0, delta_wp)) - 3.1669)


# ── Stockfish detection ────────────────────────────────────────────────────────

def find_stockfish(override: str | None = None) -> str:
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"Stockfish not found at: {override}")
        return override

    import shutil
    sf = shutil.which("stockfish")
    if sf:
        return sf

    candidates = [
        r"C:\Program Files\Stockfish\stockfish.exe",
        r"C:\Users\ozeng\AppData\Local\Programs\Stockfish\stockfish.exe",
        r"C:\stockfish\stockfish.exe",
        r"C:\tools\stockfish\stockfish.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "\nStockfish not found. Install it with:\n"
        "    winget install Stockfish.Stockfish\n"
        "Or download from: https://stockfishchess.org/download/\n"
        "Then re-run, or pass --stockfish <path> explicitly."
    )


# ── Per-game analysis (executes in a worker process) ──────────────────────────

def analyze_one(args: tuple) -> dict:
    """
    Analyze a single game.  Designed to run in a subprocess — opens and closes
    its own Stockfish engine so workers don't share state.
    """
    game_id, pgn_text, color, sf_path, depth = args
    result: dict = {"game_id": game_id, "error": None}
    engine = None

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            result["error"] = "PGN parse failed"
            return result

        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"Threads": 1, "Hash": 16})
        limit = chess.engine.Limit(depth=depth)

        board = game.board()

        # Evaluate the starting position
        info  = engine.analyse(board, limit)
        score = info["score"].white().score(mate_score=10000)

        my_accs,  opp_accs  = [], []
        my_cpls,  opp_cpls  = [], []
        my_blunders = my_mistakes = my_inaccuracies = 0

        node = game
        while node.variations:
            node      = node.variations[0]
            is_white  = (board.turn == chess.WHITE)
            is_mine   = (color == "white") == is_white

            # Win% BEFORE this move, from the mover's point of view
            wp_before = _win_pct(score if is_white else -score)

            board.push(node.move)

            info        = engine.analyse(board, limit)
            score_after = info["score"].white().score(mate_score=10000)

            # Win% AFTER this move, still from the mover's point of view
            wp_after = _win_pct(score_after if is_white else -score_after)

            delta_wp = wp_before - wp_after                      # positive = bad move
            acc      = _move_accuracy(delta_wp)

            # Centipawn loss (from each player's perspective)
            cpl = (max(0.0, score - score_after)  if is_white
                   else max(0.0, score_after - score))

            if is_mine:
                my_accs.append(acc)
                my_cpls.append(cpl)
                if   delta_wp >= 20: my_blunders    += 1
                elif delta_wp >= 10: my_mistakes     += 1
                elif delta_wp >=  5: my_inaccuracies += 1
            else:
                opp_accs.append(acc)
                opp_cpls.append(cpl)

            score = score_after

        avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else None
        result.update({
            "accuracy_me":     avg(my_accs),
            "accuracy_opp":    avg(opp_accs),
            "acpl_me":         avg(my_cpls),
            "acpl_opp":        avg(opp_cpls),
            "blunders_me":     my_blunders,
            "mistakes_me":     my_mistakes,
            "inaccuracies_me": my_inaccuracies,
        })

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if engine:
            try:
                engine.quit()
            except Exception:
                pass

    return result


# ── DB helpers ─────────────────────────────────────────────────────────────────

def ensure_columns(con: sqlite3.Connection) -> None:
    existing = {row[1] for row in con.execute("PRAGMA table_info(games)")}
    new_cols = {
        "accuracy_me":     "REAL",
        "accuracy_opp":    "REAL",
        "acpl_me":         "REAL",
        "acpl_opp":        "REAL",
        "blunders_me":     "INTEGER",
        "mistakes_me":     "INTEGER",
        "inaccuracies_me": "INTEGER",
    }
    for col, dtype in new_cols.items():
        if col not in existing:
            con.execute(f"ALTER TABLE games ADD COLUMN {col} {dtype}")
    con.commit()


def load_pending(con: sqlite3.Connection, since: str) -> list[tuple]:
    return con.execute(
        "SELECT game_id, pgn, color FROM games "
        "WHERE accuracy_me IS NULL AND date >= ? ORDER BY date",
        (since,),
    ).fetchall()


def save_result(con: sqlite3.Connection, r: dict) -> None:
    con.execute(
        """UPDATE games SET
            accuracy_me=:accuracy_me,   accuracy_opp=:accuracy_opp,
            acpl_me=:acpl_me,           acpl_opp=:acpl_opp,
            blunders_me=:blunders_me,   mistakes_me=:mistakes_me,
            inaccuracies_me=:inaccuracies_me
           WHERE game_id=:game_id""",
        r,
    )
    con.commit()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze chess games with Stockfish.")
    ap.add_argument("--since",     default="2026-05-01",
                    help="Analyze games on/after this date (default: 2026-05-01)")
    ap.add_argument("--all",       action="store_true",
                    help="Analyze all games, ignoring --since")
    ap.add_argument("--depth",     type=int, default=16,
                    help="Stockfish depth (default 16; use 12-14 for a quick pass)")
    ap.add_argument("--workers",   type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallel worker processes (default: CPU count − 1)")
    ap.add_argument("--stockfish", default=None,
                    help="Path to stockfish executable (auto-detected if omitted)")
    args = ap.parse_args()

    since   = "1900-01-01" if args.all else args.since
    sf_path = find_stockfish(args.stockfish)

    print(f"Stockfish : {sf_path}")
    print(f"Since     : {since}")
    print(f"Depth     : {args.depth}")
    print(f"Workers   : {args.workers}")
    print()

    con = sqlite3.connect(DB_PATH)
    ensure_columns(con)
    pending = load_pending(con, since)

    if not pending:
        print("Nothing to analyze — all matching games already have accuracy data.")
        con.close()
        return

    total = len(pending)
    print(f"{total} game(s) to analyze…\n")

    work = [
        (gid, pgn, color, sf_path, args.depth)
        for gid, pgn, color in pending
    ]

    done = errors = 0
    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_one, w): w[0] for w in work}

        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            seq = done + errors + 1

            if r.get("error"):
                errors += 1
                print(f"  [{seq}/{total}] ERROR {r['game_id']}: {r['error']}")
            else:
                save_result(con, r)
                done += 1
                elapsed   = time.time() - t0
                rate      = done / elapsed
                remaining = (total - done - errors) / rate if rate else 0
                print(
                    f"  [{seq:>4}/{total}]  "
                    f"acc={r['accuracy_me']:5.1f}%  "
                    f"acpl={r['acpl_me']:5.1f}  "
                    f"B{r['blunders_me']} M{r['mistakes_me']} I{r['inaccuracies_me']}  "
                    f"| {rate:.2f} g/s  ~{remaining:.0f}s left",
                    flush=True,
                )

    con.close()
    elapsed = time.time() - t0
    print(f"\nFinished — {done} analyzed, {errors} errors — {elapsed:.0f}s total")

    if done:
        con2 = sqlite3.connect(DB_PATH)
        row = con2.execute(
            "SELECT AVG(accuracy_me), AVG(acpl_me), SUM(blunders_me) "
            "FROM games WHERE accuracy_me IS NOT NULL AND date >= ?",
            (since,),
        ).fetchone()
        con2.close()
        print(f"\nBatch summary — avg accuracy: {row[0]:.1f}%  "
              f"avg ACPL: {row[1]:.1f}  total blunders: {row[2]}")


if __name__ == "__main__":
    main()
