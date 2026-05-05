"""Streamlit chess dashboard — reads from data/chess.db."""
import pathlib
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH   = pathlib.Path("data/chess.db")
USERNAME  = "ozengnr"
EST_OFFSET = -5   # UTC → EST (UTC-5, no DST)
DAYS_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
PHASE_ORDER = ["Opening (≤10 moves)", "Middlegame (11–25 moves)", "Endgame (26+ moves)"]
ECO_LABELS = {
    "A": "A: Flank / English",
    "B": "B: Semi-open (1.e4 ≠ e5)",
    "C": "C: Open (1.e4 e5)",
    "D": "D: Closed / Semi-closed",
    "E": "E: Indian defences",
}

# ── Helper functions ───────────────────────────────────────────────────────────
def _norm_term(t: str) -> str:
    t = t.lower()
    if "checkmate"                         in t: return "checkmate"
    if "timeout vs insufficient" in t or "time vs insufficient" in t:
                                               return "time vs insufficient"
    if "on time"                           in t: return "on time"
    if "resign"                            in t: return "resignation"
    if "stalemate"                         in t: return "stalemate"
    if "repetition"                        in t: return "repetition"
    if "agreement"                         in t: return "agreed"
    if "insufficient"                      in t: return "insufficient material"
    if "50"                                in t: return "50-move rule"
    if "abandon"                           in t: return "abandoned"
    if "3rd time" in t or "three"          in t: return "three-check"
    return t

def _phase(n: int) -> str:
    if n <= 10: return "Opening (≤10 moves)"
    if n <= 25: return "Middlegame (11–25 moves)"
    return "Endgame (26+ moves)"

def _fmt_h(h) -> str:
    if pd.isna(h): return "?"
    h = int(h)
    if h == 0:  return "12am"
    if h < 12:  return f"{h}am"
    if h == 12: return "12pm"
    return f"{h-12}pm"

def _wr(s: pd.Series) -> float:
    return (s == "win").sum() / len(s) if len(s) else float("nan")

def _agg_wr(df, col, observed=True) -> pd.DataFrame:
    return (
        df.groupby(col, observed=observed)
        .agg(games=("result","count"), wins=("result", lambda x:(x=="win").sum()))
        .assign(win_rate=lambda d: d["wins"]/d["games"])
        .reset_index()
    )

def _rdylgn_bar(df, x, y="win_rate", xlabel=None, height=360, x_dtick=None):
    fig = px.bar(df, x=x, y=y, text_auto=".1%",
                 color=y, color_continuous_scale="RdYlGn", range_color=[0.3, 0.7],
                 labels={y:"Win Rate", x: xlabel or x}, height=height)
    fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%", yaxis_range=[0,1])
    if x_dtick:
        fig.update_xaxes(dtick=x_dtick)
    return fig

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM games", con)
    con.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    hour_utc = df["pgn"].str.extract(r'\[UTCTime "(\d+):\d+:\d+"', expand=False).astype(float)
    df["hour_est"] = (hour_utc + EST_OFFSET) % 24

    df["day_of_week"]    = pd.Categorical(df["date"].dt.day_name(), categories=DAYS_ORDER, ordered=True)
    df["opening_label"]  = df["opening"].where(df["opening"].str.len() > 0, df["eco"])
    df["termination_clean"] = df["termination"].apply(_norm_term)
    df["game_phase"]     = pd.Categorical(df["num_moves"].apply(_phase), categories=PHASE_ORDER, ordered=True)
    df["eco_family"]     = df["eco"].str[0].map(ECO_LABELS).fillna("Unknown")
    df["month"]          = df["date"].dt.to_period("M").dt.to_timestamp()

    # Accuracy columns are added by analyze_games.py; ensure they exist even if
    # that script hasn't been run yet so the Accuracy tab doesn't crash.
    for col in ["accuracy_me","accuracy_opp","acpl_me","acpl_opp",
                "blunders_me","mistakes_me","inaccuracies_me"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df[df["result"].isin(["win","loss","draw"])].copy()


# ── Coaching tips ──────────────────────────────────────────────────────────────
def coaching_tips(fdf: pd.DataFrame, section: str) -> list[str]:
    n = len(fdf)
    if n < 5:
        return ["Not enough games in this filter to generate tips."]

    wr   = (fdf["result"] == "win").mean()
    wr_w = _wr(fdf.loc[fdf["color"]=="white","result"])
    wr_b = _wr(fdf.loc[fdf["color"]=="black","result"])
    tips = []

    if section == "overview":
        tips.append(f"Overall win rate **{wr:.1%}** across **{n:,}** games.")
        diff = abs(wr_w - wr_b)
        worse = "Black" if wr_w > wr_b else "White"
        if diff > 0.03:
            tips.append(f"You score **{wr_w:.1%} as White** vs **{wr_b:.1%} as Black** — a {diff:.0%} gap. Focus your opening study on **{worse}**.")
        top_loss_term = fdf.loc[fdf["result"]=="loss","termination_clean"].value_counts()
        if len(top_loss_term):
            tips.append(f"Most common way to lose: **{top_loss_term.index[0]}** ({top_loss_term.iloc[0]} games). Targeted practice here yields the fastest gains.")
        recent = fdf[fdf["date"] >= fdf["date"].max() - pd.Timedelta(days=30)]
        if len(recent) >= 10:
            rwr = (recent["result"]=="win").mean()
            arrow = "📈" if rwr > wr else "📉"
            tips.append(f"Last-30-day win rate: **{rwr:.1%}** {arrow} (all-time avg {wr:.1%}).")

    elif section == "rating":
        for tc in sorted(fdf["time_class"].unique()):
            sub = fdf[fdf["time_class"]==tc].sort_values("date")
            if len(sub) < 20: continue
            peak = sub["my_rating"].max()
            curr = sub["my_rating"].iloc[-1]
            sign = "+" if curr >= peak else ""
            tips.append(f"**{tc.capitalize()}** — peak {peak}, current {curr} ({sign}{curr-peak} from peak).")
        opp_all  = fdf["opp_rating"].mean()
        opp_100  = fdf.sort_values("date").tail(100)["opp_rating"].mean()
        direction = "stronger" if opp_100 > opp_all else "weaker"
        tips.append(f"Avg opponent (all-time): **{opp_all:.0f}** | last 100 games: **{opp_100:.0f}** — playing {direction} opposition recently.")

    elif section == "openings":
        op = _agg_wr(fdf, "opening_label")
        op10 = op[op["games"] >= 10]
        if len(op10):
            best = op10.loc[op10["win_rate"].idxmax()]
            worst = op10.loc[op10["win_rate"].idxmin()]
            tips.append(f"Best opening (≥10 games): **{best['opening_label']}** — {best['win_rate']:.1%} in {best['games']:.0f} games.")
            tips.append(f"Worst opening (≥10 games): **{worst['opening_label']}** — {worst['win_rate']:.1%} in {worst['games']:.0f} games. Consider replacing it.")
        top_op = fdf["opening_label"].value_counts().index[0]
        top_wr_val = op.loc[op["opening_label"]==top_op,"win_rate"].values
        if len(top_wr_val):
            verdict = "keep playing it" if top_wr_val[0] > 0.50 else "below 50% — study alternatives"
            tips.append(f"Most played opening **{top_op}**: {top_wr_val[0]:.1%} win rate → {verdict}.")
        tips.append(f"White {wr_w:.1%} vs Black {wr_b:.1%} — opening prep is {'stronger as White' if wr_w > wr_b else 'stronger as Black'}.")

    elif section == "splits":
        hr = _agg_wr(fdf.dropna(subset=["hour_est"]), "hour_est")
        if len(hr) >= 3:
            bh = int(hr.loc[hr["win_rate"].idxmax(),"hour_est"])
            wh = int(hr.loc[hr["win_rate"].idxmin(),"hour_est"])
            tips.append(f"Best time to play: **{_fmt_h(bh)} EST** ({hr['win_rate'].max():.1%} win rate). Avoid **{_fmt_h(wh)} EST** ({hr['win_rate'].min():.1%}).")
        dow = _agg_wr(fdf, "day_of_week")
        if len(dow):
            bd = dow.loc[dow["win_rate"].idxmax(),"day_of_week"]
            tips.append(f"Best day: **{bd}** ({dow['win_rate'].max():.1%}). Schedule important sessions then.")
        above = fdf[fdf["opp_rating"] > fdf["my_rating"]]
        below = fdf[fdf["opp_rating"] <= fdf["my_rating"]]
        if len(above) >= 10 and len(below) >= 10:
            wr_up = (above["result"]=="win").mean()
            wr_dn = (below["result"]=="win").mean()
            tips.append(f"vs higher-rated: **{wr_up:.1%}** | vs lower-rated: **{wr_dn:.1%}**. {'Good upset ability!' if wr_up > 0.35 else 'Room to improve against stronger players.'}")
        phase_wr = _agg_wr(fdf, "game_phase")
        if len(phase_wr):
            weak = phase_wr.loc[phase_wr["win_rate"].idxmin(),"game_phase"]
            tips.append(f"Weakest game phase: **{weak}** ({phase_wr['win_rate'].min():.1%}). This is where to invest study time.")

    elif section == "games":
        avg_m = fdf["num_moves"].mean()
        tips.append(f"Average game length: **{avg_m:.0f} moves**.")
        tl_rate = ((fdf["result"]=="loss") & (fdf["termination_clean"]=="on time")).mean()
        if tl_rate > 0.10:
            tips.append(f"**{tl_rate:.0%}** of all games are time losses. Try increment time controls (e.g. 3+2 or 5+5) to reduce flagging.")
        short_w = ((fdf["num_moves"]<=10) & (fdf["result"]=="win")).sum()
        short_l = ((fdf["num_moves"]<=10) & (fdf["result"]=="loss")).sum()
        tips.append(f"Games ≤10 moves: **{short_w} wins, {short_l} losses** — {'solid early game' if short_w > short_l else 'watch out for opening traps'}.")

    elif section == "accuracy":
        tips.append("Chess.com accuracy scores require **manual game analysis** on the website. The API only returns raw PGNs without engine evaluation.")
        tips.append("To unlock accuracy: open any game on chess.com → click **Analysis** → **Request Analysis**. Re-run `fetch_games.py` after to pick up the updated PGNs.")
        tips.append("As a proxy, games won by **checkmate** suggest tactical accuracy; games lost **on time** suggest clock management issues rather than move quality.")
        tl_rate = ((fdf["result"]=="loss") & (fdf["termination_clean"]=="on time")).mean()
        tips.append(f"You flag in **{tl_rate:.1%}** of games — time pressure is your biggest diagnosable issue without engine data.")

    return tips


def show_tips(tips: list[str]) -> None:
    with st.expander("💡 Coaching tips", expanded=False):
        for t in tips:
            st.markdown(f"- {t}")


# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chess Dashboard", page_icon="♟", layout="wide")
df = load_data()

with st.sidebar:
    st.title("♟ Chess Dashboard")
    st.caption(f"Player: **{USERNAME}**  |  {len(df):,} games")
    st.divider()
    all_classes = sorted(df["time_class"].dropna().unique())
    sel_classes = st.multiselect("Time class", all_classes, default=all_classes)
    d_min, d_max = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("Date range", value=(d_min, d_max))
    st.divider()
    if st.button("↺ Refresh data", help="Clears the cache and reloads from chess.db — use after running analyze_games.py"):
        st.cache_data.clear()
        st.rerun()

mask = df["time_class"].isin(sel_classes)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    mask &= (df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))
fdf = df[mask]

# ── Tabs ───────────────────────────────────────────────────────────────────────
T = st.tabs(["Overview", "Rating History", "Openings", "Performance Splits",
             "Games & Terminations", "Stockfish Accuracy"])
t_ov, t_rt, t_op, t_sp, t_gm, t_ac = T


# ════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
with t_ov:
    total = len(fdf)
    wins  = (fdf["result"]=="win").sum()
    losses= (fdf["result"]=="loss").sum()
    draws = (fdf["result"]=="draw").sum()
    wr    = wins/total if total else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Games", f"{total:,}")
    c2.metric("Win Rate",    f"{wr:.1%}")
    c3.metric("Wins",        f"{wins:,}")
    c4.metric("Losses / Draws", f"{losses:,} / {draws:,}")

    show_tips(coaching_tips(fdf, "overview"))

    st.subheader("Current Rating by Time Class")
    latest = fdf.sort_values("date").groupby("time_class", observed=True).last()["my_rating"]
    for col,(tc,r) in zip(st.columns(max(len(latest),1)), latest.items()):
        col.metric(str(tc).capitalize(), int(r))

    st.subheader("Monthly Win Rate")
    monthly = (
        fdf.groupby("month", observed=True)
        .agg(games=("result","count"), wins=("result", lambda x:(x=="win").sum()))
        .assign(win_rate=lambda d: d["wins"]/d["games"])
        .reset_index()
    )
    fig = px.bar(monthly, x="month", y="win_rate", color="win_rate",
                 color_continuous_scale="RdYlGn", range_color=[0.3,0.7],
                 labels={"month":"Month","win_rate":"Win Rate"})
    fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Games Played per Day")
    st.caption("Daily game count with 7-day rolling average — shows activity patterns and volume trends.")
    daily = (
        fdf.groupby("date", observed=True)
        .agg(games=("result","count"))
        .reset_index()
        .sort_values("date")
    )
    daily["roll7"] = daily["games"].rolling(7, min_periods=1).mean()
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily["date"], y=daily["games"],
        name="Daily games", marker_color="rgba(99,110,250,0.35)",
    ))
    fig_daily.add_trace(go.Scatter(
        x=daily["date"], y=daily["roll7"],
        mode="lines", name="7-day avg",
        line=dict(color="#ef553b", width=2),
    ))
    fig_daily.update_layout(
        height=320, xaxis_title="Date", yaxis_title="Games",
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — RATING HISTORY
# ════════════════════════════════════════════════════════════════
with t_rt:
    st.header("Rating History")
    show_tips(coaching_tips(fdf, "rating"))

    # My rating
    st.subheader("My Rating Over Time")
    fig_my = go.Figure()
    pal = px.colors.qualitative.Plotly
    for i, tc in enumerate(sorted(fdf["time_class"].unique())):
        sub = fdf[fdf["time_class"]==tc].sort_values("date")
        c = pal[i % len(pal)]
        roll = sub["my_rating"].rolling(20, min_periods=5).mean()
        fig_my.add_trace(go.Scatter(x=sub["date"], y=sub["my_rating"],
            mode="markers", marker=dict(size=3, color=c, opacity=0.3),
            name=f"{tc} (raw)", legendgroup=tc))
        fig_my.add_trace(go.Scatter(x=sub["date"], y=roll,
            mode="lines", line=dict(width=2.5, color=c),
            name=f"{tc} (20-game avg)", legendgroup=tc))
    fig_my.update_layout(height=420, xaxis_title="Date", yaxis_title="Rating",
                         hovermode="x unified")
    st.plotly_chart(fig_my, use_container_width=True)

    # Opponent ELO trends
    st.subheader("Opponent Rating Trend")
    st.caption("30-game rolling average of opponent ratings — rising means you're facing stronger opposition.")
    opp_sub = fdf.dropna(subset=["opp_rating"]).sort_values("date")
    fig_opp = go.Figure()
    for i, tc in enumerate(sorted(opp_sub["time_class"].unique())):
        sub = opp_sub[opp_sub["time_class"]==tc]
        c = pal[i % len(pal)]
        roll = sub["opp_rating"].rolling(30, min_periods=10).mean()
        fig_opp.add_trace(go.Scatter(x=sub["date"], y=sub["opp_rating"],
            mode="markers", marker=dict(size=3, color=c, opacity=0.2),
            name=f"{tc} (raw)", legendgroup=tc))
        fig_opp.add_trace(go.Scatter(x=sub["date"], y=roll,
            mode="lines", line=dict(width=2.5, color=c),
            name=f"{tc} (30-game avg)", legendgroup=tc))
    fig_opp.update_layout(height=400, xaxis_title="Date", yaxis_title="Opponent Rating",
                          hovermode="x unified")
    st.plotly_chart(fig_opp, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — OPENINGS
# ════════════════════════════════════════════════════════════════
with t_op:
    st.header("Openings")
    show_tips(coaching_tips(fdf, "openings"))

    # Choice distribution
    st.subheader("Opening Choice Distribution")
    c_left, c_right = st.columns([2, 1])

    with c_left:
        top15 = fdf["opening_label"].value_counts().head(15).reset_index()
        top15.columns = ["Opening", "Games"]
        top15["Win Rate"] = top15["Opening"].map(
            fdf.groupby("opening_label", observed=True)["result"].apply(_wr)
        )
        fig_dist = px.bar(top15, x="Games", y="Opening", orientation="h",
                          color="Win Rate", color_continuous_scale="RdYlGn",
                          range_color=[0.3,0.7], text="Games",
                          labels={"Opening":"","Games":"Games Played"},
                          title="Top 15 Most Played Openings")
        fig_dist.update_layout(yaxis_categoryorder="total ascending",
                               coloraxis_colorbar_title="Win %",
                               height=480)
        fig_dist.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_right:
        eco_counts = fdf["eco_family"].value_counts().reset_index()
        eco_counts.columns = ["ECO Family", "Games"]
        fig_eco = px.pie(eco_counts, names="ECO Family", values="Games",
                         title="Games by ECO Family", hole=0.4)
        fig_eco.update_traces(textposition="inside", textinfo="percent+label")
        fig_eco.update_layout(showlegend=False, height=480)
        st.plotly_chart(fig_eco, use_container_width=True)

    # Performance table
    st.subheader("Opening Performance Table")
    min_games = st.slider("Minimum games", 5, 50, 10, key="op_min")

    overall = (fdf.groupby("opening_label", observed=True)["result"]
               .agg(games="count", wins=lambda x:(x=="win").sum())
               .assign(win_rate=lambda d: d["wins"]/d["games"]))
    white_wr = (fdf[fdf["color"]=="white"]
                .groupby("opening_label", observed=True)["result"]
                .apply(_wr).rename("wr_white"))
    black_wr = (fdf[fdf["color"]=="black"]
                .groupby("opening_label", observed=True)["result"]
                .apply(_wr).rename("wr_black"))

    op_table = (overall[["games","win_rate"]]
                .join(white_wr).join(black_wr).reset_index()
                .rename(columns={"opening_label":"Opening"}))
    op_table = op_table[op_table["games"] >= min_games].sort_values("games", ascending=False)
    op_table["games"] = op_table["games"].astype(int)

    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        op_table.assign(
            win_rate=op_table["win_rate"]*100,
            wr_white=op_table["wr_white"]*100,
            wr_black=op_table["wr_black"]*100,
        ).rename(columns={"win_rate":"Win %","wr_white":"Win % White","wr_black":"Win % Black"}),
        column_config={"Win %":pct,"Win % White":pct,"Win % Black":pct},
        use_container_width=True, hide_index=True,
    )

    # Game-phase performance by opening
    st.subheader("Opening → Phase Reached Performance")
    st.caption("Groups games by the phase they ended in — a proxy for where things go wrong or right.")
    phase_perf = (
        fdf.groupby("game_phase", observed=True)
        .agg(games=("result","count"),
             wins=("result", lambda x:(x=="win").sum()),
             losses=("result", lambda x:(x=="loss").sum()),
             draws=("result", lambda x:(x=="draw").sum()))
        .assign(win_rate=lambda d: d["wins"]/d["games"])
        .reset_index()
    )
    fig_phase = go.Figure()
    for outcome, color in [("wins","#2ecc71"),("losses","#e74c3c"),("draws","#3498db")]:
        fig_phase.add_trace(go.Bar(
            x=phase_perf["game_phase"], y=phase_perf[outcome],
            name=outcome.capitalize(), marker_color=color,
        ))
    fig_phase.update_layout(barmode="group", xaxis_title="Game Phase",
                            yaxis_title="Games", height=340)
    st.plotly_chart(fig_phase, use_container_width=True)

    wr_phase = phase_perf.set_index("game_phase")["win_rate"]
    c1,c2,c3 = st.columns(3)
    for col, ph in zip([c1,c2,c3], PHASE_ORDER):
        if ph in wr_phase:
            col.metric(ph, f"{wr_phase[ph]:.1%} win rate")


# ════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE SPLITS
# ════════════════════════════════════════════════════════════════
with t_sp:
    st.header("Performance Splits")
    show_tips(coaching_tips(fdf, "splits"))

    c_l, c_r = st.columns(2)

    # Opponent rating bucket
    with c_l:
        st.subheader("vs. Opponent Rating")
        bkt = fdf.dropna(subset=["opp_rating"]).copy()
        bkt["bucket"] = (bkt["opp_rating"]//100*100).astype(int)
        bk_stats = _agg_wr(bkt, "bucket")
        bk_stats["label"] = bk_stats["bucket"].astype(str) + "–" + (bk_stats["bucket"]+99).astype(str)
        st.plotly_chart(_rdylgn_bar(bk_stats,"label", xlabel="Opponent Rating"), use_container_width=True)

    # Color
    with c_r:
        st.subheader("By Color")
        cs = _agg_wr(fdf, "color")
        fig_c = px.bar(cs, x="color", y="win_rate", text_auto=".1%",
                       color="color",
                       color_discrete_map={"white":"#f0d9b5","black":"#6e4f2b"},
                       labels={"color":"Color","win_rate":"Win Rate"}, height=360)
        fig_c.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0,1])
        st.plotly_chart(fig_c, use_container_width=True)

    # Day of week
    c_l2, c_r2 = st.columns(2)
    with c_l2:
        st.subheader("By Day of Week")
        dow = _agg_wr(fdf, "day_of_week").sort_values("day_of_week")
        st.plotly_chart(_rdylgn_bar(dow, "day_of_week", xlabel="Day"), use_container_width=True)

    # Hour — game count AND win rate side by side
    with c_r2:
        st.subheader("By Hour of Day (EST)")
        hr_df = fdf.dropna(subset=["hour_est"]).copy()
        hr_df["hour_int"] = hr_df["hour_est"].astype(int)
        hr_stats = _agg_wr(hr_df, "hour_int")
        hr_stats["label"] = hr_stats["hour_int"].map(_fmt_h)
        hr_stats = hr_stats.sort_values("hour_int")

        hi1, hi2 = st.tabs(["Win Rate", "Games Played"])
        with hi1:
            fig_hr_wr = _rdylgn_bar(hr_stats, "label", xlabel="Hour (EST)", x_dtick=1)
            st.plotly_chart(fig_hr_wr, use_container_width=True)
        with hi2:
            fig_hr_ct = px.bar(hr_stats, x="label", y="games",
                               color="games", color_continuous_scale="Blues",
                               labels={"label":"Hour (EST)","games":"Games Played"}, height=360)
            fig_hr_ct.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_hr_ct, use_container_width=True)

    # Game phase performance
    st.subheader("Win Rate by Game Phase")
    st.caption("Phases defined by game length (proxy — engine analysis not available).")
    phase_wr = _agg_wr(fdf, "game_phase").sort_values("game_phase")
    phase_wr["games_label"] = phase_wr["games"].astype(str) + " games"
    fig_ph = _rdylgn_bar(phase_wr, "game_phase", xlabel="Phase", height=300)
    for _, row in phase_wr.iterrows():
        fig_ph.add_annotation(x=row["game_phase"], y=row["win_rate"]+0.04,
                              text=f"{row['games']:,} games", showarrow=False, font_size=11)
    st.plotly_chart(fig_ph, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 5 — GAMES & TERMINATIONS
# ════════════════════════════════════════════════════════════════
with t_gm:
    st.header("Games & Terminations")
    show_tips(coaching_tips(fdf, "games"))

    result_filter = st.multiselect(
        "Filter by result", ["win","loss","draw"],
        default=["win","loss","draw"], key="gm_filter"
    )
    gdf = fdf[fdf["result"].isin(result_filter)] if result_filter else fdf

    c_l, c_r = st.columns(2)

    with c_l:
        st.subheader("Move Count Distribution")
        fig_mv = px.histogram(gdf, x="num_moves", nbins=60,
                              color="result",
                              color_discrete_map={"win":"#2ecc71","loss":"#e74c3c","draw":"#3498db"},
                              labels={"num_moves":"Moves","count":"Games","result":"Result"},
                              barmode="overlay", opacity=0.7)
        fig_mv.update_layout(yaxis_title="Games", bargap=0.05)
        st.plotly_chart(fig_mv, use_container_width=True)

    with c_r:
        st.subheader("Termination Reasons")
        term = (gdf["termination_clean"].value_counts()
                .reset_index()
                .rename(columns={"termination_clean":"Termination","count":"Games"}))
        fig_term = px.bar(term, x="Games", y="Termination", orientation="h",
                          color="Games", color_continuous_scale="Blues")
        fig_term.update_layout(coloraxis_showscale=False,
                               yaxis_categoryorder="total ascending", height=440)
        st.plotly_chart(fig_term, use_container_width=True)

    # Termination × result heatmap
    st.subheader("Termination Breakdown by Result")
    heat = (gdf.groupby(["termination_clean","result"], observed=True)
              .size().unstack(fill_value=0).reset_index())
    heat = heat.rename(columns={"termination_clean":"Termination"})
    heat = heat.sort_values("Termination")
    pct_cols = {c: st.column_config.NumberColumn(c.capitalize(), format="%d") for c in ["win","loss","draw"] if c in heat.columns}
    st.dataframe(heat, column_config=pct_cols, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 6 — STOCKFISH ACCURACY
# ════════════════════════════════════════════════════════════════
with t_ac:
    st.header("Stockfish Accuracy")

    adf = fdf.dropna(subset=["accuracy_me"]).copy()
    n_analyzed = len(adf)
    n_total    = len(fdf)

    st.caption(
        f"**{n_analyzed:,} / {n_total:,}** games analyzed with Stockfish 17 (depth 16). "
        f"Run `python analyze_games.py --all` to analyze the full archive."
    )

    if n_analyzed == 0:
        st.warning(
            "No Stockfish data in the database yet.  \n"
            "Run: `python analyze_games.py --stockfish stockfish/stockfish.exe`",
            icon="⚠️"
        )
    else:
        # ── Coaching tips for accuracy ────────────────────────────────────
        acc_tips = []
        avg_acc  = adf["accuracy_me"].mean()
        avg_acpl = adf["acpl_me"].mean()
        avg_blun = adf["blunders_me"].mean()
        acc_tips.append(f"Average accuracy: **{avg_acc:.1f}%** | Average ACPL: **{avg_acpl:.1f}** | Avg blunders/game: **{avg_blun:.2f}**")
        win_acc  = adf.loc[adf["result"]=="win",  "accuracy_me"].mean()
        loss_acc = adf.loc[adf["result"]=="loss", "accuracy_me"].mean()
        if not pd.isna(win_acc) and not pd.isna(loss_acc):
            acc_tips.append(f"You play at **{win_acc:.1f}%** accuracy in wins vs **{loss_acc:.1f}%** in losses — a {win_acc-loss_acc:.1f}-point gap.")
        high_acc = adf[adf["accuracy_me"] >= 85]
        if len(high_acc):
            acc_tips.append(f"In **{len(high_acc)}** games where your accuracy ≥ 85%: win rate = **{(high_acc['result']=='win').mean():.1%}**.")
        top_blun = adf.nlargest(1, "blunders_me").iloc[0]
        acc_tips.append(f"Your worst game had **{int(top_blun['blunders_me'])} blunders** — study that game to understand the pattern.")
        show_tips(acc_tips)

        # ── Big metrics ───────────────────────────────────────────────────
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Avg Accuracy",  f"{avg_acc:.1f}%")
        m2.metric("Avg ACPL",      f"{avg_acpl:.1f}")
        m3.metric("Avg Blunders",  f"{avg_blun:.2f}")
        m4.metric("Avg Mistakes",  f"{adf['mistakes_me'].mean():.2f}")
        m5.metric("Avg Inaccuracies", f"{adf['inaccuracies_me'].mean():.2f}")

        st.divider()

        # ── Row 1: accuracy distribution + accuracy by result ─────────────
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Accuracy Distribution")
            fig_hist = px.histogram(
                adf, x="accuracy_me", nbins=40,
                color="result",
                color_discrete_map={"win":"#2ecc71","loss":"#e74c3c","draw":"#3498db"},
                labels={"accuracy_me":"My Accuracy (%)","count":"Games","result":"Result"},
                barmode="overlay", opacity=0.75,
            )
            fig_hist.update_layout(yaxis_title="Games", xaxis_range=[0,100])
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.subheader("Accuracy by Result")
            fig_box = px.box(
                adf, x="result", y="accuracy_me",
                color="result",
                color_discrete_map={"win":"#2ecc71","loss":"#e74c3c","draw":"#3498db"},
                labels={"result":"Result","accuracy_me":"My Accuracy (%)"},
                category_orders={"result":["win","draw","loss"]},
                points="outliers",
            )
            fig_box.update_layout(showlegend=False, yaxis_range=[0,100])
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Row 2: me vs opponent + ACPL by result ────────────────────────
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("My Accuracy vs Opponent")
            st.caption("Each point is one game — dots above the diagonal mean you outplayed your opponent.")
            fig_scatter = px.scatter(
                adf, x="accuracy_opp", y="accuracy_me",
                color="result",
                color_discrete_map={"win":"#2ecc71","loss":"#e74c3c","draw":"#3498db"},
                labels={"accuracy_opp":"Opponent Accuracy (%)","accuracy_me":"My Accuracy (%)","result":"Result"},
                opacity=0.7,
            )
            # Diagonal reference line
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                                  line=dict(dash="dash", color="grey", width=1))
            fig_scatter.update_layout(xaxis_range=[0,100], yaxis_range=[0,100])
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c4:
            st.subheader("Blunders / Mistakes / Inaccuracies")
            err_data = pd.DataFrame({
                "Category": ["Blunders","Mistakes","Inaccuracies"],
                "Per Game":  [
                    adf["blunders_me"].mean(),
                    adf["mistakes_me"].mean(),
                    adf["inaccuracies_me"].mean(),
                ],
            })
            fig_err = px.bar(
                err_data, x="Category", y="Per Game", text_auto=".2f",
                color="Category",
                color_discrete_sequence=["#e74c3c","#e67e22","#f1c40f"],
                labels={"Per Game":"Avg per Game"},
            )
            fig_err.update_layout(showlegend=False)
            st.plotly_chart(fig_err, use_container_width=True)

        # ── Row 3: accuracy vs opp rating + accuracy over time ────────────
        c5, c6 = st.columns(2)

        with c5:
            st.subheader("Accuracy vs Opponent Rating")
            st.caption("Are you more/less accurate against stronger players?")
            sub = adf.dropna(subset=["opp_rating"]).copy()
            sub["opp_bucket"] = (sub["opp_rating"] // 100 * 100).astype(int)
            bkt_acc = (sub.groupby("opp_bucket")
                       .agg(games=("accuracy_me","count"), avg_acc=("accuracy_me","mean"))
                       .reset_index())
            bkt_acc["label"] = bkt_acc["opp_bucket"].astype(str) + "–" + (bkt_acc["opp_bucket"]+99).astype(str)
            fig_opp_acc = px.bar(
                bkt_acc, x="label", y="avg_acc", text_auto=".1f",
                color="avg_acc", color_continuous_scale="RdYlGn", range_color=[50, 90],
                labels={"label":"Opponent Rating","avg_acc":"Avg Accuracy (%)"},
            )
            fig_opp_acc.update_layout(coloraxis_showscale=False, yaxis_range=[0,100])
            st.plotly_chart(fig_opp_acc, use_container_width=True)

        with c6:
            st.subheader("Accuracy Over Time")
            st.caption("10-game rolling average of your accuracy — shows improvement trends.")
            acc_time = adf.sort_values("date").copy()
            acc_time["roll10"] = acc_time["accuracy_me"].rolling(10, min_periods=3).mean()
            fig_at = go.Figure()
            fig_at.add_trace(go.Scatter(
                x=acc_time["date"], y=acc_time["accuracy_me"],
                mode="markers", marker=dict(size=4, color="#636efa", opacity=0.3),
                name="Per game",
            ))
            fig_at.add_trace(go.Scatter(
                x=acc_time["date"], y=acc_time["roll10"],
                mode="lines", line=dict(color="#ef553b", width=2.5),
                name="10-game avg",
            ))
            fig_at.update_layout(
                height=360, yaxis_range=[0, 100],
                xaxis_title="Date", yaxis_title="Accuracy (%)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_at, use_container_width=True)

        # ── Full game table ───────────────────────────────────────────────
        st.subheader("Analyzed Games")
        tbl = (adf[["date","color","opponent","opp_rating","my_rating","result",
                     "time_class","accuracy_me","accuracy_opp","acpl_me",
                     "blunders_me","mistakes_me","inaccuracies_me","opening"]]
               .sort_values("date", ascending=False)
               .rename(columns={
                   "accuracy_me":"Acc Me","accuracy_opp":"Acc Opp",
                   "acpl_me":"ACPL","blunders_me":"Blunders",
                   "mistakes_me":"Mistakes","inaccuracies_me":"Inaccuracies",
               }))
        pct2 = st.column_config.NumberColumn(format="%.1f")
        st.dataframe(
            tbl,
            column_config={"Acc Me": pct2, "Acc Opp": pct2, "ACPL": pct2},
            use_container_width=True, hide_index=True,
        )
