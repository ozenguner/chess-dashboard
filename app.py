import pathlib
import re
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Chess Dashboard", page_icon="♟", layout="wide")

DB_PATH = pathlib.Path("data/chess.db")

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _normalize_termination(t: str) -> str:
    t = t.lower()
    if "checkmate" in t:
        return "checkmate"
    if "timeout vs insufficient" in t or "time vs insufficient" in t:
        return "time vs insufficient"
    if "on time" in t:
        return "on time"
    if "resign" in t:
        return "resignation"
    if "stalemate" in t:
        return "stalemate"
    if "repetition" in t:
        return "repetition"
    if "agreement" in t:
        return "agreed"
    if "insufficient" in t:
        return "insufficient material"
    if "50" in t:
        return "50-move rule"
    if "abandon" in t:
        return "abandoned"
    if "3rd time" in t or "three" in t:
        return "three-check"
    return t


@st.cache_data
def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM games", con)
    con.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = (
        df["pgn"].str.extract(r'\[UTCTime "(\d+):\d+:\d+"', expand=False).astype(float)
    )
    df["day_of_week"] = pd.Categorical(
        df["date"].dt.day_name(), categories=DAYS_ORDER, ordered=True
    )
    # Use opening name when present, fall back to ECO code
    df["opening_label"] = df["opening"].where(df["opening"].str.len() > 0, df["eco"])
    df["termination_clean"] = df["termination"].apply(_normalize_termination)
    # Exclude non-standard variants from win-rate calculations
    df = df[df["result"].isin(["win", "loss", "draw"])].copy()
    return df


df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Filters")
    all_classes = sorted(df["time_class"].dropna().unique())
    selected_classes = st.multiselect("Time class", all_classes, default=all_classes)
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date))

mask = df["time_class"].isin(selected_classes)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    mask &= (df["date"] >= pd.Timestamp(date_range[0])) & (
        df["date"] <= pd.Timestamp(date_range[1])
    )
fdf = df[mask]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_rating, tab_openings, tab_splits, tab_games = st.tabs(
    ["Overview", "Rating History", "Openings", "Performance Splits", "Games & Terminations"]
)

# ── Overview ─────────────────────────────────────────────────────────────────
with tab_overview:
    total = len(fdf)
    wins = (fdf["result"] == "win").sum()
    losses = (fdf["result"] == "loss").sum()
    draws = (fdf["result"] == "draw").sum()
    win_rate = wins / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Games", f"{total:,}")
    c2.metric("Win Rate", f"{win_rate:.1%}")
    c3.metric("Wins", f"{wins:,}")
    c4.metric("Losses / Draws", f"{losses:,} / {draws:,}")

    st.subheader("Current Rating by Time Class")
    latest = (
        fdf.sort_values("date").groupby("time_class", observed=True).last()["my_rating"]
    )
    rating_cols = st.columns(max(len(latest), 1))
    for col, (tc, rating) in zip(rating_cols, latest.items()):
        col.metric(tc.capitalize(), int(rating))

    st.divider()

    st.subheader("Win Rate Over Time (monthly)")
    monthly = (
        fdf.assign(month=fdf["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", observed=True)
        .agg(games=("result", "count"), wins=("result", lambda x: (x == "win").sum()))
        .reset_index()
    )
    monthly["win_rate"] = monthly["wins"] / monthly["games"]
    fig_monthly = px.bar(
        monthly, x="month", y="win_rate",
        labels={"month": "Month", "win_rate": "Win Rate"},
        color="win_rate", color_continuous_scale="RdYlGn", range_color=[0.3, 0.7],
    )
    fig_monthly.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig_monthly, use_container_width=True)


# ── Rating History ────────────────────────────────────────────────────────────
with tab_rating:
    st.header("Rating History")

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for i, tc in enumerate(sorted(fdf["time_class"].unique())):
        sub = fdf[fdf["time_class"] == tc].sort_values("date")
        color = palette[i % len(palette)]
        roll = sub["my_rating"].rolling(20, min_periods=5).mean()

        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["my_rating"],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.3),
            name=f"{tc} (raw)", legendgroup=tc, showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=sub["date"], y=roll,
            mode="lines",
            line=dict(width=2.5, color=color),
            name=f"{tc} (20-game avg)", legendgroup=tc,
        ))

    fig.update_layout(
        height=520, xaxis_title="Date", yaxis_title="Rating",
        hovermode="x unified", legend_title="Time Class",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Openings ──────────────────────────────────────────────────────────────────
with tab_openings:
    st.header("Openings")
    min_games = st.slider("Minimum games to show", 5, 50, 10)

    def _wr(s: pd.Series) -> float:
        return (s == "win").sum() / len(s) if len(s) else float("nan")

    overall = (
        fdf.groupby("opening_label", observed=True)["result"]
        .agg(games="count", wins=lambda x: (x == "win").sum())
        .assign(win_rate=lambda d: d["wins"] / d["games"])
    )
    white_wr = (
        fdf[fdf["color"] == "white"]
        .groupby("opening_label", observed=True)["result"]
        .apply(_wr)
        .rename("win_rate_white")
    )
    black_wr = (
        fdf[fdf["color"] == "black"]
        .groupby("opening_label", observed=True)["result"]
        .apply(_wr)
        .rename("win_rate_black")
    )

    op = (
        overall[["games", "win_rate"]]
        .join(white_wr)
        .join(black_wr)
        .reset_index()
        .rename(columns={"opening_label": "Opening"})
    )
    op = op[op["games"] >= min_games].sort_values("games", ascending=False)
    op["games"] = op["games"].astype(int)

    # Keep floats for sortable columns; format via column_config
    pct_col = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        op.assign(
            win_rate=op["win_rate"] * 100,
            win_rate_white=op["win_rate_white"] * 100,
            win_rate_black=op["win_rate_black"] * 100,
        ).rename(columns={
            "win_rate": "Win %",
            "win_rate_white": "Win % White",
            "win_rate_black": "Win % Black",
        }),
        column_config={"Win %": pct_col, "Win % White": pct_col, "Win % Black": pct_col},
        use_container_width=True,
        hide_index=True,
    )


# ── Performance Splits ────────────────────────────────────────────────────────
with tab_splits:
    st.header("Performance Splits")

    def bar_winrate(data, x, y="win_rate", xlabel=None, **kwargs):
        fig = px.bar(
            data, x=x, y=y, text_auto=".1%",
            labels={y: "Win Rate", x: xlabel or x},
            color=y, color_continuous_scale="RdYlGn", range_color=[0.3, 0.7],
            **kwargs,
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
        return fig

    def agg_winrate(grp_col, df_=fdf, observed=True):
        return (
            df_.groupby(grp_col, observed=observed)
            .agg(games=("result", "count"), wins=("result", lambda x: (x == "win").sum()))
            .assign(win_rate=lambda d: d["wins"] / d["games"])
            .reset_index()
        )

    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("vs. Opponent Rating Bucket")
        sub = fdf.dropna(subset=["opp_rating"]).copy()
        sub["bucket"] = (sub["opp_rating"] // 100 * 100).astype(int)
        bkt = agg_winrate("bucket", df_=sub)
        bkt["label"] = bkt["bucket"].astype(str) + "–" + (bkt["bucket"] + 99).astype(str)
        st.plotly_chart(bar_winrate(bkt, "label", xlabel="Opponent Rating"), use_container_width=True)

    with c_right:
        st.subheader("By Color")
        color_stats = agg_winrate("color")
        fig_c = px.bar(
            color_stats, x="color", y="win_rate", text_auto=".1%",
            color="color",
            color_discrete_map={"white": "#f0d9b5", "black": "#7a5230"},
            labels={"color": "Color", "win_rate": "Win Rate"},
        )
        fig_c.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig_c, use_container_width=True)

    c_left2, c_right2 = st.columns(2)

    with c_left2:
        st.subheader("By Day of Week")
        dow = agg_winrate("day_of_week").sort_values("day_of_week")
        st.plotly_chart(bar_winrate(dow, "day_of_week", xlabel="Day"), use_container_width=True)

    with c_right2:
        st.subheader("By Hour of Day (UTC)")
        hour_df = fdf.dropna(subset=["hour"])
        hr = agg_winrate("hour", df_=hour_df)
        hr["hour"] = hr["hour"].astype(int)
        fig_h = bar_winrate(hr, "hour", xlabel="Hour (UTC)")
        fig_h.update_layout(xaxis=dict(dtick=1, tickmode="linear"))
        st.plotly_chart(fig_h, use_container_width=True)


# ── Games & Terminations ──────────────────────────────────────────────────────
with tab_games:
    st.header("Games & Terminations")

    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Move Count Distribution")
        fig_moves = px.histogram(
            fdf, x="num_moves", nbins=60,
            color="time_class",
            labels={"num_moves": "Moves", "count": "Games", "time_class": "Time Class"},
            barmode="overlay", opacity=0.75,
        )
        fig_moves.update_layout(yaxis_title="Games", bargap=0.05)
        st.plotly_chart(fig_moves, use_container_width=True)

    with c_right:
        st.subheader("Termination Reasons")
        term = (
            fdf["termination_clean"].value_counts()
            .reset_index()
            .rename(columns={"termination_clean": "Termination", "count": "Games"})
        )
        fig_term = px.bar(
            term, x="Games", y="Termination", orientation="h",
            color="Games", color_continuous_scale="Blues",
        )
        fig_term.update_layout(
            coloraxis_showscale=False,
            yaxis_categoryorder="total ascending",
            height=420,
        )
        st.plotly_chart(fig_term, use_container_width=True)
