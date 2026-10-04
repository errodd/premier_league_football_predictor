"""Leakage-safe team-history feature block.

Single source of truth for the historical block. It lives here, outside the
notebooks, because `01_alcance_y_frontera_eda_vs_preparacion.md` reserves the
construction of derived columns for stage 2 and requires the artifact to be an
executable module rather than code inside an exploratory notebook.

Every value comes from fixtures strictly earlier than the row being built:
`shift(1)` inside each (Team, Season) before the rolling window, so a match
never contributes to its own features and the summer break is never counted as
league congestion.

    from team_history import K, HIST_COLS, build_team_history
"""

import numpy as np
import pandas as pd

# Size of the recent-form window.
K = 5

# 8 team-history metrics. Home/away points-per-game splits were evaluated and
# EXCLUDED: they are a noisy subdivision of the same points signal already
# captured by Hist_Pts_k, with no distinct association (see notebook section 7.3).
HIST_COLS = ["Hist_Pts_k", "Hist_W_k", "Hist_D_k", "Hist_L_k",
             "Hist_GF_k", "Hist_GA_k", "Hist_GD_k", "RestDays"]


def _team_match_long(df):
    """One row per (team, match), with goals/points from that team's view."""
    def stack(is_home):
        tcol, ocol = ("HomeTeam", "AwayTeam") if is_home else ("AwayTeam", "HomeTeam")
        gs, gc = ("FTHG", "FTAG") if is_home else ("FTAG", "FTHG")
        s = df[["Season", "Date_dt", tcol, ocol, gs, gc, "FTR"]].copy()
        s.columns = ["Season", "Date_dt", "Team", "Opponent", "GF", "GA", "FTR"]
        s["is_home"] = is_home
        win = ((s["FTR"] == "H") & is_home) | ((s["FTR"] == "A") & (not is_home))
        s["Pts"] = np.where(s["FTR"] == "D", 1, np.where(win, 3, 0))
        s["W"] = (s["Pts"] == 3).astype(int)
        s["D"] = (s["Pts"] == 1).astype(int)
        s["L"] = (s["Pts"] == 0).astype(int)
        return s
    return (pd.concat([stack(True), stack(False)])
            .sort_values(["Team", "Season", "Date_dt"]).reset_index(drop=True))


def _past_roll(tm, col, how="sum", k=K):
    """shift(1) then rolling(k): strictly historical, reset every season."""
    shifted = tm.groupby(["Team", "Season"])[col].shift(1)
    return (shifted.groupby([tm["Team"], tm["Season"]])
                   .rolling(k, min_periods=1).agg(how)
                   .reset_index(level=[0, 1], drop=True))


def build_team_history(df, k=K, cols=HIST_COLS):
    """Attach Home_/Away_/Diff_ historical columns to a match-level frame.

    Uses only past fixtures within each season. Idempotent: drops its own
    columns before merging, so it can be called repeatedly / in any order.
    """
    n_before = len(df)
    df = df.copy()
    if "Date_dt" not in df.columns:
        df["Date_dt"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    tm = _team_match_long(df)
    for c in ["Pts", "W", "D", "L", "GF", "GA"]:
        tm["Hist_" + c + "_k"] = _past_roll(tm, c, "sum", k)
    tm["Hist_GD_k"] = tm["Hist_GF_k"] - tm["Hist_GA_k"]
    tm["RestDays"] = tm.groupby(["Team", "Season"])["Date_dt"].diff().dt.days

    family = [p + c for p in ("Home_", "Away_", "Diff_") for c in cols]
    df = df.drop(columns=[c for c in family if c in df.columns], errors="ignore")

    home = (tm[tm["is_home"]][["Season", "Date_dt", "Team"] + cols]
            .rename(columns={"Team": "HomeTeam", **{c: "Home_" + c for c in cols}}))
    away = (tm[~tm["is_home"]][["Season", "Date_dt", "Team"] + cols]
            .rename(columns={"Team": "AwayTeam", **{c: "Away_" + c for c in cols}}))

    df = df.merge(home, on=["Season", "Date_dt", "HomeTeam"], how="left", validate="m:1")
    df = df.merge(away, on=["Season", "Date_dt", "AwayTeam"], how="left", validate="m:1")
    diff = pd.DataFrame({"Diff_" + c: df["Home_" + c] - df["Away_" + c] for c in cols})
    df = pd.concat([df, diff], axis=1)

    # --- internal self-check: no fan-out, and season openers are NaN ---
    assert len(df) == n_before, f"merge changed row count: {n_before} -> {len(df)}"
    _open = tm.groupby(["Team", "Season"]).head(1)
    assert _open["Hist_Pts_k"].isna().all(), "season opener should have no history"
    assert _open["RestDays"].isna().all(), "season opener should have no rest days"
    return df


if __name__ == "__main__":
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "processed", "premier_league_2021_2026.csv")
    out = build_team_history(pd.read_csv(path))
    fam = [p + c for p in ("Home_", "Away_", "Diff_") for c in HIST_COLS]
    print(f"{len(out)} rows x {len(out.columns)} columns")
    print(f"history block: {len(fam)} columns, {out[fam].notna().all(axis=1).sum()} "
          f"rows complete, {len(out) - out[fam].notna().all(axis=1).sum()} season openers")