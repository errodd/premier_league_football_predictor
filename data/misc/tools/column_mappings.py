# Canonical column mappings + bookmaker registry, derived from data/documentation.txt.
# Single source of truth so future seasons with new/renamed columns are recognized
# without touching the merge logic.

# Exact column-name aliases -> canonical name (same meaning, two labels).
ALIASES = {
    # results
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
    # Pinnacle
    "PH": "PSH",
    "PD": "PSD",
    "PA": "PSA",
    # VC Bet -> BetVictor (pre-closing)
    "VCH": "BVH",
    "VCD": "BVD",
    "VCA": "BVA",
    # VC Bet -> BetVictor (closing)
    "VCCH": "BVCH",
    "VCCD": "BVCD",
    "VCCA": "BVCA",
}

# Bookmaker abbreviation -> name. Includes legacy bookmakers absent from current data.
BOOKMAKERS = {
    "1XB": "1XBet",
    "B365": "Bet365",
    "BF": "Betfair",
    "BFD": "Betfred",
    "BMGM": "BetMGM",
    "BV": "BetVictor",
    "BS": "Blue Square",
    "BW": "Bet&Win",
    "CL": "Coral",
    "GB": "Gamebookers",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "PP": "Paddy Power",
    "PS": "Pinnacle",
    "SK": "Skybet",
    "SO": "Sporting Odds",
    "SB": "Sportingbet",
    "SJ": "Stan James",
    "SY": "Stanleybet",
    "VC": "VC Bet (now BetVictor)",
    "WH": "William Hill",
    "BFE": "Betfair Exchange",
}

# Columns documented but not present in the current CSVs (recognized, never renamed).
LEGACY_COLUMNS = frozenset({
    # match statistics
    "Attendance", "HHW", "AHW", "HFKC", "AFKC", "HO", "AO", "HBP", "ABP",
    # BetBrain aggregates (superseded by Max/Avg market odds)
    "Bb1X2", "BbMxH", "BbAvH", "BbMxD", "BbAvD", "BbMxA", "BbAvA",
    "BbOU", "BbMx>2.5", "BbAv>2.5", "BbMx<2.5", "BbAv<2.5",
    "BbAH", "BbAHh", "BbMxAHH", "BbAvAHH", "BbMxAHA", "BbAvAHA",
    # bookmaker-specific handicap sizes / legacy over-under
    "B365AH", "GBAH", "LBAH", "GB>2.5", "GB<2.5",
})
