"""Canonical column mappings + documented column vocabulary.

Derived from data/documentation.txt. Single source of truth so future
seasons with new/renamed columns are recognized without touching the
merge logic.
"""

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

# Glossary: bookmaker abbreviation -> human-readable name. Explanatory only;
# these abbreviations never appear as standalone columns (only as prefixes of
# e.g. B365H, BWCH), so this dict is never used for renaming.
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

# Columns documented but not present in the current CSVs (recognized,
# never renamed).
LEGACY_COLUMNS = frozenset({
    # match statistics
    "Attendance", "HHW", "AHW", "HFKC", "AFKC", "HO", "AO", "HBP", "ABP",
    # BetBrain aggregates (superseded by Max/Avg market odds)
    "Bb1X2", "BbMxH", "BbAvH", "BbMxD", "BbAvD", "BbMxA", "BbAvA",
    "BbOU", "BbMx>2.5", "BbAv>2.5", "BbMx<2.5", "BbAv<2.5",
    "BbAH", "BbAHh", "BbMxAHH", "BbAvAHH", "BbMxAHA", "BbAvAHA",
    # bookmaker-specific handicap odds/sizes / legacy over-under
    "GBAHH", "GBAHA", "GBAH", "LBAHH", "LBAHA", "LBAH",
    "B365AH", "GB>2.5", "GB<2.5",
    # legacy bookmaker pre-closing odds (H/D/A), absent from current data
    "BSH", "BSD", "BSA", "GBH", "GBD", "GBA", "PPH", "PPD", "PPA",
    "SBH", "SBD", "SBA", "SJH", "SJD", "SJA", "SKH", "SKD", "SKA",
    "SOH", "SOD", "SOA", "SYH", "SYD", "SYA",
})
