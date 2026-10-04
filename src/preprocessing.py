"""Feature contract and preprocessing pipeline for notebook 02.

Single source of truth for *which* raw columns may reach the model and *how*
each one is prepared. It lives here, outside the notebooks, because notebook 03
rebuilds the pipeline by importing this module rather than by importing a
notebook.

Two ideas drive the whole module.

1. **Positive whitelist.** `ALLOWED_RAW_COLUMNS` names every raw column the
   model is allowed to see. A column that is not on the list cannot leak, so
   leakage stops being something you have to remember to check and becomes
   something the code refuses. `remainder="drop"` is the second line of
   defence, not the first.

2. **The label travels through `y`, never through `X`.** The historical block
   needs `FTR` to compute points and W/D/L. `FeatureBuilder` takes the label in
   `fit`, keeps it in `self.y_` and returns an explicit projection of the
   allowed features, so `FTR` is structurally absent from `X` and cannot appear
   in the output even if the code changes.

The inventory of the 157 raw columns is *generated* from `FAMILIES` plus
`EXCEPCIONES` rather than written by hand, so a typo becomes an execution error
instead of a silently mislabelled column. The prose motive for each family
lives in the notebook table, not here.

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from preprocessing import ALLOWED_RAW_COLUMNS, build_model_pipeline
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from team_history import K, HIST_COLS, build_team_history

# --------------------------------------------------------------------------
# Raw-column inventory: 18 families, 157 columns, no overlap, no orphans.
# Each entry is (action, columns, one-line motive).
# --------------------------------------------------------------------------
FAMILIES = {
    "G-01_identificadores": (
        "excluir",
        ["Season", "Div", "Date", "Time", "HomeTeam", "AwayTeam"],
        "identidad; uso crudo infla cardinalidad y no generaliza",
    ),
    "G-02_metadatos": (
        "excluir",
        ["Referee"],
        "identidad nominal, pocos partidos por arbitro",
    ),
    "G-03_resultado": (
        "excluir",
        ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"],
        "post-partido; FTR es el objetivo, el resto es fuente de historial",
    ),
    "G-04_estadisticas": (
        "excluir",
        ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
         "HY", "AY", "HR", "AR"],
        "post-partido; 0 % ausentes no las legitima",
    ),
    "G-05_1x2_apertura": (
        "excluir",
        ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA",
         "IWH", "IWD", "IWA", "PSH", "PSD", "PSA",
         "WHH", "WHD", "WHA", "BVH", "BVD", "BVA"],
        "Avg* es el promedio de esta familia: 3 columnas, no 18",
    ),
    "G-06_1x2_agregadas": (
        "excluir",
        ["MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"],
        "Avg* alimenta el de-vig; Max* es redundante con Avg*",
    ),
    "G-07_goles_apertura": (
        "excluir",
        ["B365>2.5", "B365<2.5", "P>2.5", "P<2.5",
         "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5"],
        "senal de total de goles, objetivo de otra etapa",
    ),
    "G-08_handicap_apertura": (
        "excluir",
        ["AHh", "B365AHH", "B365AHA", "PAHH", "PAHA",
         "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA"],
        "AHh es el unico con senal medida; las otras 8 son el mismo handicap",
    ),
    "G-09_1x2_cierre": (
        "excluir",
        ["B365CH", "B365CD", "B365CA", "BWCH", "BWCD", "BWCA",
         "IWCH", "IWCD", "IWCA", "PSCH", "PSCD", "PSCA",
         "WHCH", "WHCD", "WHCA", "BVCH", "BVCD", "BVCA"],
        "disponibles en T-0; se excluyen por redundancia, no por fuga",
    ),
    "G-10_1x2_agregadas_cierre": (
        "excluir",
        ["MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA"],
        "misma magnitud que G-06 en otro momento de mercado",
    ),
    "G-11_goles_cierre": (
        "excluir",
        ["B365C>2.5", "B365C<2.5", "PC>2.5", "PC<2.5",
         "MaxC>2.5", "MaxC<2.5", "AvgC>2.5", "AvgC<2.5"],
        "igual que G-07",
    ),
    "G-12_handicap_cierre": (
        "excluir",
        ["AHCh", "B365CAHH", "B365CAHA", "PCAHH", "PCAHA",
         "MaxCAHH", "MaxCAHA", "AvgCAHH", "AvgCAHA"],
        "el handicap de cierre es la misma magnitud en otro momento",
    ),
    "G-13_intercambio_apertura": (
        "excluir",
        ["BFH", "BFD", "BFA", "1XBH", "1XBD", "1XBA",
         "BFEH", "BFED", "BFEA", "BFE>2.5", "BFE<2.5",
         "BFEAHH", "BFEAHA", "BFDH", "BFDD", "BFDA"],
        "alta proporcion de ausentes; su presencia se resume en OddsQuotes",
    ),
    "G-14_intercambio_cierre": (
        "excluir",
        ["BFCH", "BFCD", "BFCA", "1XBCH", "1XBCD", "1XBCA",
         "BFDCH", "BFDCD", "BFDCA"],
        "misma razon que G-13, con doble definicion apertura/cierre",
    ),
    "G-15_bfec": (
        "excluir",
        ["BFECH", "BFECD", "BFECA", "BFEC>2.5", "BFEC<2.5",
         "BFECAHH", "BFECAHA"],
        "tercera reiteracion del mercado 1X2, sin senal propia medida",
    ),
    "G-16_bmgm": (
        "excluir",
        ["BMGMH", "BMGMD", "BMGMA", "BMGMCH", "BMGMCD", "BMGMCA"],
        "solo en temporadas recientes; su ausencia es sesgo de disponibilidad",
    ),
    "G-17_cl": (
        "excluir",
        ["CLH", "CLD", "CLA", "CLCH", "CLCD", "CLCA"],
        "la familia con mas ausencias del dataset (hasta 1640 celdas)",
    ),
    "G-18_lb": (
        "excluir",
        ["LBH", "LBD", "LBA", "LBCH", "LBCD", "LBCA"],
        "misma razon que G-17",
    ),
}

# Column-level overrides. A column takes its family action unless it appears
# here. A column in neither its family nor here is an execution error, not a
# column without a verdict.
EXCEPCIONES = {
    "FTR": "target",
    "AvgH": "transformar",
    "AvgD": "transformar",
    "AvgA": "transformar",
    "AHh": "transformar",
}

ACCIONES_VALIDAS = {"usar", "transformar", "excluir", "investigar", "target"}

# Minimum input contract. Everything else in the raw frame is optional.
REQUIRED_COLUMNS = [
    "Season", "Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG",
    "AvgH", "AvgD", "AvgA", "AHh",
]

# The 17 columns that only exist after the final whistle. FTHG and FTAG are
# deliberately absent from ALLOWED_RAW_COLUMNS' prohibition: they enter on
# purpose, as the results of *earlier* matches, and build_team_history applies
# shift(1) so a row never consumes its own result.
POST_MATCH_COLUMNS = [
    "FTHG", "FTAG", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]

# The 42 1X2 quote columns of families G-13..G-18. Written out explicitly
# because a prefix filter would also catch the goals and handicap columns.
# This is the only block of the dataset where absence is structural, so it is
# the only place where an availability indicator has non-zero variance.
ODDS_QUOTE_COLS = [
    # apertura (12)
    "BFH", "BFD", "BFA", "1XBH", "1XBD", "1XBA",
    "BFEH", "BFED", "BFEA", "BFDH", "BFDD", "BFDA",
    # cierre (9)
    "BFCH", "BFCD", "BFCA", "1XBCH", "1XBCD", "1XBCA",
    "BFDCH", "BFDCD", "BFDCA",
    # BFEC (3)
    "BFECH", "BFECD", "BFECA",
    # BMGM (6)
    "BMGMH", "BMGMD", "BMGMA", "BMGMCH", "BMGMCD", "BMGMCA",
    # CL (6)
    "CLH", "CLD", "CLA", "CLCH", "CLCD", "CLCA",
    # LB (6)
    "LBH", "LBD", "LBA", "LBCH", "LBCD", "LBCA",
]

# The 22 -> 23 base features that survive the redundancy pruning of the
# feature catalogue. HIST_KEEP is 14 of the 24 columns produced by
# build_team_history; the 10 rejected ones are exact identities
# (D = Pts - 3W, GD = GF - GA) or restated differences.
HIST_KEEP = [
    "Home_Hist_Pts_k", "Away_Hist_Pts_k", "Diff_Hist_Pts_k",
    "Home_Hist_GF_k", "Away_Hist_GF_k", "Diff_Hist_GF_k",
    "Home_Hist_GA_k", "Away_Hist_GA_k",
    "Home_Hist_W_k", "Away_Hist_W_k", "Home_Hist_L_k", "Away_Hist_L_k",
    "Home_RestDays", "Away_RestDays",
]
MARKET_KEEP = ["p_home", "p_draw", "p_away", "Mkt_Gap", "AHh"]
CAL_KEEP = ["Home_MatchesPlayed", "Away_MatchesPlayed", "OddsQuotes"]
DOW_KEEP = ["DayOfWeek"]

BASE_FEATURES = HIST_KEEP + MARKET_KEEP + CAL_KEEP + DOW_KEEP

# Raw identity columns kept in the projection so the one-hot block can consume
# them. They are not features themselves: they exist to be encoded and are
# never passed through to the model.
CLUB_COLS = ["HomeTeam", "AwayTeam"]

# What FeatureBuilder returns: the 23 derived features plus the 2 raw club
# columns. Everything else in ALLOWED_RAW_COLUMNS is dropped by
# `remainder="drop"`, which is the second line of defence.
PIPELINE_FEATURES = BASE_FEATURES + CLUB_COLS

# Positive whitelist. `X = df.loc[:, ALLOWED_RAW_COLUMNS]` is the first line of
# defence against leakage; `remainder="drop"` is the second.
ALLOWED_RAW_COLUMNS = (
    ["Season", "Date", "Time", "HomeTeam", "AwayTeam",  # structure
     "FTHG", "FTAG",                             # PAST results
     "AvgH", "AvgD", "AvgA", "AHh"]              # market
    + ODDS_QUOTE_COLS                            # 42 cols -> OddsQuotes
)


def build_inventory(columns):
    """Build the per-column action table instead of writing 157 rows by hand.

    Parameters
    ----------
    columns : sequence of str
        Real header of the raw frame.

    Returns
    -------
    pandas.DataFrame
        One row per column with `columna`, `familia`, `accion` and `motivo`,
        ordered as the input header.
    """
    seen = {}
    for family, (_action, cols, _motive) in FAMILIES.items():
        for col in cols:
            if col in seen:
                raise ValueError(
                    f"{col} esta en {seen[col]} y en {family}: solapamiento")
            seen[col] = family

    orphans = [c for c in columns if c not in seen]
    if orphans:
        raise ValueError(f"columnas sin familia: {orphans}")

    rows = []
    for col in columns:
        family = seen[col]
        _fam_action, _cols, motive = FAMILIES[family]
        action = EXCEPCIONES.get(col, _fam_action)
        if action not in ACCIONES_VALIDAS:
            raise ValueError(f"{col}: accion invalida {action!r}")
        rows.append({"columna": col, "familia": family,
                     "accion": action, "motivo": motive})
    return pd.DataFrame(rows)


def _matches_played(df, side):
    """Prior league matches played by each team within its own season.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame already carrying `Season`, `Date_dt`, `Time` and the team column.
    side : str
        "HomeTeam" or "AwayTeam".

    Returns
    -------
    pandas.Series
        Count aligned to `df.index`. Zero on a season opener. Computed on a
        date-sorted copy and reindexed back, because `cumcount` follows row
        order and the raw frame is not guaranteed to be chronological.
    """
    order = df.sort_values(["Season", "Date_dt", "Time"]).index
    counted = df.loc[order].groupby(["Season", side]).cumcount()
    return counted.reindex(df.index)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds every derived feature, then projects onto `keep`.

    The label enters through `y` in `fit` and is held in `self.y_` for
    `transform`, never through `X`. `X` only needs results of *earlier*
    matches, which are available at T-0. `build_team_history` applies
    `shift(1)` inside `(Team, Season)`, so a row never consumes its own result.

    Parameters
    ----------
    k : int
        Size of the recent-form window.
    keep : list of str or None
        Columns to return. Defaults to `PIPELINE_FEATURES` (25).

    Attributes
    ----------
    y_ : numpy.ndarray of shape (n_samples,)
        Label memorised in `fit`.
    """

    def __init__(self, k=K, keep=None):
        self.k = k
        self.keep = PIPELINE_FEATURES if keep is None else list(keep)

    def fit(self, X, y=None):
        """Memorise the label and validate the input contract.

        Parameters
        ----------
        X : pandas.DataFrame
            Match frame without the target column.
        y : pandas.Series
            `FTR`, index-aligned with `X`.

        Returns
        -------
        FeatureBuilder
            The fitted transformer.
        """
        if y is None:
            raise ValueError("FeatureBuilder necesita la etiqueta en y")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)} filas) e y ({len(y)}) miden "
                             f"distinto")
        if not (X.index == y.index).all():
            raise ValueError("X e y desalineados")
        if "FTR" in X.columns:
            raise ValueError("FTR no debe viajar dentro de X")
        # FTR is part of the raw-frame contract but reaches this transformer
        # through `y`, so it is the one required column X must not carry.
        structure = [c for c in REQUIRED_COLUMNS if c != "FTR"]
        missing = [c for c in structure if c not in X.columns]
        if missing:
            raise ValueError(f"faltan columnas requeridas: {missing}")
        self.y_ = np.asarray(y)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        """Return the allowed features, one row per row of `X`.

        Parameters
        ----------
        X : pandas.DataFrame
            Match frame without the target column.

        Returns
        -------
        pandas.DataFrame
            Same rows as `X`, columns `self.keep` and nothing else.
        """
        base = X.assign(FTR=self.y_)
        if "Date_dt" not in base.columns:
            base["Date_dt"] = pd.to_datetime(base["Date"],
                                            format="%d/%m/%Y")
        out = self._row_wise(build_team_history(base, k=self.k))
        absent = [c for c in self.keep if c not in out.columns]
        if absent:
            raise ValueError(f"features no construidas: {absent}")
        return out[self.keep].copy()

    @staticmethod
    def _row_wise(df):
        """Add the features that depend only on the current match record.

        Parameters
        ----------
        df : pandas.DataFrame
            Frame carrying the historical block and the raw columns.

        Returns
        -------
        pandas.DataFrame
            The same frame plus the market and calendar features.
        """
        inv = 1.0 / df[["AvgH", "AvgD", "AvgA"]].to_numpy(dtype=float)
        p = inv / inv.sum(axis=1, keepdims=True)
        out = df.assign(p_home=p[:, 0], p_draw=p[:, 1], p_away=p[:, 2])
        out["Mkt_Gap"] = out["AvgH"] - out["AvgA"]
        out["DayOfWeek"] = out["Date_dt"].dt.dayofweek
        out["Home_MatchesPlayed"] = _matches_played(out, "HomeTeam")
        out["Away_MatchesPlayed"] = _matches_played(out, "AwayTeam")
        out["OddsQuotes"] = out[ODDS_QUOTE_COLS].notna().sum(axis=1)
        return out


def build_preprocessor():
    """Build the ColumnTransformer of notebook 02.

    Returns
    -------
    ColumnTransformer
        Three numeric blocks (median imputation + standardisation), clubs and
        day of week in one-hot with `handle_unknown="ignore"`, and
        `remainder="drop"` as the second line of defence. `verbose_feature_
        names_out=False` keeps output names bare so the containment check
        against the whitelist is a real test rather than a tautology.
    """
    numeric = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler())])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ct = ColumnTransformer(
        [("hist", numeric, HIST_KEEP),
         ("market", numeric, MARKET_KEEP),
         ("cal", numeric, CAL_KEEP),
         ("clubs", ohe, CLUB_COLS),
         ("dow", ohe, DOW_KEEP)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def build_model_pipeline(estimator, k=K):
    """Return the full pipeline plus an unfitted clone of the estimator.

    Notebook 02 does not use this to fit anything: it calls it with a
    `DummyClassifier` only to demonstrate the signature contract and to check
    that the estimator handed in was not modified (hence `clone`). The fit
    happens in notebook 03.

    Parameters
    ----------
    estimator : scikit-learn estimator
        Model to place as the last step. Cloned, not used.
    k : int
        Size of the recent-form window.

    Returns
    -------
    Pipeline
        Pipeline with the steps `feat`, `prep` and `model`.
    """
    return Pipeline([("feat", FeatureBuilder(k=k)),
                     ("prep", build_preprocessor()),
                     ("model", clone(estimator))])


if __name__ == "__main__":
    import os

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = os.path.join(root, "data", "processed",
                        "premier_league_2021_2026.csv")
    raw = pd.read_csv(path)

    # --- the inventory partitions the real header -----------------------
    inv = build_inventory(raw.columns)
    counts = inv["accion"].value_counts().to_dict()
    print(f"inventario: {len(inv)} columnas, {inv['familia'].nunique()} "
          f"familias, acciones={counts}")
    assert len(inv) == 157, "el inventario debe cubrir las 157 columnas"
    assert counts == {"excluir": 152, "transformar": 4, "target": 1}, counts

    # --- the whitelist is the first line of defence ---------------------
    assert len(ODDS_QUOTE_COLS) == 42
    assert len(set(ODDS_QUOTE_COLS)) == 42
    assert set(ODDS_QUOTE_COLS) <= set(raw.columns)
    assert len(set(ALLOWED_RAW_COLUMNS)) == len(ALLOWED_RAW_COLUMNS)
    assert "FTR" not in ALLOWED_RAW_COLUMNS
    leaked = set(ALLOWED_RAW_COLUMNS) & set(POST_MATCH_COLUMNS)
    assert leaked == {"FTHG", "FTAG"}, f"post-partido filtrado: {leaked}"
    print(f"lista blanca: {len(ALLOWED_RAW_COLUMNS)} columnas crudas, "
          f"post-partido permitido solo como historial = {sorted(leaked)}")

    # --- the derived block ----------------------------------------------
    assert len(BASE_FEATURES) == 23 == len(set(BASE_FEATURES))
    assert len(PIPELINE_FEATURES) == 25 == len(set(PIPELINE_FEATURES))
    assert set(REQUIRED_COLUMNS) <= set(raw.columns), "faltan columnas"
    frame = raw.loc[:, ALLOWED_RAW_COLUMNS]
    feat = FeatureBuilder(k=K).fit(frame, raw["FTR"])
    built = feat.transform(frame)
    assert list(built.columns) == PIPELINE_FEATURES
    assert len(built) == len(raw)
    openers = int(built["Home_Hist_Pts_k"].isna().sum())
    print(f"bloque derivado: {built.shape[0]} x {built.shape[1]}, "
          f"openers de local = {openers} "
          f"({openers // 5} por temporada)")

    # --- the full pipeline reaches a matrix -----------------------------
    prep = build_preprocessor()
    out = prep.fit_transform(built, raw["FTR"])
    names = list(out.columns)
    bare = set(HIST_KEEP + MARKET_KEEP + CAL_KEEP)
    assert all(n in bare or n.startswith(("HomeTeam_", "AwayTeam_",
                                         "DayOfWeek_")) for n in names)
    assert not set(names) & set(POST_MATCH_COLUMNS + ["FTR"])
    print(f"ColumnTransformer: {built.shape[1]} features base -> "
          f"{out.shape[1]} columnas de salida")
    print("OK: src/preprocessing.py")