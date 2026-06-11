"""Central configuration for the v2 pipeline.

Every numeric value that remains here is either a physical rule of the game
(period lengths, foul limits) or a modeling hyperparameter. Behavioral
quantities (rebound rates, pace, fatigue, substitution patterns) are fitted
from data and live in the artifact files, not here.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_v2"
EVENTS_DIR = DATA_DIR / "events"          # per-season raw event parquet
PLAYERS_DIR = DATA_DIR / "players"        # per-season game rosters/starters
TABLES_DIR = DATA_DIR / "tables"          # derived training tables
ARTIFACTS_DIR = ROOT / "models_v2"        # fitted model artifacts

CHANCES_FILE = TABLES_DIR / "chances.parquet"
GAMES_FILE = TABLES_DIR / "games.parquet"
SUB_EVENTS_FILE = TABLES_DIR / "sub_decisions.parquet"
FT_FILE = TABLES_DIR / "free_throws.parquet"

FEATURE_STORE_FILE = ARTIFACTS_DIR / "feature_store.pkl"
MODEL_FILE = ARTIFACTS_DIR / "possession_model_v2.pt"
DURATION_FILE = ARTIFACTS_DIR / "duration_model.pkl"
SUB_MODEL_FILE = ARTIFACTS_DIR / "substitution_model.pkl"

# ---------------------------------------------------------------- seasons
# Extend backwards for more data; era drift is handled by a season index
# feature in the model.
SEASONS = ["2022-23", "2023-24", "2024-25"]

# ---------------------------------------------------------------- game rules
REGULATION_PERIODS = 4
PERIOD_SECONDS = 12 * 60
OT_SECONDS = 5 * 60
FOUL_OUT_LIMIT = 6
TEAM_FOULS_FOR_BONUS = 5          # 5th team foul of a period -> bonus FTs
SHOT_CLOCK_SECONDS = 24

# ---------------------------------------------------------------- PlayByPlayV3 action types
ACT_MADE = "Made Shot"
ACT_MISSED = "Missed Shot"
ACT_FT = "Free Throw"
ACT_REBOUND = "Rebound"
ACT_TURNOVER = "Turnover"
ACT_FOUL = "Foul"
ACT_SUB = "Substitution"
ACT_TIMEOUT = "Timeout"
ACT_JUMPBALL = "Jump Ball"
ACT_PERIOD = "period"
ACT_VIOLATION = "Violation"

# ---------------------------------------------------------------- model hyperparameters
STAT_DIM = 15              # per-player stat feature vector size (see features.py)
PLAYER_EMB_DIM = 16        # residual player embedding (cross-season)
SEASON_EMB_DIM = 4
PLAYER_TOKEN_DIM = 32      # encoded player token size
CONTEXT_DIM = 8            # see dataset.py build_context
HIDDEN_DIM = 128
DROPOUT = 0.2
PLAYER_DROPOUT = 0.05      # probability a lineup slot is masked to UNK during training
EMB_WEIGHT_DECAY = 1e-3    # weight decay applied to embedding tables specifically
LR = 1e-3
BATCH_SIZE = 512
VAL_FRACTION = 0.15        # fraction of GAMES (not rows) held out

# Empirical-Bayes shrinkage pseudo-counts for player rate stats
SHRINK_ATTEMPTS = 50.0     # shooting percentages
SHRINK_POSSESSIONS = 200.0 # usage / action rates

# Actions (stage 2 of the factorized model)
ACTIONS = ["2pt_attempt", "3pt_attempt", "turnover", "off_foul", "drawn_foul"]
# Shot results (stage 3, conditioned on a 2pt/3pt attempt)
SHOT_RESULTS = ["make", "miss", "andone", "shooting_foul"]
