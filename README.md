# NBA Game Simulation Engine

A sophisticated NBA game simulation system that uses machine learning to predict possession outcomes and simulates complete games with realistic player behaviors, fatigue systems, and substitution patterns.

🚀 **Now with modern data engineering**: Snowflake integration, separate ETL pipelines, and cloud-ready architecture!

## What This Does

This project simulates NBA games at the possession level, using a neural network trained on real NBA play-by-play data to predict how each possession will end. It goes far beyond basic statistical simulation by incorporating:

- **Player-specific tendencies** based on their actual NBA performance
- **Lineup chemistry** effects through player embeddings
- **Fatigue and energy systems** that affect performance over time
- **Realistic substitution patterns** based on game situations
- **Complete season simulation** capabilities

## Key Features

### Machine Learning Foundation
- Uses PyTorch neural network trained on NBA API play-by-play data
- Player-season embeddings capture individual player tendencies
- Predicts 11 different possession outcomes (2pt/3pt makes/misses, turnovers, fouls, etc.)
- Accounts for game context (score margin, quarter, etc.)

### Player-Aware Simulation
- **Usage Rate Modeling**: Star players get the ball more often than role players
- **Shot Distribution**: Each player takes shots according to their real tendencies
- **Shooting Percentages**: Uses actual career shooting percentages for realistic outcomes
- **Fatigue Effects**: Tired players perform worse and handle the ball less

### Complete Game Engine
- **Real-time substitutions** based on energy levels and game situations
- **Fatigue tracking** with energy depletion and recovery
- **Timeout management** for strategic rest periods
- **Overtime support** for tied games
- **Detailed boxscores** with full statistical tracking

### Season Simulation
- **Full season scheduling** with customizable games per team
- **League standings** and statistics tracking
- **Player and team performance** across multiple games
- **Statistical leaderboards** for various categories

## Project Structure

```
nba_simulation/
├── main.py                              # Main game simulation engine
├── train_possession_model.py            # Legacy ML training pipeline
├── train_with_data_engine.py            # Modern ML training with Snowflake
├── player_aware_predictor.py            # Legacy player prediction logic
├── modern_player_aware_predictor.py     # Modern player predictor
├── setup_data_engineering.py            # Setup script for new architecture
├── data_engine/                         # Modern data engineering package
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # Configuration management
│   ├── snowflake_client.py              # Snowflake database client
│   ├── data_ingestion.py                # Raw data collection pipeline
│   ├── data_transformation.py           # Data processing pipeline
│   └── pipeline.py                      # Pipeline orchestrator
├── data/                                # Local data storage (legacy)
│   ├── pbp_data.csv                    # Play-by-play training data
│   └── checkpoint.pkl                  # Data collection progress
├── features/                            # Feature storage
│   ├── feature_engineer.pkl             # Legacy feature mappings
│   └── modern_feature_engineer.pkl      # Modern feature mappings
├── models/                              # Trained models
│   ├── possession_model.pt              # Legacy trained model
│   └── modern_possession_model.pt       # Modern trained model
├── env_example.txt                      # Environment variables template
└── requirements_data_engine.txt         # Additional dependencies
```

## Modern Data Engineering Architecture

This project now features a modern data engineering architecture that separates concerns and provides better scalability:

### 🔄 ETL Pipeline Architecture

1. **Data Ingestion** (`data_engine/data_ingestion.py`)
   - Collects raw NBA API data (play-by-play events, lineups, game metadata)
   - Stores unprocessed data directly to Snowflake data warehouse
   - Includes retry logic, rate limiting, and error handling

2. **Data Transformation** (`data_engine/data_transformation.py`)
   - Processes raw data into ML-ready features
   - Extracts possessions from events and computes player statistics
   - Can be re-run independently when transformation logic changes

3. **Data Warehouse** (Snowflake)
   - Persistent storage for raw and processed data
   - No need to re-collect data when changing models or features
   - Easy querying and analysis capabilities

### 🏀 Database Schema Overview

This database models NBA game data at multiple levels of granularity — from individual play-by-play **events**, to team **possessions**, to aggregated **box scores** for players and teams.  
It’s designed to be **normalized**, **historically accurate**, and easily extendable for predictive modeling or advanced analytics.

---

#### 🧱 Core Tables

##### **Games**
Stores basic metadata for each game.

| Field | Description |
|--------|--------------|
| `id` | Unique game identifier |
| `date` | Date of the game |
| `home_team_id`, `away_team_id` | Foreign keys to `Teams` |
| `season` | Season identifier (e.g. `'2024-25'`) |

---

##### **Teams**
Contains franchise information.  
Historical relocations or renames can be represented as new rows.

| Field | Description |
|--------|--------------|
| `id` | Unique team identifier |
| `team_name`, `team_abbr` | Display name and abbreviation |
| `city`, `state` | Location metadata |
| `year_founded` | Historical metadata |

---

##### **Players**
Static player information. Team membership over time is handled separately.

| Field | Description |
|--------|--------------|
| `id` | Unique player identifier |
| `name` | Full player name |
| `college`, `year_drafted` | Background info |

---

##### **PlayerTeamStints**
Tracks which team a player was on and when, enabling accurate lineup reconstruction and season-based joins.

| Field | Description |
|--------|--------------|
| `id` | Surrogate primary key |
| `player_id`, `team_id` | Entity relationships |
| `season` | Season label |
| `start_date`, `end_date` | Duration of stint |

> A player can appear multiple times per season if traded or signed midyear.

---

###### **Events**
The atomic unit of game activity. Each record corresponds to a discrete play-by-play event, such as a made shot, turnover, rebound, foul, or substitution.

| Field | Description |
|--------|--------------|
| `id` | Event identifier |
| `game_id`, `team_id` | Context for the event |
| `period`, `clock_remaining`, `sequence` | Temporal ordering within the game |
| `event_type` | e.g. `'shot'`, `'rebound'`, `'turnover'`, `'foul'`, `'substitution'` |
| `player1_id`, `player2_id` | Main and secondary actors (e.g. shooter and assister) |
| `shot_value`, `shot_made`, `rebound_type`, `turnover_type`, `foul_type` | Event-specific details |
| `possession_id` | (Optional) links the event to its corresponding possession once derived |

This table forms the **source of truth** for all other derived data (possessions, box scores, etc.).

---

##### **Possessions**
Represents a team’s offensive possession, spanning one or more events.  
A possession starts when a team gains control of the ball and ends when they lose it (via score, turnover, or period end).

| Field | Description |
|--------|--------------|
| `id` | Unique possession identifier |
| `game_id`, `period` | Context |
| `offense_team_id`, `defense_team_id` | Teams involved |
| `start_event_id`, `end_event_id` | Boundary events in the play-by-play |
| `points` | Total points scored in this possession |
| `result` | Outcome classification (e.g. `'made_3'`, `'turnover'`, `'end_period'`) |
| `assist_player_id` | Optional convenience link if a shot was assisted |
| `duration_seconds` | Derived duration between start and end events |

> Possession-level stats are derived from the underlying event stream but cached here for fast querying.

---

##### **Substitutions**
Can either exist as a standalone table or as a filtered view on the `Events` table where `event_type='substitution'`.

| Field | Description |
|--------|--------------|
| `id` | Substitution identifier |
| `game_id`, `team_id` | Context |
| `player_out_id`, `player_in_id` | Players involved |
| `period`, `clock_remaining` | Timing information |

---

##### **PlayerPerformances** & **TeamPerformances**
Materialized aggregates built from `Events` for faster box-score queries.

| Field | Description |
|--------|--------------|
| `game_id` | Foreign key |
| `player_id` / `team_id` | Entity reference |
| `points`, `rebounds`, `assists`, `turnovers`, etc. | Derived stats |
| `plus_minus`, `win`, `home` | Derived or cached indicators |

> These tables are **derived**, not raw sources. They can be rebuilt programmatically from events to maintain data integrity.

---

### 🚀 Quick Start with Modern Architecture

1. **Setup Snowflake connection:**
   ```bash
   # Install dependencies
   pip install -r requirements_data_engine.txt
   
   # Set environment variables (copy from env_example.txt)
   export SNOWFLAKE_ACCOUNT=your_account.region
   export SNOWFLAKE_USER=your_username
   export SNOWFLAKE_PASSWORD=your_password
   
   # Run setup script
   python setup_data_engineering.py
   ```

2. **Run data pipeline:**
   ```bash
   # Collect and process data
   python -m data_engine.pipeline
   
   # Train model with processed data
   python train_with_data_engine.py
   ```

3. **Use modern predictor:**
   ```python
   from modern_player_aware_predictor import create_predictor
   predictor = create_predictor()  # Auto-detects modern vs legacy files
   ```

## How It Works

### 1. Data Collection & Training
The system first collects real NBA play-by-play data from the NBA API and trains a neural network to predict possession outcomes based on:
- 10-player lineup context (5 offensive + 5 defensive players)
- Player-season embeddings learned from historical data
- Game context (score, quarter, time remaining)

### 2. Game Simulation Process
For each possession, the system:
1. **Selects the ball handler** using usage rate and fatigue levels
2. **Predicts the outcome** using the trained neural network
3. **Updates player energy** based on activity level
4. **Manages substitutions** when players are tired or in foul trouble
5. **Tracks statistics** and updates the boxscore

### 3. Realistic Game Flow
- **48-minute games** broken into 12-minute quarters
- **Overtime periods** when games are tied
- **Strategic timeout calling** for rest and substitution opportunities
- **Context-aware substitutions** (starters play more in close games, rest in blowouts)

## Usage Examples

### Simulate a Single Game
```python
from main import GameEngine

# Lakers vs Warriors
game = GameEngine(hometeam_id=1610612747, awayteam_id=1610612744, season='2023-24')
game.simulate_game()

# View final boxscore
print(game.get_boxscore_display('home'))
```

### Run a Full Season
```python
from main import run_season_simulation

# Simulate 2024-25 season with 10 games per team (for testing)
season = run_season_simulation(season='2024-25', max_games=10)

# View standings and league leaders
season.print_season_summary()
season.get_leaders('PTS', limit=10)  # Top 10 scorers
```

### Train Your Own Model
```python
# Run the training pipeline
python train_possession_model.py
```

## Technical Details

### Machine Learning Architecture
- **Input**: 10-player lineups (5v5) + game context
- **Model**: Player embedding layers + fully connected network
- **Output**: 11-class classification for possession outcomes
- **Training**: Cross-entropy loss with early stopping and learning rate scheduling

### Performance Optimizations
- **API rate limiting** with exponential backoff
- **Checkpointing** for data collection resume capability
- **Batch processing** for efficient model training
- **GPU support** for faster training (CUDA/MPS)

## Requirements

```bash
pip install torch pandas numpy nba-api
```

## Getting Started

1. **Clone and setup**:
   ```bash
   git clone <repo>
   cd nba_simulation
   pip install -r requirements.txt
   ```

2. **Train the model** (first run):
   ```bash
   python train_possession_model.py
   ```
   This will collect NBA data and train the possession prediction model.

3. **Run simulations**:
   ```bash
   python main.py
   ```

## Example Output

```
Q1 - LAL 32 vs GSW 28
Q2 - LAL 58 vs GSW 54  
Q3 - LAL 89 vs GSW 81
Q4 - LAL 118 vs GSW 115

Final Boxscore - Lakers
Player         MIN  PTS  REB  AST  FGA   FGM  FG%   3PA  3PM  3P%
LeBron James    36   28    8    7   18    11  61.1%   5    3   60.0%
Anthony Davis   34   22   12    2   14     8  57.1%   1    0    0.0%
Russell Westbrook 31   18    6   12   12     7  58.3%   2    1   50.0%
```

## Future Improvements

- **Advanced substitution AI** using ML instead of rule-based logic
- **Injury simulation** and player availability tracking  
- **Advanced analytics** integration (RAPM, BPM, etc.)
- **Play-by-play commentary** generation
- **Team chemistry** effects beyond individual player stats

