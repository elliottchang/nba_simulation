# NBA Play-by-Play Game Simulator

A realistic NBA game simulation engine that predicts possession outcomes using machine learning trained on historical play-by-play data.

## Overview

This project simulates NBA games at the possession level, using a hybrid neural network model that combines:
- **Season-specific player embeddings** (captures player improvement over time, e.g., SGA 2020 vs 2025)
- **Statistical features** (shooting percentages, usage rates, turnover rates)
- **Game context** (score differential, quarter, time remaining)

The model is trained on real NBA play-by-play data and can accurately predict how possessions will end based on the players on the court.

## Project Structure

```
nba_simulation/
├── main.py                      # Core game engine classes (Team, Player, GameEngine)
├── ingest_nba_data.py          # Data ingestion utilities
├── train_possession_model.py    # ML model training pipeline
├── possession_predictor.py      # Inference interface for trained model
├── pbp_model.ipynb             # Jupyter notebook for experimentation
├── pbp_data.csv                # Collected play-by-play data (generated)
├── possession_model.pt         # Trained model weights (generated)
└── feature_engineer.pkl        # Feature mappings and vocabularies (generated)
```

## Key Features

### 1. Season-Specific Modeling
The model uses player-season embeddings, meaning each player has a different representation for each season. This allows the simulation to accurately reflect:
- Player development (young players improving)
- Peak years vs decline
- Different team contexts and roles

**Example:** Shai Gilgeous-Alexander (SGA)
- 2020 SGA: ~19 PPG, developing playmaker
- 2025 SGA: ~30+ PPG, MVP candidate
- The model captures this difference automatically

### 2. Robust to Player Combinations
The model works with any combination of players because it:
- Learns individual player tendencies through embeddings
- Doesn't require specific lineup combinations to be seen during training
- Aggregates player features to understand lineup strengths

### 3. Realistic Possession Outcomes
Predicts multiple outcome types:
- `2PT_make` / `2PT_miss` - Two-point field goals
- `3PT_make` / `3PT_miss` - Three-point field goals
- `FT_make` / `FT_miss` - Free throws
- `TO` - Turnovers
- And who performed the action

## Core Classes

### Team Class
```python
Team(season: str, team_id: int = None, team_abbr: str = None, team_name: str = None)
```
Represents an NBA team for a specific season, including:
- Roster information
- Automatic starter selection based on historical data
- Team metadata (name, abbreviation, location)

### Player Class
```python
Player(player_id: int, season: str)
```
Represents a player for a specific season with:
- Full game log data
- Statistical information
- Season-specific performance metrics

### GameEngine Class
```python
GameEngine(hometeam_id: int, awayteam_id: int, season: str)
```
The main simulation engine that:
- Manages game state (quarter, possession, score)
- Tracks both team boxscores
- Handles lineup management
- Simulates possessions using the ML model

## Machine Learning Model

### Architecture: Hybrid Stats + Player Embeddings

```
Input Layer:
├── Player-Season Embeddings (learned, 64-dim)
│   └── Captures unique characteristics of each player in each season
├── Statistical Features (6 features)
│   ├── Usage rate
│   ├── 2PT/3PT attempt rates
│   ├── Shooting percentages
│   └── Turnover rate
└── Game Context (2 features)
    ├── Score differential
    └── Current quarter

Hidden Layers:
├── Fully connected (128 units) + BatchNorm + Dropout
└── Fully connected (128 units) + BatchNorm + Dropout

Output Layer:
└── Possession outcome probabilities (multi-class classification)
```

### Why This Approach?

**Problem:** Traditional basketball simulations use fixed ratings that don't capture:
- Player development over seasons
- Context-dependent performance
- Complex player interactions

**Solution:** Our hybrid model learns:
1. **Player embeddings** that evolve across seasons (SGA_2020 ≠ SGA_2025)
2. **Statistical tendencies** from real play-by-play data
3. **Game context effects** (clutch situations, blowouts, etc.)

**Benefits:**
- ✅ Historically accurate (trained on real data)
- ✅ Season-specific (different embeddings per year)
- ✅ Robust to any lineup combination
- ✅ Captures realistic possession flow
- ✅ Probabilistic (adds natural variance)

## Usage

### Step 1: Train the Model

```bash
python train_possession_model.py
```

This will:
1. Collect play-by-play data from NBA API
2. Engineer features and build vocabularies
3. Train the neural network
4. Save model checkpoint and feature mappings

**Configuration options** (edit in `main()` function):
- `SEASONS`: List of seasons to train on (e.g., `['2022-23', '2023-24']`)
- `MAX_GAMES_PER_SEASON`: Limit games for testing (set to `None` for full dataset)
- `NUM_EPOCHS`: Training epochs
- `EMBEDDING_DIM`: Size of player embeddings
- `HIDDEN_DIM`: Size of hidden layers

**Expected output:**
```
pbp_data.csv              # Raw play-by-play data
feature_engineer.pkl      # Feature mappings
possession_model.pt       # Trained model weights
```

### Step 2: Use the Predictor

```python
from possession_predictor import PossessionPredictor

# Load trained model
predictor = PossessionPredictor(
    model_path='possession_model.pt',
    feature_engineer_path='feature_engineer.pkl'
)

# Make a prediction
result = predictor.predict_possession(
    player_id=203999,  # Nikola Jokic
    season='2023-24',
    score_margin=5,
    period=2,
    return_probabilities=True
)

print(f"Predicted outcome: {result['outcome']}")
print(f"Confidence: {result['confidence']:.2%}")

# Sample from probability distribution (for simulation)
outcome = predictor.sample_possession(
    player_id=203999,
    season='2023-24',
    score_margin=0,
    period=1,
    temperature=1.0  # Higher = more random
)
```

### Step 3: Integrate with Game Engine

```python
from main import GameEngine
from possession_predictor import PossessionPredictor

# Initialize game
game = GameEngine(
    hometeam_id=1610612738,  # Boston Celtics
    awayteam_id=1610612747,  # Los Angeles Lakers
    season='2023-24'
)

# Load predictor
predictor = PossessionPredictor()

# Simulate possessions
for _ in range(10):
    # Get current player with ball (simplified)
    current_player = game.home_lineup[0][0]  # First starter's ID
    
    # Predict possession outcome
    outcome = predictor.sample_possession(
        player_id=current_player,
        season='2023-24',
        score_margin=game.hometeam_boxscore['PTS'].sum() - game.awayteam_boxscore['PTS'].sum(),
        period=game.quarter
    )
    
    print(f"Possession {game.possession}: {outcome}")
    game.step()
```

## Demo & Testing

Run the inference demo:
```bash
python possession_predictor.py
```

This demonstrates:
- Deterministic prediction with confidence scores
- Stochastic sampling for realistic simulation
- Player tendency analysis
- Season-over-season comparisons

## Data Requirements

### From NBA API (`nba_api` package)
- Play-by-play data: `playbyplayv2.PlayByPlayV2`
- Game finder: `leaguegamefinder.LeagueGameFinder`
- Team rosters: `commonteamroster.CommonTeamRoster`
- Player game logs: `playergamelog.PlayerGameLog`

### Collected Features
- Possession outcomes (shot results, turnovers)
- Player involved in each possession
- Game context (score, time, quarter)
- Player-season statistics (usage, shooting percentages)

## Future Enhancements

### Short-term improvements:
1. **Full lineup modeling** - Use all 10 players on court (not just the player ending possession)
2. **Defensive matchups** - Model defender impact on possession outcomes
3. **More context features** - Shot clock, home/away, rest days, injuries
4. **Better stat features** - Pull advanced stats from NBA API instead of computing from possessions

### Advanced features:
1. **Sequence modeling** - Use LSTM/Transformer to model entire possession sequences (pass → screen → shot)
2. **Spatial features** - Incorporate shot location, court position (requires tracking data)
3. **Lineup synergies** - Learn specific player combination effects (pick-and-roll partnerships)
4. **Fatigue modeling** - Adjust predictions based on minutes played
5. **Coaching strategies** - Model play types (isolation, pick-and-roll, post-up)

## Installation

```bash
# Required packages
pip install pandas numpy torch nba_api

# Optional for visualization
pip install matplotlib seaborn
```

## Performance Metrics

After training, the model reports:
- **Training loss**: Cross-entropy loss on training set
- **Validation loss**: Cross-entropy loss on held-out validation set
- **Validation accuracy**: Percentage of correct outcome predictions

Typical performance (will vary based on data size and configuration):
- Validation accuracy: 40-60% (multi-class classification is inherently challenging)
- Key insight: The model learns realistic probability distributions, not just majority class

## License

MIT License - Feel free to use and modify for your projects!

## Acknowledgments

- NBA API for providing access to play-by-play data
- PyTorch for the deep learning framework
- The NBA statistics community for inspiring this project