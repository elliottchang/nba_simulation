# NBA Game Simulation Engine

A sophisticated NBA game simulation system that uses machine learning to predict possession outcomes and simulates complete games with realistic player behaviors, fatigue systems, and substitution patterns.

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
├── main.py                    # Main game simulation engine
├── train_possession_model.py  # ML model training pipeline
├── player_aware_predictor.py  # Player-specific prediction logic
├── data/
│   ├── pbp_data.csv          # Play-by-play training data
│   └── checkpoint.pkl        # Data collection progress
├── features/
│   └── feature_engineer.pkl   # Feature mappings and statistics
└── models/
    └── possession_model.pt    # Trained neural network
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

