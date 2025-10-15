# Player-Aware Possession Modeling

### Stage 1: WHO Ends the Possession
```python
# Select player weighted by usage rate
player_id = predictor.select_player(offensive_lineup, season)

# Curry: 30% usage → 30% chance to end possession
# Kuminga: 15% usage → 15% chance to end possession
```

### Stage 2: WHAT They Do
```python
# Determine action based on player's shot distribution + lineup context
action = predictor.predict_player_action(player_id, season, ...)

# Curry: 55% 3PT attempts, 35% 2PT attempts, 10% turnovers
# Kuminga: 25% 3PT attempts, 65% 2PT attempts, 10% turnovers
```

### Stage 3: SUCCESS Rate
```python
# Determine outcome based on player's actual shooting %
outcome = predictor.predict_outcome(player_id, action, season)

# Curry taking 3PT: 43% make rate
# Kuminga taking 3PT: 33% make rate
```

## How It Works

### Data Sources

All from your training data (`feature_engineer.pkl`):

```python
# For each player-season pair, we have:
{
    'usage_rate': 0.285,      # % of team possessions they use
    '2pt_rate': 0.145,        # % of possessions ending in 2PT
    '3pt_rate': 0.087,        # % of possessions ending in 3PT
    'to_rate': 0.032,         # % of possessions ending in TO
    '2pt_pct': 0.623,         # 2PT shooting %
    '3pt_pct': 0.356,         # 3PT shooting %
}
```

### Prediction Flow

```
Possession Starts
       ↓
[Stage 1: WHO] → Select player (usage-weighted)
       ↓
[Stage 2: WHAT] → Determine action (player tendencies + lineup context)
       ↓
[Stage 3: SUCCESS] → Make/miss (player shooting %)
       ↓
Return detailed result
```

### Why This Works

1. **Usage Rates** → Natural from possession-ending data
   - High usage players end more possessions in training data
   - We extract this as `usage_rate`

2. **Shot Distribution** → From player's attempts
   - 3PT specialists have high `3pt_rate`
   - Post players have high `2pt_rate`

3. **Shooting %** → From make/miss ratios
   - Good shooters have high `2pt_pct` / `3pt_pct`
   - Poor shooters have low percentages

## Usage

### Basic Example

```python
from player_aware_predictor import PlayerAwarePredictor

predictor = PlayerAwarePredictor()

# Simulate possession
result = predictor.sample_possession_detailed(
    offensive_lineup=[curry, klay, draymond, wiggins, looney],
    defensive_lineup=[lebron, ad, reaves, rui, dlo],
    season='2023-24',
    score_margin=0,
    period=1
)

print(f"Player {result['player_id']}: {result['outcome']}")
# Example output: "Player 201939: 3PT_make"

# Update boxscore
boxscore.loc[result['player_id'], 'PTS'] += result['points']
boxscore.loc[result['player_id'], 'FGA'] += result['fga']
boxscore.loc[result['player_id'], '3PA'] += result['3pa']
boxscore.loc[result['player_id'], '3PM'] += result['3pm']
```

### Compare Players

```python
# See how Curry vs Kuminga perform in same situation
comparison = predictor.compare_players(
    player_ids=[curry_id, kuminga_id],
    season='2023-24',
    offensive_lineup=lineup,
    defensive_lineup=defense,
    num_possessions=100
)

print(f"Curry: {comparison[curry_id]['avg_points']:.2f} pts/poss")
print(f"Kuminga: {comparison[kuminga_id]['avg_points']:.2f} pts/poss")
```

### Integration with Game Engine

```python
from main import GameEngine
from player_aware_predictor import PlayerAwarePredictor

# Setup
game = GameEngine(hometeam_id=1610612744, awayteam_id=1610612747, season='2023-24')
predictor = PlayerAwarePredictor()

# Get lineups
home_lineup = [pid for pid, _ in game.home_lineup]
away_lineup = [pid for pid, _ in game.away_lineup]

# Simulate quarter
for poss in range(24):  # ~24 possessions per quarter
    if poss % 2 == 0:
        # Home offense
        result = predictor.sample_possession_detailed(
            offensive_lineup=home_lineup,
            defensive_lineup=away_lineup,
            season='2023-24',
            score_margin=game.home_score - game.away_score,
            period=1
        )
        
        # Update home boxscore
        player_id = result['player_id']
        game.hometeam_boxscore.loc[player_id, 'PTS'] += result['points']
        game.hometeam_boxscore.loc[player_id, 'FGA'] += result['fga']
        game.hometeam_boxscore.loc[player_id, 'FGM'] += result['fgm']
        game.hometeam_boxscore.loc[player_id, '3PA'] += result['3pa']
        game.hometeam_boxscore.loc[player_id, '3PM'] += result['3pm']
        
    else:
        # Away offense (similar)
        pass
```

## Key Differences vs Base Model

| Feature | Base Model | Player-Aware Model |
|---------|-----------|-------------------|
| **Player Selection** | Random | Usage-weighted |
| **Shot Selection** | Team average | Player-specific |
| **Success Rate** | Team average | Player-specific |
| **Curry Usage** | 20% (uniform) | 30% (realistic) |
| **Curry 3PT Freq** | 30% (team avg) | 55% (his actual) |
| **Curry 3PT %** | 36% (team avg) | 43% (his actual) |

## Expected Results

Running 100 possessions with Warriors lineup:

**Curry:**
- Ends ~30-35% of possessions (high usage)
- Takes ~55% 3-pointers when he shoots
- Makes ~43% of his 3-pointers
- Scores ~1.2 points per possession he uses

**Kuminga:**
- Ends ~10-15% of possessions (lower usage)
- Takes ~25% 3-pointers when he shoots  
- Makes ~33% of his 3-pointers
- Scores ~0.8 points per possession he uses

## Run the Demo

```bash
# See player-aware modeling in action
python player_aware_predictor.py

# Compare Curry vs Kuminga
python example_curry_vs_kuminga.py
```

## Technical Details

### Combining Player + Lineup Context

The model uses a weighted combination:

```python
# 60% player tendencies, 40% lineup-adjusted probabilities
player_3pt_rate = player_stats['3pt_rate'] * 0.6
lineup_3pt_boost = lineup_context['3pt_prob'] * 0.4
final_3pt_rate = player_3pt_rate + lineup_3pt_boost
```

This means:
- Player tendencies dominate (their natural game)
- But lineup context adjusts (floor spacing, matchups)

### Why Not Just Use Player Stats?

We could ignore the neural network and only use player stats, but:

❌ Ignores defensive matchups
❌ Ignores floor spacing effects
❌ Ignores game situation

The hybrid approach:
✅ Player tendencies (what they naturally do)
✅ Lineup context (how the matchup affects it)
✅ Realistic and flexible

## Future Enhancements

### Short-term
1. **Usage-weighted by game situation**
   - Stars get more touches in clutch
   - Role players get more in blowouts

2. **Shot location**
   - Model where players shoot from
   - Some players better from corners, others from top

### Long-term
1. **Player synergies**
   - Curry-Draymond pick-and-roll
   - Explicit passer-shooter connections

2. **Defensive assignments**
   - Model 1-on-1 matchups explicitly
   - Elite defenders reduce shooter %

3. **Hot hand / momentum**
   - Players on hot streaks shoot better
   - Confidence effects

## Files

1. **`player_aware_predictor.py`** - Main implementation
2. **`example_curry_vs_kuminga.py`** - Demonstrates player differences
3. **`PLAYER_AWARE_MODEL.md`** - This documentation

## Summary

The player-aware model finally gives you what you wanted:

✅ **Curry gets the ball more than Kuminga** (usage-weighted selection)  
✅ **Curry takes more 3-pointers** (player-specific shot distribution)  
✅ **Curry makes more 3-pointers** (player-specific shooting %)  

All while maintaining:
- Lineup-aware context (defensive matchups matter)
- Season-specific modeling (2020 Curry ≠ 2025 Curry)
- Easy integration with your game engine

Happy simulating! 🏀

