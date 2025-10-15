# Substitution & Fatigue System Design

## Current State Analysis

Your `GameEngine` has:
- ✅ `self.deadball` - tracks when ball is not in play
- ✅ `self.home_lineup` / `self.away_lineup` - current 5-man units
- ✅ `self.quarter_duration` - time tracking
- ✅ Plus/minus tracking (just added!)
- ✅ Integration with `PlayerAwarePredictor`

**Missing:**
- ❌ Minutes played per player
- ❌ Fatigue tracking
- ❌ Substitution logic
- ❌ Timeout mechanics
- ❌ Available bench players

---

## 1. Fatigue Modeling Approaches

### Option A: Linear Fatigue (Simple)

**Concept:** Performance decreases linearly with minutes played

```python
class GameEngine:
    def __init__(self, ...):
        # Track minutes for each player
        self.player_minutes = defaultdict(float)  # player_id → minutes
        self.fatigue_threshold = 6.0  # Minutes before fatigue kicks in
    
    def get_fatigue_multiplier(self, player_id: int) -> float:
        """
        Returns multiplier for player performance (0.7 to 1.0)
        """
        minutes = self.player_minutes[player_id]
        
        if minutes < self.fatigue_threshold:
            return 1.0  # Fresh
        
        # Linear decay: 1.0 at 6 min → 0.7 at 12 min
        minutes_fatigued = minutes - self.fatigue_threshold
        decay_rate = 0.05  # Lose 5% per minute
        multiplier = max(0.7, 1.0 - (minutes_fatigued * decay_rate))
        
        return multiplier
```

**Pros:**
- ✅ Simple to understand
- ✅ Easy to tune
- ✅ Predictable behavior

**Cons:**
- ❌ Not realistic (fatigue isn't linear)
- ❌ Doesn't capture "second wind"

---

### Option B: Exponential Fatigue (Realistic)

**Concept:** Fatigue accelerates - fresh → OK → tired → exhausted

```python
def get_fatigue_multiplier(self, player_id: int) -> float:
    """
    Exponential fatigue curve - gradual then rapid decline
    """
    minutes = self.player_minutes[player_id]
    
    # Exponential decay curve
    # At 0 min: 1.0 (100%)
    # At 6 min: 0.95 (95%)
    # At 10 min: 0.82 (82%)
    # At 15 min: 0.65 (65%)
    
    return 1.0 * (0.95 ** (minutes / 2))
```

**Curve visualization:**
```
Performance
    100% |████████▓▓▓▒▒░░
     90% |        ████▓▓▓▒▒░░
     80% |            ████▓▓▒▒░
     70% |                ████▒░
     60% |                    ██░
         +----------------------
         0   3   6   9   12  15 (minutes)
```

**Pros:**
- ✅ Realistic fatigue pattern
- ✅ Elite players can play longer before major drop
- ✅ Captures urgency of substitution

**Cons:**
- ❌ More complex
- ❌ Harder to tune parameters

---

### Option C: Energy Pool System (Game-like)

**Concept:** Each player has an energy pool that depletes

```python
class GameEngine:
    def __init__(self, ...):
        # Each player starts with 100 energy
        self.player_energy = {pid: 100.0 for pid in all_players}
        
    def step(self):
        # ... existing code ...
        
        # Deplete energy for all players on court
        for player_id in offensive_lineup + defensive_lineup:
            # Base depletion per possession
            energy_cost = 1.0
            
            # Extra cost for high-intensity plays
            if player_id == result['player_id']:
                energy_cost += 0.5  # Player who acted works harder
            
            if result['outcome'] in ['2PT_make', '2PT_miss']:
                energy_cost += 0.3  # Driving to basket is tiring
            
            self.player_energy[player_id] -= energy_cost
            self.player_energy[player_id] = max(0, self.player_energy[player_id])
    
    def get_fatigue_multiplier(self, player_id: int) -> float:
        energy = self.player_energy[player_id]
        
        if energy > 80:
            return 1.0  # Fresh
        elif energy > 50:
            return 0.85 + (energy - 50) * 0.005  # 0.85 to 1.0
        elif energy > 20:
            return 0.70 + (energy - 20) * 0.005  # 0.70 to 0.85
        else:
            return 0.50 + energy * 0.01  # 0.50 to 0.70
    
    def rest_player(self, player_id: int):
        """Called when player is subbed out - regenerate energy"""
        # Regenerate 10 energy per "possession" of rest
        self.player_energy[player_id] = min(100, self.player_energy[player_id] + 10)
```

**Pros:**
- ✅ Intuitive (like video games)
- ✅ Can model high-intensity plays costing more energy
- ✅ Natural rest/recovery mechanic
- ✅ Can show energy bars to user

**Cons:**
- ❌ Most complex
- ❌ Requires tuning energy costs
- ❌ May not match real NBA patterns

---

## 2. Integrating Fatigue with Possession Model

### Approach A: Adjust Player Stats (Recommended)

**Modify player statistics based on fatigue before prediction:**

```python
def step(self):
    # ... get lineups ...
    
    # Apply fatigue to the prediction
    result = self.predictor.sample_possession_detailed_with_fatigue(
        offensive_lineup=offensive_lineup,
        defensive_lineup=defensive_lineup,
        season=self.season,
        score_margin=score_margin,
        period=self.quarter,
        fatigue_multipliers={
            pid: self.get_fatigue_multiplier(pid) 
            for pid in offensive_lineup + defensive_lineup
        }
    )
```

**In PlayerAwarePredictor:**

```python
def sample_possession_detailed_with_fatigue(
    self,
    offensive_lineup,
    defensive_lineup,
    season,
    fatigue_multipliers,
    ...
):
    # Select player (fatigue affects who gets ball - tired players used less)
    adjusted_usage_rates = []
    for player_id in offensive_lineup:
        base_usage = self.get_player_tendencies(player_id, season)['usage_rate']
        fatigue_mult = fatigue_multipliers.get(player_id, 1.0)
        
        # Tired players get ball less often
        adjusted_usage = base_usage * fatigue_mult
        adjusted_usage_rates.append(adjusted_usage)
    
    # Select player with adjusted weights
    player_id = self.select_player_weighted(offensive_lineup, adjusted_usage_rates)
    
    # Predict outcome with adjusted shooting %
    tendencies = self.get_player_tendencies(player_id, season)
    fatigue_mult = fatigue_multipliers.get(player_id, 1.0)
    
    # Fatigue reduces shooting % and increases turnovers
    adjusted_2pt_pct = tendencies['2pt_pct'] * fatigue_mult
    adjusted_3pt_pct = tendencies['3pt_pct'] * fatigue_mult
    adjusted_to_rate = tendencies['to_rate'] * (2.0 - fatigue_mult)  # More TOs when tired
    
    # ... make prediction with adjusted stats ...
```

**What gets affected by fatigue:**
- ❌ Lower shooting % (makes fewer shots)
- ❌ Lower usage rate (gets ball less)
- ❌ Higher turnover rate (loses ball more)
- ❌ Lower assist rate (worse passing decisions)

**Pros:**
- ✅ Works with existing model
- ✅ Realistic effects
- ✅ Easy to tune multipliers
- ✅ No retraining needed

**Cons:**
- ❌ Requires modifying predictor
- ❌ Need to pass fatigue data through

---

### Approach B: Temperature Adjustment (Quick Hack)

**Use temperature parameter to add randomness for fatigued players:**

```python
def get_temperature(self, offensive_lineup):
    """
    Higher temperature = more random = worse performance
    """
    avg_fatigue = np.mean([
        self.get_fatigue_multiplier(pid) 
        for pid in offensive_lineup
    ])
    
    # Fresh team: temp = 1.0 (realistic)
    # Tired team: temp = 1.5 (more random/worse)
    temperature = 1.0 + (1.0 - avg_fatigue) * 0.5
    return temperature

def step(self):
    # ...
    result = self.predictor.sample_possession_detailed(
        ...,
        temperature=self.get_temperature(offensive_lineup)
    )
```

**Pros:**
- ✅ Very simple
- ✅ No predictor changes needed
- ✅ Works immediately

**Cons:**
- ❌ Not realistic (just adds randomness)
- ❌ Doesn't distinguish effects (shooting vs turnovers)
- ❌ Crude approximation

---

### Approach C: Post-Process Results (Hacky but Works)

**Adjust results after prediction based on fatigue:**

```python
def step(self):
    # ... get prediction ...
    result = self.predictor.sample_possession_detailed(...)
    
    # Apply fatigue effects to result
    player_id = result['player_id']
    fatigue_mult = self.get_fatigue_multiplier(player_id)
    
    # If fatigued and made a shot, maybe turn it into a miss
    if result['fgm'] == 1 and random.random() > fatigue_mult:
        # Fatigue caused miss!
        result['fgm'] = 0
        result['points'] = 0
        
        if result['3pm'] == 1:
            result['3pm'] = 0
            result['outcome'] = '3PT_miss'
        else:
            result['outcome'] = '2PT_miss'
    
    # If fatigued, higher chance of turnover
    if random.random() > fatigue_mult ** 2:  # Exponential for TOs
        result['to'] = 1
        result['outcome'] = 'TO'
        result['points'] = 0
        result['fga'] = 0
        result['fgm'] = 0
```

**Pros:**
- ✅ No predictor changes
- ✅ Works with existing code
- ✅ Easy to implement

**Cons:**
- ❌ Statistically questionable
- ❌ Can break model's learned distributions
- ❌ Feels like a hack

---

## 3. Substitution Mechanics

### Option A: Rule-Based Substitutions (Realistic NBA)

**Simulate real NBA rotation patterns:**

```python
class GameEngine:
    def __init__(self, ...):
        # Define rotation patterns (minutes each player should play per quarter)
        self.rotation_plan = {
            'starters': {
                'target_minutes_per_quarter': 9,  # Play 9 of 12 minutes
                'rest_start': 6.0,  # Sub out at 6 min mark
                'return_time': 0.0  # Return start of next quarter
            },
            'bench': {
                'target_minutes_per_quarter': 6,
                'rotation_group': []  # Which bench players sub in
            }
        }
    
    def check_substitutions(self):
        """
        Called during deadballs - check if subs needed
        """
        if not self.deadball:
            return
        
        # Check home team
        for i, player_id in enumerate(self.home_lineup):
            minutes = self.player_minutes[player_id]
            
            # Time for regular rotation?
            if 5.5 < minutes < 6.5:  # Around 6-minute mark
                # Find bench player to sub in
                bench_player = self.get_next_bench_player('home')
                if bench_player:
                    self.substitute_player('home', i, bench_player)
        
        # Same for away team
        # ...
    
    def get_next_bench_player(self, team):
        """
        Find bench player who needs minutes
        """
        roster = self.hometeam.roster_df if team == 'home' else self.awayteam.roster_df
        lineup = self.home_lineup if team == 'home' else self.away_lineup
        
        # Find players not currently on court
        bench = [pid for pid in roster['PLAYER_ID'] if pid not in lineup]
        
        # Sort by minutes played (least → most)
        bench.sort(key=lambda pid: self.player_minutes[pid])
        
        return bench[0] if bench else None
```

**Substitution triggers:**
- ✅ Time-based (6-minute mark, end of quarter)
- ✅ Fatigue-based (player too tired)
- ✅ Foul trouble (player has 5 fouls)
- ✅ Performance-based (player cold/hot)

---

### Option B: Coach Strategy System

**Different strategies for different situations:**

```python
class SubstitutionStrategy:
    def __init__(self, coach_style='balanced'):
        self.style = coach_style  # 'aggressive', 'balanced', 'conservative'
    
    def should_substitute(self, player_id, game_state, player_minutes, fatigue):
        """
        Determine if player should be subbed based on strategy
        """
        if self.style == 'aggressive':
            # Play starters heavy minutes
            return player_minutes > 10 or fatigue < 0.6
        
        elif self.style == 'balanced':
            # Regular NBA rotations
            return player_minutes > 7 or fatigue < 0.7
        
        elif self.style == 'conservative':
            # Short minutes, fresh legs
            return player_minutes > 5 or fatigue < 0.8
        
        # Context-aware subs
        if game_state['close_game'] and game_state['late']:
            # Keep starters in close games
            return fatigue < 0.5  # Only sub if exhausted
        
        if game_state['blowout']:
            # Rest starters in blowouts
            return player_minutes > 4  # Quick hook
        
        return False
```

---

### Option C: Performance-Based (Dynamic)

**Sub based on how players are performing:**

```python
def track_player_performance(self):
    """Track recent performance for substitution decisions"""
    # Add to GameEngine
    self.player_recent_performance = defaultdict(lambda: {
        'last_5_fga': [],
        'last_5_fgm': [],
        'recent_turnovers': 0
    })

def update_performance_tracking(self, result):
    player_id = result['player_id']
    perf = self.player_recent_performance[player_id]
    
    # Track last 5 shots
    if result['fga'] > 0:
        perf['last_5_fga'].append(result['fga'])
        perf['last_5_fgm'].append(result['fgm'])
        
        # Keep only last 5
        if len(perf['last_5_fga']) > 5:
            perf['last_5_fga'].pop(0)
            perf['last_5_fgm'].pop(0)
    
    if result['to'] > 0:
        perf['recent_turnovers'] += 1

def is_player_cold(self, player_id):
    """Player shooting poorly - maybe sub them out"""
    perf = self.player_recent_performance[player_id]
    
    if len(perf['last_5_fga']) >= 3:
        makes = sum(perf['last_5_fgm'])
        attempts = sum(perf['last_5_fga'])
        
        if makes == 0 and attempts >= 3:
            return True  # 0 for last 3+ → cold!
    
    if perf['recent_turnovers'] >= 3:
        return True  # Struggling with ball
    
    return False
```

---

## 4. Timeout System

### Basic Implementation:

```python
class GameEngine:
    def __init__(self, ...):
        # NBA rules: 7 timeouts per game
        self.home_timeouts = 7
        self.away_timeouts = 7
        
        # Track when timeouts were called
        self.timeout_log = []
    
    def call_timeout(self, team: str) -> bool:
        """
        Call a timeout - forces deadball and allows substitutions
        
        Returns:
            True if timeout successful, False if no timeouts left
        """
        if team == 'home':
            if self.home_timeouts <= 0:
                return False
            self.home_timeouts -= 1
        else:
            if self.away_timeouts <= 0:
                return False
            self.away_timeouts -= 1
        
        # Force deadball
        self.deadball = True
        
        # Log it
        self.timeout_log.append({
            'team': team,
            'quarter': self.quarter,
            'time_remaining': self.quarter_duration,
            'score': f"{self.home_score}-{self.away_score}"
        })
        
        # Now team can make substitutions!
        self.check_substitutions()
        
        return True
```

### Strategic Timeout Triggers:

```python
def should_call_timeout(self, team: str) -> bool:
    """
    AI decision: when to call timeout
    """
    if team == 'home':
        timeouts_left = self.home_timeouts
        our_score = self.home_score
        their_score = self.away_score
    else:
        timeouts_left = self.away_timeouts
        our_score = self.away_score
        their_score = self.home_score
    
    # No timeouts left
    if timeouts_left <= 0:
        return False
    
    # Reason 1: Team is tired and no natural deadball
    if not self.deadball:
        avg_fatigue = self.get_team_average_fatigue(team)
        if avg_fatigue < 0.65:  # Team exhausted
            return True
    
    # Reason 2: Opponent on a run
    # TODO: Track scoring runs
    
    # Reason 3: Late game management
    if self.quarter == 4 and self.quarter_duration < 120:  # Last 2 minutes
        score_diff = abs(our_score - their_score)
        if score_diff <= 5:  # Close game
            # Save timeouts for end
            return False
    
    # Reason 4: Need to make substitutions but ball won't stop
    # ...
    
    return False
```

---

## 5. Complete Integration Example

### Recommended Approach:

```python
class GameEngine:
    def __init__(self, hometeam_id, awayteam_id, season):
        # ... existing init ...
        
        # NEW: Minutes tracking
        all_home = self.hometeam.roster_df['PLAYER_ID'].tolist()
        all_away = self.awayteam.roster_df['PLAYER_ID'].tolist()
        self.player_minutes = {pid: 0.0 for pid in all_home + all_away}
        
        # NEW: Fatigue (using exponential model)
        self.fatigue_enabled = True
        
        # NEW: Timeouts
        self.home_timeouts = 7
        self.away_timeouts = 7
        
        # NEW: Substitution strategy
        self.sub_check_interval = 2.0  # Check every 2 minutes
        self.last_sub_check = {1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0}
    
    def step(self):
        # ... existing step logic ...
        
        # NEW: Track minutes
        possession_time = np.random.binomial(n=24, p=0.7) / 60.0  # Convert to minutes
        for player_id in offensive_lineup + defensive_lineup:
            self.player_minutes[player_id] += possession_time / 10  # Split among players
        
        # NEW: Check for substitutions at deadballs
        if self.deadball:
            self.check_substitutions()
        
        # NEW: Check if timeout needed
        if self.should_call_timeout('home'):
            self.call_timeout('home')
        if self.should_call_timeout('away'):
            self.call_timeout('away')
    
    def get_fatigue_multiplier(self, player_id):
        """Exponential fatigue"""
        minutes = self.player_minutes[player_id]
        return 1.0 * (0.95 ** (minutes / 2))
    
    def substitute_player(self, team, lineup_position, new_player_id):
        """Make a substitution"""
        if team == 'home':
            old_player = self.home_lineup[lineup_position]
            self.home_lineup[lineup_position] = new_player_id
        else:
            old_player = self.away_lineup[lineup_position]
            self.away_lineup[lineup_position] = new_player_id
        
        print(f"{team.upper()} SUB: OUT #{old_player}, IN #{new_player_id}")
```

---

## 6. Recommended Implementation Order

### Phase 1: Basic Fatigue
1. ✅ Add `player_minutes` tracking
2. ✅ Implement `get_fatigue_multiplier()` (exponential)
3. ✅ Display fatigue in game state

### Phase 2: Fatigue Effects
4. ✅ Modify predictor to accept fatigue multipliers
5. ✅ Apply fatigue to shooting %, usage, turnovers
6. ✅ Test that fatigued players perform worse

### Phase 3: Basic Substitutions
7. ✅ Implement `substitute_player()` method
8. ✅ Add rule-based subs (time-based)
9. ✅ Sub players at 6-minute mark and between quarters

### Phase 4: Timeouts
10. ✅ Add timeout tracking
11. ✅ Implement `call_timeout()`
12. ✅ Basic timeout strategy (fatigue-based)

### Phase 5: Advanced
13. ✅ Performance-based substitutions
14. ✅ Strategic timeout timing
15. ✅ Coach style/strategy system

---

## 7. Data Structures Summary

```python
# Add to GameEngine.__init__():
self.player_minutes = {}           # player_id → minutes played
self.home_timeouts = 7             # Timeouts remaining
self.away_timeouts = 7
self.timeout_log = []              # Record of timeouts
self.sub_log = []                  # Record of substitutions
self.fatigue_enabled = True        # Toggle fatigue on/off

# Optional advanced features:
self.player_energy = {}            # Energy pool system
self.player_recent_performance = {} # Hot/cold tracking
self.scoring_runs = []             # Momentum tracking
```

---

## Summary: Quick Start Recommendation

**Start with this simple but effective approach:**

1. **Exponential fatigue** - realistic and tunable
2. **Stat adjustment** - modify shooting % and usage before prediction
3. **Time-based substitutions** - sub at 6-min mark like real NBA
4. **Timeout for forced subs** - when no natural deadball and team tired

This gives you 80% of realism with 20% of the complexity!

Would you like me to implement any of these approaches?

