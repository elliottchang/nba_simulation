# Substitution Strategy Guide: When and Who to Sub

## Core Question: When Should Players Rest?

Basketball is about **balancing freshness with having your best players on court**. Here are the key decision factors:

---

## 1. WHEN to Substitute (Triggers)

### A. Time-Based Triggers (Most Common in NBA)

#### **The 6-Minute Mark** (Classic NBA Pattern)
```python
# Quarter starts at 12:00
# Starters play until ~6:00 remaining
# Bench plays 6:00 to 0:00
# Starters return next quarter

if self.quarter_duration <= 6*60 and self.quarter_duration > 5*60:
    # Between 6:00 and 5:00 - sub window
    if not already_subbed_this_quarter:
        make_subs('home')
        make_subs('away')
```

**Why 6 minutes?**
- Gives starters ~6 minutes rest before next quarter
- Allows bench to play meaningful minutes
- Standard across NBA

#### **Between Quarters** (Always)
```python
if self.quarter_duration <= 0:
    # End of quarter - always bring back starters
    self.home_lineup = self.hometeam.starters
    self.away_lineup = self.awayteam.starters
    self.quarter += 1
```

#### **2-Minute Windows**
```python
# Some coaches like frequent rotations
sub_windows = [10*60, 8*60, 6*60, 4*60, 2*60]  # Every 2 minutes

for sub_time in sub_windows:
    if abs(self.quarter_duration - sub_time) < 30:  # Within 30 seconds
        consider_subs()
```

---

### B. Fatigue-Based Triggers

#### **Energy Threshold**
```python
def check_fatigue_subs(self):
    """Sub out exhausted players"""
    for i, player_id in enumerate(self.home_lineup):
        energy = self.player_energy[player_id]
        fatigue_mult = self.get_fatigue_multiplier(player_id)
        
        # Critical fatigue - must sub
        if energy < 30 or fatigue_mult < 0.65:
            bench_player = self.find_best_bench_player('home', position=i)
            if bench_player:
                self.substitute_player('home', i, bench_player)
        
        # Moderate fatigue - sub if good bench option
        elif energy < 50 or fatigue_mult < 0.80:
            bench_player = self.find_best_bench_player('home', position=i)
            if bench_player and self.player_energy[bench_player] > 80:
                self.substitute_player('home', i, bench_player)
```

#### **Team Average Fatigue**
```python
def team_needs_rest(self, team):
    """Check if whole lineup is tired"""
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    avg_energy = np.mean([self.player_energy[pid] for pid in lineup])
    
    # If team average < 60 energy, need multiple subs
    if avg_energy < 60:
        return True
    
    return False
```

---

### C. Performance-Based Triggers

#### **Player is Cold (0 for last 3+ shots)**
```python
def is_player_cold(self, player_id):
    """Track recent performance"""
    # Look at last 5 FGA
    recent_fga = self.player_recent_fga[player_id][-5:]
    recent_fgm = self.player_recent_fgm[player_id][-5:]
    
    if len(recent_fga) >= 3:
        if sum(recent_fgm) == 0 and sum(recent_fga) >= 3:
            return True  # 0 for last 3+
    
    return False

# In step():
if is_player_cold(player_id) and self.deadball:
    # Give them a break, maybe they'll heat up later
    substitute_player(...)
```

#### **Excessive Turnovers**
```python
# If player has 3+ TOs in short span
if self.player_turnovers_this_quarter[player_id] >= 3:
    # Pull them out before it gets worse
    substitute_player(...)
```

---

### D. Foul Trouble Triggers

```python
def check_foul_trouble(self):
    """Sub out players with too many fouls"""
    for player_id in self.home_lineup:
        fouls = self.hometeam_boxscore.loc[player_id, 'PF']
        
        # NBA logic:
        if fouls >= 5:
            # Must sub out (1 foul from fouling out)
            bench_player = self.find_best_bench_player('home')
            self.substitute_player('home', position, bench_player)
        
        elif fouls >= 3 and self.quarter <= 2:
            # Early foul trouble - sit them
            if self.deadball:
                substitute_player(...)
        
        elif fouls >= 4 and self.quarter <= 3:
            # 4 fouls in Q3 - risky to keep in
            if self.deadball:
                substitute_player(...)
```

---

### E. Game Situation Triggers

#### **Blowout Game**
```python
def is_blowout(self):
    """Check if game is out of reach"""
    margin = abs(self.home_score - self.away_score)
    
    # Q3 or Q4 with 20+ point lead = blowout
    if self.quarter >= 3 and margin >= 20:
        return True
    
    # Q4 with <3 min and 15+ lead = over
    if self.quarter == 4 and self.quarter_duration < 180 and margin >= 15:
        return True
    
    return False

# Sub strategy:
if is_blowout():
    if winning_team:
        # Rest all starters, play bench
        sub_in_all_bench()
    else:
        # Losing team: pull starters to avoid injury
        sub_in_all_bench()
```

#### **Close Game (Keep Stars In)**
```python
def is_close_game(self):
    """Check if game is competitive"""
    margin = abs(self.home_score - self.away_score)
    
    # Close if within 5 points in Q4 last 6 minutes
    if self.quarter == 4 and self.quarter_duration < 360 and margin <= 5:
        return True
    
    return False

# Sub strategy:
if is_close_game():
    # Keep best 5 in, even if tired
    # Only sub if player is REALLY exhausted (energy < 20)
    if player_energy < 20:
        sub_out()  # Even in close games, can't play at 0%
```

---

## 2. WHO to Take Out (Selection Criteria)

### Priority System

```python
def select_player_to_sub_out(self, lineup, team):
    """
    Prioritize who to sub out
    
    Priority (highest first):
    1. Foul trouble (5 fouls)
    2. Exhausted (energy < 25)
    3. Very tired (energy < 40)
    4. Playing poorly (0-5 shooting)
    5. Regular rotation (by minutes threshold)
    """
    candidates = []
    
    for i, player_id in enumerate(lineup):
        # Check foul trouble
        fouls = boxscore.loc[player_id, 'PF']
        if fouls >= 5:
            return (i, player_id, 'foul_trouble')
        
        # Check exhaustion
        energy = self.player_energy[player_id]
        fatigue_mult = self.get_fatigue_multiplier(player_id)
        
        if energy < 25:
            candidates.append((i, player_id, 'exhausted', energy))
        elif energy < 40:
            candidates.append((i, player_id, 'very_tired', energy))
        elif energy < 55:
            candidates.append((i, player_id, 'tired', energy))
    
    # Return most tired player
    if candidates:
        candidates.sort(key=lambda x: x[3])  # Sort by energy (lowest first)
        return candidates[0]
    
    # No urgent needs - return None or do time-based rotation
    return None
```

---

### NBA Rotation Patterns

#### **Starters vs Bench Split**
```python
# Typical NBA: Starters play 32-36 minutes, Bench 12-16 minutes

def get_target_minutes(self, player_id):
    """How many minutes should this player play?"""
    # Check if starter
    if player_id in self.hometeam.starters:
        return 34  # Starters: ~34 minutes/game
    else:
        # Bench players by role
        if is_sixth_man(player_id):
            return 24  # Key bench player
        elif is_rotation_player(player_id):
            return 15  # Regular rotation
        else:
            return 5   # Deep bench (garbage time only)

def should_rest_based_on_minutes(self, player_id):
    current_minutes = self.player_minutes[player_id]
    target_per_quarter = self.get_target_minutes(player_id) / 4
    
    # If played more than expected for this quarter
    if current_minutes > target_per_quarter * self.quarter:
        return True
    
    return False
```

---

### Position-Based Substitutions

```python
def get_player_position(self, player_id):
    """Get player's primary position"""
    # From roster data
    roster = self.hometeam.roster_df
    position = roster[roster['PLAYER_ID'] == player_id]['POSITION'].iloc[0]
    return position

def find_bench_player_for_position(self, team, position_needed):
    """
    Find bench player who can play this position
    
    NBA positions: PG, SG, SF, PF, C
    Can have flexibility:
    - Guards: PG/SG interchangeable
    - Wings: SG/SF/PF can overlap
    - Bigs: PF/C interchangeable
    """
    roster = self.hometeam.roster_df if team == 'home' else self.awayteam.roster_df
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    
    # Get bench players
    bench = roster[~roster['PLAYER_ID'].isin(lineup)]
    
    # Filter by position
    if position_needed in ['PG', 'SG']:
        # Look for guards
        position_match = bench[bench['POSITION'].isin(['PG', 'SG', 'G'])]
    elif position_needed in ['SF', 'PF']:
        # Look for forwards
        position_match = bench[bench['POSITION'].isin(['SF', 'PF', 'F'])]
    else:  # Center
        position_match = bench[bench['POSITION'].isin(['C', 'F-C'])]
    
    if len(position_match) > 0:
        # Return freshest player at that position
        best = max(position_match['PLAYER_ID'].tolist(), 
                  key=lambda p: self.player_energy[p])
        return best
    
    # No position match? Return freshest available
    return max(bench['PLAYER_ID'].tolist(), 
              key=lambda p: self.player_energy[p])
```

---

## 3. WHO to Bring In (Bench Selection)

### Strategy A: Fresh Legs Priority

**Simple: Bring in whoever has most energy**

```python
def find_best_bench_player(self, team, position=None):
    """
    Find best bench player to sub in
    
    Priority:
    1. High energy (>80)
    2. Not in current lineup
    3. Position fit (if specified)
    4. Quality (from training data stats)
    """
    roster = self.hometeam.roster_df if team == 'home' else self.awayteam.roster_df
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    
    # Get all bench players
    bench_ids = [pid for pid in roster['PLAYER_ID'] if pid not in lineup]
    
    if not bench_ids:
        return None
    
    # Filter by energy (must be reasonably fresh)
    fresh_bench = [pid for pid in bench_ids if self.player_energy[pid] > 70]
    
    if not fresh_bench:
        # Emergency: take whoever has most energy
        return max(bench_ids, key=lambda p: self.player_energy[p])
    
    # From fresh players, pick best by quality
    # Use their player stats to determine quality
    best_player = max(fresh_bench, key=lambda p: self.get_player_quality(p))
    
    return best_player

def get_player_quality(self, player_id):
    """
    Rate player quality for substitution decisions
    """
    season = self.season
    key = (player_id, season)
    
    if key not in self.predictor.fe.player_season_stats:
        return 0.0
    
    stats = self.predictor.fe.player_season_stats[key]
    
    # Simple quality score: usage + efficiency
    quality = (
        stats['usage_rate'] * 0.5 +  # 50% weight on usage
        stats['2pt_pct'] * 0.25 +      # 25% weight on 2PT%
        stats['3pt_pct'] * 0.25        # 25% weight on 3PT%
    )
    
    return quality
```

---

### Strategy B: Role-Based Substitutions

**Match player roles - don't sub star for deep bench**

```python
class PlayerRole:
    """Define player tiers"""
    STAR = 1         # Top 2-3 players (32-38 min/game)
    STARTER = 2      # Other starters (28-32 min/game)
    SIXTH_MAN = 3    # First off bench (22-26 min/game)
    ROTATION = 4     # Regular rotation (12-20 min/game)
    DEEP_BENCH = 5   # Rarely plays (0-10 min/game)

def get_player_role(self, player_id):
    """Determine player's role on team"""
    # Get their season stats
    key = (player_id, self.season)
    if key not in self.predictor.fe.player_season_stats:
        return PlayerRole.DEEP_BENCH
    
    stats = self.predictor.fe.player_season_stats[key]
    usage = stats['usage_rate']
    
    if usage > 0.28:
        return PlayerRole.STAR
    elif usage > 0.22:
        return PlayerRole.STARTER
    elif usage > 0.18:
        return PlayerRole.SIXTH_MAN
    elif usage > 0.12:
        return PlayerRole.ROTATION
    else:
        return PlayerRole.DEEP_BENCH

def find_bench_replacement(self, player_id_out, team):
    """
    Find appropriate replacement based on role
    """
    role_out = self.get_player_role(player_id_out)
    roster = self.hometeam.roster_df if team == 'home' else self.awayteam.roster_df
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    
    bench = [pid for pid in roster['PLAYER_ID'] if pid not in lineup]
    
    # Match role level (sub star for sixth man, not deep bench)
    if role_out == PlayerRole.STAR:
        # Star needs rest → bring in sixth man or best bench
        candidates = [p for p in bench 
                     if self.get_player_role(p) in [PlayerRole.SIXTH_MAN, PlayerRole.ROTATION]]
    
    elif role_out in [PlayerRole.STARTER, PlayerRole.SIXTH_MAN]:
        # Regular rotation → bring in rotation player
        candidates = [p for p in bench 
                     if self.get_player_role(p) in [PlayerRole.ROTATION, PlayerRole.DEEP_BENCH]]
    
    else:
        # Bench player → bring in anyone fresh
        candidates = bench
    
    if not candidates:
        candidates = bench
    
    # From candidates, pick freshest
    return max(candidates, key=lambda p: self.player_energy[p])
```

---

### Strategy C: Lineup Balance

**Maintain team balance - don't have all guards or all bigs**

```python
def check_lineup_balance(self, lineup):
    """
    Ensure lineup has good position distribution
    
    Typical NBA lineup:
    - 1-2 guards (PG/SG)
    - 2-3 wings (SF/PF)
    - 1-2 bigs (C/PF)
    """
    positions = [self.get_player_position(pid) for pid in lineup]
    
    guards = sum(1 for p in positions if p in ['PG', 'SG', 'G'])
    wings = sum(1 for p in positions if p in ['SF', 'PF', 'F'])
    bigs = sum(1 for p in positions if p in ['C', 'F-C'])
    
    # Check for bad compositions
    if guards == 0:
        return False, "Need a guard"
    if guards >= 4:
        return False, "Too many guards"
    if bigs == 0:
        return False, "Need a big"
    if bigs >= 4:
        return False, "Too many bigs"
    
    return True, "Balanced"

def substitute_maintaining_balance(self, team, position_idx):
    """
    Sub while keeping lineup balanced
    """
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    player_out = lineup[position_idx]
    position_out = self.get_player_position(player_out)
    
    # Try to replace with same position
    replacement = self.find_bench_player_for_position(team, position_out)
    
    # Check if new lineup is balanced
    test_lineup = lineup.copy()
    test_lineup[position_idx] = replacement
    
    is_balanced, msg = self.check_lineup_balance(test_lineup)
    
    if is_balanced:
        self.substitute_player(team, position_idx, replacement)
    else:
        # Try different bench player
        # ...
```

---

## 4. Complete Substitution Strategies

### Strategy 1: Simple Rotation (Easiest)

```python
def simple_rotation_subs(self):
    """
    NBA-style simple rotation:
    - Starters play 0-6 min, rest 6-12 min
    - Bench plays 6-12 min
    - Repeat each quarter
    """
    # Check if we're at the 6-minute mark
    if 5.5*60 < self.quarter_duration < 6.5*60:
        if not self.subs_made_this_quarter:
            # Sub out all starters for bench
            for i in range(5):
                bench_player = self.find_best_bench_player('home')
                self.substitute_player('home', i, bench_player)
            
            for i in range(5):
                bench_player = self.find_best_bench_player('away')
                self.substitute_player('away', i, bench_player)
            
            self.subs_made_this_quarter = True
    
    # Reset flag each quarter
    if self.quarter_duration > 11*60:
        self.subs_made_this_quarter = False
```

**Pros:**
- ✅ Simple and predictable
- ✅ Matches real NBA patterns
- ✅ Easy to implement

**Cons:**
- ❌ Not adaptive to game situation
- ❌ Ignores actual fatigue levels

---

### Strategy 2: Fatigue-Driven (Adaptive)

```python
def fatigue_driven_subs(self):
    """
    Substitute based on actual energy levels
    More realistic and adaptive
    """
    if not self.deadball:
        return
    
    for team in ['home', 'away']:
        lineup = self.home_lineup if team == 'home' else self.away_lineup
        
        for i, player_id in enumerate(lineup):
            energy = self.player_energy[player_id]
            
            # Tier 1: Emergency (must sub immediately)
            if energy < 25:
                bench = self.find_best_bench_player(team)
                if bench and self.player_energy[bench] > 60:
                    self.substitute_player(team, i, bench)
                    continue
            
            # Tier 2: Should sub (if good replacement available)
            if energy < 45:
                bench = self.find_best_bench_player(team)
                if bench and self.player_energy[bench] > 75:
                    self.substitute_player(team, i, bench)
                    continue
            
            # Tier 3: Maintenance (spread minutes)
            minutes_in_quarter = self.player_minutes[player_id] % 3  # Reset each quarter
            if minutes_in_quarter > 7:  # Played 7+ minutes this quarter
                bench = self.find_best_bench_player(team)
                if bench and self.player_energy[bench] > 80:
                    self.substitute_player(team, i, bench)
```

**Pros:**
- ✅ Adaptive to actual game flow
- ✅ Prevents exhaustion
- ✅ Dynamic and realistic

**Cons:**
- ❌ More complex
- ❌ Can lead to unusual patterns

---

### Strategy 3: Hybrid (Recommended)

**Combine time-based windows with fatigue triggers:**

```python
def hybrid_substitution_strategy(self):
    """
    Best of both worlds:
    - Time windows (6-min mark, between quarters)
    - Emergency fatigue subs
    - Game situation awareness
    """
    if not self.deadball:
        # Can only sub during deadballs
        # Exception: call timeout if emergency
        if self.needs_emergency_timeout():
            self.call_timeout_for_subs()
        return
    
    # Between quarters - always reset to starters
    if self.quarter_duration <= 0:
        self.home_lineup = self.hometeam.starters.copy()
        self.away_lineup = self.awayteam.starters.copy()
        return
    
    # Check game situation
    close_game = self.is_close_game()
    blowout = self.is_blowout()
    
    for team in ['home', 'away']:
        lineup = self.home_lineup if team == 'home' else self.away_lineup
        
        # EMERGENCY SUBS (always)
        for i, player_id in enumerate(lineup):
            # Foul trouble
            if boxscore.loc[player_id, 'PF'] >= 5:
                self.substitute_player(team, i, self.find_best_bench_player(team))
                continue
            
            # Exhaustion
            if self.player_energy[player_id] < 20:
                self.substitute_player(team, i, self.find_best_bench_player(team))
                continue
        
        # REGULAR ROTATION SUBS (if not close game)
        if not close_game:
            # 6-minute mark
            if 5.5*60 < self.quarter_duration < 6.5*60:
                for i, player_id in enumerate(lineup):
                    if self.player_energy[player_id] < 70:
                        bench = self.find_best_bench_player(team)
                        if bench and self.player_energy[bench] > 80:
                            self.substitute_player(team, i, bench)
        
        # BLOWOUT SUBS (rest everyone)
        if blowout:
            # Winning by 20+? Rest all starters
            for i, player_id in enumerate(lineup):
                if player_id in self.hometeam.starters:
                    bench = self.find_best_bench_player(team)
                    self.substitute_player(team, i, bench)
```

---

## 5. Timeout Strategy

### When to Call Timeout

```python
def should_call_timeout(self, team):
    """
    Decide when to use a timeout
    
    Timeout purposes:
    1. Force deadball for substitutions
    2. Stop opponent momentum
    3. Advance ball (late game)
    4. Ice opponent free throw shooter
    """
    if team == 'home':
        timeouts = self.home_timeouts
        lineup = self.home_lineup
    else:
        timeouts = self.away_timeouts
        lineup = self.away_lineup
    
    if timeouts <= 0:
        return False
    
    # Reason 1: Team exhausted, no natural deadball coming
    avg_energy = np.mean([self.player_energy[pid] for pid in lineup])
    if avg_energy < 45 and not self.deadball:
        # Team is dying, force a sub opportunity
        return True
    
    # Reason 2: Opponent on scoring run (TODO: track runs)
    if self.opponent_scoring_run >= 8:  # 8-0 run
        return True
    
    # Reason 3: Late game management
    if self.quarter == 4 and self.quarter_duration < 120:
        margin = abs(self.home_score - self.away_score)
        if margin <= 3:
            # Save timeouts for final possession
            return False
    
    # Reason 4: Multiple players need rest
    tired_players = sum(1 for pid in lineup if self.player_energy[pid] < 50)
    if tired_players >= 3:
        return True
    
    return False

def call_timeout(self, team):
    """
    Execute a timeout
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
    
    # Time doesn't pass during timeout (clock stops)
    # ... 
    
    # Make substitutions now that we have deadball
    self.check_substitutions()
    
    # Players recover small amount during timeout
    for player_id in self.player_energy.keys():
        self.player_energy[player_id] = min(100, self.player_energy[player_id] + 1.0)
    
    return True
```

---

## 6. Realistic NBA Rotation Patterns

### Standard 9-Man Rotation

```python
"""
Typical NBA team uses 9-10 players regularly:
- 5 starters (32-36 min each = 160-180 min total)
- 3-4 bench (12-20 min each = 36-80 min total)
- Deep bench (garbage time only)

Total: 240 minutes per game (48 min × 5 positions)
"""

def build_rotation(self):
    """
    Define which players are in the rotation
    """
    roster = self.hometeam.roster_df['PLAYER_ID'].tolist()
    
    # Rank by quality
    roster_ranked = sorted(roster, 
                          key=lambda p: self.get_player_quality(p), 
                          reverse=True)
    
    # Top 5 are starters (already set)
    self.starters = roster_ranked[:5]
    
    # Next 4 are rotation (get real minutes)
    self.rotation = roster_ranked[5:9]
    
    # Rest are deep bench (garbage time only)
    self.deep_bench = roster_ranked[9:]
```

### Quarter-by-Quarter Pattern

```python
"""
Q1: Starters play 0:00-6:00, Bench plays 6:00-12:00
Q2: Starters play 0:00-6:00, Bench plays 6:00-12:00
Q3: Starters play 0:00-8:00, Bench plays 8:00-12:00
Q4: Starters play 0:00-2:00, situational 2:00-12:00
"""

def get_rotation_plan_for_quarter(self, quarter):
    """
    Return expected rotation pattern
    """
    if quarter in [1, 2]:
        return {
            'starters_until': 6*60,   # Starters for first 6 min
            'bench_until': 0,          # Bench finishes quarter
            'starters_return': 0       # Starters return next Q
        }
    
    elif quarter == 3:
        return {
            'starters_until': 8*60,   # Starters play more in Q3
            'bench_until': 0,
            'starters_return': 0
        }
    
    elif quarter == 4:
        # Q4 is situational
        if self.is_close_game():
            return {
                'starters_until': -1,  # -1 = play whole quarter
                'bench_until': -1,
                'starters_return': -1
            }
        else:
            return {
                'starters_until': 6*60,
                'bench_until': 0,
                'starters_return': 0
            }
```

---

## 7. Edge Cases & Special Situations

### Foul Out (Player DQ'd)

```python
def check_foul_outs(self):
    """Remove players who fouled out"""
    for player_id in self.home_lineup:
        if self.hometeam_boxscore.loc[player_id, 'PF'] >= 6:
            # Player fouled out - permanent removal
            position = self.home_lineup.index(player_id)
            bench = self.find_best_bench_player('home')
            
            if bench:
                self.substitute_player('home', position, bench)
                # Mark player as fouled out
                self.players_fouled_out.append(player_id)
```

### No Bench Available

```python
def find_best_bench_player(self, team):
    """Handle case where no bench players available"""
    roster = self.hometeam.roster_df if team == 'home' else self.awayteam.roster_df
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    
    bench = [pid for pid in roster['PLAYER_ID'] if pid not in lineup]
    
    if not bench:
        # No bench left (injuries, foul outs, etc.)
        return None
    
    # Return freshest available
    return max(bench, key=lambda p: self.player_energy[p])
```

### Injury (Future)

```python
def handle_injury(self, player_id):
    """
    Random injury occurrence
    """
    # Small chance each possession
    injury_chance = 0.0001  # 0.01% per possession
    
    if random.random() < injury_chance:
        # Player injured - remove from game
        if player_id in self.home_lineup:
            position = self.home_lineup.index(player_id)
            bench = self.find_best_bench_player('home')
            self.substitute_player('home', position, bench)
            
            # Mark as injured
            self.injured_players.append(player_id)
        
        return True
    
    return False
```

---

## 8. Complete Example Implementation

### Full Substitution Logic

```python
def manage_substitutions(self):
    """
    Complete substitution management
    Called every possession (but only acts during deadballs)
    """
    # Step 1: Check for emergency situations
    self.check_foul_outs()
    
    # Step 2: Check if timeout needed for subs
    if self.should_call_timeout('home'):
        self.call_timeout('home')
    if self.should_call_timeout('away'):
        self.call_timeout('away')
    
    # Step 3: Make substitutions (only during deadballs)
    if self.deadball:
        # Between quarters - always return starters
        if self.quarter_duration <= 0:
            self.home_lineup = self.hometeam.starters.copy()
            self.away_lineup = self.awayteam.starters.copy()
        
        # Regular game - check fatigue and timing
        else:
            self.make_rotation_subs('home')
            self.make_rotation_subs('away')

def make_rotation_subs(self, team):
    """Execute substitutions for one team"""
    lineup = self.home_lineup if team == 'home' else self.away_lineup
    
    # Get game context
    close_game = self.is_close_game()
    blowout = self.is_blowout()
    
    for i, player_id in enumerate(lineup):
        energy = self.player_energy[player_id]
        minutes = self.player_minutes[player_id]
        fouls = self.get_player_fouls(player_id, team)
        
        should_sub = False
        
        # Critical situations (always sub)
        if fouls >= 5:
            should_sub = True
        elif energy < 20:
            should_sub = True
        
        # Regular situations (sub if not close game)
        elif not close_game:
            if energy < 40:
                should_sub = True
            elif 5.5*60 < self.quarter_duration < 6.5*60:
                should_sub = True  # Regular rotation
        
        # Blowout (rest everyone)
        elif blowout:
            if player_id in self.hometeam.starters:
                should_sub = True
        
        # Execute substitution
        if should_sub:
            bench_player = self.find_best_bench_player(team)
            if bench_player:
                self.substitute_player(team, i, bench_player)
```

---

## 9. Summary: Recommended Approach

### Start Simple, Add Complexity

**Phase 1: Basic Time-Based**
```python
# Sub at 6-minute mark, between quarters
# Bring back starters each quarter
# Simple and works
```

**Phase 2: Add Fatigue**
```python
# Sub when energy < 40 (tired)
# Emergency sub when energy < 20 (exhausted)
# Keep stars in close games even if tired
```

**Phase 3: Add Context**
```python
# Close games: play starters more
# Blowouts: rest everyone
# Foul trouble: pull players early
```

**Phase 4: Add Quality/Role**
```python
# Match replacement quality to player going out
# Maintain position balance
# Track hot/cold performance
```

---

## Quick Decision Tree

```
Is it a deadball?
├─ NO → Can only sub if:
│       └─ Team exhausted (avg energy < 30) → Call timeout
│
└─ YES → Check priority:
    ├─ 1. Foul out (6 fouls) → MUST SUB
    ├─ 2. Exhausted (energy < 20) → MUST SUB
    ├─ 3. Quarter end → BRING BACK STARTERS
    ├─ 4. Close game + Q4 → KEEP STARTERS (even if tired)
    ├─ 5. Blowout → REST STARTERS
    ├─ 6. Very tired (energy < 40) → SHOULD SUB
    └─ 7. 6-minute mark → REGULAR ROTATION
```

---

## Key Metrics to Track

```python
# For each player:
player_energy[player_id]          # Current energy (0-100)
player_minutes[player_id]         # Minutes played
get_fatigue_multiplier(player_id) # Performance % (0.5-1.0)

# For substitution decisions:
get_player_quality(player_id)     # How good is this player?
get_player_role(player_id)        # Star, rotation, or deep bench?
is_player_cold(player_id)         # Shooting poorly?
```

---

## Summary

**WHEN to sub:**
- ✅ Fatigue (energy < 40)
- ✅ Time windows (6-min mark)
- ✅ Between quarters (always)
- ✅ Foul trouble (5-6 fouls)
- ✅ Poor performance (0-5 shooting)

**WHO to take out:**
- ✅ Most tired player
- ✅ Foul trouble
- ✅ Worst current performance
- ✅ Longest time since rest

**WHO to bring in:**
- ✅ Freshest bench player (energy > 80)
- ✅ Same position if possible
- ✅ Quality match (don't sub star for deep bench)
- ✅ Maintain lineup balance

**Timeouts for:**
- ✅ Force deadball when team exhausted
- ✅ Make multiple subs at once
- ✅ Strategic (stop runs, late game)

Start with **simple time-based + fatigue triggers**, then add complexity as needed! 🏀

