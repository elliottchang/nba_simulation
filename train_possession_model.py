"""
NBA Possession Prediction Model - Hybrid Stats + Player Embeddings
This module implements a neural network that predicts how NBA possessions end,
using season-specific player embeddings and lineup statistics.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
from datetime import datetime
import time
import os

from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.library.parameters import SeasonAll


# ============================================================================
# Data Collection
# ============================================================================

def retry_api_call(func, max_retries: int = 3, base_delay: float = 2.0, timeout: int = 60):
    """
    Retry an API call with exponential backoff.
    
    Args:
        func: Function to call (should be a lambda or callable with no args)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be doubled for each retry)
        timeout: Timeout for the API call in seconds
    
    Returns:
        Result of the function call, or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            
            delay = base_delay * (2 ** attempt)
            print(f"  Attempt {attempt + 1} failed: {e}")
            print(f"  Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
    
    return None


class PlayByPlayDataCollector:
    """Collects and processes play-by-play data from NBA API."""
    
    def __init__(self, seasons: List[str], checkpoint_file: str = 'data/checkpoint.pkl'):
        """
        Args:
            seasons: List of seasons in format 'YYYY-YY' (e.g., ['2022-23', '2023-24'])
            checkpoint_file: Path to save/load checkpoint data
        """
        self.seasons = seasons
        self.pbp_data = []
        self.player_stats = {}
        self.checkpoint_file = checkpoint_file
        self.processed_games = set()  # Track which games we've already processed
        self.failed_games = {}  # Track failed games and retry counts
        
        # Adaptive rate limiting
        self.base_delay = 1.5  # Increased from 1.0
        self.current_delay = self.base_delay
        self.consecutive_errors = 0
        self.error_threshold = 3  # Number of consecutive errors before cooldown
    
    def handle_api_error(self, save_data: bool = True):
        """Handle API error by increasing delay and saving progress.
        
        Args:
            save_data: If True, save checkpoint with CSV backup
        """
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= self.error_threshold:
            print(f"\n⚠️  {self.consecutive_errors} consecutive errors detected!")
            print(f"   Saving progress before cooldown...")
            
            # Save checkpoint with CSV backup
            if save_data and len(self.pbp_data) > 0:
                self.save_checkpoint(save_csv=True, reason="error_recovery")
            
            cooldown = 30 + (self.consecutive_errors - self.error_threshold) * 15
            print(f"   Entering cooldown period: {cooldown} seconds...")
            print(f"   This gives the NBA API time to recover.")
            print(f"   💡 You can also stop (Ctrl+C) and restart later - progress is saved!")
            time.sleep(cooldown)
            # Reset after cooldown
            self.consecutive_errors = 0
            self.current_delay = self.base_delay
        else:
            # Gradually increase delay
            self.current_delay = min(self.current_delay * 1.5, 5.0)
            print(f"   Adjusted API delay to {self.current_delay:.1f}s")
    
    def handle_api_success(self):
        """Handle successful API call by resetting error tracking."""
        if self.consecutive_errors > 0:
            print(f"   ✓ Connection restored! Resetting error count.")
        self.consecutive_errors = 0
        # Gradually decrease delay back to normal
        self.current_delay = max(self.current_delay * 0.9, self.base_delay)
        
    def collect_games_for_season(self, season: str, max_games: Optional[int] = None):
        """Collect all games for a given season."""
        print(f"\nCollecting games for {season} season...")
        
        try:
            # Get all games for the season with retry logic
            games_df = retry_api_call(
                lambda: leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season',
                    timeout=60
                ).get_data_frames()[0],
                max_retries=3,
                base_delay=2.0
            )
            
            # Get unique game IDs (each game appears twice, once per team)
            game_ids = games_df['GAME_ID'].unique()
            
            if max_games:
                game_ids = game_ids[:max_games]
            
            print(f"Found {len(game_ids)} games for {season}")
            return game_ids
            
        except Exception as e:
            print(f"Error collecting games for {season}: {e}")
            return []
    
    def parse_possession_outcome(self, events: pd.DataFrame, poss_events: pd.DataFrame) -> Dict:
        """
        Parse a possession to determine outcome and player who ended it.
        
        New outcome categories:
        - 3pt_make, 2pt_make: Clean made baskets
        - 3pt_miss, 2pt_miss: Missed shots (no foul)
        - 3pt_andone, 2pt_andone: Made basket + drawn foul (and-one)
        - 3pt_foul, 2pt_foul: Shooting foul, no basket made
        - TO: Turnover
        - def_foul: Defensive foul (non-shooting)
        - off_foul: Offensive foul
        
        Returns:
            Dict with keys: 'outcome', 'player_id', 'player_name', 'team_id'
        """
        if len(poss_events) == 0:
            return None
        
        # NBA API Event types:
        # 1 = Made shot
        # 2 = Missed shot
        # 3 = Free throw
        # 5 = Turnover
        # 6 = Foul
        
        # Get the last event and check for fouls in the possession
        last_event = poss_events.iloc[-1]
        event_type = last_event['EVENTMSGTYPE']
        
        # Helper function to check if shot is 3PT
        def is_3pt(event):
            desc = str(event.get('HOMEDESCRIPTION', '')) + str(event.get('VISITORDESCRIPTION', ''))
            return '3PT' in desc
        
        # Check for fouls in the possession
        has_foul = any(poss_events['EVENTMSGTYPE'] == 6)
        
        # Check for made shots in the possession  
        made_shot_events = poss_events[poss_events['EVENTMSGTYPE'] == 1]
        has_made_shot = len(made_shot_events) > 0
        
        # Special case: Foul after made shot = And-one
        if event_type == 6 and has_made_shot:
            # Last event is foul, but there's a made shot before it
            made_shot_event = made_shot_events.iloc[-1]
            is_three = is_3pt(made_shot_event)
            
            # Check if this is a defensive foul (not offensive)
            foul_desc = str(last_event.get('HOMEDESCRIPTION', '')) + str(last_event.get('VISITORDESCRIPTION', ''))
            if 'OFF.FOUL' not in foul_desc.upper() and 'OFFENSIVE' not in foul_desc.upper():
                # And-one: made basket + defensive foul
                outcome = '3pt_andone' if is_three else '2pt_andone'
                # Use the shooter as the player, not the fouler
                return {
                    'outcome': outcome,
                    'player_id': made_shot_event.get('PLAYER1_ID'),
                    'player_name': made_shot_event.get('PLAYER1_NAME'),
                    'team_id': made_shot_event.get('PLAYER1_TEAM_ID')
                }
        
        # Determine outcome based on event sequence
        if event_type == 1:  # Made shot (no foul after)
            is_three = is_3pt(last_event)
            outcome = '3pt_make' if is_three else '2pt_make'
                
        elif event_type == 2:  # Missed shot
            is_three = is_3pt(last_event)
            
            # Check if there's a shooting foul
            # Look for fouls around the missed shot
            if has_foul:
                # Check if it's a shooting foul by looking at event descriptions
                foul_events = poss_events[poss_events['EVENTMSGTYPE'] == 6]
                for _, foul_event in foul_events.iterrows():
                    foul_desc = str(foul_event.get('HOMEDESCRIPTION', '')) + str(foul_event.get('VISITORDESCRIPTION', ''))
                    if 'SHOOTING' in foul_desc.upper() or 'S.FOUL' in foul_desc.upper():
                        # Shooting foul on missed shot
                        outcome = '3pt_foul' if is_three else '2pt_foul'
                        break
                else:
                    # Regular missed shot
                    outcome = '3pt_miss' if is_three else '2pt_miss'
            else:
                # Regular missed shot (no foul)
                outcome = '3pt_miss' if is_three else '2pt_miss'
                
        elif event_type == 5:  # Turnover
            # Check if it's an offensive foul (which is a type of turnover)
            to_desc = str(last_event.get('HOMEDESCRIPTION', '')) + str(last_event.get('VISITORDESCRIPTION', ''))
            if 'OFF.FOUL' in to_desc.upper() or 'OFFENSIVE FOUL' in to_desc.upper() or 'CHARGE' in to_desc.upper():
                outcome = 'off_foul'
            else:
                outcome = 'TO'
                
        elif event_type == 6:  # Foul
            # Foul ends the possession
            foul_desc = str(last_event.get('HOMEDESCRIPTION', '')) + str(last_event.get('VISITORDESCRIPTION', ''))
            
            # Check foul type
            if 'OFF.FOUL' in foul_desc.upper() or 'OFFENSIVE' in foul_desc.upper() or 'CHARGE' in foul_desc.upper():
                outcome = 'off_foul'
            elif 'SHOOTING' in foul_desc.upper() or 'S.FOUL' in foul_desc.upper():
                # Shooting foul - try to determine if 2pt or 3pt
                # Look at preceding events for shot attempts
                if len(poss_events) > 1:
                    prev_event = poss_events.iloc[-2]
                    if prev_event['EVENTMSGTYPE'] == 2:  # Preceded by missed shot
                        is_three = is_3pt(prev_event)
                        outcome = '3pt_foul' if is_three else '2pt_foul'
                    else:
                        # Default to 2pt foul if can't determine
                        outcome = '2pt_foul'
                else:
                    outcome = '2pt_foul'
            else:
                # Non-shooting defensive foul
                outcome = 'def_foul'
                
        elif event_type == 3:  # Free throw
            # Free throws typically don't end possessions in our data collection
            # But if they do, we need to trace back to what caused them
            # For now, classify based on make/miss and try to infer shot type
            ft_desc = str(last_event.get('HOMEDESCRIPTION', '')) + str(last_event.get('VISITORDESCRIPTION', ''))
            
            # Look back for the shot that led to free throws
            shot_events = poss_events[poss_events['EVENTMSGTYPE'].isin([1, 2])]
            if len(shot_events) > 0:
                shot_event = shot_events.iloc[-1]
                is_three = is_3pt(shot_event)
                if shot_event['EVENTMSGTYPE'] == 1:  # Made shot (and-one)
                    outcome = '3pt_andone' if is_three else '2pt_andone'
                else:  # Missed shot (shooting foul)
                    outcome = '3pt_foul' if is_three else '2pt_foul'
            else:
                # No shot found, might be non-shooting foul leading to FTs
                outcome = 'def_foul'
        else:
            # Other event types - skip
            return None
        
        return {
            'outcome': outcome,
            'player_id': last_event.get('PLAYER1_ID'),
            'player_name': last_event.get('PLAYER1_NAME'),
            'team_id': last_event.get('PLAYER1_TEAM_ID')
        }
    
    def get_starting_lineups(self, game_id: str) -> Dict:
        """
        Get starting lineups from boxscore.
        
        Returns:
            Dict with 'home' and 'away' lists of player IDs
        """
        try:
            # Use adaptive delay
            time.sleep(self.current_delay)
            
            # Use retry logic for API call
            player_stats = retry_api_call(
                lambda: boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=game_id, timeout=60
                ).get_data_frames()[0],
                max_retries=3,
                base_delay=2.0
            )
            
            # Mark success
            self.handle_api_success()
            
            if player_stats.empty or 'START_POSITION' not in player_stats.columns:
                return None
            
            # Get unique team IDs
            team_ids = player_stats['TEAM_ID'].unique()
            if len(team_ids) != 2:
                return None
            
            home_team_id = team_ids[0]
            away_team_id = team_ids[1]
            
            # Get starters (players with non-empty START_POSITION)
            starters = player_stats[player_stats['START_POSITION'].notna()]
            
            home_starters = starters[starters['TEAM_ID'] == home_team_id]['PLAYER_ID'].tolist()
            away_starters = starters[starters['TEAM_ID'] == away_team_id]['PLAYER_ID'].tolist()
            
            # Filter to exactly 5 starters per team (sometimes there's inconsistent data)
            # Prioritize by minutes played
            if len(home_starters) != 5 or len(away_starters) != 5:
                # Fall back to top 5 by minutes (convert MIN to numeric first)
                player_stats['MIN_NUMERIC'] = pd.to_numeric(player_stats['MIN'], errors='coerce').fillna(0)
                home_players = player_stats[player_stats['TEAM_ID'] == home_team_id].nlargest(5, 'MIN_NUMERIC')
                away_players = player_stats[player_stats['TEAM_ID'] == away_team_id].nlargest(5, 'MIN_NUMERIC')
                home_starters = home_players['PLAYER_ID'].tolist()
                away_starters = away_players['PLAYER_ID'].tolist()
            
            return {
                'home': home_starters[:5],  # Ensure exactly 5
                'away': away_starters[:5],
                'home_team_id': home_team_id,
                'away_team_id': away_team_id
            }
            
        except Exception as e:
            print(f"  Warning: Could not get starters for game {game_id}: {e}")
            self.handle_api_error()
            return None
    
    def extract_possessions_from_game(self, game_id: str, season: str) -> List[Dict]:
        """Extract possession-level data with full lineup tracking."""
        try:
            # Use adaptive delay
            time.sleep(self.current_delay)
            
            # Use retry logic for API call
            events = retry_api_call(
                lambda: playbyplayv2.PlayByPlayV2(
                    game_id=game_id, timeout=60
                ).get_data_frames()[0],
                max_retries=3,
                base_delay=2.0
            )
            
            # Mark success
            self.handle_api_success()
            
            if events.empty:
                return []
            
            # Get starting lineups
            lineup_data = self.get_starting_lineups(game_id)
            if lineup_data is None:
                # Skip games where we can't get lineups
                print(f"  Skipping game {game_id}: no lineup data")
                return []
            
            # Initialize lineups
            home_lineup = set(lineup_data['home'])
            away_lineup = set(lineup_data['away'])
            home_team_id = lineup_data['home_team_id']
            away_team_id = lineup_data['away_team_id']
            
            possessions = []
            current_possession = []
            
            for idx, event in events.iterrows():
                event_type = event['EVENTMSGTYPE']
                
                # Track substitutions (event type 8)
                if event_type == 8:
                    player_in = event.get('PLAYER1_ID')
                    player_out = event.get('PLAYER2_ID')
                    team_id = event.get('PLAYER1_TEAM_ID')
                    
                    if pd.notna(player_in) and pd.notna(player_out) and pd.notna(team_id):
                        if team_id == home_team_id:
                            if player_out in home_lineup:
                                home_lineup.remove(player_out)
                            home_lineup.add(player_in)
                        elif team_id == away_team_id:
                            if player_out in away_lineup:
                                away_lineup.remove(player_out)
                            away_lineup.add(player_in)
                
                # Events that end possessions
                # 1=made shot, 2=missed shot, 3=free throw, 5=turnover, 6=foul
                if event_type in [1, 2, 5, 6]:  # Made shot, missed shot, turnover, foul
                    current_possession.append(event)
                    
                    # Special case: If this is a made shot, check if next event is a foul (and-one)
                    should_continue = False
                    if event_type == 1 and idx < len(events) - 1:  # Made shot and not last event
                        next_event = events.iloc[idx + 1]
                        if next_event['EVENTMSGTYPE'] == 6:  # Next event is a foul
                            should_continue = True  # Don't process yet, wait for foul
                    
                    if should_continue:
                        continue  # Add foul to possession in next iteration
                    
                    # Process the possession
                    if len(current_possession) > 0:
                        poss_df = pd.DataFrame(current_possession)
                        outcome = self.parse_possession_outcome(events, poss_df)
                        
                        if outcome and outcome['player_id'] is not None and outcome['outcome'] is not None:
                            # Determine offensive/defensive teams
                            is_home_offense = outcome['team_id'] == home_team_id
                            
                            # Get current lineups (ensure exactly 5 players)
                            current_home = list(home_lineup)[:5]
                            current_away = list(away_lineup)[:5]
                            
                            # Skip if lineups are incomplete
                            if len(current_home) < 5 or len(current_away) < 5:
                                current_possession = []
                                continue
                            
                            offensive_lineup = current_home if is_home_offense else current_away
                            defensive_lineup = current_away if is_home_offense else current_home
                            
                            # Get score differential and time remaining
                            score_margin = event.get('SCOREMARGIN', '0')
                            try:
                                score_margin = int(score_margin) if score_margin not in ['TIE', None, ''] else 0
                            except:
                                score_margin = 0
                            
                            possessions.append({
                                'game_id': game_id,
                                'season': season,
                                'period': event['PERIOD'],
                                'time_remaining': event.get('PCTIMESTRING', '12:00'),
                                'score_margin': score_margin,
                                'offensive_team_id': outcome['team_id'],
                                'outcome': outcome['outcome'],
                                'player_id': outcome['player_id'],
                                'player_name': outcome['player_name'],
                                # NEW: Full lineup data
                                'offensive_lineup': offensive_lineup,
                                'defensive_lineup': defensive_lineup,
                            })
                    
                    # Reset for next possession
                    current_possession = []
                elif event_type == 3:  # Free throw
                    # Add to possession but don't end it (unless it's the last FT)
                    current_possession.append(event)
                    
                    # Check if this is the last free throw in the sequence
                    # If it is and possession should end, we can process it
                    # For now, we'll keep accumulating until a shot/turnover/foul
                else:
                    # Other events (rebounds, etc.) - add to possession
                    current_possession.append(event)
            
            return possessions
            
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            self.handle_api_error()
            return []
    
    def save_checkpoint(self, save_csv: bool = False, reason: str = "periodic"):
        """Save current progress to checkpoint file and optionally CSV.
        
        Args:
            save_csv: If True, also save data to CSV
            reason: Reason for checkpoint (for logging)
        """
        checkpoint = {
            'pbp_data': self.pbp_data,
            'processed_games': self.processed_games,
            'failed_games': self.failed_games,
            'current_delay': self.current_delay,
            'consecutive_errors': self.consecutive_errors
        }
        
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save pickle checkpoint
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  💾 Checkpoint saved ({reason}): {len(self.processed_games)} games processed, {len(self.pbp_data)} possessions")
        
        # Save CSV if requested
        if save_csv and len(self.pbp_data) > 0:
            csv_path = 'data/pbp_data_partial.csv'
            os.makedirs('data', exist_ok=True)
            df = pd.DataFrame(self.pbp_data)
            df.to_csv(csv_path, index=False)
            print(f"  📊 Partial data saved to {csv_path}")
    
    def load_checkpoint(self):
        """Load progress from checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.pbp_data = checkpoint.get('pbp_data', [])
                self.processed_games = checkpoint.get('processed_games', set())
                self.failed_games = checkpoint.get('failed_games', {})
                self.current_delay = checkpoint.get('current_delay', self.base_delay)
                self.consecutive_errors = checkpoint.get('consecutive_errors', 0)
                
                print(f"\n✅ Resuming from checkpoint:")
                print(f"  - {len(self.processed_games)} games already processed")
                print(f"  - {len(self.pbp_data)} possessions collected")
                print(f"  - {len(self.failed_games)} games previously failed")
                print(f"  - Current delay: {self.current_delay:.1f}s")
                
                # Check if there's also a partial CSV
                partial_csv = 'data/pbp_data_partial.csv'
                if os.path.exists(partial_csv):
                    print(f"  - Partial CSV backup found: {partial_csv}")
                
                return True
            except Exception as e:
                print(f"Could not load checkpoint: {e}")
                return False
        return False
    
    def collect_all_data(self, max_games_per_season: Optional[int] = None):
        """Collect play-by-play data for all specified seasons with checkpointing."""
        # Try to load existing checkpoint
        self.load_checkpoint()
        
        all_possessions = self.pbp_data if isinstance(self.pbp_data, list) else []
        
        for season in self.seasons:
            game_ids = self.collect_games_for_season(season, max_games_per_season)
            
            # Filter out already processed games
            remaining_games = [g for g in game_ids if g not in self.processed_games]
            
            print(f"Processing {len(remaining_games)} games for {season} (skipping {len(game_ids) - len(remaining_games)} already processed)...")
            
            for i, game_id in enumerate(remaining_games):
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(remaining_games)} games...")
                
                # Skip games that have failed too many times
                if game_id in self.failed_games and self.failed_games[game_id] >= 3:
                    print(f"  Skipping game {game_id}: failed {self.failed_games[game_id]} times")
                    continue
                
                try:
                    possessions = self.extract_possessions_from_game(game_id, season)
                    all_possessions.extend(possessions)
                    self.processed_games.add(game_id)
                    
                    # Reset failure count on success
                    if game_id in self.failed_games:
                        del self.failed_games[game_id]
                    
                except Exception as e:
                    print(f"❌ Error processing game {game_id}: {e}")
                    self.failed_games[game_id] = self.failed_games.get(game_id, 0) + 1
                    
                    # Save immediately on error with CSV backup
                    self.pbp_data = all_possessions
                    print(f"   💾 Saving progress after error...")
                    self.save_checkpoint(save_csv=True, reason="after_error")
                
                # Save checkpoint more frequently if errors are occurring
                checkpoint_frequency = 20 if self.consecutive_errors > 0 else 50
                if (i + 1) % checkpoint_frequency == 0:
                    self.pbp_data = all_possessions
                    # Save with CSV if in error mode
                    save_csv = self.consecutive_errors > 0
                    self.save_checkpoint(save_csv=save_csv, reason="periodic")
            
            print(f"Collected {len(all_possessions)} total possessions so far")
            
            # Save checkpoint after each season with CSV backup
            self.pbp_data = all_possessions
            self.save_checkpoint(save_csv=True, reason="season_complete")
        
        # Final save
        self.pbp_data = all_possessions
        self.save_checkpoint(save_csv=True, reason="collection_complete")
        
        print(f"\n✅ Data collection complete!")
        print(f"  Total possessions collected: {len(all_possessions)}")
        print(f"  Successfully processed games: {len(self.processed_games)}")
        print(f"  Failed/skipped games: {len(self.failed_games)}")
        
        self.pbp_data = pd.DataFrame(all_possessions)
        return self.pbp_data
    
    def save_data(self, filepath: str):
        """Save collected data to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else 'data', exist_ok=True)
        self.pbp_data.to_csv(filepath, index=False)
        print(f"Saved {len(self.pbp_data)} possessions to {filepath}")
        
        # If saving the final file, clean up partial CSV
        if 'partial' not in filepath:
            partial_csv = 'data/pbp_data_partial.csv'
            if os.path.exists(partial_csv):
                os.remove(partial_csv)
                print(f"  🧹 Cleaned up partial CSV file")
    
    def load_data(self, filepath: str):
        """Load previously collected data."""
        self.pbp_data = pd.read_csv(filepath)
        print(f"Loaded {len(self.pbp_data)} possessions from {filepath}")
        return self.pbp_data
    
    def get_collection_status(self):
        """Get current collection status."""
        return {
            'games_processed': len(self.processed_games),
            'possessions_collected': len(self.pbp_data) if isinstance(self.pbp_data, list) else len(self.pbp_data),
            'games_failed': len(self.failed_games),
            'current_delay': self.current_delay,
            'consecutive_errors': self.consecutive_errors
        }


# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureEngineer:
    """Extracts and processes features for the possession prediction model."""
    
    def __init__(self, pbp_data: pd.DataFrame):
        self.pbp_data = pbp_data
        self.player_season_stats = {}
        self.player_to_idx = {}
        self.idx_to_player = {}
        self.outcome_to_idx = {}
        self.idx_to_outcome = {}
        
    def build_player_season_vocab(self):
        """Create mapping of (player_id, season) tuples to indices."""
        unique_combos = set()
        
        for _, row in self.pbp_data.iterrows():
            player_id = row['player_id']
            season = row['season']
            if pd.notna(player_id):
                unique_combos.add((int(player_id), season))
        
        # Add special tokens
        self.player_to_idx[('PAD', 'PAD')] = 0
        self.player_to_idx[('UNK', 'UNK')] = 1
        
        for i, combo in enumerate(sorted(unique_combos), start=2):
            self.player_to_idx[combo] = i
            self.idx_to_player[i] = combo
        
        print(f"Built vocabulary with {len(self.player_to_idx)} player-season pairs")
        return self.player_to_idx
    
    def build_outcome_vocab(self):
        """Create mapping of outcomes to indices."""
        unique_outcomes = self.pbp_data['outcome'].unique()
        
        self.outcome_to_idx = {outcome: i for i, outcome in enumerate(sorted(unique_outcomes))}
        self.idx_to_outcome = {i: outcome for outcome, i in self.outcome_to_idx.items()}
        
        print(f"Built outcome vocabulary with {len(self.outcome_to_idx)} classes:")
        for outcome, idx in self.outcome_to_idx.items():
            count = len(self.pbp_data[self.pbp_data['outcome'] == outcome])
            print(f"  {outcome}: {count} samples")
        
        return self.outcome_to_idx
    
    def compute_player_season_stats(self):
        """Compute aggregate statistics for each player-season pair."""
        # Compute stats from possession data using new outcome categories
        
        stats = defaultdict(lambda: {
            'possessions': 0,
            '2pt_attempts': 0,
            '2pt_makes': 0,
            '3pt_attempts': 0,
            '3pt_makes': 0,
            'turnovers': 0,
            'fouls_drawn': 0,
            'fouls_committed': 0,
            'andones': 0,
        })
        
        for _, row in self.pbp_data.iterrows():
            key = (int(row['player_id']), row['season'])
            stats[key]['possessions'] += 1
            
            outcome = row['outcome']
            
            # 2-point attempts and makes
            if outcome in ['2pt_make', '2pt_andone']:
                stats[key]['2pt_attempts'] += 1
                stats[key]['2pt_makes'] += 1
                if outcome == '2pt_andone':
                    stats[key]['andones'] += 1
                    stats[key]['fouls_drawn'] += 1
            elif outcome in ['2pt_miss', '2pt_foul']:
                stats[key]['2pt_attempts'] += 1
                if outcome == '2pt_foul':
                    stats[key]['fouls_drawn'] += 1
            
            # 3-point attempts and makes
            elif outcome in ['3pt_make', '3pt_andone']:
                stats[key]['3pt_attempts'] += 1
                stats[key]['3pt_makes'] += 1
                if outcome == '3pt_andone':
                    stats[key]['andones'] += 1
                    stats[key]['fouls_drawn'] += 1
            elif outcome in ['3pt_miss', '3pt_foul']:
                stats[key]['3pt_attempts'] += 1
                if outcome == '3pt_foul':
                    stats[key]['fouls_drawn'] += 1
            
            # Turnovers
            elif outcome == 'TO':
                stats[key]['turnovers'] += 1
            
            # Fouls
            elif outcome == 'off_foul':
                stats[key]['fouls_committed'] += 1
                stats[key]['turnovers'] += 1  # Offensive fouls are turnovers
            elif outcome == 'def_foul':
                # This is tracked for defensive players, not offensive
                pass
        
        # Compute rates
        for key, stat_dict in stats.items():
            poss = max(stat_dict['possessions'], 1)
            stat_dict['usage_rate'] = stat_dict['possessions'] / poss
            stat_dict['2pt_rate'] = stat_dict['2pt_attempts'] / poss
            stat_dict['3pt_rate'] = stat_dict['3pt_attempts'] / poss
            stat_dict['to_rate'] = stat_dict['turnovers'] / poss
            
            # Shooting percentages
            if stat_dict['2pt_attempts'] > 0:
                stat_dict['2pt_pct'] = stat_dict['2pt_makes'] / stat_dict['2pt_attempts']
            else:
                stat_dict['2pt_pct'] = 0.0
                
            if stat_dict['3pt_attempts'] > 0:
                stat_dict['3pt_pct'] = stat_dict['3pt_makes'] / stat_dict['3pt_attempts']
            else:
                stat_dict['3pt_pct'] = 0.0
            
            # Additional rates
            stat_dict['foul_drawn_rate'] = stat_dict['fouls_drawn'] / poss
            stat_dict['foul_commit_rate'] = stat_dict['fouls_committed'] / poss
            stat_dict['andone_rate'] = stat_dict['andones'] / poss
        
        self.player_season_stats = dict(stats)
        print(f"Computed stats for {len(self.player_season_stats)} player-season pairs")
        return self.player_season_stats
    
    def get_stat_features(self, player_id: int, season: str) -> np.ndarray:
        """Get statistical features for a player-season."""
        key = (player_id, season)
        
        if key not in self.player_season_stats:
            # Return league average or zeros for unknown players
            return np.zeros(9)  # Updated size
        
        stats = self.player_season_stats[key]
        return np.array([
            stats['usage_rate'],
            stats['2pt_rate'],
            stats['3pt_rate'],
            stats['to_rate'],
            stats['2pt_pct'],
            stats['3pt_pct'],
            stats['foul_drawn_rate'],
            stats['foul_commit_rate'],
            stats['andone_rate'],
        ], dtype=np.float32)


# ============================================================================
# PyTorch Dataset
# ============================================================================

class PossessionDataset(Dataset):
    """PyTorch dataset for possession prediction."""
    
    def __init__(self, pbp_data: pd.DataFrame, feature_engineer: FeatureEngineer):
        self.data = pbp_data.reset_index(drop=True)
        self.fe = feature_engineer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        season = row['season']
        
        # Parse lineup data (stored as lists)
        # If data was loaded from CSV, lineups might be strings
        offensive_lineup = row['offensive_lineup']
        defensive_lineup = row['defensive_lineup']
        
        if isinstance(offensive_lineup, str):
            import ast
            offensive_lineup = ast.literal_eval(offensive_lineup)
            defensive_lineup = ast.literal_eval(defensive_lineup)
        
        # Convert to player-season indices
        unk_idx = self.fe.player_to_idx[('UNK', 'UNK')]
        
        offensive_indices = []
        for player_id in offensive_lineup[:5]:  # Ensure exactly 5
            key = (int(player_id), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            offensive_indices.append(idx_val)
        
        defensive_indices = []
        for player_id in defensive_lineup[:5]:  # Ensure exactly 5
            key = (int(player_id), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            defensive_indices.append(idx_val)
        
        # Pad if necessary (should not happen, but safety check)
        while len(offensive_indices) < 5:
            offensive_indices.append(unk_idx)
        while len(defensive_indices) < 5:
            defensive_indices.append(unk_idx)
        
        # Context features
        context = np.array([
            row['score_margin'],
            row['period'],
        ], dtype=np.float32)
        
        # Target
        outcome_idx = self.fe.outcome_to_idx[row['outcome']]
        
        return {
            'offensive_lineup': torch.tensor(offensive_indices, dtype=torch.long),  # (5,)
            'defensive_lineup': torch.tensor(defensive_indices, dtype=torch.long),  # (5,)
            'context': torch.tensor(context, dtype=torch.float32),
            'outcome': torch.tensor(outcome_idx, dtype=torch.long),
        }


# ============================================================================
# Neural Network Model
# ============================================================================

class PossessionPredictionModel(nn.Module):
    """
    Lineup-aware model using player embeddings for full 10-player context.
    Predicts possession outcome based on 5 offensive + 5 defensive players.
    """
    
    def __init__(
        self,
        num_player_seasons: int,
        num_outcomes: int,
        embedding_dim: int = 64,
        context_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Player-season embeddings (the key innovation!)
        self.player_embeddings = nn.Embedding(num_player_seasons, embedding_dim, padding_idx=0)
        
        # Combine embeddings from 5 offensive + 5 defensive players + context
        # We'll aggregate each lineup to a single vector
        combined_dim = embedding_dim * 2 + context_dim  # offensive_agg + defensive_agg + context
        
        # Neural network layers
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output head for possession outcome
        self.outcome_head = nn.Linear(hidden_dim, num_outcomes)
        
        self.relu = nn.ReLU()
        
    def forward(self, offensive_lineup, defensive_lineup, context):
        """
        Args:
            offensive_lineup: (batch_size, 5) - indices of 5 offensive player-season pairs
            defensive_lineup: (batch_size, 5) - indices of 5 defensive player-season pairs
            context: (batch_size, context_dim) - game context features
        
        Returns:
            outcome_logits: (batch_size, num_outcomes) - logits for possession outcomes
        """
        # Get embeddings for all players
        offensive_emb = self.player_embeddings(offensive_lineup)  # (batch_size, 5, embedding_dim)
        defensive_emb = self.player_embeddings(defensive_lineup)  # (batch_size, 5, embedding_dim)
        
        # Aggregate lineup embeddings (mean pooling)
        # Could also try: sum, max, attention mechanism
        offensive_agg = offensive_emb.mean(dim=1)  # (batch_size, embedding_dim)
        defensive_agg = defensive_emb.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Concatenate offensive, defensive, and context
        x = torch.cat([offensive_agg, defensive_agg, context], dim=1)
        
        # Forward pass through neural network
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Predict outcome
        outcome_logits = self.outcome_head(x)
        
        return outcome_logits


# ============================================================================
# Training Loop
# ============================================================================

class ModelTrainer:
    """Handles training and evaluation of the possession prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        weight_decay: float = 1e-5,
        patience: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        
        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            offensive_lineup = batch['offensive_lineup'].to(self.device)
            defensive_lineup = batch['defensive_lineup'].to(self.device)
            context = batch['context'].to(self.device)
            outcome = batch['outcome'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outcome_logits = self.model(offensive_lineup, defensive_lineup, context)
            
            # Compute loss
            loss = self.criterion(outcome_logits, outcome)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                offensive_lineup = batch['offensive_lineup'].to(self.device)
                defensive_lineup = batch['defensive_lineup'].to(self.device)
                context = batch['context'].to(self.device)
                outcome = batch['outcome'].to(self.device)
                
                # Forward pass
                outcome_logits = self.model(offensive_lineup, defensive_lineup, context)
                
                # Compute loss
                loss = self.criterion(outcome_logits, outcome)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outcome_logits, 1)
                total += outcome.size(0)
                correct += (predicted == outcome).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, early_stopping: bool = True):
        """
        Full training loop with early stopping and learning rate scheduling.
        
        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping: If True, stop when validation loss stops improving
        """
        print(f"\nStarting training for up to {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping: {'Enabled' if early_stopping else 'Disabled'} (patience={self.patience})")
        print(f"Learning rate scheduling: Enabled")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Check if this is the best model so far
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                improved = True
                
                # Save best model state
                self.best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
            else:
                self.epochs_without_improvement += 1
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2f}%")
            if improved:
                print(f"  ✓ New best model! (saved)")
            else:
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if early_stopping and self.epochs_without_improvement >= self.patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Best model was at epoch {self.best_epoch}")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                print(f"   Best val accuracy: {self.best_val_acc:.2f}%")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            print(f"\n✓ Restoring best model from epoch {self.best_epoch}")
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
        
        print("\nTraining complete!")
        print(f"Final results:")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best val accuracy: {self.best_val_acc:.2f}%")
        
        return self.train_losses, self.val_losses, self.val_accuracies
    
    def save_model(self, filepath: str):
        """Save model checkpoint with best model information."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }, filepath)
        print(f"Model saved to {filepath}")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best val accuracy: {self.best_val_acc:.2f}%")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("="*80)
    print("NBA Possession Prediction Model - Training Pipeline")
    print("="*80)
    
    # Configuration
    SEASONS = ['2022-23', '2023-24', '2024-25']  # Start with recent seasons
    MAX_GAMES_PER_SEASON = None  # Limit for testing; set to None for full dataset
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4  # L2 regularization (increased from 1e-5)
    NUM_EPOCHS = 50  # Max epochs (early stopping will likely stop earlier)
    EARLY_STOP_PATIENCE = 7  # Stop if no improvement for 7 epochs
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    DROPOUT = 0.4  # Increased from 0.3 to reduce overfitting
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 1: Collect data
    print("\n" + "="*80)
    print("STEP 1: Data Collection")
    print("="*80)
    
    collector = PlayByPlayDataCollector(seasons=SEASONS)
    
    # Try to load existing data first
    try:
        pbp_data = collector.load_data('data/pbp_data.csv')
    except:
        print("No existing data found. Collecting from NBA API...")
        pbp_data = collector.collect_all_data(max_games_per_season=MAX_GAMES_PER_SEASON)
        collector.save_data('data/pbp_data.csv')
    
    if len(pbp_data) == 0:
        print("No data collected. Exiting.")
        return
    
    # Step 2: Feature engineering
    print("\n" + "="*80)
    print("STEP 2: Feature Engineering")
    print("="*80)
    
    fe = FeatureEngineer(pbp_data)
    fe.build_player_season_vocab()
    fe.build_outcome_vocab()
    fe.compute_player_season_stats()
    
    # Save feature engineer for later use
    os.makedirs('features', exist_ok=True)
    with open('features/feature_engineer.pkl', 'wb') as f:
        pickle.dump(fe, f)
    print("Feature engineer saved to features/feature_engineer.pkl")
    
    # Step 3: Create datasets
    print("\n" + "="*80)
    print("STEP 3: Dataset Creation")
    print("="*80)
    
    # Split into train/val
    train_size = int(0.8 * len(pbp_data))
    train_data = pbp_data[:train_size]
    val_data = pbp_data[train_size:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    train_dataset = PossessionDataset(train_data, fe)
    val_dataset = PossessionDataset(val_data, fe)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 4: Initialize model
    print("\n" + "="*80)
    print("STEP 4: Model Initialization")
    print("="*80)
    
    model = PossessionPredictionModel(
        num_player_seasons=len(fe.player_to_idx),
        num_outcomes=len(fe.outcome_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dropout rate: {DROPOUT}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    
    # Step 5: Train model
    print("\n" + "="*80)
    print("STEP 5: Model Training")
    print("="*80)
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=EARLY_STOP_PATIENCE,
        device=device
    )
    
    trainer.train(num_epochs=NUM_EPOCHS, early_stopping=True)
    
    # Step 6: Save model
    print("\n" + "="*80)
    print("STEP 6: Saving Model")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/possession_model.pt')
    
    print("\n" + "="*80)
    print("Training pipeline complete!")
    print("="*80)
    print("\nSaved files:")
    print("  - data/pbp_data.csv: Raw play-by-play data")
    print("  - features/feature_engineer.pkl: Feature mappings and vocabularies")
    print("  - models/possession_model.pt: Trained model checkpoint")
    print("\nYou can now use this model in your GameEngine to predict possessions!")


if __name__ == "__main__":
    main()

