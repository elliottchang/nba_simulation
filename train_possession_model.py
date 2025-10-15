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

from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.library.parameters import SeasonAll


# ============================================================================
# Data Collection
# ============================================================================

class PlayByPlayDataCollector:
    """Collects and processes play-by-play data from NBA API."""
    
    def __init__(self, seasons: List[str]):
        """
        Args:
            seasons: List of seasons in format 'YYYY-YY' (e.g., ['2022-23', '2023-24'])
        """
        self.seasons = seasons
        self.pbp_data = []
        self.player_stats = {}
        
    def collect_games_for_season(self, season: str, max_games: Optional[int] = None):
        """Collect all games for a given season."""
        print(f"\nCollecting games for {season} season...")
        
        try:
            # Get all games for the season
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games_df = gamefinder.get_data_frames()[0]
            
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
        
        Returns:
            Dict with keys: 'outcome', 'player_id', 'player_name', 'team_id'
        """
        if len(poss_events) == 0:
            return None
        
        # Get the last meaningful event in the possession
        last_event = poss_events.iloc[-1]
        event_type = last_event['EVENTMSGTYPE']
        
        # Event types: 1=shot_made, 2=shot_miss, 3=ft, 5=turnover, etc.
        outcome_map = {
            1: {  # Made shot
                1: '2PT_make',  # Jump shot, layup, etc.
                2: '3PT_make',  # 3-pointer
            },
            2: {  # Missed shot
                1: '2PT_miss',
                2: '3PT_miss',
            },
            3: 'FT',  # Free throw
            5: 'TO',  # Turnover
        }
        
        # Determine outcome
        if event_type == 1:  # Made shot
            shot_type = 1 if '3PT' in str(last_event.get('HOMEDESCRIPTION', '')) or \
                           '3PT' in str(last_event.get('VISITORDESCRIPTION', '')) else 1
            outcome = '3PT_make' if '3PT' in str(last_event.get('HOMEDESCRIPTION', '')) or \
                                     '3PT' in str(last_event.get('VISITORDESCRIPTION', '')) else '2PT_make'
        elif event_type == 2:  # Missed shot
            outcome = '3PT_miss' if '3PT' in str(last_event.get('HOMEDESCRIPTION', '')) or \
                                     '3PT' in str(last_event.get('VISITORDESCRIPTION', '')) else '2PT_miss'
        elif event_type == 3:  # Free throw
            outcome = 'FT_make' if 'MISS' not in str(last_event.get('HOMEDESCRIPTION', '')) and \
                                   'MISS' not in str(last_event.get('VISITORDESCRIPTION', '')) else 'FT_miss'
        elif event_type == 5:  # Turnover
            outcome = 'TO'
        else:
            outcome = 'other'
        
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
            time.sleep(0.6)  # Rate limit
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = boxscore.get_data_frames()[0]
            
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
            return None
    
    def extract_possessions_from_game(self, game_id: str, season: str) -> List[Dict]:
        """Extract possession-level data with full lineup tracking."""
        try:
            # Add delay to respect API rate limits
            time.sleep(0.6)
            
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            events = pbp.get_data_frames()[0]
            
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
                if event_type in [1, 2, 5]:  # Made shot, missed shot, turnover
                    current_possession.append(event)
                    
                    # Process the possession
                    if len(current_possession) > 0:
                        poss_df = pd.DataFrame(current_possession)
                        outcome = self.parse_possession_outcome(events, poss_df)
                        
                        if outcome and outcome['player_id'] is not None:
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
                else:
                    current_possession.append(event)
            
            return possessions
            
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            return []
    
    def collect_all_data(self, max_games_per_season: Optional[int] = None):
        """Collect play-by-play data for all specified seasons."""
        all_possessions = []
        
        for season in self.seasons:
            game_ids = self.collect_games_for_season(season, max_games_per_season)
            
            print(f"Processing {len(game_ids)} games for {season}...")
            for i, game_id in enumerate(game_ids):
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(game_ids)} games...")
                
                possessions = self.extract_possessions_from_game(game_id, season)
                all_possessions.extend(possessions)
            
            print(f"Collected {len(all_possessions)} total possessions so far")
        
        self.pbp_data = pd.DataFrame(all_possessions)
        return self.pbp_data
    
    def save_data(self, filepath: str):
        """Save collected data to disk."""
        self.pbp_data.to_csv(filepath, index=False)
        print(f"Saved {len(self.pbp_data)} possessions to {filepath}")
    
    def load_data(self, filepath: str):
        """Load previously collected data."""
        self.pbp_data = pd.read_csv(filepath)
        print(f"Loaded {len(self.pbp_data)} possessions from {filepath}")
        return self.pbp_data


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
        # For now, we'll compute basic stats from the possession data
        # In a production system, you'd pull from player stats APIs
        
        stats = defaultdict(lambda: {
            'possessions': 0,
            '2pt_attempts': 0,
            '2pt_makes': 0,
            '3pt_attempts': 0,
            '3pt_makes': 0,
            'turnovers': 0,
            'ft_attempts': 0,
        })
        
        for _, row in self.pbp_data.iterrows():
            key = (int(row['player_id']), row['season'])
            stats[key]['possessions'] += 1
            
            outcome = row['outcome']
            if '2PT' in outcome:
                stats[key]['2pt_attempts'] += 1
                if 'make' in outcome:
                    stats[key]['2pt_makes'] += 1
            elif '3PT' in outcome:
                stats[key]['3pt_attempts'] += 1
                if 'make' in outcome:
                    stats[key]['3pt_makes'] += 1
            elif outcome == 'TO':
                stats[key]['turnovers'] += 1
            elif 'FT' in outcome:
                stats[key]['ft_attempts'] += 1
        
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
        
        self.player_season_stats = dict(stats)
        print(f"Computed stats for {len(self.player_season_stats)} player-season pairs")
        return self.player_season_stats
    
    def get_stat_features(self, player_id: int, season: str) -> np.ndarray:
        """Get statistical features for a player-season."""
        key = (player_id, season)
        
        if key not in self.player_season_stats:
            # Return league average or zeros for unknown players
            return np.zeros(6)
        
        stats = self.player_season_stats[key]
        return np.array([
            stats['usage_rate'],
            stats['2pt_rate'],
            stats['3pt_rate'],
            stats['to_rate'],
            stats['2pt_pct'],
            stats['3pt_pct'],
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
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
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
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2f}%")
        
        print("\nTraining complete!")
        return self.train_losses, self.val_losses, self.val_accuracies
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        print(f"Model saved to {filepath}")


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
    MAX_GAMES_PER_SEASON = 50  # Limit for testing; set to None for full dataset
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 1: Collect data
    print("\n" + "="*80)
    print("STEP 1: Data Collection")
    print("="*80)
    
    collector = PlayByPlayDataCollector(seasons=SEASONS)
    
    # Try to load existing data first
    try:
        pbp_data = collector.load_data('pbp_data.csv')
    except:
        print("No existing data found. Collecting from NBA API...")
        pbp_data = collector.collect_all_data(max_games_per_season=MAX_GAMES_PER_SEASON)
        collector.save_data('pbp_data.csv')
    
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
    with open('feature_engineer.pkl', 'wb') as f:
        pickle.dump(fe, f)
    print("Feature engineer saved to feature_engineer.pkl")
    
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
        hidden_dim=HIDDEN_DIM
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Train model
    print("\n" + "="*80)
    print("STEP 5: Model Training")
    print("="*80)
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # Step 6: Save model
    print("\n" + "="*80)
    print("STEP 6: Saving Model")
    print("="*80)
    
    trainer.save_model('possession_model.pt')
    
    print("\n" + "="*80)
    print("Training pipeline complete!")
    print("="*80)
    print("\nSaved files:")
    print("  - pbp_data.csv: Raw play-by-play data")
    print("  - feature_engineer.pkl: Feature mappings and vocabularies")
    print("  - possession_model.pt: Trained model checkpoint")
    print("\nYou can now use this model in your GameEngine to predict possessions!")


if __name__ == "__main__":
    main()

