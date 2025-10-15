"""
Player-Aware Possession Predictor
This module adds player-specific modeling on top of the lineup-aware base model.
It correctly models WHO ends possessions and their individual tendencies.
"""

import torch
import torch.nn.functional as F
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from train_possession_model import PossessionPredictionModel, FeatureEngineer
import random


class PlayerAwarePredictor:
    """
    Enhanced predictor that models individual player tendencies:
    1. WHO ends the possession (usage-weighted)
    2. WHAT they do (player's shot distribution)
    3. SUCCESS RATE (player's shooting percentages)
    """
    
    def __init__(
        self,
        model_path: str = 'possession_model.pt',
        feature_engineer_path: str = 'feature_engineer.pkl',
        device: str = 'cpu'
    ):
        """Load model and feature engineer."""
        self.device = device
        
        # Load feature engineer
        with open(feature_engineer_path, 'rb') as f:
            self.fe = pickle.load(f)
        
        print(f"Loaded feature engineer with {len(self.fe.player_to_idx)} player-season pairs")
        
        # Initialize model
        self.model = PossessionPredictionModel(
            num_player_seasons=len(self.fe.player_to_idx),
            num_outcomes=len(self.fe.outcome_to_idx),
            embedding_dim=64,
            hidden_dim=128
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def get_player_tendencies(self, player_id: int, season: str) -> Dict:
        """
        Get a player's statistical tendencies.
        
        Returns:
            Dict with usage_rate, shot_distribution, shooting_percentages
        """
        key = (int(player_id), season)
        
        if key not in self.fe.player_season_stats:
            # Return league average for unknown players
            return {
                'usage_rate': 0.20,  # League average ~20%
                '2pt_rate': 0.40,
                '3pt_rate': 0.30,
                'to_rate': 0.10,
                '2pt_pct': 0.50,
                '3pt_pct': 0.35,
            }
        
        stats = self.fe.player_season_stats[key]
        return {
            'usage_rate': stats['usage_rate'],
            '2pt_rate': stats['2pt_rate'],
            '3pt_rate': stats['3pt_rate'],
            'to_rate': stats['to_rate'],
            '2pt_pct': stats['2pt_pct'],
            '3pt_pct': stats['3pt_pct'],
        }
    
    def select_player(
        self,
        offensive_lineup: List[int],
        season: str
    ) -> int:
        """
        Select which player ends the possession, weighted by usage rate.
        
        Args:
            offensive_lineup: List of 5 player IDs
            season: Season string
        
        Returns:
            Selected player ID
        """
        # Get usage rates for all players
        usage_rates = []
        for player_id in offensive_lineup[:5]:
            tendencies = self.get_player_tendencies(player_id, season)
            usage_rates.append(tendencies['usage_rate'])
        
        # Normalize to probabilities
        total_usage = sum(usage_rates)
        if total_usage == 0:
            # Fallback to uniform
            return random.choice(offensive_lineup[:5])
        
        probabilities = [u / total_usage for u in usage_rates]
        
        # Sample player weighted by usage
        selected_player = random.choices(offensive_lineup[:5], weights=probabilities, k=1)[0]
        return int(selected_player)
    
    def select_player_with_fatigue(
        self,
        offensive_lineup: List[int],
        season: str,
        fatigue_multipliers: Dict[int, float]
    ) -> int:
        """
        Select which player ends the possession, weighted by usage rate and fatigue.
        Tired players get the ball less often.
        
        Args:
            offensive_lineup: List of 5 player IDs
            season: Season string
            fatigue_multipliers: Dict of player_id → fatigue multiplier (0.5 to 1.0)
        
        Returns:
            Selected player ID
        """
        # Get usage rates adjusted for fatigue
        adjusted_usage_rates = []
        for player_id in offensive_lineup[:5]:
            tendencies = self.get_player_tendencies(player_id, season)
            base_usage = tendencies['usage_rate']
            fatigue_mult = fatigue_multipliers.get(player_id, 1.0)
            
            # Tired players get ball less often
            adjusted_usage = base_usage * fatigue_mult
            adjusted_usage_rates.append(adjusted_usage)
        
        # Normalize to probabilities
        total_usage = sum(adjusted_usage_rates)
        if total_usage == 0:
            # Fallback to uniform
            return random.choice(offensive_lineup[:5])
        
        probabilities = [u / total_usage for u in adjusted_usage_rates]
        
        # Sample player weighted by fatigue-adjusted usage
        selected_player = random.choices(offensive_lineup[:5], weights=probabilities, k=1)[0]
        return int(selected_player)
    
    def predict_player_action(
        self,
        player_id: int,
        season: str,
        offensive_lineup: List[int],
        defensive_lineup: List[int],
        score_margin: int = 0,
        period: int = 1
    ) -> str:
        """
        Predict what action a specific player will take.
        
        This uses:
        1. Lineup context (from neural network)
        2. Player's individual shot distribution
        
        Args:
            player_id: The player who will end the possession
            season: Season string
            offensive_lineup: Offensive lineup
            defensive_lineup: Defensive lineup
            score_margin: Score differential
            period: Quarter
        
        Returns:
            Action type: '2PT_attempt', '3PT_attempt', 'TO'
        """
        # Get player tendencies
        tendencies = self.get_player_tendencies(player_id, season)
        
        # Get lineup-adjusted probabilities from neural network
        unk_idx = self.fe.player_to_idx[('UNK', 'UNK')]
        
        offensive_indices = []
        for pid in offensive_lineup[:5]:
            key = (int(pid), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            offensive_indices.append(idx_val)
        
        defensive_indices = []
        for pid in defensive_lineup[:5]:
            key = (int(pid), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            defensive_indices.append(idx_val)
        
        # Pad if necessary
        while len(offensive_indices) < 5:
            offensive_indices.append(unk_idx)
        while len(defensive_indices) < 5:
            defensive_indices.append(unk_idx)
        
        context = np.array([score_margin, period], dtype=np.float32)
        
        # Get lineup-level outcome distribution
        offensive_tensor = torch.tensor([offensive_indices], dtype=torch.long).to(self.device)
        defensive_tensor = torch.tensor([defensive_indices], dtype=torch.long).to(self.device)
        context_tensor = torch.tensor(context.reshape(1, -1), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outcome_logits = self.model(offensive_tensor, defensive_tensor, context_tensor)
            lineup_probs = F.softmax(outcome_logits, dim=1)[0]
        
        # Combine lineup context with player tendencies
        # Weight: 60% player tendencies, 40% lineup context
        player_weight = 0.6
        lineup_weight = 0.4
        
        # Build action distribution
        action_probs = {
            '2PT_attempt': tendencies['2pt_rate'] * player_weight,
            '3PT_attempt': tendencies['3pt_rate'] * player_weight,
            'TO': tendencies['to_rate'] * player_weight,
        }
        
        # Add lineup influence
        for outcome, prob in self.fe.idx_to_outcome.items():
            if '2PT' in prob:
                action_probs['2PT_attempt'] += lineup_probs[outcome].item() * lineup_weight / 2
            elif '3PT' in prob:
                action_probs['3PT_attempt'] += lineup_probs[outcome].item() * lineup_weight / 2
            elif 'TO' in prob:
                action_probs['TO'] += lineup_probs[outcome].item() * lineup_weight
        
        # Normalize
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v/total for k, v in action_probs.items()}
        
        # Sample action
        actions = list(action_probs.keys())
        weights = list(action_probs.values())
        selected_action = random.choices(actions, weights=weights, k=1)[0]
        
        return selected_action
    
    def predict_outcome(
        self,
        player_id: int,
        action: str,
        season: str
    ) -> str:
        """
        Predict if the player succeeds at the action.
        
        Uses the player's actual shooting percentages.
        
        Args:
            player_id: Player taking the action
            action: '2PT_attempt', '3PT_attempt', or 'TO'
            season: Season string
        
        Returns:
            Full outcome: '2PT_make', '2PT_miss', etc.
        """
        if action == 'TO':
            return 'TO'
        
        # Get player's shooting percentage
        tendencies = self.get_player_tendencies(player_id, season)
        
        if action == '2PT_attempt':
            make_prob = tendencies['2pt_pct']
            if random.random() < make_prob:
                return '2PT_make'
            else:
                return '2PT_miss'
        
        elif action == '3PT_attempt':
            make_prob = tendencies['3pt_pct']
            if random.random() < make_prob:
                return '3PT_make'
            else:
                return '3PT_miss'
        
        return 'TO'
    
    def predict_outcome_with_fatigue(
        self,
        player_id: int,
        action: str,
        season: str,
        fatigue_multiplier: float
    ) -> str:
        """
        Predict outcome with fatigue affecting performance.
        
        Fatigue effects:
        - Reduces shooting percentages (tired players miss more)
        - Increases turnover chance (tired players lose ball more)
        
        Args:
            player_id: Player taking the action
            action: '2PT_attempt', '3PT_attempt', or 'TO'
            season: Season string
            fatigue_multiplier: 0.5 (exhausted) to 1.0 (fresh)
        
        Returns:
            Full outcome with fatigue effects applied
        """
        if action == 'TO':
            return 'TO'
        
        # Get player's base shooting percentage
        tendencies = self.get_player_tendencies(player_id, season)
        
        if action == '2PT_attempt':
            # Apply fatigue to shooting percentage
            base_pct = tendencies['2pt_pct']
            fatigued_pct = base_pct * fatigue_multiplier
            
            # Tired players also might turn it over instead
            turnover_chance = (1.0 - fatigue_multiplier) * 0.05  # Up to 5% chance when exhausted
            
            if random.random() < turnover_chance:
                return 'TO'
            elif random.random() < fatigued_pct:
                return '2PT_make'
            else:
                return '2PT_miss'
        
        elif action == '3PT_attempt':
            # Apply fatigue to shooting percentage
            base_pct = tendencies['3pt_pct']
            fatigued_pct = base_pct * fatigue_multiplier
            
            # Tired players also might turn it over instead
            turnover_chance = (1.0 - fatigue_multiplier) * 0.05
            
            if random.random() < turnover_chance:
                return 'TO'
            elif random.random() < fatigued_pct:
                return '3PT_make'
            else:
                return '3PT_miss'
        
        return 'TO'
    
    def sample_possession_detailed(
        self,
        offensive_lineup: List[int],
        defensive_lineup: List[int],
        season: str,
        score_margin: int = 0,
        period: int = 1,
        fatigue_multipliers: Optional[Dict[int, float]] = None
    ) -> Dict:
        """
        Full player-aware possession prediction with fatigue effects.
        
        Process:
        1. Select which player (usage-weighted, adjusted for fatigue)
        2. Determine action type (player tendencies + lineup context)
        3. Determine success (player shooting percentage, reduced by fatigue)
        
        Args:
            offensive_lineup: 5 offensive player IDs
            defensive_lineup: 5 defensive player IDs
            season: Season string
            score_margin: Score differential
            period: Quarter
            fatigue_multipliers: Optional dict of player_id → fatigue multiplier (0.5 to 1.0)
                                 If None, assumes all players at 1.0 (fresh)
        
        Returns:
            Detailed dictionary with player-specific outcome
        """
        if fatigue_multipliers is None:
            fatigue_multipliers = {}
        
        # Step 1: Select player (usage-weighted with fatigue adjustment)
        player_id = self.select_player_with_fatigue(offensive_lineup, season, fatigue_multipliers)
        
        # Step 2: Determine action
        action = self.predict_player_action(
            player_id=player_id,
            season=season,
            offensive_lineup=offensive_lineup,
            defensive_lineup=defensive_lineup,
            score_margin=score_margin,
            period=period
        )
        
        # Step 3: Determine outcome (with fatigue affecting shooting %)
        fatigue_mult = fatigue_multipliers.get(player_id, 1.0)
        outcome = self.predict_outcome_with_fatigue(player_id, action, season, fatigue_mult)
        
        # Build detailed result
        result = {
            'outcome': outcome,
            'player_id': int(player_id),
            'points': 0,
            'fga': 0,
            'fgm': 0,
            '3pa': 0,
            '3pm': 0,
            'fta': 0,
            'ftm': 0,
            'to': 0,
            'possession_changes': True
        }
        
        # Map outcome to stats
        if outcome == '2PT_make':
            result['points'] = 2
            result['fga'] = 1
            result['fgm'] = 1
        elif outcome == '2PT_miss':
            result['fga'] = 1
        elif outcome == '3PT_make':
            result['points'] = 3
            result['fga'] = 1
            result['fgm'] = 1
            result['3pa'] = 1
            result['3pm'] = 1
        elif outcome == '3PT_miss':
            result['fga'] = 1
            result['3pa'] = 1
        elif outcome == 'TO':
            result['to'] = 1
        
        return result
    
    def compare_players(
        self,
        player_ids: List[int],
        season: str,
        offensive_lineup: List[int],
        defensive_lineup: List[int],
        num_possessions: int = 100
    ) -> Dict:
        """
        Compare how different players would perform in the same situation.
        
        Useful for answering: "What if Curry took this shot vs Kuminga?"
        """
        results = {}
        
        for player_id in player_ids:
            # Force this player to end possessions
            outcomes = {
                '2PT_make': 0,
                '2PT_miss': 0,
                '3PT_make': 0,
                '3PT_miss': 0,
                'TO': 0
            }
            
            for _ in range(num_possessions):
                action = self.predict_player_action(
                    player_id=player_id,
                    season=season,
                    offensive_lineup=offensive_lineup,
                    defensive_lineup=defensive_lineup
                )
                outcome = self.predict_outcome(player_id, action, season)
                if outcome in outcomes:
                    outcomes[outcome] += 1
            
            # Calculate summary stats
            total_shots = outcomes['2PT_make'] + outcomes['2PT_miss'] + outcomes['3PT_make'] + outcomes['3PT_miss']
            makes = outcomes['2PT_make'] + outcomes['3PT_make']
            
            results[player_id] = {
                'outcomes': outcomes,
                'shooting_pct': makes / total_shots if total_shots > 0 else 0,
                '3pt_frequency': (outcomes['3PT_make'] + outcomes['3PT_miss']) / num_possessions,
                'to_rate': outcomes['TO'] / num_possessions,
                'avg_points': (outcomes['2PT_make']*2 + outcomes['3PT_make']*3) / num_possessions
            }
        
        return results


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate player-aware predictions."""
    
    print("="*80)
    print("Player-Aware Possession Predictor Demo")
    print("="*80)
    
    try:
        predictor = PlayerAwarePredictor()
    except FileNotFoundError:
        print("\nError: Model not trained yet!")
        print("Run: python train_possession_model.py")
        return
    
    # Example lineup
    season = '2022-23'
    
    # Get some players from the data
    sample_players = list(predictor.fe.player_season_stats.keys())[:10]
    if len(sample_players) < 10:
        print("Not enough training data for demo")
        return
    
    offensive_lineup = [pid for pid, _ in sample_players[:5]]
    defensive_lineup = [pid for pid, _ in sample_players[5:10]]
    
    print(f"\nOffensive lineup: {offensive_lineup}")
    print(f"Defensive lineup: {defensive_lineup}")
    
    # Show player tendencies
    print("\n" + "="*80)
    print("Player Tendencies (Offense)")
    print("="*80)
    
    for i, player_id in enumerate(offensive_lineup):
        tendencies = predictor.get_player_tendencies(player_id, season)
        print(f"\nPlayer {player_id}:")
        print(f"  Usage: {tendencies['usage_rate']:.1%}")
        print(f"  2PT rate: {tendencies['2pt_rate']:.1%} (makes {tendencies['2pt_pct']:.1%})")
        print(f"  3PT rate: {tendencies['3pt_rate']:.1%} (makes {tendencies['3pt_pct']:.1%})")
        print(f"  TO rate: {tendencies['to_rate']:.1%}")
    
    # Simulate possessions
    print("\n" + "="*80)
    print("Simulating 20 Possessions")
    print("="*80)
    
    player_stats = {pid: {'possessions': 0, 'points': 0, 'fga': 0} for pid in offensive_lineup}
    
    for i in range(20):
        result = predictor.sample_possession_detailed(
            offensive_lineup=offensive_lineup,
            defensive_lineup=defensive_lineup,
            season=season,
            score_margin=0,
            period=1
        )
        
        player_id = result['player_id']
        player_stats[player_id]['possessions'] += 1
        player_stats[player_id]['points'] += result['points']
        player_stats[player_id]['fga'] += result['fga']
        
        print(f"Possession {i+1}: Player {player_id} - {result['outcome']} (+{result['points']} pts)")
    
    # Show distribution
    print("\n" + "="*80)
    print("Possession Distribution")
    print("="*80)
    
    for player_id in offensive_lineup:
        stats = player_stats[player_id]
        if stats['possessions'] > 0:
            print(f"Player {player_id}: {stats['possessions']}/20 possessions ({stats['possessions']/20*100:.0f}%), "
                  f"{stats['points']} pts, {stats['fga']} FGA")
    
    print("\n" + "="*80)
    print("Key Feature: Player-Specific Modeling")
    print("="*80)
    print("""
The model now correctly captures individual player tendencies:

1. WHO ends possessions: Weighted by each player's usage rate
   - High usage players (stars) get more possessions
   - Role players get fewer possessions

2. WHAT they do: Based on their shot distribution
   - 3PT shooters take more 3s
   - Post players take more 2s
   
3. SUCCESS RATE: Based on their actual shooting %
   - Good shooters make more shots
   - Poor shooters make fewer shots

Example: Steph Curry vs Jonathan Kuminga
- Curry: Higher usage (more possessions)
- Curry: Higher 3PT rate (more 3s when he shoots)
- Curry: Higher 3PT% (makes more when he takes 3s)
    """)


if __name__ == "__main__":
    demo()

