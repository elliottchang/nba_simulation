"""
Possession Predictor - Inference Module
This module loads the trained possession prediction model and provides
an interface for predicting possession outcomes during game simulation.
"""

import torch
import torch.nn.functional as F
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from train_possession_model import PossessionPredictionModel, FeatureEngineer


class PossessionPredictor:
    """
    Wrapper class for making predictions with the trained possession model.
    Integrates with GameEngine to predict how possessions will end.
    """
    
    def __init__(
        self,
        model_path: str = 'possession_model.pt',
        feature_engineer_path: str = 'feature_engineer.pkl',
        device: str = 'cpu'
    ):
        """
        Load trained model and feature engineer.
        
        Args:
            model_path: Path to saved model checkpoint
            feature_engineer_path: Path to saved feature engineer
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
        """
        self.device = device
        
        # Load feature engineer
        with open(feature_engineer_path, 'rb') as f:
            self.fe = pickle.load(f)
        
        print(f"Loaded feature engineer with {len(self.fe.player_to_idx)} player-season pairs")
        print(f"Outcome classes: {list(self.fe.outcome_to_idx.keys())}")
        
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
        print(f"Using device: {device}")
    
    def predict_possession(
        self,
        offensive_lineup: List[int],
        defensive_lineup: List[int],
        season: str,
        score_margin: int = 0,
        period: int = 1,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Predict the outcome of a possession based on full lineups.
        
        Args:
            offensive_lineup: List of 5 player IDs for offensive team
            defensive_lineup: List of 5 player IDs for defensive team
            season: Season in format 'YYYY-YY' (e.g., '2023-24')
            score_margin: Current score differential (positive = offensive team is winning)
            period: Current quarter/period
            return_probabilities: If True, return probability distribution
        
        Returns:
            Dictionary with prediction results:
                - 'outcome': Predicted outcome string
                - 'outcome_idx': Outcome class index
                - 'confidence': Confidence score (0-1)
                - 'probabilities': (optional) Full probability distribution
        """
        # Convert player IDs to player-season indices
        unk_idx = self.fe.player_to_idx[('UNK', 'UNK')]
        
        offensive_indices = []
        for player_id in offensive_lineup[:5]:
            key = (int(player_id), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            offensive_indices.append(idx_val)
        
        defensive_indices = []
        for player_id in defensive_lineup[:5]:
            key = (int(player_id), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            defensive_indices.append(idx_val)
        
        # Pad if necessary (should not happen if lineups are valid)
        while len(offensive_indices) < 5:
            offensive_indices.append(unk_idx)
        while len(defensive_indices) < 5:
            defensive_indices.append(unk_idx)
        
        # Prepare context
        context = np.array([score_margin, period], dtype=np.float32)
        
        # Convert to tensors
        offensive_tensor = torch.tensor([offensive_indices], dtype=torch.long).to(self.device)
        defensive_tensor = torch.tensor([defensive_indices], dtype=torch.long).to(self.device)
        context_tensor = torch.tensor([context], dtype=torch.float32).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outcome_logits = self.model(offensive_tensor, defensive_tensor, context_tensor)
            probabilities = F.softmax(outcome_logits, dim=1)
            
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        predicted_outcome = self.fe.idx_to_outcome[predicted_idx]
        
        result = {
            'outcome': predicted_outcome,
            'outcome_idx': predicted_idx,
            'confidence': confidence,
        }
        
        if return_probabilities:
            prob_dict = {
                self.fe.idx_to_outcome[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
            result['probabilities'] = prob_dict
        
        return result
    
    def sample_possession(
        self,
        offensive_lineup: List[int],
        defensive_lineup: List[int],
        season: str,
        score_margin: int = 0,
        period: int = 1,
        temperature: float = 1.0
    ) -> str:
        """
        Sample a possession outcome from the predicted distribution.
        This adds randomness for more realistic simulation.
        
        Args:
            offensive_lineup: List of 5 player IDs for offensive team
            defensive_lineup: List of 5 player IDs for defensive team
            season: Season in format 'YYYY-YY'
            score_margin: Current score differential
            period: Current quarter/period
            temperature: Sampling temperature (higher = more random)
                         1.0 = use model probabilities as-is
                         >1.0 = more uniform (more randomness)
                         <1.0 = more peaked (more deterministic)
        
        Returns:
            Sampled outcome string (e.g., '2PT_make', 'TO', etc.)
        """
        # Convert player IDs to player-season indices
        unk_idx = self.fe.player_to_idx[('UNK', 'UNK')]
        
        offensive_indices = []
        for player_id in offensive_lineup[:5]:
            key = (int(player_id), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            offensive_indices.append(idx_val)
        
        defensive_indices = []
        for player_id in defensive_lineup[:5]:
            key = (int(player_id), season)
            idx_val = self.fe.player_to_idx.get(key, unk_idx)
            defensive_indices.append(idx_val)
        
        # Pad if necessary
        while len(offensive_indices) < 5:
            offensive_indices.append(unk_idx)
        while len(defensive_indices) < 5:
            defensive_indices.append(unk_idx)
        
        # Prepare context
        context = np.array([score_margin, period], dtype=np.float32)
        
        # Convert to tensors
        offensive_tensor = torch.tensor([offensive_indices], dtype=torch.long).to(self.device)
        defensive_tensor = torch.tensor([defensive_indices], dtype=torch.long).to(self.device)
        context_tensor = torch.tensor([context], dtype=torch.float32).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outcome_logits = self.model(offensive_tensor, defensive_tensor, context_tensor)
            
            # Apply temperature
            if temperature != 1.0:
                outcome_logits = outcome_logits / temperature
            
            probabilities = F.softmax(outcome_logits, dim=1)
            
            # Sample from the distribution
            sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
        
        return self.fe.idx_to_outcome[sampled_idx]
    
    def sample_possession_detailed(
        self,
        offensive_lineup: List[int],
        defensive_lineup: List[int],
        season: str,
        score_margin: int = 0,
        period: int = 1,
        temperature: float = 1.0
    ) -> Dict:
        """
        Sample a possession and return detailed stats for boxscore updating.
        
        Args:
            offensive_lineup: List of 5 player IDs for offensive team
            defensive_lineup: List of 5 player IDs for defensive team
            season: Season in format 'YYYY-YY'
            score_margin: Current score differential
            period: Current quarter/period
            temperature: Sampling temperature
        
        Returns:
            Dictionary with boxscore-ready stats:
            {
                'outcome': '2PT_make',
                'player_id': 203999,  # Player who ended possession (random selection from lineup)
                'points': 2,
                'fga': 1,
                'fgm': 1,
                '3pa': 0,
                '3pm': 0,
                'fta': 0,
                'ftm': 0,
                'to': 0,
                'possession_changes': True
            }
        """
        # Get the outcome
        outcome = self.sample_possession(
            offensive_lineup=offensive_lineup,
            defensive_lineup=defensive_lineup,
            season=season,
            score_margin=score_margin,
            period=period,
            temperature=temperature
        )
        
        # Randomly select player from offensive lineup (weighted by usage in future)
        import random
        player_id = random.choice(offensive_lineup[:5])
        
        # Parse outcome into boxscore stats
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
        elif outcome == 'FT_make':
            result['points'] = 1
            result['fta'] = 1
            result['ftm'] = 1
            result['possession_changes'] = False  # Simplified
        elif outcome == 'FT_miss':
            result['fta'] = 1
            result['possession_changes'] = False  # Simplified
        elif outcome == 'TO':
            result['to'] = 1
        
        return result
    
    def get_player_season_tendencies(self, player_id: int, season: str) -> Dict:
        """
        Get a player's statistical tendencies for a given season.
        Useful for understanding what the model knows about a player.
        
        Args:
            player_id: Player ID
            season: Season in format 'YYYY-YY'
        
        Returns:
            Dictionary of player statistics
        """
        key = (int(player_id), season)
        
        if key not in self.fe.player_season_stats:
            return {
                'found': False,
                'message': f'No data for player {player_id} in season {season}'
            }
        
        stats = self.fe.player_season_stats[key]
        return {
            'found': True,
            'player_id': player_id,
            'season': season,
            'possessions': stats['possessions'],
            'usage_rate': stats['usage_rate'],
            '2pt_rate': stats['2pt_rate'],
            '3pt_rate': stats['3pt_rate'],
            'to_rate': stats['to_rate'],
            '2pt_pct': stats['2pt_pct'],
            '3pt_pct': stats['3pt_pct'],
        }
    
    def compare_player_across_seasons(self, player_id: int, seasons: List[str]) -> Dict:
        """
        Compare a player's tendencies across multiple seasons.
        Useful for seeing how players improve (e.g., SGA 2020 vs 2025).
        
        Args:
            player_id: Player ID
            seasons: List of seasons to compare
        
        Returns:
            Dictionary with stats for each season
        """
        comparison = {}
        
        for season in seasons:
            comparison[season] = self.get_player_season_tendencies(player_id, season)
        
        return comparison


# ============================================================================
# Example Usage
# ============================================================================

def demo_inference():
    """Demonstrate how to use the PossessionPredictor."""
    
    print("="*80)
    print("Possession Predictor - Demo")
    print("="*80)
    
    # Initialize predictor
    try:
        predictor = PossessionPredictor(
            model_path='possession_model.pt',
            feature_engineer_path='feature_engineer.pkl'
        )
    except FileNotFoundError:
        print("\nError: Model files not found!")
        print("Please run 'python train_possession_model.py' first to train the model.")
        return
    
    # Example 1: Make a deterministic prediction
    print("\n" + "="*80)
    print("Example 1: Deterministic Prediction with Full Lineups")
    print("="*80)
    
    # Example lineups (these are placeholder IDs - in real use, you'd get from team rosters)
    season = '2022-23'
    
    # Get a sample of player IDs from the loaded data
    sample_players = list(predictor.fe.player_season_stats.keys())[:10]
    if len(sample_players) >= 10:
        offensive_lineup = [pid for pid, _ in sample_players[:5]]
        defensive_lineup = [pid for pid, _ in sample_players[5:10]]
    else:
        print("Not enough players in training data for demo. Need to train model first.")
        return
    
    print(f"\nOffensive lineup (5 players): {offensive_lineup}")
    print(f"Defensive lineup (5 players): {defensive_lineup}")
    
    result = predictor.predict_possession(
        offensive_lineup=offensive_lineup,
        defensive_lineup=defensive_lineup,
        season=season,
        score_margin=5,
        period=2,
        return_probabilities=True
    )
    
    print(f"\nPredicted outcome: {result['outcome']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nFull probability distribution:")
    for outcome, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {prob:.2%}")
    
    # Example 2: Sample multiple possessions (more realistic for simulation)
    print("\n" + "="*80)
    print("Example 2: Sampling Possessions")
    print("="*80)
    
    print(f"\nSimulating 10 possessions with these lineups:")
    outcomes = []
    for i in range(10):
        outcome = predictor.sample_possession(
            offensive_lineup=offensive_lineup,
            defensive_lineup=defensive_lineup,
            season=season,
            score_margin=0,
            period=1,
            temperature=1.0
        )
        outcomes.append(outcome)
        print(f"  Possession {i+1}: {outcome}")
    
    # Show distribution
    from collections import Counter
    counts = Counter(outcomes)
    print("\nOutcome distribution:")
    for outcome, count in counts.most_common():
        print(f"  {outcome}: {count}/10 ({count*10:.0f}%)")
    
    # Example 3: Get player tendencies
    print("\n" + "="*80)
    print("Example 3: Player Season Tendencies")
    print("="*80)
    
    example_player_id = offensive_lineup[0]
    tendencies = predictor.get_player_season_tendencies(example_player_id, season)
    
    if tendencies['found']:
        print(f"\nPlayer {example_player_id} in {season}:")
        print(f"  Total possessions: {tendencies['possessions']}")
        print(f"  Usage rate: {tendencies['usage_rate']:.3f}")
        print(f"  2PT rate: {tendencies['2pt_rate']:.3f} (pct: {tendencies['2pt_pct']:.3f})")
        print(f"  3PT rate: {tendencies['3pt_rate']:.3f} (pct: {tendencies['3pt_pct']:.3f})")
        print(f"  Turnover rate: {tendencies['to_rate']:.3f}")
    else:
        print(tendencies['message'])
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)


if __name__ == "__main__":
    demo_inference()

