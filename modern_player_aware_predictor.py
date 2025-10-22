"""
Modern Player-Aware Possession Predictor
Updated to work with the new data engineering infrastructure and Snowflake.
"""

import torch
import torch.nn.functional as F
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from train_possession_model import PossessionPredictionModel
import random
import logging

# Import new data engineering components
try:
    from data_engine.config import SnowflakeConfig
    from data_engine.snowflake_client import SnowflakeClient
    from data_engine.data_transformation import NBADataTransformation, DataCollectionConfig
    DATA_ENGINE_AVAILABLE = True
except ImportError:
    DATA_ENGINE_AVAILABLE = False
    logging.warning("Data engine components not available. Falling back to local files only.")

logger = logging.getLogger(__name__)

class ModernPlayerAwarePredictor:
    """
    Enhanced predictor that can work with both Snowflake and local data sources.
    Automatically detects which system is available and uses appropriate data backend.
    """
    
    def __init__(
        self,
        model_path: str = 'models/modern_possession_model.pt',
        feature_engineer_path: str = 'features/modern_feature_engineer.pkl',
        device: str = 'cpu',
        use_snowflake: bool = True
    ):
        """
        Initialize predictor with automatic backend detection.
        
        Args:
            model_path: Path to trained model
            feature_engineer_path: Path to feature engineer pickle file
            device: Device to run model on
            use_snowflake: Whether to attempt Snowflake connection
        """
        self.device = device
        self.use_snowflake = use_snowflake and DATA_ENGINE_AVAILABLE
        
        # Try to load feature engineer from file first
        try:
            with open(feature_engineer_path, 'rb') as f:
                self.fe = pickle.load(f)
            logger.info(f"Loaded feature engineer from {feature_engineer_path}")
        except FileNotFoundError:
            if self.use_snowflake:
                logger.info("Local feature engineer not found, attempting to load from Snowflake...")
                self.fe = self._load_feature_engineer_from_snowflake()
            else:
                raise FileNotFoundError(f"Could not find feature engineer at {feature_engineer_path}")
        
        logger.info(f"Loaded feature engineer with {len(self.fe.player_to_idx)} player-season pairs")
        
        # Initialize model
        self.model = PossessionPredictionModel(
            num_player_seasons=len(self.fe.player_to_idx),
            num_outcomes=len(self.fe.outcome_to_idx),
            embedding_dim=64,
            hidden_dim=128
        ).to(device)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find model at {model_path}")
    
    def _load_feature_engineer_from_snowflake(self):
        """Load feature engineer data from Snowflake and reconstruct the object."""
        if not DATA_ENGINE_AVAILABLE:
            raise ImportError("Snowflake components not available")
        
        try:
            snowflake_config = SnowflakeConfig.from_env()
        except Exception as e:
            raise RuntimeError(f"Could not load Snowflake config: {e}")
        
        # This is a simplified version - in practice you might want to store
        # the full feature engineer state in Snowflake or reconstruct it
        with SnowflakeClient(snowflake_config) as client:
            # Query to get player stats and reconstruct feature engineer
            # This is a placeholder - you'd need to implement the full reconstruction
            # based on how you store the feature engineer state
            raise NotImplementedError("Feature engineer reconstruction from Snowflake not yet implemented")
    
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
    
    def select_player(self, offensive_lineup: List[int], season: str) -> int:
        """Select which player ends the possession, weighted by usage rate."""
        usage_rates = []
        for player_id in offensive_lineup[:5]:
            tendencies = self.get_player_tendencies(player_id, season)
            usage_rates.append(tendencies['usage_rate'])
        
        total_usage = sum(usage_rates)
        if total_usage == 0:
            return random.choice(offensive_lineup[:5])
        
        probabilities = [u / total_usage for u in usage_rates]
        selected_player = random.choices(offensive_lineup[:5], weights=probabilities, k=1)[0]
        return int(selected_player)
    
    def select_player_with_fatigue(
        self,
        offensive_lineup: List[int],
        season: str,
        fatigue_multipliers: Dict[int, float]
    ) -> int:
        """Select which player ends the possession, weighted by usage rate and fatigue."""
        adjusted_usage_rates = []
        for player_id in offensive_lineup[:5]:
            tendencies = self.get_player_tendencies(player_id, season)
            base_usage = tendencies['usage_rate']
            fatigue_mult = fatigue_multipliers.get(player_id, 1.0)
            
            # Tired players get ball less often
            adjusted_usage = base_usage * fatigue_mult
            adjusted_usage_rates.append(adjusted_usage)
        
        total_usage = sum(adjusted_usage_rates)
        if total_usage == 0:
            return random.choice(offensive_lineup[:5])
        
        probabilities = [u / total_usage for u in adjusted_usage_rates]
        selected_player = random.choices(offensive_lineup[:5], weights=probabilities, k=1)[0]
        return int(selected_player)
    
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
        1. Use trained model to predict from outcome categories
        2. Select which player (usage-weighted, adjusted for fatigue)
        
        Returns:
            Detailed dictionary with player-specific outcome
        """
        if fatigue_multipliers is None:
            fatigue_multipliers = {}
        
        # Step 1: Use trained model to predict outcome directly
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
        
        # Get model prediction
        offensive_tensor = torch.tensor([offensive_indices], dtype=torch.long).to(self.device)
        defensive_tensor = torch.tensor([defensive_indices], dtype=torch.long).to(self.device)
        context_tensor = torch.tensor(context.reshape(1, -1), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outcome_logits = self.model(offensive_tensor, defensive_tensor, context_tensor)
            outcome_probs = F.softmax(outcome_logits, dim=1)[0]
            
            # Sample from the outcome categories
            outcome_indices = list(range(len(self.fe.outcome_to_idx)))
            weights = outcome_probs.cpu().numpy()
            selected_idx = random.choices(outcome_indices, weights=weights, k=1)[0]
            outcome = self.fe.idx_to_outcome[selected_idx]
        
        # Step 2: Select player (usage-weighted with fatigue adjustment)
        player_id = self.select_player_with_fatigue(offensive_lineup, season, fatigue_multipliers)
        
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
        
        # Map outcome to stats - handle all outcome categories
        if outcome == '2pt_make':
            result['points'] = 2
            result['fga'] = 1
            result['fgm'] = 1
        elif outcome == '2pt_miss':
            result['fga'] = 1
        elif outcome == '3pt_make':
            result['points'] = 3
            result['fga'] = 1
            result['fgm'] = 1
            result['3pa'] = 1
            result['3pm'] = 1
        elif outcome == '3pt_miss':
            result['fga'] = 1
            result['3pa'] = 1
        elif outcome == '2pt_andone':
            result['points'] = 2
            result['fga'] = 1
            result['fgm'] = 1
            result['fta'] = 1
        elif outcome == '3pt_andone':
            result['points'] = 3
            result['fga'] = 1
            result['fgm'] = 1
            result['3pa'] = 1
            result['3pm'] = 1
            result['fta'] = 1
        elif outcome == '2pt_foul':
            result['fga'] = 1
            result['fta'] = 2
        elif outcome == '3pt_foul':
            result['fga'] = 1
            result['3pa'] = 1
            result['fta'] = 3
        elif outcome == 'TO':
            result['to'] = 1
        elif outcome == 'off_foul':
            result['to'] = 1  # Offensive foul is a turnover
        
        return result


# Backward compatibility wrapper
def create_predictor(model_path: str = None, feature_engineer_path: str = None) -> ModernPlayerAwarePredictor:
    """
    Factory function to create predictor with automatic path detection.
    Tries modern paths first, falls back to legacy paths.
    """
    if model_path is None:
        modern_model_path = 'models/modern_possession_model.pt'
        legacy_model_path = 'models/possession_model.pt'
        
        try:
            import os
            if os.path.exists(modern_model_path):
                model_path = modern_model_path
            else:
                model_path = legacy_model_path
        except:
            model_path = legacy_model_path
    
    if feature_engineer_path is None:
        modern_fe_path = 'features/modern_feature_engineer.pkl'
        legacy_fe_path = 'features/feature_engineer.pkl'
        
        try:
            import os
            if os.path.exists(modern_fe_path):
                feature_engineer_path = modern_fe_path
            else:
                feature_engineer_path = legacy_fe_path
        except:
            feature_engineer_path = legacy_fe_path
    
    return ModernPlayerAwarePredictor(
        model_path=model_path,
        feature_engineer_path=feature_engineer_path
    )
