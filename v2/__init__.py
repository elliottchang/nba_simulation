"""
NBA simulation v2: data-driven, factorized possession modeling.

Pipeline:
    collect    -> raw play-by-play events + rotation-based lineups (data_v2/events/)
    transform  -> chance table, rebound/FT/substitution/stint tables (data_v2/tables/)
    features   -> shrunken per-player-season stat vectors + vocabularies
    train      -> factorized possession model (ball-ender / action / success / rebound)
    substitution -> learned substitution-exit policy
    engine     -> game/season simulation driven entirely by fitted components
    calibrate  -> backtest simulated games against held-out real games
"""

__version__ = "2.0.0"
