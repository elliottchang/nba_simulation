# Simulation v2 — data-driven factorized possession model

Replaces the legacy pipeline's hand-tuned constants and disconnected
model/attribution with fitted components at every stage.

## What changed vs. legacy

| Legacy | v2 |
|---|---|
| Outcome sampled at lineup level, player attributed by (broken) usage lottery | Factorized model: P(ball-ender) → P(action \| ball-ender) → P(result \| shot, defense) — stats follow from who actually did what |
| `usage_rate` always 1.0 | Usage = ended chances ÷ team chances while on floor, EB-shrunk |
| Lineups tracked from sub events (drift across quarters, ~60% of possessions lost) | Substitution tracking + period-start inference + self-healing anchors (pilot: 236 chances/game, 0.8% skipped) |
| Shooting/defensive fouls attributed to the defender (wrong team's lineups) | Fouled player is the ball-ender, identified by who shoots the ensuing FTs; defender only charged the foul |
| Hardcoded rebound rates, uniform rebounder | Learned OREB head + rebounder head over real rebounds |
| Energy meter with invented thresholds | Stint/cumulative-minutes features in the model, fitted on real data |
| Rule-based substitutions gated by a `deadball` flag that never reset | Logistic exit policy fitted on real (deadball × player) coach decisions |
| `Binomial(24, 0.7)` possession clock | Empirical duration distributions by action and clock state |
| FT% fabricated from 2PT% | Real (shrunken) FT% per player-season |
| No bonus, foul-out at 5, lineups reset to starters each quarter | Team fouls/penalty tracked per period, foul-out at 6, no resets |
| AST/STL/BLK always zero | Attributed from real assist/steal/block rates |
| Mean-pooled embeddings, per-player-season vocab | Stat-anchored tokens + cross-season player embeddings + cross-attention matchup layer |
| Random-row train/val split (leaks) | Game-level split, marginal baseline reported |
| No realism measurement | `calibrate.py` backtests simulated vs. held-out real games |

> **API note (June 2026):** the NBA retired `PlayByPlayV2` and `GameRotation`
> (both return empty JSON), which also broke the legacy collector. v2 uses
> `PlayByPlayV3` + `BoxScoreTraditionalV3` — two calls per game. V3 foul
> descriptions name the *referee*, not the fouled player, so fouled-player
> attribution goes through the ensuing free-throw shooter; non-bonus drawn
> fouls (no FTs) are kept at lineup level (`ball_ender = -1`) and handled
> with a lineup-mean token in the action head.

## Pilot validation (20 real 2024-25 games)

Extraction quality, measured before committing to full collection:
accumulated scores match the official play-by-play exactly in 19/20 games;
236 chances/game (legacy: 89) with 0.8% skipped; steal capture 367/369 vs
official; FG% 45.5 / 3PA share 40.4% / OREB 24.0% / FT% 80.3 / assisted-on-
make 61.7% — all consistent with NBA rates. Training and the calibration
loop run end to end on this pilot; expect simulated quality to be poor until
full seasons are collected (embeddings and the substitution policy are
data-hungry).

## Workflow

```bash
pip install -r requirements_v2.txt

# 1. Collect raw events + rotations (network, hours; checkpointed, resumable)
python -m v2.collect                      # all seasons in config.SEASONS
python -m v2.collect --season 2023-24 --max-games 100   # partial run

# 2. Derive training tables (offline, re-runnable)
python -m v2.transform

# 3. Train possession model + duration + substitution models
python -m v2.train --epochs 30

# 4. Measure realism against held-out games
python -m v2.calibrate --games 40 --sims 5
```

Simulate a game (after training; no network needed):

```python
from v2.engine import GameEngineV2
game = GameEngineV2(1610612747, 1610612744, "2023-24", verbose=True)
result = game.simulate_game()
print(result["home_box"])
```

Verify the pipeline without real data:

```bash
python -m v2.smoke_test
```

## Remaining simplifications (documented, not hidden)

* The defender charged with a foul is uniform-random; a who-fouls model is a
  natural next head (fouler identity is in the raw events).
* Replacement choice on substitution is minutes-share × freshness; the exit
  decision is learned but the entry decision is a prior. A learned entry
  model would close the loop.
* No intentional-foul / hold-for-last-shot end-game logic; the duration model
  only captures faster late-clock play.
* `MADE_SHOT_DEADBALL_SAMPLE` thins sub-decision rows at made-shot deadballs
  for table size; it does not bias the fitted conditional probabilities.
* Timeouts are not simulated (they matter mainly through substitution
  opportunities, which the exit policy already absorbs).

## Dimensionality guards

Three seasons ≈ 2,900 player-seasons is thin for per-player learning, so:
stat vectors carry the bulk of the signal (shrunk with pseudo-counts);
learned embeddings are small (16-dim), cross-season, and heavily
weight-decayed; lineup slots are randomly masked to UNK during training;
matchup interactions go through a single cross-attention layer over latent
types rather than pairwise player terms; and validation is by game with a
marginal baseline so interaction capacity has to prove itself out of sample.
Extending `config.SEASONS` backwards is the cheapest way to buy more signal.
