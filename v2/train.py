"""Train the factorized possession model + fit auxiliary models.

    python -m v2.train [--epochs 30] [--max-rows N]

Reports validation log-loss against a marginal-distribution baseline so the
lineup model has to prove it earns its parameters.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import ChanceDataset, split_by_game, ACTION_TO_IDX
from .duration import fit_duration_model
from .features import build_feature_store, resolve_names
from .model import PossessionModelV2, make_optimizer
from .substitution import fit_substitution_model


def marginal_baseline_logloss(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Log-loss of predicting the league-marginal action distribution."""
    probs = train_df["action"].value_counts(normalize=True)
    eps = 1e-9
    return float(-np.log(val_df["action"].map(probs).fillna(eps) + eps).mean())


def run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    totals, counts = {}, {}
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, parts, _ = model.loss(batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


def train(epochs=30, max_rows=None, device=None, patience=4):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    chances = pd.read_parquet(config.CHANCES_FILE)
    if max_rows:
        chances = chances.iloc[:max_rows]
    fts = pd.read_parquet(config.FT_FILE)
    subs = pd.read_parquet(config.SUB_EVENTS_FILE)
    print(f"{len(chances):,} chances, {chances['game_id'].nunique():,} games")

    train_df, val_df = split_by_game(chances)
    print(f"Split: {train_df['game_id'].nunique()} train games, "
          f"{val_df['game_id'].nunique()} val games")
    print(f"Baseline (marginal action) val log-loss: "
          f"{marginal_baseline_logloss(train_df, val_df):.4f}")

    # Feature store is fitted on TRAINING games only to keep validation honest.
    fs = build_feature_store(train_df, fts, subs)
    resolve_names(fs)
    fs.save()

    train_ds = ChanceDataset(train_df, fs, training=True)
    val_ds = ChanceDataset(val_df, fs, training=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = PossessionModelV2(fs.num_players, fs.num_seasons).to(device)
    optimizer = make_optimizer(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    best_val, bad_epochs = float("inf"), 0
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device)
        va = run_epoch(model, val_loader, None, device)
        val_total = sum(va.values())
        line = " ".join(f"{k}={v:.3f}" for k, v in sorted(va.items()))
        print(f"epoch {epoch:02d} ({time.time()-t0:.0f}s) "
              f"train={sum(tr.values()):.3f} val={val_total:.3f} [{line}]")

        if val_total < best_val - 1e-4:
            best_val, bad_epochs = val_total, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_players": fs.num_players,
                "num_seasons": fs.num_seasons,
                "val_loss": best_val,
            }, config.MODEL_FILE)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (no improvement for {patience} epochs)")
                break
    print(f"Best val loss {best_val:.4f} -> {config.MODEL_FILE}")

    # ------------------------------------------------------ auxiliary models
    fit_duration_model(train_df)
    fit_substitution_model(subs[subs["game_id"].isin(train_df["game_id"].unique())])
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
