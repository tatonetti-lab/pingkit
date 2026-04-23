import pathlib, os, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from pingkit.embedding  import embed_dataset
from pingkit.extraction import extract_token_vectors
from pingkit.model      import (
    cross_validate, fit, save_artifacts,
    predict, load_npz_features, _evaluate, load_artifacts
)

from sklearn.metrics     import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import argparse

parser = argparse.ArgumentParser(description="Train and evaluate MMLU model with bootstrap CIs")

# Modified to accept multiple layers
parser.add_argument("--layer", type=int, nargs='+', default=[41], 
                    help="Model layer(s) to extract features from. Can specify multiple layers (e.g., --layer 20 30 40)")
parser.add_argument("--stack", action="store_true", help="Whether to stack features before training")
parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset directory")
parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset directory")
parser.add_argument("--train_response_df", type=str, required=True, help="CSV with train responses (id, answer)")
parser.add_argument("--test_response_df", type=str, required=True, help="CSV with test responses (id, answer)")
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--n_epochs", type=int, default=100, help="Maximum number of training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--random_state", type=int, default=405, help="Random seed for training split/model init")
parser.add_argument("--bootstrap_B", type=int, default=1000, help="Number of bootstrap resamples")
parser.add_argument("--bootstrap_seed", type=int, default=8675309, help="Random seed for bootstrap resampling")
parser.add_argument("--ece_bins", type=int, default=10, help="Number of uniform bins for ECE computation")
parser.add_argument("--model", type=str, default="mlp", help="Number of uniform bins for ECE computation")

args = parser.parse_args()

# Handle layers - now it's a list
layers = args.layer  # This is now a list of integers
layer = layers  # Keep the variable name for compatibility
stack = args.stack
train_dir = pathlib.Path(args.train_dir)
test_dir = pathlib.Path(args.test_dir)
train_response_df = args.train_response_df
test_response_df = args.test_response_df
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate
random_state = args.random_state

# bootstrap settings
B_BOOTSTRAP = args.bootstrap_B
BOOTSTRAP_SEED = args.bootstrap_seed
ECE_BINS = np.linspace(0.0, 1.0, args.ece_bins + 1)  # fixed binning

# Create layer string for file naming
layer_str = "_".join(map(str, sorted(layers)))
artifact_root = pathlib.Path(train_dir / f"artifacts/L{layer_str}")


# ── helpers ─────────────────────────────────────────────────────────────────
def mlp(input_dim, num_classes, hidden_dim=512):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )

def ece_uniform(correct, p_hat, bins):
    """Histogram ECE (uniform bin edges). Bins fixed across bootstraps."""
    inds = np.digitize(p_hat, bins) - 1
    ece_sum, n = 0.0, len(p_hat)
    for b in range(len(bins)-1):
        m = (inds == b)
        nb = int(m.sum())
        if nb == 0:
            continue
        frac_pos  = correct[m].mean()
        mean_pred = p_hat[m].mean()
        ece_sum  += nb * abs(frac_pos - mean_pred)
    return ece_sum / n

def brier_binary(correct, p_hat):
    """Brier score on correctness-as-label with the selected-class probability."""
    return np.mean((p_hat - correct)**2)

def percentile_ci(samples, alpha=0.05):
    lo, hi = np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

# ── feature extraction (if needed) ───────────────────────────────────────────
if stack:
    train_npz = extract_token_vectors(
        str(train_dir),
        output_file=f"{train_dir}/condensed/features_rs_L{layer_str}.npz",
        layers=layer,  # Now passing a list of layers
        parts="rs",
        n_jobs=os.cpu_count(),
    )
    print(f"✅   stacked features for layers {layers}:", train_npz)

    test_npz = extract_token_vectors(
        str(test_dir),
        output_file=f"{test_dir}/condensed/features_rs_L{layer_str}.npz",
        layers=layer,  # Now passing a list of layers
        parts="rs",
        n_jobs=os.cpu_count(),
    )
    print(f"✅   stacked features for layers {layers}:", test_npz)

# ── load train ───────────────────────────────────────────────────────────────
X_train_df, meta = load_npz_features(train_dir / "condensed" / f"features_rs_L{layer_str}.npz")

y_train = pd.read_csv(train_response_df, index_col="id")["answer"]
y_train = y_train.map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

# Align indices
common_idx   = X_train_df.index.intersection(y_train.index)
X_train_df   = X_train_df.loc[common_idx]
y_train      = y_train.loc[common_idx]
print("Train set :", X_train_df.shape, "labels:", y_train.shape)

# ── load test ────────────────────────────────────────────────────────────────
X_test_df, _ = load_npz_features(test_dir / "condensed" / f"features_rs_L{layer_str}.npz")

y_test = pd.read_csv(test_response_df, index_col="id")["answer"]
y_test = y_test.map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

common_idx   = X_test_df.index.intersection(y_test.index)
X_test_df    = X_test_df.loc[common_idx]
y_test       = y_test.loc[common_idx]
print("Test set  :", X_test_df.shape, "labels:", y_test.shape)

# ── train ────────────────────────────────────────────────────────────────────
model, history = fit(
    X_train_df,
    y_train.values,
    model              = mlp,  # or other compatible constructor
    meta               = meta,
    num_classes        = 4,
    metric             = "loss",
    batch_size         = batch_size,
    learning_rate      = learning_rate,
    patience           = 25,
    n_epochs           = n_epochs,
    early_stopping     = True,
    val_split          = 0.2,
    random_state       = random_state,
)

save_artifacts(model, path=str(artifact_root), meta=meta)

# ── evaluate on test ─────────────────────────────────────────────────────────
device     = next(model.parameters()).device
X_test_np  = X_test_df.values.astype(np.float32)

probs, test_acc, _ = _evaluate(
    model,
    X_test_np,
    y_test.values,
    model_type="mlp",
    metric_fn=lambda y, p: accuracy_score(y, p.argmax(1)),
    ce_loss=None,
    device=device,
)

pred_labels = probs.argmax(1)
p_selected  = probs[np.arange(len(probs)), pred_labels]
p_selected  = np.clip(p_selected, 1e-12, 1 - 1e-12)  # numerical safety
is_correct  = (pred_labels == y_test.values).astype(int)

# Point estimates (use the same definitions as bootstrap for consistency)
acc_point   = float(is_correct.mean())
brier_point = float(brier_binary(is_correct.astype(float), p_selected))
ece_point   = float(ece_uniform(is_correct.astype(float), p_selected, ECE_BINS))

# (Optional) you can still compute sklearn calibration_curve for plotting
frac_pos, mean_pred = calibration_curve(
    is_correct, p_selected, n_bins=10, strategy='uniform'
)

# ── stratified bootstrap CIs (B=1000) ────────────────────────────────────────
rng = np.random.default_rng(BOOTSTRAP_SEED)
y_true = y_test.values
pred   = pred_labels
conf   = p_selected

classes   = np.unique(y_true)
idx_by_c  = {c: np.where(y_true == c)[0] for c in classes}
n_by_c    = {c: len(idx_by_c[c]) for c in classes}

acc_samples   = np.empty(B_BOOTSTRAP, dtype=np.float64)
brier_samples = np.empty(B_BOOTSTRAP, dtype=np.float64)
ece_samples   = np.empty(B_BOOTSTRAP, dtype=np.float64)

for b in range(B_BOOTSTRAP):
    idx_boot = np.concatenate([
        rng.choice(idx_by_c[c], size=n_by_c[c], replace=True)
        for c in classes
    ])
    yb   = y_true[idx_boot]
    pb   = pred[idx_boot]
    cb   = conf[idx_boot]
    corr = (pb == yb).astype(float)

    acc_samples[b]   = corr.mean()
    brier_samples[b] = brier_binary(corr, cb)
    ece_samples[b]   = ece_uniform(corr, cb, ECE_BINS)

acc_ci   = percentile_ci(acc_samples,   alpha=0.05)
brier_ci = percentile_ci(brier_samples, alpha=0.05)
ece_ci   = percentile_ci(ece_samples,   alpha=0.05)

# ── report ───────────────────────────────────────────────────────────────────
results_text = f"""Model Evaluation Results
========================
Layer: {layer}

Point Estimates:
- Test Accuracy: {acc_point:.4f}
- Brier Score:   {brier_point:.4f}
- Binary ECE:    {ece_point:.4f}

95% Stratified Bootstrap CIs (B={B_BOOTSTRAP}, seed={BOOTSTRAP_SEED}):
- Accuracy CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]
- Brier CI:    [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]
- ECE CI:      [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]

Model Configuration:
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}
- Epochs: {n_epochs}
- Hidden Dim: 512
- Random State (train): {random_state}

Dataset Info:
- Train Set: {X_train_df.shape}
- Test Set:  {X_test_df.shape}
"""

artifact_root.mkdir(parents=True, exist_ok=True)
results_file = artifact_root / "evaluation_results.txt"
with open(results_file, 'w') as f:
    f.write(results_text)

# Also save machine-readable metrics & CI
metrics_json = {
    "layer": layer,
    "point": {"accuracy": acc_point, "brier": brier_point, "ece": ece_point},
    "ci_95": {
        "accuracy": {"low": acc_ci[0], "high": acc_ci[1]},
        "brier":    {"low": brier_ci[0], "high": brier_ci[1]},
        "ece":      {"low": ece_ci[0], "high": ece_ci[1]},
    },
    "bootstrap": {"B": B_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "bins": ECE_BINS.tolist()},
}
with open(artifact_root / "metrics_with_ci.json", "w") as f:
    json.dump(metrics_json, f, indent=2)

print(f"Results saved to: {results_file}")
print(json.dumps(metrics_json, indent=2))

# ── save predictions ─────────────────────────────────────────────────────────
predictions_df = pd.DataFrame({
    'id': X_test_df.index,
    'true_label': y_test.values,
    'pred_label': pred_labels,
    'confidence': p_selected,
    'is_correct': is_correct,
    'prob_A': probs[:, 0],
    'prob_B': probs[:, 1],
    'prob_C': probs[:, 2],
    'prob_D': probs[:, 3]
})
predictions_file = artifact_root / "predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved to: {predictions_file}")

# Still print to console (point + CI)
print(
    f"ACC {acc_point:.4f}  Brier {brier_point:.4f}  Binary-ECE {ece_point:.4f} | "
    f"ACC 95% CI [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]  "
    f"Brier 95% CI [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]  "
    f"ECE 95% CI [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]"
)
