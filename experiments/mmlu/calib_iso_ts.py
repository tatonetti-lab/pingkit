import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats

# ── Settings ──────────────────────────────────────────────────────────────────
INPUT_FILE = "mmlu_gemma_9b_test_CoT.csv"
CALIBRATION_SPLIT = 0.1
N_BINS = 10  # Number of bins for ECE calculation
N_BOOTSTRAP = 1000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95  # Confidence level for intervals
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(INPUT_FILE)
# ── Helper Functions ──────────────────────────────────────────────────────────

def clean_token(tok):
    """Clean token by removing tokenizer artifacts and normalizing."""
    if not isinstance(tok, str):
        return None
    # Remove common tokenizer prefixes and strip whitespace, then uppercase
    letter = tok.lstrip("▁Ġ ").upper()
    # Return only if it's a valid MMLU answer choice
    if letter in ['A', 'B', 'C', 'D']:
        return letter
    return None


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities (numpy array)
        labels: True binary labels (1 for correct, 0 for incorrect)
        n_bins: Number of bins for calibration

    Returns:
        ece: Expected Calibration Error
        bin_data: Dictionary with bin statistics for plotting
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    bin_data = {
        'bin_centers': [],
        'bin_accuracies': [],
        'bin_confidences': [],
        'bin_counts': []
    }

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices of samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()

            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            # Store for visualization
            bin_data['bin_centers'].append((bin_lower + bin_upper) / 2)
            bin_data['bin_accuracies'].append(accuracy_in_bin)
            bin_data['bin_confidences'].append(avg_confidence_in_bin)
            bin_data['bin_counts'].append(in_bin.sum())

    return ece, bin_data


def bootstrap_ece(probs, labels, n_bins=15, n_bootstrap=1000, random_state=101):
    """
    Compute bootstrapped ECE with confidence intervals.

    Args:
        probs: Predicted probabilities (numpy array)
        labels: True binary labels (1 for correct, 0 for incorrect)
        n_bins: Number of bins for calibration
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        dict with 'ece_values', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    np.random.seed(random_state)
    n_samples = len(probs)
    ece_values = []

    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_probs = probs[bootstrap_idx]
        bootstrap_labels = labels[bootstrap_idx]

        ece, _ = compute_ece(bootstrap_probs, bootstrap_labels, n_bins)
        ece_values.append(ece)

    ece_values = np.array(ece_values)

    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(ece_values, 100 * alpha / 2)
    ci_upper = np.percentile(ece_values, 100 * (1 - alpha / 2))

    return {
        'ece_values': ece_values,
        'mean': np.mean(ece_values),
        'std': np.std(ece_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def compute_brier_score(probs, labels):
    """
    Compute Brier score for binary predictions.

    Args:
        probs: Predicted probabilities (numpy array)
        labels: True binary labels (1 for correct, 0 for incorrect)

    Returns:
        float: Mean squared error between probabilities and labels
    """
    return np.mean((probs - labels) ** 2)


def bootstrap_brier(probs, labels, n_bootstrap=1000, random_state=101):
    """
    Compute bootstrapped Brier score with confidence intervals.

    Args:
        probs: Predicted probabilities (numpy array)
        labels: True binary labels (1 for correct, 0 for incorrect)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        dict with 'brier_values', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    np.random.seed(random_state)
    n_samples = len(probs)
    brier_values = []

    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_probs = probs[bootstrap_idx]
        bootstrap_labels = labels[bootstrap_idx]
        brier = compute_brier_score(bootstrap_probs, bootstrap_labels)
        brier_values.append(brier)

    brier_values = np.array(brier_values)

    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(brier_values, 100 * alpha / 2)
    ci_upper = np.percentile(brier_values, 100 * (1 - alpha / 2))

    return {
        'brier_values': brier_values,
        'mean': np.mean(brier_values),
        'std': np.std(brier_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def compute_accuracy(labels):
    """
    Compute accuracy from binary correctness labels.

    Args:
        labels: Binary array where 1 = correct and 0 = incorrect

    Returns:
        float: Accuracy
    """
    return np.mean(labels)


def bootstrap_accuracy(labels, n_bootstrap=1000, random_state=101):
    """
    Compute bootstrapped accuracy with confidence intervals.

    Args:
        labels: True binary labels (1 for correct, 0 for incorrect)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        dict with 'accuracy_values', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    np.random.seed(random_state)
    n_samples = len(labels)
    accuracy_values = []

    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_labels = labels[bootstrap_idx]
        accuracy = compute_accuracy(bootstrap_labels)
        accuracy_values.append(accuracy)

    accuracy_values = np.array(accuracy_values)

    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(accuracy_values, 100 * alpha / 2)
    ci_upper = np.percentile(accuracy_values, 100 * (1 - alpha / 2))

    return {
        'accuracy_values': accuracy_values,
        'mean': np.mean(accuracy_values),
        'std': np.std(accuracy_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


class TemperatureScaling(nn.Module):
    """
    Temperature scaling calibration.
    A single learnable parameter T is optimized on validation data.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def fit(self, probs, labels, max_iter=1000, lr=0.01):
        """
        Fit temperature parameter using NLL loss.

        Args:
            probs: Predicted probabilities (numpy array)
            labels: True binary labels
            max_iter: Maximum iterations for optimization
            lr: Learning rate
        """
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))

        logits_tensor = torch.tensor(logits, dtype=torch.float32).to(DEVICE)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(DEVICE)

        self.to(DEVICE)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_tensor)
            scaled_probs = torch.sigmoid(scaled_logits)
            loss = nn.BCELoss()(scaled_probs, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimized temperature: {self.temperature.item():.4f}")

    def predict(self, probs):
        """Apply temperature scaling to probabilities."""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        logits_tensor = torch.tensor(logits, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            scaled_logits = self.forward(logits_tensor)
            scaled_probs = torch.sigmoid(scaled_logits).cpu().numpy()

        return scaled_probs


def plot_calibration_with_bootstrap(
    original_data, isotonic_data, temp_data,
    original_boot, isotonic_boot, temp_boot,
    save_path=None
):
    """
    Plot calibration curves with bootstrap confidence intervals.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    methods = ['Original', 'Isotonic Regression', 'Temperature Scaling']
    data_list = [original_data, isotonic_data, temp_data]
    boot_list = [original_boot, isotonic_boot, temp_boot]

    for ax, method, data, boot in zip(axes[0], methods, data_list, boot_list):
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

        if data['bin_centers']:
            ax.scatter(
                data['bin_centers'],
                data['bin_accuracies'],
                s=np.array(data['bin_counts']) * 2,
                alpha=0.7,
                label='Calibration curve'
            )

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ece_str = (
            f"ECE: {boot['mean']:.4f} ± {boot['std']:.4f}\n"
            f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]"
        )
        ax.set_title(f'{method}\n{ece_str}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    for ax, method, boot in zip(axes[1], methods, boot_list):
        ax.hist(
            boot['ece_values'],
            bins=50,
            alpha=0.7,
            density=True,
            color='skyblue',
            edgecolor='black'
        )

        ax.axvline(
            boot['mean'],
            color='red',
            linestyle='-',
            linewidth=2,
            label=f"Mean: {boot['mean']:.4f}"
        )
        ax.axvline(boot['ci_lower'], color='orange', linestyle='--', linewidth=2)
        ax.axvline(
            boot['ci_upper'],
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f"{CONFIDENCE_LEVEL*100:.0f}% CI"
        )

        ax.set_xlabel('ECE')
        ax.set_ylabel('Density')
        ax.set_title(f'{method} - Bootstrap Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_metric_comparison(methods, means, stds, ylabel, title, save_path=None):
    """
    Generic comparison plot with error bars.
    """
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    bars = plt.bar(
        methods,
        means,
        yerr=stds,
        capsize=10,
        alpha=0.7,
        color=colors,
        edgecolor='black'
    )

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std + 0.001,
            f'{mean:.4f}±{std:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ece_comparison(original_boot, isotonic_boot, temp_boot, save_path=None):
    methods = ['Original', 'Isotonic\nRegression', 'Temperature\nScaling']
    means = [original_boot['mean'], isotonic_boot['mean'], temp_boot['mean']]
    stds = [original_boot['std'], isotonic_boot['std'], temp_boot['std']]

    plot_metric_comparison(
        methods,
        means,
        stds,
        ylabel='Expected Calibration Error (ECE)',
        title=(
            f'ECE Comparison with Bootstrap {CONFIDENCE_LEVEL*100:.0f}% Confidence Intervals\n'
            f'(Lower is better)'
        ),
        save_path=save_path
    )


def plot_brier_comparison(original_boot, isotonic_boot, temp_boot, save_path=None):
    methods = ['Original', 'Isotonic\nRegression', 'Temperature\nScaling']
    means = [original_boot['mean'], isotonic_boot['mean'], temp_boot['mean']]
    stds = [original_boot['std'], isotonic_boot['std'], temp_boot['std']]

    plot_metric_comparison(
        methods,
        means,
        stds,
        ylabel='Brier Score',
        title=(
            f'Brier Score Comparison with Bootstrap {CONFIDENCE_LEVEL*100:.0f}% Confidence Intervals\n'
            f'(Lower is better)'
        ),
        save_path=save_path
    )


def plot_accuracy_comparison(original_boot, isotonic_boot, temp_boot, save_path=None):
    methods = ['Original', 'Isotonic\nRegression', 'Temperature\nScaling']
    means = [original_boot['mean'], isotonic_boot['mean'], temp_boot['mean']]
    stds = [original_boot['std'], isotonic_boot['std'], temp_boot['std']]

    plot_metric_comparison(
        methods,
        means,
        stds,
        ylabel='Accuracy',
        title=(
            f'Accuracy Comparison with Bootstrap {CONFIDENCE_LEVEL*100:.0f}% Confidence Intervals\n'
            f'(Higher is better)'
        ),
        save_path=save_path
    )


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # Clean tokens using robust token processing
    df['cleaned_top_token'] = df['top_token'].apply(clean_token)
    df['cleaned_correct'] = df['correct'].apply(clean_token)

    # Check if predictions are correct (only for valid tokens)
    df['is_correct'] = (
        (df['cleaned_top_token'].notna()) &
        (df['cleaned_correct'].notna()) &
        (df['cleaned_top_token'] == df['cleaned_correct'])
    ).astype(int)

    # Set probability to 0 for invalid tokens
    df['prob_correct'] = np.where(
        df['cleaned_top_token'].notna(),
        df['top_token_prob'],
        0.0
    )

    probs = df['prob_correct'].values
    labels = df['is_correct'].values

    probs = np.clip(probs, 1e-12, 1 - 1e-12)

    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {df['cleaned_top_token'].notna().sum()}")
    print(f"Invalid predictions (set to prob=0): {df['cleaned_top_token'].isna().sum()}")
    print(f"Accuracy: {labels.mean():.4f}")
    print(f"Mean confidence: {probs.mean():.4f}")

    # Split data for calibration and testing
    indices = np.arange(len(df))
    cal_idx, test_idx = train_test_split(
        indices,
        test_size=1 - CALIBRATION_SPLIT,
        random_state=42,
        stratify=labels
    )

    cal_probs, cal_labels = probs[cal_idx], labels[cal_idx]
    test_probs, test_labels = probs[test_idx], labels[test_idx]

    print(f"\nCalibration set: {len(cal_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")
    print(f"Bootstrap samples: {N_BOOTSTRAP}")

    # ── Original metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ORIGINAL (UNCALIBRATED)")
    print("=" * 60)

    original_ece, original_bin_data = compute_ece(test_probs, test_labels, N_BINS)
    original_bin_data['ece'] = original_ece

    print("Computing bootstrapped ECE...")
    original_boot = bootstrap_ece(test_probs, test_labels, N_BINS, N_BOOTSTRAP)
    print(f"ECE: {original_boot['mean']:.4f} ± {original_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{original_boot['ci_lower']:.4f}, {original_boot['ci_upper']:.4f}]")

    original_brier = compute_brier_score(test_probs, test_labels)
    print("Computing bootstrapped Brier score...")
    original_brier_boot = bootstrap_brier(test_probs, test_labels, N_BOOTSTRAP)
    print(f"Brier score: {original_brier_boot['mean']:.4f} ± {original_brier_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{original_brier_boot['ci_lower']:.4f}, {original_brier_boot['ci_upper']:.4f}]")

    original_accuracy = compute_accuracy(test_labels)
    print("Computing bootstrapped accuracy...")
    original_accuracy_boot = bootstrap_accuracy(test_labels, N_BOOTSTRAP)
    print(f"Accuracy: {original_accuracy_boot['mean']:.4f} ± {original_accuracy_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{original_accuracy_boot['ci_lower']:.4f}, {original_accuracy_boot['ci_upper']:.4f}]")

    # ── Isotonic Regression ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ISOTONIC REGRESSION CALIBRATION")
    print("=" * 60)

    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_reg.fit(cal_probs, cal_labels)

    test_probs_isotonic = iso_reg.predict(test_probs)

    isotonic_ece, isotonic_bin_data = compute_ece(test_probs_isotonic, test_labels, N_BINS)
    isotonic_bin_data['ece'] = isotonic_ece

    print("Computing bootstrapped ECE...")
    isotonic_boot = bootstrap_ece(test_probs_isotonic, test_labels, N_BINS, N_BOOTSTRAP)
    print(f"ECE after Isotonic Regression: {isotonic_boot['mean']:.4f} ± {isotonic_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{isotonic_boot['ci_lower']:.4f}, {isotonic_boot['ci_upper']:.4f}]")
    print(f"ECE improvement: {original_boot['mean'] - isotonic_boot['mean']:.4f}")

    isotonic_brier = compute_brier_score(test_probs_isotonic, test_labels)
    print("Computing bootstrapped Brier score...")
    isotonic_brier_boot = bootstrap_brier(test_probs_isotonic, test_labels, N_BOOTSTRAP)
    print(f"Brier after Isotonic Regression: {isotonic_brier_boot['mean']:.4f} ± {isotonic_brier_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{isotonic_brier_boot['ci_lower']:.4f}, {isotonic_brier_boot['ci_upper']:.4f}]")
    print(f"Brier improvement: {original_brier_boot['mean'] - isotonic_brier_boot['mean']:.4f}")

    # Accuracy should not change because labels/predicted class identity do not change
    isotonic_accuracy = compute_accuracy(test_labels)
    print("Computing bootstrapped accuracy...")
    isotonic_accuracy_boot = bootstrap_accuracy(test_labels, N_BOOTSTRAP)
    print(f"Accuracy after Isotonic Regression: {isotonic_accuracy_boot['mean']:.4f} ± {isotonic_accuracy_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{isotonic_accuracy_boot['ci_lower']:.4f}, {isotonic_accuracy_boot['ci_upper']:.4f}]")
    print(f"Accuracy improvement: {isotonic_accuracy_boot['mean'] - original_accuracy_boot['mean']:.4f}")

    # ── Temperature Scaling ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 60)

    temp_scaling = TemperatureScaling()
    temp_scaling.fit(cal_probs, cal_labels)

    test_probs_temp = temp_scaling.predict(test_probs)

    temp_ece, temp_bin_data = compute_ece(test_probs_temp, test_labels, N_BINS)
    temp_bin_data['ece'] = temp_ece

    print("Computing bootstrapped ECE...")
    temp_boot = bootstrap_ece(test_probs_temp, test_labels, N_BINS, N_BOOTSTRAP)
    print(f"ECE after Temperature Scaling: {temp_boot['mean']:.4f} ± {temp_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{temp_boot['ci_lower']:.4f}, {temp_boot['ci_upper']:.4f}]")
    print(f"ECE improvement: {original_boot['mean'] - temp_boot['mean']:.4f}")

    temp_brier = compute_brier_score(test_probs_temp, test_labels)
    print("Computing bootstrapped Brier score...")
    temp_brier_boot = bootstrap_brier(test_probs_temp, test_labels, N_BOOTSTRAP)
    print(f"Brier after Temperature Scaling: {temp_brier_boot['mean']:.4f} ± {temp_brier_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{temp_brier_boot['ci_lower']:.4f}, {temp_brier_boot['ci_upper']:.4f}]")
    print(f"Brier improvement: {original_brier_boot['mean'] - temp_brier_boot['mean']:.4f}")

    temp_accuracy = compute_accuracy(test_labels)
    print("Computing bootstrapped accuracy...")
    temp_accuracy_boot = bootstrap_accuracy(test_labels, N_BOOTSTRAP)
    print(f"Accuracy after Temperature Scaling: {temp_accuracy_boot['mean']:.4f} ± {temp_accuracy_boot['std']:.4f}")
    print(f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{temp_accuracy_boot['ci_lower']:.4f}, {temp_accuracy_boot['ci_upper']:.4f}]")
    print(f"Accuracy improvement: {temp_accuracy_boot['mean'] - original_accuracy_boot['mean']:.4f}")

    # ── Statistical Significance Tests ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    # ECE tests
    _, p_isotonic = stats.ttest_rel(original_boot['ece_values'], isotonic_boot['ece_values'])
    _, p_temp = stats.ttest_rel(original_boot['ece_values'], temp_boot['ece_values'])
    _, p_methods = stats.ttest_rel(isotonic_boot['ece_values'], temp_boot['ece_values'])

    print("ECE tests")
    print(f"Original vs Isotonic p-value: {p_isotonic:.6f}")
    print(f"Original vs Temperature p-value: {p_temp:.6f}")
    print(f"Isotonic vs Temperature p-value: {p_methods:.6f}")

    # Brier tests
    _, p_isotonic_brier = stats.ttest_rel(
        original_brier_boot['brier_values'],
        isotonic_brier_boot['brier_values']
    )
    _, p_temp_brier = stats.ttest_rel(
        original_brier_boot['brier_values'],
        temp_brier_boot['brier_values']
    )
    _, p_methods_brier = stats.ttest_rel(
        isotonic_brier_boot['brier_values'],
        temp_brier_boot['brier_values']
    )

    print("\nBrier score tests")
    print(f"Original vs Isotonic p-value: {p_isotonic_brier:.6f}")
    print(f"Original vs Temperature p-value: {p_temp_brier:.6f}")
    print(f"Isotonic vs Temperature p-value: {p_methods_brier:.6f}")

    # Accuracy tests
    _, p_isotonic_acc = stats.ttest_rel(
        original_accuracy_boot['accuracy_values'],
        isotonic_accuracy_boot['accuracy_values']
    )
    _, p_temp_acc = stats.ttest_rel(
        original_accuracy_boot['accuracy_values'],
        temp_accuracy_boot['accuracy_values']
    )
    _, p_methods_acc = stats.ttest_rel(
        isotonic_accuracy_boot['accuracy_values'],
        temp_accuracy_boot['accuracy_values']
    )

    print("\nAccuracy tests")
    print(f"Original vs Isotonic p-value: {p_isotonic_acc:.6f}")
    print(f"Original vs Temperature p-value: {p_temp_acc:.6f}")
    print(f"Isotonic vs Temperature p-value: {p_methods_acc:.6f}")

    alpha = 0.05
    print(f"\nUsing significance level α = {alpha}")
    print(f"Isotonic ECE improvement is {'significant' if p_isotonic < alpha else 'not significant'}")
    print(f"Temperature ECE improvement is {'significant' if p_temp < alpha else 'not significant'}")
    print(f"ECE difference between methods is {'significant' if p_methods < alpha else 'not significant'}")

    print(f"Isotonic Brier improvement is {'significant' if p_isotonic_brier < alpha else 'not significant'}")
    print(f"Temperature Brier improvement is {'significant' if p_temp_brier < alpha else 'not significant'}")
    print(f"Brier difference between methods is {'significant' if p_methods_brier < alpha else 'not significant'}")

    print(f"Isotonic Accuracy improvement is {'significant' if p_isotonic_acc < alpha else 'not significant'}")
    print(f"Temperature Accuracy improvement is {'significant' if p_temp_acc < alpha else 'not significant'}")
    print(f"Accuracy difference between methods is {'significant' if p_methods_acc < alpha else 'not significant'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Original ECE:              {original_boot['mean']:.4f} ± {original_boot['std']:.4f}")
    print(f"Isotonic Regression ECE:   {isotonic_boot['mean']:.4f} ± {isotonic_boot['std']:.4f} (↓ {original_boot['mean'] - isotonic_boot['mean']:.4f})")
    print(f"Temperature Scaling ECE:   {temp_boot['mean']:.4f} ± {temp_boot['std']:.4f} (↓ {original_boot['mean'] - temp_boot['mean']:.4f})")

    print()
    print(f"Original Brier:            {original_brier_boot['mean']:.4f} ± {original_brier_boot['std']:.4f}")
    print(f"Isotonic Regression Brier: {isotonic_brier_boot['mean']:.4f} ± {isotonic_brier_boot['std']:.4f} (↓ {original_brier_boot['mean'] - isotonic_brier_boot['mean']:.4f})")
    print(f"Temperature Scaling Brier: {temp_brier_boot['mean']:.4f} ± {temp_brier_boot['std']:.4f} (↓ {original_brier_boot['mean'] - temp_brier_boot['mean']:.4f})")

    print()
    print(f"Original Accuracy:         {original_accuracy_boot['mean']:.4f} ± {original_accuracy_boot['std']:.4f}")
    print(f"Isotonic Regression Acc:   {isotonic_accuracy_boot['mean']:.4f} ± {isotonic_accuracy_boot['std']:.4f} (Δ {isotonic_accuracy_boot['mean'] - original_accuracy_boot['mean']:.4f})")
    print(f"Temperature Scaling Acc:   {temp_accuracy_boot['mean']:.4f} ± {temp_accuracy_boot['std']:.4f} (Δ {temp_accuracy_boot['mean'] - original_accuracy_boot['mean']:.4f})")

    best_method_ece = "Isotonic Regression" if isotonic_boot['mean'] < temp_boot['mean'] else "Temperature Scaling"
    best_method_brier = "Isotonic Regression" if isotonic_brier_boot['mean'] < temp_brier_boot['mean'] else "Temperature Scaling"
    best_method_acc = "No difference" if np.isclose(isotonic_accuracy_boot['mean'], temp_accuracy_boot['mean']) else (
        "Isotonic Regression" if isotonic_accuracy_boot['mean'] > temp_accuracy_boot['mean'] else "Temperature Scaling"
    )

    print(f"\nBest calibration method by ECE: {best_method_ece}")
    print(f"Best calibration method by Brier: {best_method_brier}")
    print(f"Best calibration method by Accuracy: {best_method_acc}")

    # Confidence interval overlap checks
    iso_ci = (isotonic_boot['ci_lower'], isotonic_boot['ci_upper'])
    temp_ci = (temp_boot['ci_lower'], temp_boot['ci_upper'])
    if iso_ci[1] < temp_ci[0] or temp_ci[1] < iso_ci[0]:
        print("ECE confidence intervals do not overlap - difference is likely significant")
    else:
        print("ECE confidence intervals overlap - difference may not be significant")

    iso_brier_ci = (isotonic_brier_boot['ci_lower'], isotonic_brier_boot['ci_upper'])
    temp_brier_ci = (temp_brier_boot['ci_lower'], temp_brier_boot['ci_upper'])
    if iso_brier_ci[1] < temp_brier_ci[0] or temp_brier_ci[1] < iso_brier_ci[0]:
        print("Brier confidence intervals do not overlap - difference is likely significant")
    else:
        print("Brier confidence intervals overlap - difference may not be significant")

    iso_acc_ci = (isotonic_accuracy_boot['ci_lower'], isotonic_accuracy_boot['ci_upper'])
    temp_acc_ci = (temp_accuracy_boot['ci_lower'], temp_accuracy_boot['ci_upper'])
    if iso_acc_ci[1] < temp_acc_ci[0] or temp_acc_ci[1] < iso_acc_ci[0]:
        print("Accuracy confidence intervals do not overlap - difference is likely significant")
    else:
        print("Accuracy confidence intervals overlap - difference may not be significant")

    # ── Visualization ─────────────────────────────────────────────────────────
    print("\nGenerating calibration plots...")
    plot_calibration_with_bootstrap(
        original_bin_data, isotonic_bin_data, temp_bin_data,
        original_boot, isotonic_boot, temp_boot,
        save_path="calibration_comparison_bootstrap.png"
    )

    print("Generating ECE comparison plot...")
    plot_ece_comparison(
        original_boot, isotonic_boot, temp_boot,
        save_path="ece_comparison_bootstrap.png"
    )

    print("Generating Brier score comparison plot...")
    plot_brier_comparison(
        original_brier_boot, isotonic_brier_boot, temp_brier_boot,
        save_path="brier_comparison_bootstrap.png"
    )

    print("Generating accuracy comparison plot...")
    plot_accuracy_comparison(
        original_accuracy_boot, isotonic_accuracy_boot, temp_accuracy_boot,
        save_path="accuracy_comparison_bootstrap.png"
    )

    # ── Save calibrated results ───────────────────────────────────────────────
    df_test = df.iloc[test_idx].copy()
    df_test['prob_isotonic'] = test_probs_isotonic
    df_test['prob_temp_scaled'] = test_probs_temp

    bootstrap_stats = pd.DataFrame({
        'method': ['original', 'isotonic', 'temperature'],

        'ece_mean': [original_boot['mean'], isotonic_boot['mean'], temp_boot['mean']],
        'ece_std': [original_boot['std'], isotonic_boot['std'], temp_boot['std']],
        'ece_ci_lower': [original_boot['ci_lower'], isotonic_boot['ci_lower'], temp_boot['ci_lower']],
        'ece_ci_upper': [original_boot['ci_upper'], isotonic_boot['ci_upper'], temp_boot['ci_upper']],

        'brier_mean': [original_brier_boot['mean'], isotonic_brier_boot['mean'], temp_brier_boot['mean']],
        'brier_std': [original_brier_boot['std'], isotonic_brier_boot['std'], temp_brier_boot['std']],
        'brier_ci_lower': [original_brier_boot['ci_lower'], isotonic_brier_boot['ci_lower'], temp_brier_boot['ci_lower']],
        'brier_ci_upper': [original_brier_boot['ci_upper'], isotonic_brier_boot['ci_upper'], temp_brier_boot['ci_upper']],

        'accuracy_mean': [original_accuracy_boot['mean'], isotonic_accuracy_boot['mean'], temp_accuracy_boot['mean']],
        'accuracy_std': [original_accuracy_boot['std'], isotonic_accuracy_boot['std'], temp_accuracy_boot['std']],
        'accuracy_ci_lower': [original_accuracy_boot['ci_lower'], isotonic_accuracy_boot['ci_lower'], temp_accuracy_boot['ci_lower']],
        'accuracy_ci_upper': [original_accuracy_boot['ci_upper'], isotonic_accuracy_boot['ci_upper'], temp_accuracy_boot['ci_upper']],

        'p_value_ece_vs_original': [np.nan, p_isotonic, p_temp],
        'p_value_brier_vs_original': [np.nan, p_isotonic_brier, p_temp_brier],
        'p_value_accuracy_vs_original': [np.nan, p_isotonic_acc, p_temp_acc]
    })

    output_file = INPUT_FILE.replace('.csv', '_calibrated_dropped.csv')
    bootstrap_file = INPUT_FILE.replace('.csv', '_bootstrap_stats_dropped.csv')

    df_test.to_csv(output_file, index=False)
    bootstrap_stats.to_csv(bootstrap_file, index=False)

    print(f"\nSaved calibrated results to: {output_file}")
    print(f"Saved bootstrap statistics to: {bootstrap_file}")


if __name__ == "__main__":
    main()