import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, ttest_rel


# =========================================================
# User paths
# =========================================================
INPUT_CSV = "~/work/gail_ppo_ridesharing-main/data/benchmark_episode_comparison.csv"
OUT_DIR = "~/work/gail_ppo_ridesharing-main/data/benchmark_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# Config
# =========================================================
# Lower is better for these metrics
METRICS = [
    ("avg_wait_time", "Average Wait Time"),
    ("avg_in_vehicle_time", "Average In-Vehicle Time"),
    ("avg_service_time", "Average Service Time"),
]

METHODS = {
    "offline": "offline",
    "online_nearest": "online_nearest",
    "online_wait_aware": "online_wait_aware",
}

RL_PREFIX = "rl"

EPS = 1e-9


# =========================================================
# Helpers
# =========================================================
def safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def mean_ci_95(x: pd.Series):
    x = x.dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if n == 1:
        return m, m, m
    se = x.std(ddof=1) / math.sqrt(n)
    half = 1.96 * se
    return m, m - half, m + half


def win_loss_tie_counts(diff_series: pd.Series, lower_is_better=True):
    """
    diff = comparator - rl
    If lower is better:
      diff > 0  => RL better
      diff < 0  => comparator better
      diff == 0 => tie
    """
    s = diff_series.dropna()

    if lower_is_better:
        rl_wins = (s > 0).sum()
        rl_losses = (s < 0).sum()
    else:
        rl_wins = (s < 0).sum()
        rl_losses = (s > 0).sum()

    ties = (s == 0).sum()
    total = len(s)

    if total == 0:
        return {
            "n": 0,
            "rl_wins": 0,
            "rl_losses": 0,
            "ties": 0,
            "rl_win_rate": np.nan,
            "rl_loss_rate": np.nan,
            "tie_rate": np.nan,
        }

    return {
        "n": total,
        "rl_wins": int(rl_wins),
        "rl_losses": int(rl_losses),
        "ties": int(ties),
        "rl_win_rate": rl_wins / total,
        "rl_loss_rate": rl_losses / total,
        "tie_rate": ties / total,
    }


def paired_tests(a: pd.Series, b: pd.Series):
    """
    a = comparator
    b = RL
    """
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 2:
        return {
            "n_used": len(tmp),
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
            "ttest_stat": np.nan,
            "ttest_p": np.nan,
        }

    a_clean = tmp.iloc[:, 0]
    b_clean = tmp.iloc[:, 1]

    # Wilcoxon can fail if all differences are exactly zero
    try:
        w_stat, w_p = wilcoxon(a_clean, b_clean, zero_method="wilcox", alternative="two-sided")
    except Exception:
        w_stat, w_p = np.nan, np.nan

    try:
        t_stat, t_p = ttest_rel(a_clean, b_clean, nan_policy="omit")
    except Exception:
        t_stat, t_p = np.nan, np.nan

    return {
        "n_used": len(tmp),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
    }


def save_distribution_plots(diff: pd.Series, comparator_label: str, metric_slug: str, metric_label: str):
    s = diff.dropna()
    if len(s) == 0:
        return

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=40)
    plt.axvline(0, linestyle="--")
    plt.title(f"Distribution of Difference: {comparator_label} - RL\n{metric_label}")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_{comparator_label}_{metric_slug}.png"), dpi=200)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot(s.dropna(), vert=True)
    plt.axhline(0, linestyle="--")
    plt.title(f"Boxplot of Difference: {comparator_label} - RL\n{metric_label}")
    plt.ylabel("Difference")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"box_{comparator_label}_{metric_slug}.png"), dpi=200)
    plt.close()

    # ECDF
    x = np.sort(s.values)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.axvline(0, linestyle="--")
    plt.title(f"ECDF of Difference: {comparator_label} - RL\n{metric_label}")
    plt.xlabel("Difference")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ecdf_{comparator_label}_{metric_slug}.png"), dpi=200)
    plt.close()


def save_scatter_plot(df_plot: pd.DataFrame, xcol: str, ycol: str, comparator_label: str, metric_slug: str, metric_label: str):
    tmp = df_plot[[xcol, ycol]].dropna()
    if len(tmp) == 0:
        return

    x = tmp[xcol]
    y = tmp[ycol]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5)
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(f"{comparator_label}")
    plt.ylabel("RL")
    plt.title(f"Paired Episode Values\n{metric_label}: {comparator_label} vs RL")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"scatter_{comparator_label}_{metric_slug}.png"), dpi=200)
    plt.close()


# =========================================================
# Load data
# =========================================================
df = pd.read_csv(INPUT_CSV)

# Force numeric
for col in df.columns:
    if col != "episode_id":
        df[col] = safe_series(df[col])

# Optional: keep a copy with all pairwise columns added
df_out = df.copy()


# =========================================================
# Main analysis
# =========================================================
summary_rows = []
test_rows = []
ci_rows = []
method_level_rows = []

for method_key, method_prefix in METHODS.items():
    # Service rate comparison too
    service_col = f"{method_prefix}_service_rate"
    rl_service_col = f"{RL_PREFIX}_service_rate"

    if service_col in df.columns and rl_service_col in df.columns:
        diff_sr = df[service_col] - df[rl_service_col]
        abs_diff_sr = diff_sr.abs()

        df_out[f"diff_{method_key}_vs_rl_service_rate"] = diff_sr
        df_out[f"abs_diff_{method_key}_vs_rl_service_rate"] = abs_diff_sr

        wl = win_loss_tie_counts(diff_sr, lower_is_better=False)
        mean_diff, ci_lo, ci_hi = mean_ci_95(diff_sr)
        mean_abs, mae_lo, mae_hi = mean_ci_95(abs_diff_sr)

        summary_rows.append({
            "comparison": f"{method_key}_vs_rl",
            "metric": "service_rate",
            "n": wl["n"],
            "mean_signed_diff": diff_sr.mean(),
            "median_signed_diff": diff_sr.median(),
            "std_signed_diff": diff_sr.std(ddof=1),
            "mean_abs_diff": abs_diff_sr.mean(),
            "median_abs_diff": abs_diff_sr.median(),
            "rmse": np.sqrt((diff_sr ** 2).mean()),
            "rl_win_rate": wl["rl_win_rate"],
            "rl_loss_rate": wl["rl_loss_rate"],
            "tie_rate": wl["tie_rate"],
        })

        ci_rows.append({
            "comparison": f"{method_key}_vs_rl",
            "metric": "service_rate",
            "signed_diff_mean": mean_diff,
            "signed_diff_ci95_low": ci_lo,
            "signed_diff_ci95_high": ci_hi,
            "abs_diff_mean": mean_abs,
            "abs_diff_ci95_low": mae_lo,
            "abs_diff_ci95_high": mae_hi,
        })

        tests = paired_tests(df[service_col], df[rl_service_col])
        test_rows.append({
            "comparison": f"{method_key}_vs_rl",
            "metric": "service_rate",
            **tests
        })

        save_distribution_plots(diff_sr, method_key, "service_rate", "Service Rate Difference")
        save_scatter_plot(df, service_col, rl_service_col, method_key, "service_rate", "Service Rate")

    for metric_slug, metric_label in METRICS:
        comp_col = f"{method_prefix}_{metric_slug}"
        rl_col = f"{RL_PREFIX}_{metric_slug}"

        if comp_col not in df.columns or rl_col not in df.columns:
            continue

        # comparator - RL
        diff = df[comp_col] - df[rl_col]
        abs_diff = diff.abs()

        # Symmetric percent difference is safer than dividing by RL only
        # 2*(A-B)/(A+B)
        spd = 2 * (df[comp_col] - df[rl_col]) / (df[comp_col] + df[rl_col] + EPS)

        # RL-relative percent difference retained too, if you still want it
        pct_rel_to_rl = (df[comp_col] - df[rl_col]) / (df[rl_col] + EPS)

        # Add to episode-level output
        df_out[f"diff_{method_key}_vs_rl_{metric_slug}"] = diff
        df_out[f"abs_diff_{method_key}_vs_rl_{metric_slug}"] = abs_diff
        df_out[f"spd_{method_key}_vs_rl_{metric_slug}"] = spd
        df_out[f"pct_rel_to_rl_{method_key}_vs_rl_{metric_slug}"] = pct_rel_to_rl

        # Win/loss/tie
        wl = win_loss_tie_counts(diff, lower_is_better=True)

        # CIs
        mean_diff, ci_lo, ci_hi = mean_ci_95(diff)
        mean_abs, mae_lo, mae_hi = mean_ci_95(abs_diff)

        # Summary row
        summary_rows.append({
            "comparison": f"{method_key}_vs_rl",
            "metric": metric_slug,
            "n": wl["n"],
            "mean_signed_diff": diff.mean(),
            "median_signed_diff": diff.median(),
            "std_signed_diff": diff.std(ddof=1),
            "mean_abs_diff": abs_diff.mean(),
            "median_abs_diff": abs_diff.median(),
            "rmse": np.sqrt((diff ** 2).mean()),
            "mean_symmetric_percent_diff": spd.mean(),
            "median_symmetric_percent_diff": spd.median(),
            "mean_percent_diff_rel_to_rl": pct_rel_to_rl.mean(),
            "median_percent_diff_rel_to_rl": pct_rel_to_rl.median(),
            "rl_win_rate": wl["rl_win_rate"],
            "rl_loss_rate": wl["rl_loss_rate"],
            "tie_rate": wl["tie_rate"],
        })

        # CI row
        ci_rows.append({
            "comparison": f"{method_key}_vs_rl",
            "metric": metric_slug,
            "signed_diff_mean": mean_diff,
            "signed_diff_ci95_low": ci_lo,
            "signed_diff_ci95_high": ci_hi,
            "abs_diff_mean": mean_abs,
            "abs_diff_ci95_low": mae_lo,
            "abs_diff_ci95_high": mae_hi,
        })

        # Statistical tests
        tests = paired_tests(df[comp_col], df[rl_col])
        test_rows.append({
            "comparison": f"{method_key}_vs_rl",
            "metric": metric_slug,
            **tests
        })

        # Plots
        save_distribution_plots(diff, method_key, metric_slug, metric_label)
        save_scatter_plot(df, comp_col, rl_col, method_key, metric_slug, metric_label)

# =========================================================
# Method-level raw performance table
# =========================================================
for method_prefix in ["offline", "online_nearest", "online_wait_aware", "rl"]:
    for metric_slug, metric_label in METRICS:
        col = f"{method_prefix}_{metric_slug}"
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                m, lo, hi = mean_ci_95(s)
                method_level_rows.append({
                    "method": method_prefix,
                    "metric": metric_slug,
                    "n": len(s),
                    "mean": s.mean(),
                    "median": s.median(),
                    "std": s.std(ddof=1),
                    "min": s.min(),
                    "q25": s.quantile(0.25),
                    "q75": s.quantile(0.75),
                    "max": s.max(),
                    "ci95_low": lo,
                    "ci95_high": hi,
                })

    service_col = f"{method_prefix}_service_rate"
    if service_col in df.columns:
        s = df[service_col].dropna()
        if len(s) > 0:
            m, lo, hi = mean_ci_95(s)
            method_level_rows.append({
                "method": method_prefix,
                "metric": "service_rate",
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(ddof=1),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "q75": s.quantile(0.75),
                "max": s.max(),
                "ci95_low": lo,
                "ci95_high": hi,
            })


# =========================================================
# Save outputs
# =========================================================
summary_df = pd.DataFrame(summary_rows)
tests_df = pd.DataFrame(test_rows)
ci_df = pd.DataFrame(ci_rows)
method_df = pd.DataFrame(method_level_rows)

summary_df.to_csv(os.path.join(OUT_DIR, "benchmark_summary.csv"), index=False)
tests_df.to_csv(os.path.join(OUT_DIR, "benchmark_paired_tests.csv"), index=False)
ci_df.to_csv(os.path.join(OUT_DIR, "benchmark_confidence_intervals.csv"), index=False)
method_df.to_csv(os.path.join(OUT_DIR, "benchmark_method_level_stats.csv"), index=False)
df_out.to_csv(os.path.join(OUT_DIR, "benchmark_episode_level_with_diffs.csv"), index=False)

print(f"Saved analysis outputs to: {OUT_DIR}")
print("Files created:")
print(" - benchmark_summary.csv")
print(" - benchmark_paired_tests.csv")
print(" - benchmark_confidence_intervals.csv")
print(" - benchmark_method_level_stats.csv")
print(" - benchmark_episode_level_with_diffs.csv")
print(" - distribution/scatter plot PNGs")