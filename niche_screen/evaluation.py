"""Site-cluster bootstrap for assessed site-niche binary decisions."""
import numpy as np


def f1(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    pairs = [(int(y), int(p)) for y, p in zip(labels, predictions) if y is not None]
    if any(y not in (0, 1) or p not in (0, 1) for y, p in pairs):
        raise ValueError("assessed labels and predictions must be binary")
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def site_bootstrap(rows, replicates=1000, seed=7):
    """Rows have site, label, prediction; None labels are unassessed.

    Resample entire sites with replacement, preserving all niche rows per draw.
    Returns micro F1 and an unweighted percentile interval; for stratified or
    unequal-probability samples use a design-aware weighted estimator instead.
    """
    if replicates < 1 or not rows:
        raise ValueError("provide rows and at least one replicate")
    groups = {}
    for row in rows:
        groups.setdefault(row["site"], []).append(row)
    if not any(row["label"] is not None for row in rows):
        raise ValueError("no assessed labels")
    sites = list(groups)
    rng = np.random.default_rng(seed)
    def score(sample):
        return f1([r["label"] for s in sample for r in groups[s]],
                  [r["prediction"] for s in sample for r in groups[s]])
    estimates = [score([sites[i] for i in rng.integers(0, len(sites), len(sites))]) for _ in range(replicates)]
    return {"f1": score(sites), "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
            "sites": len(sites), "replicates": replicates}
