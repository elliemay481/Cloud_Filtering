import numpy as np

def calibration_type_1(y_pred_quantiles, y_true, quantiles):
    """
    Finds the fraction of cases that fall inside prediction intervals.
    These prediction intervals are formed by pairs of quantiles.

    For example:
    The 50% interval is the interval between the 0.25 and 0.75 quantile.
    This interval should contain the true value 50% of the time.
    (100% of the true cases should fall between quantiles 0.0 and 1.0).

    Interpretation:
    Curve above the diagonal: interval contains too many true values, model is overly cautious.
    """
    n_quantiles = len(quantiles)
    n_intervals = n_quantiles // 2

    # Pair quantiles: left = qs[i], right = qs[-(i+1)]
    # Example: q0<->q98, q1<->q97, q2<->q96 ...
    left_idxs  = np.arange(0, n_intervals)
    right_idxs = np.arange(n_quantiles - 1, n_quantiles - 1 - n_intervals, -1)

    # Compute interval widths (nominal coverages)
    # e.g., (0.99-0.01)=0.98, (0.98-0.02)=0.96, ...
    intervals = quantiles[right_idxs] - quantiles[left_idxs]

    fractions = np.zeros(n_intervals)

    for i in range(n_intervals):
        lower = y_pred_quantiles[:, left_idxs[i]]
        upper = y_pred_quantiles[:, right_idxs[i]]

        # Count truths inside this interval
        inside = (y_true >= lower) & (y_true <= upper)
        fractions[i] = inside.sum() / len(y_true)

    # Reverse so intervals go from small → large (same as quantnn)
    return intervals[::-1], fractions[::-1]


def calibration_type_2(y_pred_quantiles, y_true, quantiles):
    """
    Finds the fraction of true cases lying below the nth quantile.

    For example:
    50% of the true values should fall below the 0.5 quantile.
    100% of cases should fall below the 1.0 quantile.

    Interpretation
    Curve above the diagonal: too many true values lie below the quantile,
    the quantile is too high, model is overly cautious.

    Essentially, the main difference is that the second method looks at a single quantile and the number of true cases below it.
    The first method looks at the interval formed between two quantiles and the true values within this interval.
    
    """
    n_quantiles = len(quantiles)
    quantile_idxs = np.arange(0, n_quantiles + 1, 1)
    fractions = np.zeros(n_quantiles)
    for i in range(len(quantiles)):
        upper_quantile = y_pred_quantiles[:, quantile_idxs[i]]
        fractions[i] = len(np.where(y_true <= upper_quantile)[0]) / len(y_true)

    return fractions