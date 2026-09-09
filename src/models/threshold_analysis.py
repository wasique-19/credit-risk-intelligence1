from sklearn.metrics import f1_score, precision_score, recall_score


def find_best_threshold(
    probabilities,
    y_true,
    thresholds=None,
):
    """Find the threshold with the highest F1 score."""

    if thresholds is None:
        thresholds = [i / 100 for i in range(5, 51, 5)]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return max(results, key=lambda x: x["f1"])