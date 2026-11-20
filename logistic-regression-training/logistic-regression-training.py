import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """

    # 1. Get number of samples and features
    n_samples, n_features = X.shape

    # 2. Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0

    # 3. Training loop
    for _ in range(steps):

        # 4. Calculate linear predictions (z)
        z = X @ w + b

        # 5. Apply sigmoid
        p = _sigmoid(z)

        # 6. Calculate gradients for weights and bias
        error = p - y
        dw = (X.T @ error) / n_samples
        db = np.mean(error)

        # 7. Update weights and bias
        w -= lr * dw
        b -= lr * db

        pass

    return w, b