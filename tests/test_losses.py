import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.grader_utils import GradingSuite, compute_numerical_gradient, relative_error


def get_mse_class():
    try:
        from src.losses.mse import MSELoss
        return MSELoss
    except ImportError:
        try:
            from src.losses.mse import MSE
            return MSE
        except ImportError as e:
            raise NotImplementedError(f"MSELoss / MSE not found in src.losses.mse: {e}")


def get_ce_class():
    try:
        from src.losses.cross_entropy import CrossEntropyLoss
        return CrossEntropyLoss
    except ImportError:
        try:
            from src.losses.cross_entropy import CrossEntropy
            return CrossEntropy
        except ImportError as e:
            raise NotImplementedError(f"CrossEntropyLoss not found in src.losses.cross_entropy: {e}")


# --- Test Functions ---

def test_mse_forward():
    MSELoss = get_mse_class()
    criterion = MSELoss()
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[1.0, 1.0], [2.0, 2.0]])
    # diff = [[0, 1], [1, 2]] -> squared = [[0, 1], [1, 4]] -> mean = 6 / 4 = 1.5
    loss = criterion.forward(y_pred, y_true)
    assert loss is not None, "MSELoss.forward returned None"
    assert np.isscalar(loss) or loss.ndim == 0, f"Loss must be a scalar, got shape {getattr(loss, 'shape', None)}"
    assert np.isclose(float(loss), 1.5, atol=1e-5), f"MSE loss mismatch: got {float(loss)}, expected 1.5"


def test_mse_backward():
    MSELoss = get_mse_class()
    criterion = MSELoss()
    y_pred = np.array([[2.0, -1.0], [3.0, 0.5]])
    y_true = np.array([[1.0, 0.0], [1.0, 1.0]])

    criterion.forward(y_pred, y_true)
    # Check if backward accepts args or uses cached forward
    try:
        grad = criterion.backward()
    except TypeError:
        grad = criterion.backward(y_pred, y_true)

    assert grad is not None, "MSELoss.backward returned None"
    assert grad.shape == y_pred.shape, f"grad shape {grad.shape} != y_pred shape {y_pred.shape}"

    # Numerical grad check w.r.t y_pred
    f = lambda pred: float(criterion.forward(pred, y_true))
    grad_num = compute_numerical_gradient(f, y_pred.copy())
    err = relative_error(grad, grad_num)
    assert err < 1e-5, f"MSE backward gradient error too high: {err:.2e} (expected < 1e-5)"


def test_cross_entropy_forward():
    CrossEntropyLoss = get_ce_class()
    criterion = CrossEntropyLoss()
    
    # 2 samples, 3 classes (one-hot targets)
    # Softmax probabilities:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_true = np.array([[1, 0, 0], [0, 1, 0]])

    # Loss = -1/2 * (ln(0.7) + ln(0.8)) = -0.5 * (-0.35667 + -0.22314) = 0.2899
    loss = criterion.forward(probs, y_true)
    assert loss is not None, "CrossEntropy.forward returned None"
    assert not np.isnan(loss), "CrossEntropy loss is NaN"
    expected = -0.5 * (np.log(0.7) + np.log(0.8))
    assert np.isclose(float(loss), expected, atol=1e-4), f"CrossEntropy mismatch: got {float(loss)}, expected {expected:.4f}"

    # Stability test with extreme values near 0
    extreme_probs = np.array([[1e-15, 1.0 - 1e-15]])
    extreme_true = np.array([[1, 0]])
    try:
        stab_loss = criterion.forward(extreme_probs, extreme_true)
        assert not np.isinf(stab_loss) and not np.isnan(stab_loss), "Loss became inf or nan (needs eps clipping)"
    except Exception as e:
        raise AssertionError(f"Extreme probability handling failed: {e}")


def test_cross_entropy_backward():
    CrossEntropyLoss = get_ce_class()
    criterion = CrossEntropyLoss()

    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    criterion.forward(probs, y_true)
    try:
        grad = criterion.backward()
    except TypeError:
        grad = criterion.backward(probs, y_true)

    assert grad is not None, "CrossEntropy.backward returned None"
    assert grad.shape == probs.shape, f"grad shape {grad.shape} != probs shape {probs.shape}"

    # Numerical grad check
    f = lambda p: float(criterion.forward(p, y_true))
    grad_num = compute_numerical_gradient(f, probs.copy())
    err = relative_error(grad, grad_num)
    assert err < 1e-5, f"CrossEntropy backward gradient error too high: {err:.2e} (expected < 1e-5)"


def build_suite() -> GradingSuite:
    suite = GradingSuite("Part 3: Loss Functions (25 Pts)")
    suite.add_test("MSE Loss Forward", 5.0, test_mse_forward)
    suite.add_test("MSE Loss Backward (Grad Check)", 5.0, test_mse_backward)
    suite.add_test("CrossEntropy Forward (Stability & Value)", 7.5, test_cross_entropy_forward)
    suite.add_test("CrossEntropy Backward (Grad Check)", 7.5, test_cross_entropy_backward)
    return suite


if __name__ == "__main__":
    suite = build_suite()
    earned, total, results = suite.run()
    print(f"\n--- {suite.name}: {earned}/{total} pts ---")
    for r in results:
        print(f"[{r['status']}] {r['name']} ({r['earned']}/{r['max']} pts): {r['detail']}")
