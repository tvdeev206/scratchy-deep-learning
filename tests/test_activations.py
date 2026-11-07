import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.grader_utils import GradingSuite, compute_numerical_gradient, relative_error


def get_relu_class():
    try:
        from src.activations.relu import ReLU
        return ReLU
    except ImportError:
        try:
            from src.layers.activations import ReLU
            return ReLU
        except ImportError:
            raise NotImplementedError("ReLU class not found in src.activations.relu or src.layers.activations")


def get_sigmoid_class():
    try:
        from src.activations.sigmoid import Sigmoid
        return Sigmoid
    except ImportError:
        try:
            from src.layers.activations import Sigmoid
            return Sigmoid
        except ImportError:
            raise NotImplementedError("Sigmoid class not found in src.activations.sigmoid or src.layers.activations")


def get_softmax_class():
    try:
        from src.activations.softmax import Softmax
        return Softmax
    except ImportError:
        try:
            from src.layers.activations import Softmax
            return Softmax
        except ImportError:
            raise NotImplementedError("Softmax class not found in src.activations.softmax or src.layers.activations")


# --- Test Functions ---

def test_relu_forward():
    ReLU = get_relu_class()
    layer = ReLU()
    x = np.array([[-2.0, -0.5, 0.0], [1.5, 3.0, -1.0]])
    expected = np.array([[0.0, 0.0, 0.0], [1.5, 3.0, 0.0]])
    out = layer.forward(x)
    assert out is not None, "ReLU forward returned None"
    assert np.allclose(out, expected, atol=1e-6), f"ReLU forward mismatch: got\n{out}\nexpected\n{expected}"


def test_relu_backward():
    ReLU = get_relu_class()
    layer = ReLU()
    x = np.array([[-1.5, 2.5], [0.5, -3.0]])
    grad_out = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[0.0, 2.0], [3.0, 0.0]])
    
    layer.forward(x)
    grad_in = layer.backward(grad_out)
    assert grad_in is not None, "ReLU backward returned None"
    assert np.allclose(grad_in, expected, atol=1e-6), f"ReLU backward mismatch: got\n{grad_in}\nexpected\n{expected}"


def test_sigmoid_forward():
    Sigmoid = get_sigmoid_class()
    layer = Sigmoid()
    x = np.array([0.0, -100.0, 100.0])
    out = layer.forward(x)
    assert out is not None, "Sigmoid forward returned None"
    assert np.isclose(out[0], 0.5, atol=1e-5), f"Sigmoid(0) should be 0.5, got {out[0]}"
    assert np.isclose(out[1], 0.0, atol=1e-5), f"Sigmoid(-100) should be ~0.0, got {out[1]}"
    assert np.isclose(out[2], 1.0, atol=1e-5), f"Sigmoid(100) should be ~1.0, got {out[2]}"


def test_sigmoid_backward():
    Sigmoid = get_sigmoid_class()
    layer = Sigmoid()
    np.random.seed(42)
    x = np.random.randn(3, 4)
    grad_out = np.random.randn(3, 4)
    
    out = layer.forward(x)
    grad_analytic = layer.backward(grad_out)
    assert grad_analytic is not None, "Sigmoid backward returned None"

    # Numerical gradient check
    f = lambda x_val: np.sum(layer.forward(x_val) * grad_out)
    grad_num = compute_numerical_gradient(f, x.copy())
    err = relative_error(grad_analytic, grad_num)
    assert err < 1e-5, f"Sigmoid backward gradient error too high: {err:.2e} (expected < 1e-5)"


def test_softmax_forward():
    Softmax = get_softmax_class()
    layer = Softmax()
    x = np.array([[1000.0, 1000.0], [1.0, 2.0]])  # test numerical stability with large numbers
    out = layer.forward(x)
    assert out is not None, "Softmax forward returned None"
    assert not np.isnan(out).any(), "Softmax resulted in NaN (missing numerical stability shift)"
    assert np.allclose(np.sum(out, axis=-1), np.ones(2), atol=1e-6), "Softmax probabilities must sum to 1"
    assert np.all(out >= 0.0) and np.all(out <= 1.0), "Softmax probabilities must be in [0, 1]"


def test_softmax_backward():
    Softmax = get_softmax_class()
    layer = Softmax()
    np.random.seed(42)
    x = np.random.randn(2, 4)
    grad_out = np.random.randn(2, 4)
    
    layer.forward(x)
    grad_analytic = layer.backward(grad_out)
    assert grad_analytic is not None, "Softmax backward returned None"

    f = lambda x_val: np.sum(layer.forward(x_val) * grad_out)
    grad_num = compute_numerical_gradient(f, x.copy())
    err = relative_error(grad_analytic, grad_num)
    assert err < 1e-5, f"Softmax backward gradient error too high: {err:.2e} (expected < 1e-5)"


def build_suite() -> GradingSuite:
    suite = GradingSuite("Part 1: Activation Functions (15 Pts)")
    suite.add_test("ReLU Forward Pass", 2.0, test_relu_forward)
    suite.add_test("ReLU Backward Pass", 3.0, test_relu_backward)
    suite.add_test("Sigmoid Forward Pass", 2.0, test_sigmoid_forward)
    suite.add_test("Sigmoid Backward Pass (Grad Check)", 3.0, test_sigmoid_backward)
    suite.add_test("Softmax Forward Pass (Stability & Sum=1)", 2.0, test_softmax_forward)
    suite.add_test("Softmax Backward Pass (Grad Check)", 3.0, test_softmax_backward)
    return suite


if __name__ == "__main__":
    suite = build_suite()
    earned, total, results = suite.run()
    print(f"\n--- {suite.name}: {earned}/{total} pts ---")
    for r in results:
        print(f"[{r['status']}] {r['name']} ({r['earned']}/{r['max']} pts): {r['detail']}")
