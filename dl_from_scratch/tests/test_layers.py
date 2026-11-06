import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.grader_utils import GradingSuite, compute_numerical_gradient, relative_error


def get_dense_class():
    try:
        from src.layers.dense import Dense
        return Dense
    except ImportError as e:
        raise NotImplementedError(f"Dense layer could not be imported: {e}")


def get_flatten_class():
    try:
        from src.layers.flatten import Flatten
        return Flatten
    except ImportError as e:
        raise NotImplementedError(f"Flatten layer could not be imported: {e}")


def get_dropout_class():
    try:
        from src.layers.dropout import Dropout
        return Dropout
    except ImportError as e:
        raise NotImplementedError(f"Dropout layer could not be imported: {e}")


# --- Test Functions ---

def test_dense_forward():
    Dense = get_dense_class()
    layer = Dense(in_features=3, out_features=2)
    # Fixed known weights and bias
    layer.params["W"] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.params["b"] = np.array([[0.5, -0.5]])

    x = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
    # Expected:
    # row 0: [1*1 + 0*3 + 1*5 + 0.5, 1*2 + 0*4 + 1*6 - 0.5] = [6.5, 7.5]
    # row 1: [0*1 + 1*3 + 2*5 + 0.5, 0*2 + 1*4 + 2*6 - 0.5] = [13.5, 15.5]
    expected = np.array([[6.5, 7.5], [13.5, 15.5]])

    out = layer.forward(x)
    assert out is not None, "Dense.forward returned None"
    assert out.shape == (2, 2), f"Dense output shape wrong: got {out.shape}, expected (2, 2)"
    assert np.allclose(out, expected, atol=1e-5), f"Dense forward output mismatch:\ngot {out}\nexpected {expected}"


def test_dense_backward_inputs():
    Dense = get_dense_class()
    np.random.seed(101)
    layer = Dense(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    grad_out = np.random.randn(5, 3)

    layer.forward(x)
    grad_in = layer.backward(grad_out)
    assert grad_in is not None, "Dense.backward returned None"
    assert grad_in.shape == x.shape, f"grad_in shape {grad_in.shape} does not match x shape {x.shape}"

    # Numerical grad check w.r.t input x
    f = lambda x_val: np.sum(layer.forward(x_val) * grad_out)
    grad_num = compute_numerical_gradient(f, x.copy())
    err = relative_error(grad_in, grad_num)
    assert err < 1e-5, f"Dense backward input gradient error too high: {err:.2e} (expected < 1e-5)"


def test_dense_backward_weights():
    Dense = get_dense_class()
    np.random.seed(102)
    layer = Dense(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    grad_out = np.random.randn(5, 3)

    layer.forward(x)
    layer.backward(grad_out)

    assert "W" in layer.grads, "layer.grads must contain 'W'"
    dW_analytic = layer.grads["W"]

    # Numerical grad check w.r.t weights W
    def f_w(w_val):
        old_w = layer.params["W"]
        layer.params["W"] = w_val
        val = np.sum(layer.forward(x) * grad_out)
        layer.params["W"] = old_w
        return val

    dW_num = compute_numerical_gradient(f_w, layer.params["W"].copy())
    err = relative_error(dW_analytic, dW_num)
    assert err < 1e-5, f"Dense backward dW gradient error too high: {err:.2e} (expected < 1e-5)"


def test_dense_backward_bias():
    Dense = get_dense_class()
    np.random.seed(103)
    layer = Dense(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    grad_out = np.random.randn(5, 3)

    layer.forward(x)
    layer.backward(grad_out)

    assert "b" in layer.grads, "layer.grads must contain 'b'"
    db_analytic = layer.grads["b"]

    def f_b(b_val):
        old_b = layer.params["b"]
        layer.params["b"] = b_val
        val = np.sum(layer.forward(x) * grad_out)
        layer.params["b"] = old_b
        return val

    db_num = compute_numerical_gradient(f_b, layer.params["b"].copy())
    err = relative_error(db_analytic, db_num)
    assert err < 1e-5, f"Dense backward db gradient error too high: {err:.2e} (expected < 1e-5)"


def test_flatten_layer():
    Flatten = get_flatten_class()
    layer = Flatten()
    x = np.random.randn(4, 3, 2, 2)
    out = layer.forward(x)
    assert out is not None, "Flatten.forward returned None"
    assert out.shape == (4, 12), f"Expected shape (4, 12), got {out.shape}"

    grad_out = np.random.randn(4, 12)
    grad_in = layer.backward(grad_out)
    assert grad_in.shape == x.shape, f"Flatten.backward expected shape {x.shape}, got {grad_in.shape}"
    assert np.allclose(grad_in.reshape(4, 12), grad_out), "Flatten backward values mismatch"


def test_dropout_layer():
    Dropout = get_dropout_class()
    layer = Dropout(p=0.5)
    np.random.seed(42)
    x = np.ones((100, 100))

    # Test training mode (some zeros, scaling factor 1/(1-p) for inverted dropout)
    out_train = layer.forward(x, training=True) if "training" in layer.forward.__code__.co_varnames else layer.forward(x)
    zero_ratio = np.mean(out_train == 0.0)
    assert 0.35 < zero_ratio < 0.65, f"Dropout ratio unexpected: {zero_ratio:.2f} (expected ~0.5)"

    # Test eval mode
    if "training" in layer.forward.__code__.co_varnames:
        out_eval = layer.forward(x, training=False)
        assert np.allclose(out_eval, x), "Dropout in eval mode must leave inputs unchanged"


def build_suite() -> GradingSuite:
    suite = GradingSuite("Part 2: Neural Network Layers (30 Pts)")
    suite.add_test("Dense Forward Pass", 5.0, test_dense_forward)
    suite.add_test("Dense Backward Input Gradients (dX)", 5.0, test_dense_backward_inputs)
    suite.add_test("Dense Backward Weight Gradients (dW)", 5.0, test_dense_backward_weights)
    suite.add_test("Dense Backward Bias Gradients (db)", 5.0, test_dense_backward_bias)
    suite.add_test("Flatten Layer Forward & Backward", 5.0, test_flatten_layer)
    suite.add_test("Dropout Layer (Inverted Dropout)", 5.0, test_dropout_layer)
    return suite


if __name__ == "__main__":
    suite = build_suite()
    earned, total, results = suite.run()
    print(f"\n--- {suite.name}: {earned}/{total} pts ---")
    for r in results:
        print(f"[{r['status']}] {r['name']} ({r['earned']}/{r['max']} pts): {r['detail']}")
