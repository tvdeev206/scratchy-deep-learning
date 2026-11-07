import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.grader_utils import GradingSuite


def get_sgd_class():
    try:
        from src.optimizers.sgd import SGD
        return SGD
    except ImportError as e:
        raise NotImplementedError(f"SGD optimizer not found in src.optimizers.sgd: {e}")


def get_adam_class():
    try:
        from src.optimizers.adam import Adam
        return Adam
    except ImportError as e:
        raise NotImplementedError(f"Adam optimizer not found in src.optimizers.adam: {e}")


class DummyLayer:
    def __init__(self):
        self.params = {
            "W": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": np.array([[0.5, 0.5]])
        }
        self.grads = {
            "W": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "b": np.array([[0.05, 0.05]])
        }


def test_sgd_standard_step():
    SGD = get_sgd_class()
    layer = DummyLayer()
    lr = 0.1

    # Support either passing layers or passing params/grads
    try:
        opt = SGD([layer], lr=lr, momentum=0.0)
    except Exception:
        opt = SGD(layer.params, layer.grads, lr=lr, momentum=0.0)

    opt.step()

    # W_new = W - lr * dW = [[1.0 - 0.01, 2.0 - 0.02], ...]
    expected_W = np.array([[0.99, 1.98], [2.97, 3.96]])
    expected_b = np.array([[0.495, 0.495]])

    assert np.allclose(layer.params["W"], expected_W, atol=1e-5), f"SGD W update mismatch:\ngot {layer.params['W']}\nexpected {expected_W}"
    assert np.allclose(layer.params["b"], expected_b, atol=1e-5), f"SGD b update mismatch:\ngot {layer.params['b']}\nexpected {expected_b}"


def test_sgd_momentum_step():
    SGD = get_sgd_class()
    layer = DummyLayer()
    lr = 0.1
    momentum = 0.9

    try:
        opt = SGD([layer], lr=lr, momentum=momentum)
    except Exception:
        opt = SGD(layer.params, layer.grads, lr=lr, momentum=momentum)

    # Step 1: v1 = momentum * 0 + lr * grad = 0.1 * grad
    # W1 = W - v1
    opt.step()
    
    # Step 2: v2 = momentum * v1 + lr * grad
    opt.step()

    # If v1 = 0.1 * 0.1 = 0.01
    # v2 = 0.9 * 0.01 + 0.1 * 0.1 = 0.009 + 0.01 = 0.019
    # W2 = 1.0 - 0.01 - 0.019 = 0.971
    expected_W_00 = 1.0 - (lr * 0.1) - (momentum * (lr * 0.1) + lr * 0.1)
    assert np.isclose(layer.params["W"][0, 0], expected_W_00, atol=1e-4), (
        f"SGD momentum W[0,0] mismatch after 2 steps: got {layer.params['W'][0, 0]}, expected {expected_W_00}"
    )


def test_adam_step():
    Adam = get_adam_class()
    layer = DummyLayer()
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    try:
        opt = Adam([layer], lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    except Exception:
        opt = Adam(layer.params, layer.grads, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    g = layer.grads["W"][0, 0] # 0.1
    w_initial = layer.params["W"][0, 0] # 1.0

    opt.step()

    # Manual Adam calculation for step t=1
    m1 = (1.0 - beta1) * g
    v1 = (1.0 - beta2) * (g ** 2)
    m_hat = m1 / (1.0 - beta1) # which is g
    v_hat = v1 / (1.0 - beta2) # which is g^2
    expected_w = w_initial - lr * m_hat / (np.sqrt(v_hat) + eps)

    assert np.isclose(layer.params["W"][0, 0], expected_w, atol=1e-4), (
        f"Adam W[0,0] update mismatch: got {layer.params['W'][0, 0]}, expected {expected_w}"
    )


def build_suite() -> GradingSuite:
    suite = GradingSuite("Part 4: Optimizers (30 Pts)")
    suite.add_test("SGD Standard Update", 7.5, test_sgd_standard_step)
    suite.add_test("SGD with Momentum Multi-Step", 7.5, test_sgd_momentum_step)
    suite.add_test("Adam Update with Bias Correction", 15.0, test_adam_step)
    return suite


if __name__ == "__main__":
    suite = build_suite()
    earned, total, results = suite.run()
    print(f"\n--- {suite.name}: {earned}/{total} pts ---")
    for r in results:
        print(f"[{r['status']}] {r['name']} ({r['earned']}/{r['max']} pts): {r['detail']}")
