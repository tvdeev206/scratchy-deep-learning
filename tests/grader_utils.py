import numpy as np
import traceback


def compute_numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient of scalar function f at x using two-sided difference:
    grad[i] = (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        fx_plus = f(x)

        x[idx] = old_val - eps
        fx_minus = f(x)

        x[idx] = old_val
        grad[idx] = (fx_plus - fx_minus) / (2.0 * eps)
        it.iternext()
    return grad


def relative_error(grad_analytic, grad_numeric, eps=1e-8):
    """
    Computes relative error:
    ||grad_analytic - grad_numeric|| / (||grad_analytic|| + ||grad_numeric|| + eps)
    """
    numerator = np.linalg.norm(grad_analytic - grad_numeric)
    denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric) + eps
    return numerator / denominator


class GradingSuite:
    def __init__(self, name: str):
        self.name = name
        self.tests = []

    def add_test(self, name: str, points: float, func):
        self.tests.append({
            "name": name,
            "points": points,
            "func": func
        })

    def run(self):
        total_earned = 0.0
        total_possible = 0.0
        results = []

        for item in self.tests:
            max_pts = item["points"]
            total_possible += max_pts
            try:
                item["func"]()
                earned = max_pts
                status = "PASS"
                detail = "All assertions passed"
            except NotImplementedError:
                earned = 0.0
                status = "TODO"
                detail = "Not implemented yet"
            except AssertionError as e:
                earned = 0.0
                status = "FAIL"
                detail = str(e) if str(e) else "Assertion failed"
            except Exception as e:
                earned = 0.0
                status = "ERROR"
                detail = f"{type(e).__name__}: {str(e)}"

            total_earned += earned
            results.append({
                "name": item["name"],
                "status": status,
                "earned": earned,
                "max": max_pts,
                "detail": detail
            })

        return total_earned, total_possible, results
