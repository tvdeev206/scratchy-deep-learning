import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_activations import build_suite as build_activations_suite
from tests.test_layers import build_suite as build_layers_suite
from tests.test_losses import build_suite as build_losses_suite
from tests.test_optimizers import build_suite as build_optimizers_suite


def run_all_grading():
    suites = [
        build_activations_suite(),
        build_layers_suite(),
        build_losses_suite(),
        build_optimizers_suite(),
    ]

    grand_total_earned = 0.0
    grand_total_possible = 0.0

    print("=" * 75)
    print("       DEEP LEARNING FROM SCRATCH - AUTOGRADER REPORT")
    print("=" * 75)

    status_symbols = {
        "PASS": "[PASS]",
        "FAIL": "[FAIL]",
        "TODO": "[TODO]",
        "ERROR": "[ERR ]"
    }

    for suite in suites:
        earned, possible, results = suite.run()
        grand_total_earned += earned
        grand_total_possible += possible

        print(f"\n>> {suite.name}  [{earned:.1f}/{possible:.1f} pts]")
        print("-" * 75)
        for r in results:
            sym = status_symbols.get(r['status'], r['status'])
            pts_str = f"{r['earned']:>4.1f}/{r['max']:<4.1f} pts"
            print(f"  {sym}  | {r['name']:<42} | {pts_str} | {r['detail']}")

    print("\n" + "=" * 75)
    percentage = (grand_total_earned / grand_total_possible * 100.0) if grand_total_possible > 0 else 0.0
    print(f"TOTAL SCORE: {grand_total_earned:.1f} / {grand_total_possible:.1f} ({percentage:.1f}%)")
    print("=" * 75)

    return grand_total_earned, grand_total_possible


if __name__ == "__main__":
    run_all_grading()
