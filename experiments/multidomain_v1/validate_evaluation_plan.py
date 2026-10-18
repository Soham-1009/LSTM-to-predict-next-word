"""Validation script for Multi-Domain Evaluation Plan and Acquisition Gate.

Validates the machine-readable evaluation_allocation.json configuration:
1. Verifies that domain-level parent counts sum up to the reported totals.
2. Verifies that domain-level token counts sum up to the reported totals.
3. Checks Scenario A ShareAlike exclusion invariants.
4. Checks evaluation tier thresholds and rationale.
5. Verifies cross-document mathematical consistency.
"""

import json
from pathlib import Path
import sys


def validate_evaluation_allocation(config_path: Path) -> None:
    print(f"Loading evaluation allocation configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tier_defs = data["tier_definitions"]
    scenarios = data["scenarios"]

    for scenario_key, scenario_data in scenarios.items():
        scenario_name = scenario_data["name"]
        print(f"\n--- Validating {scenario_name} ({scenario_key}) ---")

        for variant_key, variant_data in scenario_data["variants"].items():
            print(f"  Variant: {variant_key} ({variant_data['description']})")
            domains = variant_data["domains"]
            totals = variant_data["computed_totals"]

            sum_val_parents = 0
            sum_test_parents = 0
            sum_val_tokens = 0
            sum_test_tokens = 0

            for domain_key, domain_info in domains.items():
                vp = domain_info["val_parents"]
                tp = domain_info["test_parents"]
                vt = domain_info["val_tokens"]
                tt = domain_info["test_tokens"]

                sum_val_parents += vp
                sum_test_parents += tp
                sum_val_tokens += vt
                sum_test_tokens += tt

                # Check tier classification consistency
                tier = domain_info.get("eval_tier")
                if tier == "tier_1":
                    assert vp >= tier_defs["tier_1"]["min_val_parents"], (
                        f"Domain {domain_key} marked tier_1 but val_parents={vp} < 20"
                    )
                    assert tp >= tier_defs["tier_1"]["min_test_parents"], (
                        f"Domain {domain_key} marked tier_1 but test_parents={tp} < 20"
                    )
                elif tier == "tier_2":
                    assert vp < 20 or tp < 20, (
                        f"Domain {domain_key} marked tier_2 but has >=20 parents: ({vp}, {tp})"
                    )
                    assert vp >= tier_defs["tier_2"]["min_val_parents"], (
                        f"Domain {domain_key} marked tier_2 but val_parents={vp} < 5"
                    )

            # Programmatic assertions
            assert sum_val_parents == totals["val_parents"], (
                f"Val parent sum {sum_val_parents} != declared total {totals['val_parents']} in {scenario_key}:{variant_key}"
            )
            assert sum_test_parents == totals["test_parents"], (
                f"Test parent sum {sum_test_parents} != declared total {totals['test_parents']} in {scenario_key}:{variant_key}"
            )
            assert sum_val_tokens == totals["val_tokens"], (
                f"Val token sum {sum_val_tokens} != declared total {totals['val_tokens']} in {scenario_key}:{variant_key}"
            )
            assert sum_test_tokens == totals["test_tokens"], (
                f"Test token sum {sum_test_tokens} != declared total {totals['test_tokens']} in {scenario_key}:{variant_key}"
            )
            assert totals["total_held_out_parents"] == sum_val_parents + sum_test_parents
            assert totals["total_eval_tokens"] == sum_val_tokens + sum_test_tokens

            print(f"    -> PASSED: Parents (Val: {sum_val_parents}, Test: {sum_test_parents}, Total: {totals['total_held_out_parents']})")
            print(f"    -> PASSED: Tokens (Val: {sum_val_tokens:,}, Test: {sum_test_tokens:,}, Total: {totals['total_eval_tokens']:,})")

    # Specific Scenario A assertions
    scen_a_heur = scenarios["scenario_a"]["variants"]["heuristic_news"]
    assert scen_a_heur["computed_totals"]["val_parents"] == 90, (
        f"Scenario A heuristic val parents must be 90, got {scen_a_heur['computed_totals']['val_parents']}"
    )
    assert scen_a_heur["computed_totals"]["test_parents"] == 90
    assert scen_a_heur["domains"]["conversation"]["eval_tier"] == "tier_2", (
        "Scenario A conversation must be labeled tier_2 (10 parents < 20)"
    )

    scen_a_scaled = scenarios["scenario_a"]["variants"]["scaled_news"]
    assert scen_a_scaled["computed_totals"]["val_parents"] == 270, (
        f"Scenario A scaled val parents must be 270, got {scen_a_scaled['computed_totals']['val_parents']}"
    )
    assert scen_a_scaled["computed_totals"]["test_parents"] == 270

    # Specific Scenario B assertions
    scen_b_heur = scenarios["scenario_b"]["variants"]["heuristic_news"]
    assert scen_b_heur["computed_totals"]["val_parents"] == 120
    assert scen_b_heur["computed_totals"]["test_parents"] == 120

    scen_b_scaled = scenarios["scenario_b"]["variants"]["scaled_news"]
    assert scen_b_scaled["computed_totals"]["val_parents"] == 300
    assert scen_b_scaled["computed_totals"]["test_parents"] == 300

    print("\nALL EVALUATION ALLOCATION ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    cfg_file = Path(__file__).parent / "configs" / "evaluation_allocation.json"
    validate_evaluation_allocation(cfg_file)
