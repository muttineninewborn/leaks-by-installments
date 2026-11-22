"""Simulation results (Tables 2 and 3).

Usage: python study/simulate.py P_SKILL P_TEACH
Prints one JSON object with leak rates and honest-learner completion, as fractions and percents.
"""
import json
import sys

import ledger_sim


def pct(x):
    return round(100 * x, 1)


def main():
    ledger_sim.HON = (float(sys.argv[1]), float(sys.argv[2]))
    policies, iso_recall, input_recall, budget = ledger_sim.run()
    out = {"honest_learner": {"p_skill": ledger_sim.HON[0], "p_teach": ledger_sim.HON[1]},
           "policies": {}, "budget": {}}
    for name, v in policies.items():
        out["policies"][name] = {
            "split_leak": v["split"], "split_leak_pct": pct(v["split"]),
            "variant_leak": v["isomorphic"], "variant_leak_pct": pct(v["isomorphic"]),
            "direct_leak": v["direct"], "honest_completion": v["honest_complete"]}
    for tau, v in budget.items():
        out["budget"][str(tau)] = {
            "per_session_split_pct": pct(v["P2_split"]), "ledger_split_pct": pct(v["P3_split"]),
            "ledger_honest_pct": pct(v["P3_honest_complete"])}
    out["structure_recall_sweep"] = {str(k): v for k, v in iso_recall.items()}
    out["input_recall_sweep"] = {str(k): v for k, v in input_recall.items()}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
