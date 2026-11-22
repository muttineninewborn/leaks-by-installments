"""Live results (Table 4, Figure 2): replays every gate policy on the saved tutor replies.

Usage: python study/evaluate_live.py DRAFTS.json PARSES.json
Needs no GPU: the replies and parses were generated once by live_test.py (tutor and parse phases).
"""
import json
import sys

import live_test


def main():
    drafts, parses = live_test.load(sys.argv[1]), live_test.load(sys.argv[2])
    problems = live_test.make_problems(60)
    s = live_test.evaluate(problems, drafts, parses=parses)
    out = {"n_problems": len(problems), "policies": {}}
    for p in ("P0", "P1", "P2", "P3"):
        out["policies"][p] = {"answer_key_leak": s[p]["direct"], "split_leak": s[p]["split"],
                              "variant_leak": s[p]["iso"], "honest_completion": s[p]["honest"],
                              "hint_refused": s[p]["hint_refused"]}
    out["parser"] = s["parser"]
    out["split_sign_test_p"] = s["split_sign_test_p"]
    out["criteria"] = s["criteria"]
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
