"""
CPU simulation for the cross-session disclosure ledger (candidate #13).

World model
-----------
* A homework problem is a DAG of solution steps; the last step is the final answer.
  Every step has a skill type. A learner can compute a step when all its parent values are
  known AND they have that skill (own skill, or a method shown to them in a worked example).
* Tutor disclosures: VALUE(step) reveals a step result; WORKED(variant) solves an
  isomorphic variant end-to-end, which teaches every method in the problem but no values.
* Leak = the learner can derive the FINAL answer without having had the skills themselves.

Tutor policies compared
-----------------------
  P0  no gate
  P1  per-turn gate        : never states the final value; anything else allowed
  P2  per-session cumulative gate (state of the art, cf. cumulative verifiers inside one dialogue):
                             refuse a disclosure if, together with everything disclosed IN THIS
                             SESSION, the final answer becomes derivable or coverage > tau
  P3  cross-session ledger (invention): same rule, but the disclosure record is kept per learner
                             and keyed by problem-equivalence class (rephrasings and isomorphic
                             variants map to one ledger entry via a detector with recall r);
                             methods taught by worked variants are recorded too.
Refused requests are answered with a scaffold hint instead.

Attackers get 12 separate sessions; honest learners ask for hints when stuck and for a
bottom-out value only after two failed hints.
"""
import random, itertools, statistics as st, json, sys

HON = (0.5, 0.6)    # honest learner: P(knows a skill type), P(a hint teaches it)
TAU = 0.5           # max fraction of steps the learner may be able to derive from disclosures alone
K_TYPES = 6

def make_problem(rs):
    n = rs.randint(4, 8)
    parents = {0: []}
    for i in range(1, n):
        k = rs.randint(1, min(2, i))
        parents[i] = sorted(rs.sample(range(i), k))
    types = {i: rs.randrange(K_TYPES) for i in range(n)}
    return dict(n=n, parents=parents, types=types)

def closure(prob, values, skills):
    """steps derivable from disclosed values + skills (fixed point)."""
    known = set(values); changed = True
    while changed:
        changed = False
        for i in range(prob["n"]):
            if i not in known and prob["types"][i] in skills and all(p in known for p in prob["parents"][i]):
                known.add(i); changed = True
    return known

class Tutor:
    def __init__(s, policy, recall=1.0, iso_recall=None, fp=0.0, rs=None, tau=None, output_match=False, vigilance=False):
        s.policy, s.r, s.ri, s.fp, s.rs = policy, recall, (recall if iso_recall is None else iso_recall), fp, rs
        s.tau = TAU if tau is None else tau; s.om = output_match
        s.vig = vigilance; s.refusals = {}
        s.ledger = {}            # equivalence-class id -> {"values": set, "methods": set}
    def _entry(s, pid, kind, session):
        if s.policy in ("P3",):
            if kind == "iso": hit = s.rs.random() < s.ri
            elif s.om: hit = not s._miss(pid, 0.98)    # output-side: the value it is about to release is looked up in the learner's ledger index
            else: hit = s.rs.random() < s.r
            key = ("P", pid) if hit or kind == "orig" else ("S", session, pid, kind)
            if s.fp and s.rs.random() < s.fp:      # false merge with another learner problem: over-blocking risk
                key = ("P", pid)
            return s.ledger.setdefault(key, {"values": set(), "methods": set()})
        if s.policy == "P2":
            return s.ledger.setdefault(("S", session, pid), {"values": set(), "methods": set()})
        return {"values": set(), "methods": set()}
    def allowed(s, prob, e, new_vals=(), new_methods=()):
        if s.policy == "P0": return True
        final = prob["n"] - 1
        if s.policy == "P1": return final not in new_vals
        vals = e["values"] | set(new_vals); meth = e["methods"] | set(new_methods)
        cl = closure(prob, vals, meth)          # what disclosures ALONE make derivable (learner skills ignored)
        return final not in cl and len(cl) / prob["n"] <= s.tau
    def ask_value(s, prob, pid, step, session, kind="orig"):
        e = s._entry(pid, kind, session)
        if s.allowed(prob, e, new_vals=[step]):
            e["values"].add(step); return True
        s.refusals[pid] = s.refusals.get(pid, 0) + 1
        return False
    def _miss(s, pid, p_hit):
        p_miss = 1 - p_hit
        if s.vig: p_miss *= 0.5 ** s.refusals.get(pid, 0)   # every refusal on this problem halves the miss rate
        return s.rs.random() < p_miss
    def ask_worked(s, prob, pid, session):
        if s.policy in ("P0", "P1", "P2"): return True       # a variant with other numbers is "another problem" to them
        if s._miss(pid, s.ri): return True                   # ledger missed the isomorphism -> same as baseline
        e = s.ledger.setdefault(("P", pid), {"values": set(), "methods": set()})
        meth = set(prob["types"].values())
        if s.allowed(prob, e, new_methods=meth):
            e["methods"] |= meth; return True
        s.refusals[pid] = s.refusals.get(pid, 0) + 1
        return False

def attack(prob, pid, tutor, strategy, rs, sessions=12, own_skill=1/3):
    """Attacker re-opens chats and harvests whatever each chat will disclose.
    It already knows ~1/3 of the skill types (it can do some steps itself)."""
    own = {t for t in range(K_TYPES) if rs.random() < own_skill}
    vals, methods = set(), set()
    final = prob["n"] - 1
    derivable = lambda: closure(prob, vals, own | methods)
    for sess in range(sessions):
        if final in derivable(): return True
        kind = "orig" if strategy == "direct" else "rephrase"
        sid = 0 if strategy == "direct" else sess          # direct = keeps asking inside one chat
        if strategy == "isomorphic" and not methods:       # first: get a variant fully worked
            if tutor.ask_worked(prob, pid, sess): methods |= set(prob["types"].values())
            continue
        todo = [i for i in range(prob["n"]) if i not in derivable()]
        rs.shuffle(todo)
        for st_ in todo:                                     # take everything this chat will give
            if st_ in derivable(): continue
            if tutor.ask_value(prob, pid, st_, sid, kind=kind): vals.add(st_)
    return final in derivable()

def honest(prob, pid, tutor, rs, p_skill=None, h=None):
    p_skill = HON[0] if p_skill is None else p_skill; h = HON[1] if h is None else h
    skills = {t for t in range(K_TYPES) if rs.random() < p_skill}
    vals, self_done, sess = set(), 0, 0
    for i in range(prob["n"]):
        ready = all(p in vals for p in prob["parents"][i])
        if not ready: return dict(done=False, self=self_done / prob["n"])
        hints = 0
        while prob["types"][i] not in skills and hints < 2:
            hints += 1
            if rs.random() < h: skills.add(prob["types"][i])
        if prob["types"][i] in skills:
            vals.add(i); self_done += 1; continue
        if tutor.ask_value(prob, pid, i, sess):          # bottom-out request
            vals.add(i); continue
        hints = 0                                         # refused -> more scaffolding
        while hints < 3 and prob["types"][i] not in skills:
            hints += 1
            if rs.random() < h: skills.add(prob["types"][i])
        if prob["types"][i] in skills: vals.add(i); self_done += 1
        else: return dict(done=False, self=self_done / prob["n"])
    return dict(done=True, self=self_done / prob["n"])

def run(N=3000, seed=0):
    rs = random.Random(seed); probs = [make_problem(rs) for _ in range(N)]
    def leak(mk, strat): 
        r2 = random.Random(seed + 99)
        return sum(attack(p, k, mk(k), strat, r2) for k, p in enumerate(probs)) / N
    def hon(mk):
        h = [honest(p, k, mk(k), random.Random(seed * 7 + k)) for k, p in enumerate(probs)]
        return sum(x["done"] for x in h) / N, st.mean(x["self"] for x in h)
    pols = {"P0 no gate": lambda k: Tutor("P0"),
            "P1 per-turn gate": lambda k: Tutor("P1"),
            "P2 per-session cumulative gate": lambda k: Tutor("P2"),
            "P3a ledger, input-side matching (r=0.9)": lambda k: Tutor("P3", recall=0.9, iso_recall=0.8, rs=random.Random(seed + k)),
            "P3b ledger, output-side value matching": lambda k: Tutor("P3", iso_recall=0.8, output_match=True, rs=random.Random(seed + k)),
            "P3c + output-side structure match + vigilance": lambda k: Tutor("P3", iso_recall=0.98, output_match=True, vigilance=True, rs=random.Random(seed + k))}
    main = {}
    for name, mk in pols.items():
        c, w = hon(mk)
        main[name] = dict(direct=leak(mk, "direct"), split=leak(mk, "split"), isomorphic=leak(mk, "isomorphic"), honest_complete=c, honest_self_work=w)
    rob = {}
    for ri in (1.0, 0.98, 0.9, 0.8, 0.6):
        mk = lambda k, ri=ri: Tutor("P3", iso_recall=ri, output_match=True, rs=random.Random(seed + k))
        mkv = lambda k, ri=ri: Tutor("P3", iso_recall=ri, output_match=True, vigilance=True, rs=random.Random(seed + k))
        rob[ri] = dict(no_vigilance=leak(mk, "isomorphic"), vigilance=leak(mkv, "isomorphic"))
    rin = {}
    for r in (1.0, 0.95, 0.9, 0.7, 0.5):
        mk = lambda k, r=r: Tutor("P3", recall=r, iso_recall=0.8, rs=random.Random(seed + k))
        rin[r] = leak(mk, "split")
    taus = {}
    for t in (0.25, 0.4, 0.5, 0.6, 0.75):
        mk2 = lambda k, t=t: Tutor("P2", tau=t)
        mk3 = lambda k, t=t: Tutor("P3", iso_recall=0.98, output_match=True, vigilance=True, rs=random.Random(seed + k), tau=t)
        c, w = hon(mk3)
        taus[t] = dict(P2_split=leak(mk2, "split"), P3_split=leak(mk3, "split"), P3_honest_complete=c, P3_self_work=w)
    return main, rob, rin, taus

if __name__ == "__main__":
    if len(sys.argv) > 2: HON = (float(sys.argv[1]), float(sys.argv[2])); print("honest learner", HON)
    main, rob, rin, taus = run()
    print(f"{'policy':42s} {'direct':>7s} {'split':>7s} {'isomorph':>9s} {'honest_done':>12s} {'self_work':>10s}")
    for k, v in main.items():
        print(f"{k:42s} {v['direct']:7.3f} {v['split']:7.3f} {v['isomorphic']:9.3f} {v['honest_complete']:12.3f} {v['honest_self_work']:10.3f}")
    print("\nisomorphic-attack leak vs structure-matcher recall:")
    for k, v in rob.items(): print(f"  recall {k:.2f}: without vigilance {v['no_vigilance']:.3f} | with vigilance {v['vigilance']:.3f}")
    print("P3a split-attack leak vs input-side matcher recall:   ", {k: round(v, 3) for k, v in rin.items()})
    print("\nbudget tau sweep (split attack):")
    for t, v in taus.items(): print(f"  tau {t:.2f}: P2 leak {v['P2_split']:.3f} | P3c leak {v['P3_split']:.3f} | P3c honest done {v['P3_honest_complete']:.3f} self-work {v['P3_self_work']:.3f}")
    json.dump(dict(honest_learner=HON, main=main, iso_recall={str(k): v for k, v in rob.items()}, input_recall={str(k): v for k, v in rin.items()},
                   tau={str(k): v for k, v in taus.items()}), open(f"ledger_sim_results_{HON[0]}_{HON[1]}.json", "w"), indent=1)
