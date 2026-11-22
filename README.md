# Leaks by Installments

Code, data and paper source for **"Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense"** by Venkata Muttineni, Karthik Narayan and Christopher David.

LLM tutors are told not to hand out answers, but current safeguards judge one conversation at a time. A learner can collect an answer in pieces across sessions, or ask for a fully worked "practice" problem with different numbers. The paper measures these attacks on two open tutors and proposes a per-learner disclosure ledger that checks every reply against what the learner has already been shown.

## Repository layout

```
paper/            LaTeX source (ACL format), bibliography and figures
study/
  ledger_sim.py       simulation of tutors, attackers and honest learners
  simulate.py         prints the simulation results as JSON (Tables 2 and 3)
  live_test.py        live experiment, run 2: generation, parsing and offline gate replay
  live_test_run1.py   live experiment, run 1 (tutor model as its own parser)
  evaluate_live.py    replays every gate on the saved run-2 replies (Table 4, Figure 2)
  plot_results.py     draws Figure 2 from the results files
data/
  live/           run 2: every tutor reply (drafts_0.json Qwen2.5-3B, drafts_1.json Phi-3.5-mini),
                  parser output (parse7b.json, Qwen2.5-7B) and GPU logs
  live_run1/      run 1 GPU logs
results/          every number in the paper, as JSON with provenance
osaudit.toml      audit configuration: traces each number in the paper to its source
```

## Reproducing the results

Python 3.10 or later. The simulation and the live evaluation need no GPU and no extra packages.

```
python study/simulate.py 0.25 0.35      # weak honest learner (Tables 2 and 3), about 10 s
python study/simulate.py 0.5 0.6        # strong honest learner
python study/evaluate_live.py data/live/drafts_0.json data/live/parse7b.json   # Tutor A
python study/evaluate_live.py data/live/drafts_1.json data/live/parse7b.json   # Tutor B
pip install matplotlib && python study/plot_results.py                          # Figure 2
```

Each command prints the same values stored in `results/`.

### Regenerating the tutor replies (GPU)

The live replies were generated once on Kaggle (two NVIDIA T4 GPUs, greedy decoding) and saved in `data/live/`. To regenerate them:

```
pip install torch transformers accelerate
python study/live_test.py tutor Qwen/Qwen2.5-3B-Instruct 60 drafts_0.json
python study/live_test.py tutor microsoft/Phi-3.5-mini-instruct 60 drafts_1.json
python study/live_test.py parse Qwen/Qwen2.5-7B-Instruct 60 parse7b.json
python study/live_test.py eval drafts_0.json 60 res_0.json parse7b.json
```

The success criteria for both live runs are written at the top of `study/live_test.py`; they were fixed before each GPU run.
Run 1 did not save the raw replies, so its two results files (`results/live_run1_*.json`) are copied unchanged from the GPU run and cannot be replayed offline.

## Audit

Every number printed in the paper is traced to a results file, a declared constant or a cited source with [osaudit](https://github.com/ASSERT-KTH/osaudit):

```
osaudit                     # static checks; prints nothing when clean
osaudit results --rerun     # reruns every result and compares
```

## Building the paper

```
cd paper && latexmk -pdf paper.tex
```

## License

Code: MIT (see `LICENSE`). Paper: CC BY 4.0. The models used are released by their authors under their own licenses.

## Citation

```
@misc{muttineni2026leaks,
  title  = {Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense},
  author = {Muttineni, Venkata and Narayan, Karthik and David, Christopher},
  year   = {2026}
}
```
