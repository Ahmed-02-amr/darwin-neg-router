# GPQA-Diamond benchmark handoff

Paused cleanly on 2026-09-02 after Q69. Resume from Q70.

| Field | Value |
|---|---|
| Dataset | `fingertap/GPQA-Diamond` |
| Dataset revision | `68be7564497676e07a77a042fdb587deb88c51c3` |
| Protocol | `darwin-gpqa-reconstruction-v4` |
| Backend/model | general router / `darwin-neg-agent20` |
| Routing profile | `exact` |
| Completed | 69 of 198 |
| Correct so far | 59 |
| Interim accuracy | 85.51% |
| Last retained item | Q69 (zero-based dataset index 68) |
| Next item | Q70 (zero-based dataset index 69) |
| Solver/review allowance | 6,144 / 3,072 tokens |
| Active local output | `benchmarks/results/gpqa-diamond-v4-general-router20-adaptive-voting-full.jsonl` |

This is a partial, exploratory run, not a final benchmark estimate. The active
JSONL is intentionally untracked because it contains dataset text and model
reasoning.

The worker finished Q70 during shutdown. That record was removed from the
active output and preserved locally under `benchmarks/results/checkpoints/`, so
the resumed run will evaluate Q70 again as requested.

Resume with:

```powershell
& .\scripts\run-gpqa-general-router20.ps1
```

The launcher uses `--resume` and will start at the first missing index. Before
resuming, the active file contained exactly 69 unique records, ended at index
68, and no benchmark process remained running.
