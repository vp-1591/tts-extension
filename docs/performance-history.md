# Performance Measurement History

Server startup and TTS performance records. All numbers come from `logs/server.log`
(see ADR 0004 for the log-line semantics of each phase). Conditions noted per entry;
`logs/server.log` is not committed, so this file is the durable record.

- **Heavy imports** — `[PHASE] heavy imports took Xms` (torch/transformers/spacy/phonemizer via `from kokoro import KPipeline`).
- **Model load** — `Kokoro model loaded on <device> in Xms` (KPipeline construction; warmup reported separately in `[PHASE] TTS warmup`).
- **Total** — spawn → `model_loaded` (imports + model load + warmup), warm Ollama unless noted.
- **First chunk** — latency to the first TTS audio chunk after a request (CUDA kernel init paid at startup by warmup).

## 2026-08-30 — PRE PR #12 baseline (warm cache, cuda)

Recorded in ADR 0004 (2026-08-30 entry) as the pre-change record; reproduced in the PR #12 description.

| Metric | Value |
|---|---|
| All imports (module level) | 6.11 s |
| Socket bind | ~0 ms |
| Ollama ensure | 77 ms warm / up to 15 s cold |
| Model load | 5656 ms |
| First TTS chunk (CUDA kernel init) | 1.3–3.1 s |
| Total to `model_loaded` | ~12 s (~26.6 s with cold Ollama) |
| Total to usable TTS | ~13.1–14.9 s |

## 2026-08-30 — POST PR #12 (feat/startup-speedups, warm cache, cuda)

From the PR #12 description. Changes: heavy imports deferred into `get_pipeline()`,
`HF_HUB_OFFLINE=1` when cache is warm, ollama ensure parallel to model load, TTS warmup
at startup, faster panel/host polling.

| Metric | Before | After |
|---|---|---|
| Stdlib imports | 6.11 s | 0.01 s |
| Heavy imports (`[PHASE]`) | — | 5875 ms |
| `Ready at` (HTTP) | ~6.1 s | 0.01 s |
| Model load | 5656 ms | 2141 ms |
| TTS warmup | — | 1125 ms |
| `[PHASE] ollama ensure` | 77 ms (blocking) | 47 ms (parallel) |
| Time to `model_loaded` | ~11.8 s | ~8.0 s |
| First TTS chunk | 1.3–3.1 s | ~0.1–0.3 s |
| Cold-Ollama worst case | ~26.6 s | ~8 s + leftover |

## 2026-08-31 — POST PR #12 review fixes (warm HF cache + warm Ollama, cuda)

Re-measurement after the PR #12 review-fix commits
(`0e8965f`, `03754ac`, `f2cc10c`, `513f1b3`), from `logs/server.log`.

| Run | Heavy imports | Model load | TTS warmup | Total → `model_loaded` |
|---|---|---|---|---|
| 10:53 | 6983 ms | 2688 ms | 1530 ms | **11.2 s** |
| 10:58 | 6391 ms | 2172 ms | 1233 ms | **9.8 s** |
| 11:00 | 6297 ms | 2125 ms | 1280 ms | **9.7 s** |
| 13:18 | 6202 ms | 2156 ms | 1218 ms | **9.6 s** |
| outliers (machine under load) | 8.1–11.3 s | 2.8–7.2 s | — | 16.7–21.5 s |

**Conclusion:** warm startup settled at ~9.6–11.2 s total to `model_loaded` — about 2–4 s
faster than the pre-#12 baseline counting the eliminated first-chunk stall, ~1.5–2 s slower
than PR #12's original 8.0 s sample (import-phase variance ~6.2–8+ ms is machine-load noise,
not introduced by the review fixes). Model load 5656→~2.2 s and first TTS chunk
1.3–3.1 s→~0.2 s gains are holding. Remaining cost is dominated by heavy imports (~6.2–8 s).