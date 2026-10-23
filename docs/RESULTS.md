# Results and experiment lessons

This is a retrospective of Shasank's completed Parameter Golf experiments. It summarizes the repository's experiment ledger and checks the featured results against the final lines of each archived training log and its `submission.json`.

## Measurement conventions

- **Quality:** FineWeb validation bits per byte (bpb), lower is better. The featured numbers come from `final_int8_zlib_roundtrip_exact`, after exporting and reloading the compressed model. The log label is historical: these runs used Zstandard compression despite the `int8_zlib` name.
- **Artifact:** counted code bytes plus compressed model bytes. The limit is **16,000,000 bytes**, not 16 MiB. Values below are the historical log/metadata totals; packaging or source changes can alter a future export's size.
- **Timing:** `wallclock_seconds` in the metadata records accumulated training time. It does not include final export and evaluation, and for a resumed run it includes training before the resume.
- **Status:** the three folders are preserved non-record submission packages. Their presence is evidence of completed local/cloud runs, not proof of upstream acceptance or independent verification.

## Completed runs

| Run | Final bpb | Code bytes | Compressed weights | Total bytes | Training time | Evaluation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [Recurrence + step embeddings](../records/track_non_record_16mb/2026-04-04_Full80_Recurrence_StepEmbed_Seq1536_20k/) | 1.17631839 | 99,221 | 15,402,045 | 15,501,266 | 39,715.675 s / 11.03 h | Stride 64; 13,906.839 s |
| [Ternary + 5-bit tail](../records/track_non_record_16mb/2026-04-03_15L_HybridTail_Seq1536_20k_1x5090/) | 1.22833292 | 95,596 | 15,574,237 | 15,669,833 | 11,684.835 s / 3.25 h | Standard; 81.148 s |
| [Ternary MLP weights](../records/track_non_record_16mb/2026-03-27_15L_Ternary_MLPOnly_20k/) | 1.30378846 | 80,237 | 14,957,135 | 15,037,372 | 23,469.558 s / 6.52 h | Standard; 564.463 s |

All three stopped at 20,000 steps. Recurrence and pure ternary used one RTX 5060; hybrid used one RTX 5090. They are not matched-budget or matched-evaluation ablations. In particular, the recurrence result benefits from a different final evaluation protocol, so the table cannot isolate the effect of recurrence alone.

![Final quality and artifact size, with separate markers for the two evaluation modes](assets/results.png)

The chart is generated directly from the three `submission.json` files. Its [source script](assets/plot_results.py) is included.

## Why recurrence was the main result

The winning configuration uses `NUM_LAYERS=10` as **effective depth**. The archived implementation allocates only **8 unique transformer blocks**, and executes them in this order:

```text
0 → 1 → 2 → 3 → 4 → 3 → 4 → 5 → 6 → 7
```

Blocks 3 and 4 share their weights across two passes. A learned embedding distinguishes the passes. The logged model has 25,885,249 parameters, model width 576, eight attention heads, four KV heads, a 3× MLP, and training context length 1536.

The final recipe combined recurrence and step embeddings with a continuation from 10k to 20k steps, 4-bit bigram export, 3.3% one-shot magnitude pruning, and stride-64 evaluation. The pre-export quantized proxy was 1.1735 bpb and the final roundtrip was 1.17631839 bpb.

The earlier mixed-lowbit control finished at **1.21832265 bpb** and **15,777,592 bytes**, also using stride-64 final evaluation. The recurrence recipe improved the observed score by **0.04200426 bpb (3.45%)**. This compares complete recipes, not a recurrence-only ablation: architecture, training schedule, and export settings changed together. A [finalization excerpt from the control log](evidence/mixed-lowbit-baseline.txt) preserves its measurement.

## What the ternary experiments showed

The pure branch applies ternary quantization to **MLP weights**, while attention and other tensors retain their own precision. It is not an entirely ternary transformer. At 15 layers and 20k steps it reached 1.30378846 bpb with a 15.04 MB artifact.

The hybrid branch excludes the final MLP from the ternary path and exports that block at 5 bits. It also uses context length 1536 and a larger training batch. Its 20k RTX 5090 run reached 1.22833292 bpb. Because multiple settings and hardware changed, the full improvement cannot be attributed solely to tail precision.

The quantized-proxy and final metrics were close:

| Branch | Pre-export quantized proxy | Final roundtrip | Difference |
| --- | ---: | ---: | ---: |
| Pure ternary MLP | 1.30343904 | 1.30378846 | +0.00034942 |
| Hybrid tail | 1.22747495 | 1.22833292 | +0.00085797 |

These checks helped distinguish useful compression from misleading full-precision validation scores.

## Negative results and engineering lessons

The following observations come from the local `memory.md` experiment ledger. They provide context; only the three featured packages and the baseline excerpt are included as primary run evidence in this portfolio snapshot.

| Experiment | Observation | Lesson for this project |
| --- | --- | --- |
| Pure ternary MLP, 20k → 25k | Final bpb worsened from 1.30378846 to 1.30946402. | A longer continuation did not automatically improve the exported model. |
| 15-layer ternary, MLP width 3.6×, 5k | 1.35064656 bpb, but 16,161,996 bytes. | Better validation quality can still produce an invalid artifact. |
| Same width-control checkpoint, 4-bit bigram export | Re-export fit at 15,947,538 bytes, scoring 1.350954. | An export-only adjustment can rescue a checkpoint without retraining. |
| Stable injection with adaptive computation time (ACT), 5k | 1.29405806 bpb versus 1.28097979 for plain stable injection. | The tested ACT formulation did not earn further training. This is not a general claim about ACT. |
| Recurrence winner, eval stride 64 → 32 | 1.17631839 → 1.17631444 bpb; evaluation time roughly doubled. | Denser evaluation brought negligible benefit for this checkpoint. |

Checkpoint continuation exposed another practical issue: a finished warmdown can save a zero learning-rate scale. The successful recurrence extension explicitly used `RESUME_EXTENSION_LR_SCALE=0.2`. Recording scheduler and data-loader state makes resumed experiments easier to interpret.

## Scope and reproducibility

These are individual completed runs, without repeated seeds or confidence intervals. No claim is made about competition rank, record-track eligibility, or state-of-the-art performance. FineWeb compression is the measured task; downstream reasoning or instruction-following capability was not evaluated.

The main branch includes the three source/log/metadata packages originally prepared on separate `submit/non-record-*` branches. Those branches remain separate for their original submission workflow. The root training scripts remain competition starting points; use the record snapshots for these results.

Training checkpoints, dataset shards, and compressed weights are not bundled. The [reproduction guide](REPRODUCING.md) describes how to rerun the recurrence recipe, including its two-stage schedule. A fresh rerun is not a restoration of the original checkpoint, and exact metric reproduction has not been revalidated as part of this documentation update.
