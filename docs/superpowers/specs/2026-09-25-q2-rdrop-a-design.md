# Q2 A-Derived R-Drop Candidate Design

## Goal

Evaluate exactly one valid-only candidate P: add the R-Drop objective to control A while preserving every A data, architecture, optimizer, checkpoint, and inference setting. The run is accepted only when clean Attachment 2 valid macro-F1 is at least `0.6212527658`; Attachment 2 test is never loaded, evaluated, selected, or reported.

## Evidence And Choice

Control A is the current sequential control: its clean macro-F1 is `0.6165677806`, with class F1 `0.6541554960/0.4890109890/0.7065368567` for Negative/Neutral/Positive. It remains `0.0046849852` below the gate. Candidate O's frozen last-four-layer scalar mix modestly increased Neutral F1 to `0.5083932854`, but reduced clean macro-F1 to `0.6134493299`, increased MAE, and is retired without retuning.

Three independent options were considered:

- Fixed weighted label smoothing at epsilon `0.05` is small and easy to isolate, but it changes the hard-target supervision distribution on the same class-weight axis whose stronger inverse-frequency variant already regressed. It remains a future independent candidate.
- A mask-safe text global-context residual changes the representation/fusion path and overlaps too closely with retired text/fusion experiments.
- R-Drop leaves A's representation, class weights, labels, inference, and checkpoints unchanged. It regularizes the two stochastic training predictions of the same example. This is selected as candidate P.

R-Drop uses symmetric KL agreement between two dropout predictions. Liang et al. report `alpha=1.0` as a useful fixed choice across most of their GLUE sensitivity study; that is motivation, not an E-MOSEI performance guarantee. Source: [R-Drop](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html), [paper objective and alpha study](https://arxiv.org/html/2106.14448).

## Exact Single Treatment

Add a persisted configuration enum:

```text
dropout_consistency_variant = "none" | "rdrop_alpha_1"
```

Fresh configurations must name it. Historical saved manifests that omit it reconstruct as `"none"`. Candidate P differs from A only by:

```toml
dropout_consistency_variant = "rdrop_alpha_1"
```

It retains `seed=20260924`, 30 epochs, batch size 64, learning rate `0.001`, weight decay `0.0001`, `gated`, `identity`, `flat`, `none` temporal position, `attention` temporal pooling, no synthetic training missingness, frozen final-layer BERT, class-weight exponent `1.0`, regression weight `0.5`, consistency weight `0.0`, and every input/checkpoint rule from A.

For each original batch of 64 unique examples, the candidate performs two separate training-mode forwards with the exact same normalized tensors and masks. Its physical forwards have independent dropout masks, but the optimizer still receives one loss and 64 unique examples; it is not a batch-size change or a duplicated-data treatment. The BERT wrapper stays frozen, `eval()`, and under `torch.no_grad()`. The current model has dropout in its Transformer/attention modules and no BatchNorm running statistics, so its only intentional two-view difference is dropout.

Let `J(z, s, y, r)` be the existing complete single-view Q2 objective: weighted flat CE, plus the unchanged regression and polarity-consistency terms. Let `z1` and `z2` be the two flat classification logits from the same batch, and define KL with `batchmean` reduction. Candidate P minimizes:

```text
0.5 * [J(z1, s1, y, r) + J(z2, s2, y, r)]
+ 0.25 * [KL(log_softmax(z1), softmax(z2))
          + KL(log_softmax(z2), softmax(z1))]
```

This is the fixed R-Drop `alpha=1.0` relative weighting, uniformly scaled from the two-view paper expression. No label smoothing, text adapter, text scalar mix, fusion variant, temporal position/pooling, loss coefficient, seed, or prediction rule is combined with it. `none` retains the existing one-forward loss bit-for-bit at its call boundary.

## Persistence And Valid Reconstruction

The variant is a training semantic rather than a model-state or inference-architecture change:

- `run_q2` passes it into the training loop and writes it under `run_manifest.json` `training`.
- `check_q2` validates the fresh config but never trains or creates output.
- `evaluate-q2-valid` strictly validates a present manifest variant, defaults an absent historical field to `none`, reconstructs the existing model with `strict=True`, and does not invoke the training loss.
- No encoder state, model state key, Attachment 3 format, or valid report schema changes.

## Test Contracts

Before one real run, tests must prove:

- Fresh TOMLs require and accept only `none` or `rdrop_alpha_1`; invalid/non-string variants fail with a deterministic message, and all existing direct config fixtures explicitly use `none`.
- The one-view default path invokes the existing `_forward_split` once and produces the exact current `_joint_loss` value.
- The R-Drop path invokes it twice with identical indexes, masks, normalized inputs, labels, and scores; it computes the exact average joint loss plus the two `batchmean` KL terms from deterministic logits; gradients remain finite.
- Flat classification is required for R-Drop. A non-flat/CORN configuration is rejected before training rather than silently applying a different objective.
- A fake archive containing an inaccessible Attachment 2 `test` member can train candidate P, records `rdrop_alpha_1` in the manifest, emits 30 Attachment 3 predictions, and never reads the sentinel.
- Saved-valid accepts/validates the P manifest with strict model loading, while a historical manifest missing the field defaults to `none`; unknown manifest values fail deterministically and saved-valid never reads Attachment 3 or Attachment 2 test.

## One-Run Decision

After full regression, a new ignored A-derived TOML must preflight to `train_count=3395`, `valid_count=728`, and `attachment3_count=30` without output. It trains exactly once in a new `artifacts/q2-valid-rdrop-alpha-1/` directory and undergoes exactly one strict saved-valid reconstruction.

Required evidence is exact saved/recomputed clean metrics, valid support 728, 27 scenario rows, 30 Attachment 3 predictions, a nonempty strict `model.pt`, manifest `dropout_consistency_variant="rdrop_alpha_1"`, and a normalized A/P manifest difference containing only that field. Record clean metrics, per-class F1, scenario mean/worst, the R-Drop objective, and an explicit train/valid-only scope in the ignored comparison JSON, README Val/Test ledgers, and exploration portfolio.

Accept only clean macro-F1 at least `0.6212527658`. If it misses, retire this exact A-derived two-view dropout objective with alpha 1.0 and do not tune alpha, dropout, batch size, physical concatenation, loss scaling, KL direction/reduction, checkpoint rule, seed, or combine it with b32, label smoothing, or another retired candidate. Test remains unassessed in either outcome.
