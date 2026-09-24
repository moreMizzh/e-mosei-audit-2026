# Q2 Frozen Last-Four-Layer Scalar-Mix Candidate Design

## Goal

Evaluate exactly one valid-only Q2 candidate O that replaces the frozen BERT final-layer text feature with a trainable scalar mix of its final four frozen hidden layers. It keeps control A's train/valid split, `seed=20260924`, 30 epochs, gated fusion, no synthetic training missingness, `temporal_position_variant="none"`, `temporal_pooling_variant="attention"`, losses, checkpoint rule, and all non-text model settings. It accepts only clean valid macro-F1 `>= 0.6212527658` and never accesses Attachment 2 `test`.

## Evidence And Rationale

Control A reaches clean macro-F1 `0.6165677806`, `0.0046849852` below the gate. Its main error is the three-way text-sensitive boundary: Neutral F1 is `0.4890`, while `Neutral -> Positive` and `Positive -> Neutral` account for 134 of 263 classification mistakes. Removing text under controlled missingness has mean macro-F1 change `-0.0337`, far larger than audio (`-0.0016`) or vision (`-0.0031`).

The exact frozen final-layer output adapter b32 is retired and must not be tuned, but it yielded clean macro-F1 `0.6180331930` and modestly improved both missingness summaries. Candidate O instead selects information that the current encoder discards: the last four frozen BERT layers. It does not alter that adapter, unfreeze BERT, add a loss coefficient, or retry a retired fusion/pooling/position method.

Peters et al. define task-specific scalar mixing of frozen contextual language-model layers in [Deep Contextualized Word Representations](https://aclanthology.org/N18-1202/). BERT's feature-based analysis compares final-layer features with a learned weighted sum of the final four layers in [BERT](https://aclanthology.org/N19-1423.pdf), section 5.3/table 7. Those results are not an E-MOSEI guarantee; they motivate one controlled text-representation experiment only.

## Single Treatment

Introduce the persisted `text_encoder_variant` enum:

```text
last_hidden_state | last4_scalar_mix
```

`last_hidden_state` is required for every fresh configuration and is the historical manifest fallback. Candidate O changes only this field to `last4_scalar_mix`.

For each `[B, 3, 50]` token batch, BERT remains in `eval()` and all BERT calls remain under `torch.no_grad()`. The scalar-mix path asks the local BERT model for hidden states, takes layers 9 through 12 (the final four) with shape `[4, B, 50, 768]`, and returns:

```text
weights = softmax(layer_logits)            # layer_logits: [4], initialized to zeros
text = scale * sum_i weights[i] * layer_i  # scale: scalar, initialized to 1
```

Only `layer_logits` (four values) and `scale` (one value) receive gradients. The encoder returns the existing `[B, 50, 768]` tensor, so downstream mask handling, attachment 3 input, fusion, and model interfaces are unchanged. BERT parameters remain frozen and absent from the candidate trainable-state artifact.

## Persistence And Reconstruction

`FrozenBertEncoder` stays an explicit wrapper rather than registering BERT inside the Q2 fusion model. It exposes the two scalar-mix parameters through an explicit trainable-parameter/state interface:

- `last_hidden_state` exposes no trainable encoder parameters or state.
- `last4_scalar_mix` exposes exactly `layer_logits` and `scale`; the runner adds these to the existing AdamW parameter iterable.
- Candidate O writes `text_encoder_state.pt` containing only those two tensors. `model.pt` remains the Q2 fusion model state and is still loaded with `strict=True`.
- The run manifest records `training.text_encoder_variant`. Saved-valid reconstruction defaults a missing historical field to `last_hidden_state`; for the new variant it requires and strictly validates `text_encoder_state.pt` before evaluation.

No BERT weights, no valid examples, and no Attachment 3 labels are written into `text_encoder_state.pt`.

## Invariants And Tests

The implementation must prove all of the following before a real run:

- Fresh TOMLs require and validate `text_encoder_variant`; unsupported/non-string values fail deterministically.
- Existing default final-layer encoding remains byte-equivalent in shape and value, does not request hidden states, exposes no new trainable state, and preserves the successor RNG state.
- Scalar mix with logits zero equals the arithmetic mean of the final four supplied frozen layers at scale one; it returns `[B, 50, 768]`; both mix tensors receive finite, nonzero gradients; every BERT parameter remains `requires_grad=False` and gradient-free.
- Scalar mix state accepts only the exact two expected tensor names and shapes. BERT state is never serialized.
- The normal runner passes the configured encoder variant, optimizes the five additional parameters only for the new variant, records it in the manifest, writes 30 Attachment 3 predictions, and never accesses a fake inaccessible `test` split.
- Saved-valid evaluation reconstructs the new encoder state while retaining strict model-state loading; a historical manifest missing the field reconstructs final-layer behavior; malformed manifests or missing/malformed scalar-mix state are rejected.
- The complete suite, `train-q2 --check`, and a fresh output-directory assertion pass before the one real run.

## One-Run Decision

The one ignored local configuration changes only:

```toml
text_encoder_variant = "last4_scalar_mix"
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix"
```

It retains all A paths and training values. Preflight must report `train_count=3395`, `valid_count=728`, and `attachment3_count=30` without creating output. The candidate then trains exactly once, followed by exactly one strict saved-valid reconstruction. Required evidence is exact saved/recomputed clean metrics, 728 valid support, 27 scenario rows, 30 Attachment 3 predictions, a nonempty `model.pt`, the two-tensor encoder state, and a manifest whose only normalized A difference is `text_encoder_variant`.

If clean valid macro-F1 is below `0.6212527658`, candidate O is retired without changing scalar-mix layer count, initialization, scale parameterization, BERT mode, BERT freezing, optimizer, loss, checkpoint rule, seed, or combining it with b32. Test remains unassessed in either outcome.
