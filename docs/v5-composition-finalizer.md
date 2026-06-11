# v5 composition finalizer

This branch adds `finalize_v5_composition.py`, a reviewed replacement for the initial v5 finalization script.

## What it changes

- Preserves the original tier as `base_tier` before splitting `B` entries.
- Splits `B` into `B_safe` and `B_reservoir`.
- Computes:
  - `gap_ratio`
  - `cheap_gap_ratio`
  - `expensive_gap_ratio`
  - `effective_gap_ratio`
  - `syllable_delta`
  - `huge_deletion`
  - `usable_for_composition`
- Uses `effective_gap_ratio` for composition acceptance, so licensed gaps such as offglides, schwa, and h are de-weighted rather than punished as ordinary deletions.
- Preserves machine-readable alignment. If `alignment` is absent, it serializes the JSON `align` field.
- Backfills junction fields from IPA when missing:
  - `en_onset`, `en_coda`, `fr_onset`, `fr_coda`
  - `en_onset_class`, `en_coda_class`, `fr_onset_class`, `fr_coda_class`
- Emits a richer `composition-index.json` with indexes by:
  - pivot
  - first/final pivot class
  - EN syllable count
  - FR syllable count
  - syllable pair
  - direction
  - base tier
  - final tier
  - usability

## Run

```bash
python finalize_v5_composition.py
```

or explicitly:

```bash
python finalize_v5_composition.py \
  --input dictionary-v5.json \
  --json-out dictionary-v5.json \
  --tsv dictionary-v5.tsv \
  --index composition-index.json
```

To test TSV/index generation without overwriting the JSON:

```bash
python finalize_v5_composition.py --no-json-update --tsv dictionary-v5.review.tsv --index composition-index.review.json
```

## Acceptance rule

```python
usable_for_composition = (
    tier == "S"
    or (
        tier == "A"
        and syllable_delta <= 1
        and effective_gap_ratio <= 0.30
        and not huge_deletion
    )
    or tier == "B_safe"
)
```

`B_safe` requires:

```python
score >= 0.72
and syllable_delta <= 1
and effective_gap_ratio <= 0.20
and not huge_deletion
```
