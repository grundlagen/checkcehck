# meaning_map — hunting the convergence of sound and meaning (EN ↔ FR)

> *The grand goal: a map of meaning where every concept finds, across the
> language divide, the word that both **sounds like** it and **means** it.*

This package develops the homophonic-hunting work in
`homophone-agent-audio/` toward that goal. The homophone agent answers
*"what sounds like this?"*. `meaning_map` overlays a second relation —
*meaning* — on the same word nodes and hunts for **resonances**: pairs where
sound and meaning converge.

```
sound:   en_word  ~hom~   fr_word     (cross-lingual homophony, IPA)
meaning: en_word  ~mean~  fr_word     (translation, or one synonym hop)
resonance = both hold
```

## Run a routine

```sh
cd homophone
python3 -m meaning_map.hunt            # full lexica (~5s, no dependencies)
python3 -m meaning_map.hunt --max 40000   # quick capped pass
```

Outputs (written next to the package):

- **`MAP_OF_MEANING.md`** — the human-readable frontier report for this run.
- **`resonance_reservoir.jsonl`** — append-only memory across routines.

## How it hunts (two directions)

1. **Concept-anchored** (`meaning → sound`). For every known concept (a seed
   `EN→FR` entry plus one synonym hop) the meaning link is given; we measure
   how close the two words *sound*. This paints the whole map.
2. **Sound-anchored** (`sound → meaning`). Over the full 65k/245k IPA lexica we
   find exact cross-lingual homophones (normalised-IPA equality) and ask the
   oracle whether they also share meaning.

### Tiers

| tier | meaning | example |
|------|---------|---------|
| 🥇 **GOLD** | sound **and** meaning converge | `chic`/`chic`, `soup`/`soupe`, `ski`/`ski` |
| 🥈 **SILVER** | strong sound + confirmed meaning | `blue`/`bleu` |
| 🥉 **BRONZE** | good sound + confirmed meaning | `two`/`deux`, `head`/`tête` |
| 🛰️ **FRONTIER** | strong sound, meaning unknown | the hunting ground for the next routine |

The FRONTIER is the engine of progress: a sound-twin whose meaning we cannot
yet confirm is either a false friend or an undiscovered GOLD. The first run's
frontier yielded `chic`, `douche`, `bijou`, `boutique`, `chef`, `kiwi` … which
were confirmed by hand and promoted into the oracle. **That promotion *is* the
whittling loop.**

## Architecture

| module | role |
|--------|------|
| `phon.py` | dependency-free IPA normalise / segment / feature-weighted similarity (distilled from `src/phone_metric.py`) |
| `lexicon_index.py` | load FR/EN `word⇥IPA` lexica; build exact-sound and coarse-bucket indices |
| `semantic_oracle.py` | pluggable meaning judge; default `SeedDictOracle` (offline) |
| `seed_bilingual.py` | curated EN→FR seed + FR synonyms + confirmed cognates |
| `resonance.py` | the two hunts, tier classification, dedupe |
| `journal.py` | append-only JSONL memory (whittling) |
| `hunt.py` | CLI: run a routine, write report + reservoir |

The semantic oracle is the seam. Today it is a small curated dictionary; swap
in a multilingual-embedding or LLM backend (implement
`SemanticOracle.score`) and the same hunt scales to the whole lexicon.

See **`ROUTINE.md`** for the running log and the plan for the next routine.
