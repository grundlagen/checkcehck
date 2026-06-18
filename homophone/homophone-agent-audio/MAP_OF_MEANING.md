# The Map of Meaning

> Hunting for the points where **sound** and **meaning** touch across languages.

This document is the routine bible for the homophonic + semantic matching
effort. It explains the goal, the model, the maths, what is built today, and
what is honestly not built yet. Read it together with `NEXT_ROUTINE.md`, which
hands the baton to the next session.

---

## 1. The goal

Two phrases can relate across languages along two independent axes:

- **Sound** — they are *near-homophones* (English *envy* ≈ French *envie*).
- **Meaning** — they are *translations / synonyms* (*envy* means *envie*).

Most cross-lingual pairs satisfy at most one axis. The treasure is the rare
pair that satisfies **both at once** — a phrase that sounds like the other
language *and* means the same thing. Those points are where the two layers of
the map touch. The grand goal is a system that maps both layers over the whole
EN↔FR lexicon (and beyond), then walks **chains** through them to whittle large
candidate sets down to pairs that share a meaning — *"potentiality for
perfection in all language to one another."*

The cleanest fully-offline instance of the treasure is the **golden match**: a
French word that sounds like its own English gloss.

```
dette   /dɛt/    ⇄  debt    /dˈɛt/      (sound 1.00, meaning = its dictionary gloss)
estime  /ɛstim/  ⇄  esteem  /ɛstˈiːm/
flotte  /flɔt/   ⇄  fleet   /flˈiːt/
bœuf    /bœf/    ⇄  beef    /bˈiːf/
châle   /ʃal/    ⇄  shawl   /ʃˈɔːl/
```

These are *self-verifying*: the meaning edge is guaranteed by the bilingual
dictionary, and the sound edge is measured. They are the seed crystals of the
whole map.

---

## 2. The model — a two-layer graph

Nodes are word forms (English and French). Two edge types overlay them:

| Layer       | Edge                     | Source                              | Weight |
|-------------|--------------------------|-------------------------------------|--------|
| **SOUND**   | EN ↔ FR near-homophone   | IPA lexica + `phone_metric`         | phonetic similarity ∈ [0,1] |
| **MEANING** | FR → EN gloss            | `data/fr_en_gloss.tsv` (FreeDict)   | 1 (present) |
| **MEANING** | EN ↔ EN synonym (bridge) | two FR words sharing a gloss, or two glosses of one FR word | induced |

A **chain** is an alternating walk through these edges. The two that matter:

- **Loop-closing** `EN_w --sound--> FR_f --meaning--> EN_w'` and ask: is
  `EN_w' ≈ EN_w`? If yes, `FR_f` both sounds like and means `EN_w` — treasure.
- **Synonym whittling** `EN_w --meaning⁻¹--> FR --meaning--> EN_w'` builds a
  translation-bridge synonym set, the connective tissue that lets distinct
  surface words collapse onto a shared meaning over successive hops.

---

## 3. The maths

**Phonetic similarity.** IPA strings are first *normalised* (`src/ipa_norm.py`):
stress (`ˈ ˌ`), length (`ː`), tie bars and combining diacritics are stripped to
a bare segmental skeleton — otherwise *see* /sˈiː/ and *si* /si/ score far apart
despite being identical to the ear. Then `phone_metric.feature_distance` runs a
Levenshtein DP whose substitution cost is `1 − phone_sim(p, q)`, where
`phone_sim` rewards shared place, manner and voicing. Similarity is
`1 − distance / max(len)`. Normalisation alone lifted *see/si* from 0.50 → 1.00
and *envy/envie* from 0.52 → 0.65.

**The rendering objective.** For an English query `w`, score each French
candidate `f`:

```
score(f) = w_sound · sound(w, f) + w_meaning · meaning(w, gloss(f))
```

with defaults `w_sound = 0.6`, `w_meaning = 0.4`. `meaning` is `semantic_
relatedness`, which today recognises identity (1.0), translation-bridge
synonymy (0.7) and orthographic overlap (≤0.4). This is the **pluggable seam** —
swap in embeddings or WordNet behind the same signature (see NEXT_ROUTINE).

**Obviousness filter.** Golden matches are split into *cognates* (a
diacritic-stripped `SequenceMatcher` ratio ≥ 0.8 — *soup/soupe*) and *gems*
(everything else — *dette/debt*, *bœuf/beef*). Cognates validate the pipeline;
gems are the interesting output.

---

## 4. What is built (this routine)

| Piece | File |
|-------|------|
| IPA normalisation | `src/ipa_norm.py` |
| Two-layer engine | `src/map_of_meaning.py` |
| CLI + report | `map_of_meaning_cli.py` |
| Semantic layer seed (8.2k FR→EN) | `data/fr_en_gloss.tsv` (FreeDict, CC-BY-SA) |
| Tests | `tests/test_map_of_meaning.py` |
| Generated output | `artifacts/golden_matches.tsv`, `artifacts/map_of_meaning_report.md` |

**Corpus today:** 65k EN forms, 246k FR forms, 8.2k FR glossed, ~1,061 golden
matches at sound ≥ 0.6 of which **302 are non-cognate gems**.

### CLI

```sh
python map_of_meaning_cli.py golden --min-sim 0.85 --hide-cognates   # the gems
python map_of_meaning_cli.py render --word envy                      # sound × meaning
python map_of_meaning_cli.py homophones --word night                 # sound only
python map_of_meaning_cli.py synonyms --word love                    # bridge synonyms
python map_of_meaning_cli.py report                                  # regenerate artifacts
```

Everything is **pure standard library** — no model downloads required to run.

---

## 5. Honest limitations (read before trusting numbers)

- **Semantics is shallow.** The meaning layer is an 8.2k-entry bilingual word
  list keyed by *lemma*. It covers only ~4% of the inflected French targets in
  `pairbank.tsv`, and `semantic_relatedness` is a heuristic, not a model.
  True meaning chains need embeddings, WordNet, or a bigger dictionary.
- **IPA is coarse.** `phone_metric`'s feature classes are hand-rolled and drop
  nasal-vowel quality. Good enough to rank; not a phonological model.
- **Words, not phrases yet.** The golden-match hunt is word-level. The phrase
  pipeline (`candidate_generation_phrases.py`, `pairbank.tsv`) is the obvious
  next surface to map.
- **No back-verification.** We do not yet synthesise + re-transcribe a gem to
  confirm a human hears the match (`audio_helpers.py` stubs exist for this).

The golden rule from the project handover applies: **transparency > magic.** If
a number looks too good, it is probably a cognate; that is why we label them.

---

## 6. Why this is the right spine

The golden matches are ground truth we *generated*, not guessed. They give the
next routine a labelled set to calibrate every richer component against: a
better phonetic metric should keep ranking them high; a real semantic model
should agree they mean what the dictionary says; a phrase engine should be able
to rebuild them from sub-word pieces. The map starts from points we are sure
of and grows outward.
