# Next Routine — picking up the Map of Meaning

> Hand-off for the next autonomous session. Read `MAP_OF_MEANING.md` first; it
> is the why and the model. This file is the *what next*.

You are continuing a hunt, not starting one. The previous routine built a
two-layer (sound + meaning) cross-lingual graph and proved the headline result:
**302 non-cognate "golden matches"** — French words that sound like *and* mean
their English gloss (`dette/debt`, `estime/esteem`, `flotte/fleet`, `bœuf/beef`).
Those are ground truth. Everything below grows the map outward from them.

Work on branch `claude/vibrant-ride-bc4a71`. Commit + push when green. Keep it
runnable on pure stdlib; make heavy resources optional.

---

## The objective, restated

Map both layers over the whole lexicon, then walk chains to whittle candidate
sets down to pairs that **sound alike and share a meaning**. The golden matches
are the seed crystals; grow coverage, depth, and the phrase dimension.

---

## Tasks, in priority order

### 1. Deepen the semantic layer (biggest lever)
The meaning layer covers only ~4% of pairbank's inflected French targets. Fix
the coverage, then the depth.
- **Inflection → lemma.** `lexique.tsv` has the surface forms; fold inflected
  French targets back to their lemma so `envies`, `envient` reach `envie`'s
  gloss without the current crude suffix-stripping in `meaning_of`.
- **Bigger dictionary.** FreeDict is 8.2k headwords. Pull a Wiktionary-derived
  FR→EN set (kaikki.org) and merge into `data/fr_en_gloss.tsv`, keeping
  provenance. Target ≥40k headwords.
- **Real relatedness.** Replace the heuristic in
  `MapOfMeaning.semantic_relatedness` with a genuine signal behind the *same
  signature* `(a, b) -> (score, via)`: WordNet path similarity (offline) or a
  cached sentence-embedding cosine (the existing `src/embedding.py` already
  wraps `all-MiniLM-L6-v2`; make it the provider when installed). Gate behind a
  try/except so stdlib still works.

### 2. Lift the hunt from words to phrases
The treasure at phrase scale is far richer (this is what `pairbank.tsv`'s 39k
rows are *for*).
- Reuse `candidate_generation_phrases.suggest_phrase_swaps` to enumerate French
  phrase renderings of an English input, then run each through
  `meaningful_renderings`-style scoring (sound × meaning) using a *sentence*
  semantic model, not word glosses.
- Add `phrase_golden` to the engine + CLI: English phrase → French phrase that
  sounds like it and whose back-translation loops to the source meaning.

### 3. Multi-hop bridging
Implement the chain walk explicitly (the Audio-Asset-Manager repo already has a
"multi-hop bridging" notion — borrow it):
- `EN_w --sound--> FR_a --meaning--> EN_x --sound--> FR_b ...` bounded-depth
  search with a combined sound×meaning path cost, returning the best chain.
- Use it to *whittle*: collapse French synonyms that reach the same English
  meaning over ≤2 hops into meaning-clusters, then keep only the best-sounding
  member.

### 4. Back-verification (close the honesty gap)
A gem should be *heard*, not just scored. Wire `src/audio_helpers.py` (TTS →
ASR) so a candidate is synthesised, re-transcribed, and given a "heard-as"
bonus only if a listener would actually confuse them. The CORT and audio bonus
scaffolding already exists in `main_audio.py`.

### 5. Calibrate against ground truth
The 302 gems are a labelled set. Any change to the phonetic metric or semantic
model must be checked against them:
- Add `tests/test_regression_gems.py` asserting the known gems still rank
  ≥ 0.85 sound and `via == "direct"` meaning. Treat a drop as a regression.
- Build a tiny precision metric: sample N gems, hand-label "true homophone?"
  once, store it, and report precision each routine.

---

## Quick start for the next session

```sh
cd checkcehck/homophone/homophone-agent-audio
python3 -m pytest tests/test_map_of_meaning.py -q     # 10 tests, should pass
python3 map_of_meaning_cli.py golden --hide-cognates  # see the current gems
python3 map_of_meaning_cli.py report                  # refresh artifacts/
```

## Guardrails
- **Transparency > magic.** Keep labelling cognates vs gems; never let a
  cognate masquerade as a discovery.
- Keep the stdlib path working; optional deps stay optional.
- Commit data provenance whenever you pull an external resource.
- The golden matches are the spine — grow from what we are sure of.

Go off. Follow the footsteps, then make new ones.
