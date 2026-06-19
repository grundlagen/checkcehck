# ROUTINE — The Homophonic–Semantic Hunt for a Map of Meaning

> A self-passed baton. Read this fully before you touch anything. It is the
> previous routine *and* the next one — written so the next Claude can stand
> exactly where this one stood, understand the whole machine, and take one more
> honest step toward the grand goal.
>
> **The grand goal**: the potentiality for perfection in mapping all language
> to one another — where *sound* and *meaning* are two roads to the same place,
> and we can walk either one and arrive at the truth.

---

## 0. The dream, stated plainly

A homophonic translation is a phrase in language B that **sounds like** a phrase
in language A. Usually it's nonsense (`"un cou, deux cou"` ← *uno, dos*). The
holy grail is a homophonic translation that **also means something true** — a
pun that doesn't lie. `FR 'net'` sounds like `EN 'net'` *and* means *clean /
clear / net*. That is a **resonant bridge**: the sound road and the meaning road
land on the same English concept.

Hunt enough of these and you get a **map of meaning**: for every concept, the
cross-lingual sound-alikes that carry it. The thesis of this project:

> If you build the homophone bank (EN↔FR by sound) and the meaning bank
> (FR↔EN by gloss), lay synonyms on top, and **chain** them, you can connect
> the two roads and *whittle down the same meanings over time* until each
> meaning has its cleanest sound-alike attached. Do that across enough language
> pairs and the map approaches completeness.

---

## 1. What exists (the machinery you inherit)

This repo (`checkcehck/homophone/homophone-agent-audio`) is a dual-track
translation skeleton. Track A = literal translation, Track B = homophonic
paraphrase. The pieces, and what they actually do:

### The data
- `data/pairbank.tsv` — **the homophone bank**. 39,133 EN→FR edges
  (`30,462 homophone` + `8,671 phrase`). Columns: `src tgt src_lang tgt_lang
  tag source_file`. The `src` (English) forms are often *sound-spellings*
  (`tess`, `notte`, `ba`) rather than dictionary words — keep this in mind, it
  bites the phonetic scorer (§4).
- `data/lexique.tsv` (245k) — FR surface → IPA. `data/lexique_en.tsv` (65k) —
  EN surface → IPA. These power the IPA fallback when `phonemizer` is absent.
- `data/meaningbank.tsv` — **NEW, the meaning bank**. 119 seeded FR→EN glosses.
  Small but real. This is a *bootstrap*, meant to grow to thousands.
- `data/synonyms_en.tsv` — **NEW**. 208 seeded EN↔EN synonym links (symmetric).

### The code (`src/`)
- `phone_metric.py` — pure-Python feature-weighted IPA edit distance.
  `similarity(ipa1, ipa2) -> [0,1]`. Always available. The honest workhorse.
- `phone_distance.py` — alternative panphon-aware metric (panphon absent → falls
  back to Levenshtein).
- `orchestrator.py` — `g2p(text, lang)` (phonemizer → lexique dict → unidecode),
  plus translation/semantic/fluency **stubs**. `translate()` currently echoes.
- `embedding.py` — `semantic_similarity(a,b) -> [0,1]` (sentence-transformers →
  bag-of-words cosine fallback). No model installed here, so it's BoW today.
- `phrasebank.py` — typed loader for the homophone pairbank.
- `candidate_generation*.py` — per-token (BK-tree) and phrase-level homophone
  substitution.
- `cognate.py` — penalize trivial orthographic cognates.
- `scoring.py` / `judge.py` / `cort.py` — combine phonetic+semantic+fluency
  (+prosody+CORT complexity) into one objective; `judge` writes rationale.
- `co_optimization.py` — sketch of the A⇄B co-optimization loop.

### Sibling repos (the wider world)
- **Lingua-Sound-Wave** — the productized version: 6 scoring methods
  (`phoneme-chain` strongest), a homophone *reservoir* with tiered mining, and
  *Flit Lab* (sound-alike paraphraser with semantic verification). Read its
  `handover.md`. Its philosophy is the law here: **be honest about what works.**
- **Proto-Lingua-Weaver** — proto-forms, sound laws, etymology trees. The
  future source of *cross-family* edges (not just EN/FR).
- **Audio-Asset-Manager** — phonetic-matcher with hash-locked judge + IPA scorer.

### Environment reality
Pure stdlib only. **Missing**: `sentence_transformers, nltk, panphon, epitran,
phonemizer, rapidfuzz, unidecode, numpy`. Everything must degrade gracefully.
Don't write code that hard-requires a model.

---

## 2. What THIS routine added (your footsteps to follow)

The second pair bank and the chaining engine that connects the two roads:

- **`src/meaning_bank.py`** — `MeaningBank` (FR→EN glosses) and `SynonymBank`
  (EN↔EN, with an optional WordNet hook that's purely additive). Loaders take a
  filename and need no code change to scale to a 100k-row file.
- **`src/semantic_chain.py`** — the engine:
  - `sound_resonance(en, fr)` — IPA via `g2p` + `phone_metric.similarity`,
    falling back to a character-bigram Dice coefficient.
  - `meaning_resonance(en_src, glosses, syns)` — returns `(score, kind, meeting)`
    where `kind ∈ {resonant, latent, unglossed}`. Direct lemma identity = 1.0;
    synonym-mediated meeting ≥ 0.8; otherwise latent embedding/BoW similarity.
  - `Bridge` — one walked chain `EN_src ≈sound≈ FR_tgt =means=> {glosses}`, with
    a transparent `path` and a combined `resonance`.
  - `whittle(bridges, syns, rounds, keep_frac)` — iteratively prune to the top
    fraction, then **cluster by canonical concept** (smallest synonym-neighbour
    name) keeping the best bridge per concept. Returns survivors + a per-round
    convergence history. This *is* "whittle down the same meanings over time".
  - `meaning_map(bridges, syns)` — concept → ranked sound-alikes. The map.
- **`hunt_meaning_map.py`** — CLI that runs the whole pipeline on the real
  pairbank and writes `map_of_meaning.json` + a readable report.
- **`test_semantic_chain.py`** — dependency-free self-test (all green).

### What it produces today (honest numbers)
```
homophone edges : 39,133
meaning bank    : 119 French entries
bridges built   : 2,962   (edges whose FR target has a known gloss)
resonant bridges: 1       → EN 'net' ≈ FR 'net' → [clean, clear, net, sharp]
whittle: 2962 → 51 → 36 concept-clusters (converged)
```
**One** fully-resonant bridge. That's not a disappointment — it's the honest
floor, set by a 119-word meaning bank. The architecture is proven; the bank is
the bottleneck. That is exactly the lever the next routine should pull.

Run it yourself:
```sh
python hunt_meaning_map.py --top 30
python hunt_meaning_map.py --w-sound 0.35 --w-meaning 0.65 --rounds 5
python test_semantic_chain.py
```

---

## 3. The chain, drawn

```
        homophone bank                    meaning bank
   EN_src ───≈sound≈──▶ FR_tgt ───=means=──▶ {EN_gloss₁, EN_gloss₂, …}
     │                                                   │
     │  synonym bank (EN↔EN)            synonym bank (EN↔EN)
     ▼                                                   ▼
  {EN_src ∪ syns}  ◀──── do these neighbourhoods meet? ──── {glosses ∪ syns}
                              │
                     yes → RESONANT BRIDGE
                     near → LATENT (surfaces for next round)
```

Two roads from English to French (sound) and French back to English (meaning).
When the return trip lands where you started, the loop is closed and true.

---

## 4. Known sharp edges (don't relearn these the hard way)

1. **Sound-spelling sources.** The pairbank's English `src` are often non-words
   (`tess`, `notte`). They're absent from `lexique_en.tsv`, so `g2p` falls back
   to `unidecode` and the phonetic score collapses. Options: (a) trust the
   pairbank's `homophone` tag as a *sound prior* (these were curated as
   homophones — consider an option to set `phon=1.0` for tagged edges); (b)
   phonemize the sound-spelling directly with espeak when available; (c) carry
   IPA columns in the pairbank (the loader already supports `src_ipa/tgt_ipa`).
2. **Tiny meaning bank ⇒ few resonances.** Expected. Grow the bank (§5.1).
3. **BoW semantics is blunt.** `latent` scores are weak without a real embedder.
   Fine as a ranking nudge; don't trust absolute values.
4. **Direction is one-way.** Pairbank is EN→FR only. The user's vision is
   explicitly *bidirectional* (FR→EN homophones too). See §5.3.
5. **No proper nouns filter.** `nantes`, `jean` leak in. Add a stoplist.

---

## 5. The next routine — concrete moves, in priority order

Pick up here. Each item is sized to be a real, shippable step, not a rewrite.

### 5.1 Grow the meaning bank (highest leverage)
The whole map's richness is gated by `meaningbank.tsv`. Ways to grow it without
network deps, then with:
- Mine the **most frequent FR targets** in the pairbank (`tonte, tinte, teinte,
  tante, tente, note, …` — already the top of the distribution) and gloss them
  first; each gloss can light up dozens of edges.
- When an LLM/MT *is* wired (`orchestrator.translate` is a stub waiting), batch-
  translate the FR target vocabulary FR→EN and append to `meaningbank.tsv`.
- Optional: Wiktionary/`fr-en` dictionary import → same TSV format, no code
  change. Target: **2,500+ entries** (parity with Lingua-Sound-Wave's reservoir
  goal).

### 5.2 Make the phonetic score trustworthy on sound-spellings
Implement the **sound-prior** option in `hunt_meaning_map.py`: for edges tagged
`homophone`/`phrase`, treat the pairbank's curation as `phon≈1.0` (or blend:
`max(computed, tag_prior)`). Add `--trust-pairbank`. This will surface meaning
resonance that's currently being suppressed by bogus 0.16 sound scores.

### 5.3 Add the reverse direction (FR→EN homophones)
The user's vision is **both** banks bidirectional. Build/import a FR→EN
homophone pairbank, run the *mirror* chain (`FR_src ≈sound≈ EN_tgt =means=>
{FR_glosses}`), and **intersect** the two maps. A concept that is resonant in
*both* directions is maximally trustworthy — that intersection is the cleanest
signal we can produce and the truest "whittling."

### 5.4 Real semantics
Wire `embedding.semantic_similarity` to a real multilingual model when present
(it already tries `all-MiniLM-L6-v2`). Then `meaning_resonance` can compare
`EN_src` directly against the *French* word cross-lingually, not only against
its English glosses — a second, independent meaning road.

### 5.5 Iterate the whittle into true convergence
Today `whittle` clusters by concept once per round. Upgrade it to a fixpoint:
keep iterating synonym-merge until the cluster set is stable across two rounds,
and emit a **convergence metric** (clusters delta → 0). Track it over time in an
evolutionary journal (`evolutionary_journal.jsonl` already exists at the repo
root — append, don't overwrite).

### 5.6 Synonym chains of depth > 1
`SynonymBank.expand` is depth-1. Add bounded transitive expansion
(`note → memo → reminder`) with a decay factor so chains can *reach further*
while distant links count for less. This is the "connecting chains" the user
described — let meanings find each other across two hops.

### 5.7 Surface it in Lingua-Sound-Wave
This engine is the symbolic core that Flit Lab wants. Port `Bridge` + `whittle`
into `artifacts/api-server/src/lib/`, expose `/map-of-meaning`, and render the
concept→sound-alike map in the React explorer. Keep the honesty rule: show
component scores, never hide the count of true resonances.

---

## 6. Operating principles (the spirit to keep)

- **Honesty over magic.** One real resonant bridge beats a hundred faked ones.
  Report the floor. (This is Lingua-Sound-Wave's Golden Rule; obey it.)
- **Degrade gracefully.** Never hard-require a model. The hunt must run on bare
  stdlib and get *better*, not *broken*, when deps appear.
- **Two roads, one truth.** Always keep sound and meaning as separate
  first-class signals; only *celebrate* where they agree. Don't average them
  into mush — the agreement is the discovery.
- **Bank, then engine.** When stuck, the answer is almost always "grow the
  bank," not "add a clever heuristic."
- **Leave a baton.** Before you end, update this file: what you added, the new
  honest numbers, the next sharpest edge. The footsteps must continue.

---

## 7. Quickstart for the next Claude

```sh
cd checkcehck/homophone/homophone-agent-audio
python test_semantic_chain.py          # confirm the engine is intact
python hunt_meaning_map.py --top 30     # see the current map + honest floor
# then: open data/meaningbank.tsv and make it bigger. That's the lever.
```

Go off. Follow the footsteps. Make the map a little more complete than you
found it, and write down where you got to.

*Sound responsibly. Mean honestly.* 🎯🗣️
