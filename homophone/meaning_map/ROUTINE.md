# The Map of Meaning — Routine Log & Next-Routine Charter

> Read this fully before developing. It carries the whole arc of the
> homophonic hunt forward: from *"what sounds alike?"* to *"where do sound and
> meaning become the same thing?"* — the grand goal of perfect cross-lingual
> resonance, one language to all others, starting with EN ↔ FR.

---

## The vision, stated plainly

We are building a **map of meaning**: for every concept, the word in each
language, and the *bridges* between them. Two kinds of bridge run over the same
nodes:

- **Homophone bridges** — `en_word` *sounds like* `fr_word`.
- **Semantic bridges** — `en_word` *means* `fr_word` (translation), thickened
  by **synonym** edges inside each language.

Lay the homophone bank and the semantic bank on top of each other and chase the
chains:

```
say  --hom-->  ses /se/
say  --mean--> dire ──syn── prononcer ...        (chains that may or may not close)
chic --hom-->  chic /ʃik/
chic --mean--> chic                              (chain CLOSES → GOLD)
```

Where a homophone chain and a semantic chain **land on the same word (or
synonyms of it)**, the chain closes and a **resonance** rings. Over many
routines we *whittle*: confirmed closures accrete into the GOLD reservoir,
disproven ones go to the graveyard, and the frontier of unexplained sound-twins
shrinks toward the perfect matches. That whittling is the whole game.

---

## Routine 001 — `meaning_map` born (this routine)

**Built** a self-contained, dependency-free engine that runs in ~5s on the full
65k EN / 245k FR IPA lexica:

- `phon.py` — vendored IPA normalise + feature-weighted phonetic similarity.
- `lexicon_index.py` — exact-sound index (normalised-IPA equality) + coarse
  phonetic buckets.
- `semantic_oracle.py` / `seed_bilingual.py` — pluggable meaning judge; default
  is a curated EN→FR seed (~230 concepts) + FR synonyms.
- `resonance.py` — **two hunts** (concept-anchored `meaning→sound`,
  sound-anchored `sound→meaning`), tiering, dedupe.
- `journal.py` + `resonance_reservoir.jsonl` — append-only memory.
- `hunt.py` — CLI → `MAP_OF_MEANING.md`.

**First harvest:** 1494 resonances. The `sound→meaning` hunt produced ~1478
FRONTIER sound-twins; mining them by hand surfaced real EN↔FR borrowings
(`chic`, `douche`, `bijou`, `boutique`, `chef`, `kiwi`, `ski`, `quiche`,
`couscous`, `clique`, `niche`, `technique` …). These were promoted into the
oracle → **42 GOLD** resonances now ring. *Frontier → confirmation → gold: the
whittling loop closed once, by hand. The next routine automates it.*

**Honest limits (don't paper over these):**
- Meaning coverage is a small curated seed; recall is bounded by it, so most of
  the real lexicon sits in FRONTIER awaiting a meaning signal.
- `sound→meaning` currently uses *exact* normalised-IPA equality only; true
  near-homophones (e.g. `head`/`tête`) only appear via the concept hunt.
- French nasal vowels lose their tilde in the bucket key (scoring keeps it).
- No automated false-friend guard yet (`marque`/`manque`, `barque`/`banque`
  were rejected by hand — see `seed_bilingual.COGNATES` comment).

---

## Next routine — the charter (pick up here)

Ordered by leverage. Each is a clean, shippable step that follows the footsteps.

1. **Automate the frontier→gold promotion (the whittling, for real).**
   Replace hand-mining with a real `SemanticOracle` backend so FRONTIER pairs
   get judged automatically:
   - *Embedding backend* — multilingual sentence/word embeddings (e.g.
     LaBSE / `paraphrase-multilingual-MiniLM`); `score = cosine`. Network is
     available; cache vectors to a local file so routines stay fast.
   - *LLM backend* — a `gpt`/`deepseek` call (key via env, see repo README):
     "Do EN `x` and FR `y` share a core meaning? 0–1 + one-line gloss." Cache
     by pair key in the reservoir so each pair is paid for once.
   Keep `SeedDictOracle` as the offline default and as ground-truth for
   evaluating the new backend.

2. **A real false-friend guard.** Many high-sound pairs are traps
   (`pain`=bread≠pain, `coin`=corner≠coin, `chair`=flesh≠chair). When the oracle
   says "meaning unknown" *but* spelling/embedding says "looks related", flag
   as `FALSE_FRIEND` rather than silently dropping — these are linguistically
   valuable and belong on the map.

3. **Near-homophone `sound→meaning` hunt.** Today sound-anchoring is exact-IPA.
   Use `phon.signature` buckets to compare within-bucket with
   `phon.similarity ≥ 0.85`, capped per bucket, to catch `head`/`tête`,
   `salt`/`sel` from the *sound* side too. The bucket index already exists in
   `lexicon_index.py`.

4. **Grow the synonym graph → multi-hop chains.** The user's core idea is
   *chains* that whittle. Pull WordNet (EN) + a FR synonym source into
   `FR_SYNONYMS` / a new `EN_SYNONYMS`, then let a resonance close over **2–3
   synonym hops**, scoring `sem = decay^hops`. This is where "connecting chains
   that whittle down the same meanings over time" becomes literal.

5. **Bidirectional + the existing pairbank.** Fold
   `homophone-agent-audio/data/pairbank.tsv` (30k homophone + 8k phrase EN→FR
   pairs) in as a *third* sound source, and add an FR→EN direction so the map is
   symmetric (the user wants "fr to en … and the same … en to fr").

6. **Toward "perfection in all languages."** The seam is `SemanticOracle` +
   the two `Lexicon`s. Add a third lexicon (ES/IT/DE WikiPron) and the same
   hunt runs on any pair. Design the reservoir schema now so it is
   language-pair-tagged (it currently assumes EN/FR).

7. **Make the map a graph, then visualise.** Promote the reservoir into an
   actual graph (`networkx`): nodes = words, edges = {hom, mean, syn}. Then
   "connected components that contain ≥1 closed resonance" = **meaning
   islands**. Export to the `Lingua-Sound-Wave` / `Proto-Lingua-Weaver`
   frontends, which already render cross-lingual sound work.

### Definition of done for the next routine
Append a **Routine 002** section below, re-run `hunt.py`, and show the GOLD
count moving (and ideally a `FALSE_FRIEND` tier appearing). The reservoir must
only grow; never rewrite history — whittle by *adding* verdicts.

---

## Operating notes
- Run from `homophone/`: `python3 -m meaning_map.hunt`.
- Everything is offline-by-default and dependency-free; keep it that way for the
  core, put heavy/online backends behind the `SemanticOracle` seam.
- Be honest in the report. The frontier is not failure — it is the map's edge,
  and the edge is where the work is.

*Sound and meaning want to be the same word. Our job is to find where they
already are, and to chart the rest.* 🎯🔤
