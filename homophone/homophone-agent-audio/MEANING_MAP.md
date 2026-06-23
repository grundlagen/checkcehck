# The Map of Meaning

*A homophonic + semantic matching engine — the next leg after homophone hunting.*

> The grand goal: the potentiality for perfection in all language to one
> another. Sound is the bridge; meaning is the destination. This document is
> the routine that begins to chart the crossing.

---

## 1. Where we came from (the previous routine)

The homophone agent in `src/` hunts **sound**: given an English phrase it finds
French shapes that *sound alike*, scored by feature-aware phonetic distance,
fluency, a cognate penalty, embedding semantics, CORT complexity, and an
optional TTS "heard-as" reconfirmation. Its accumulated output is
`data/pairbank.tsv` — **39,133 EN→FR homophone bridges**.

Read in one breath, the previous routine is: *take meaning as given (the
literal translation) and search sound-space for a paraphrase that survives the
ear.* The pairbank is the residue of that search — a vast lattice of
sound-correspondences between the two languages.

But the pairbank carries **no meaning of its own**. Its English side is mostly
phonetic fragments (`see ma`, `dew pee`, `thuy`), not dictionary words. It is a
map of *sounds that rhyme across the Channel*, nothing more. That is the seam
this routine opens.

## 2. The idea: layer sense onto sound, then let chains vote

The vision (in the user's words): *match semantic meaning chains from FR→EN
pair banks and the same to homophone EN→FR pair banks, map synonyms on top,
then use these connecting chains to whittle down the same meanings over time.*

Made concrete, that is four moves — all implemented in `src/meaning_map.py`:

1. **Sound graph.** Load the pairbank as a bipartite graph: `("en", form)` and
   `("fr", form)` nodes, undirected homophone edges, weight = multiplicity
   (a correspondence seen many times is stronger evidence).

2. **Synonyms on top — *for free*.** Two English forms that both sound like the
   same French word are *phonetic synonyms*; mirror-wise for French. This needs
   no external thesaurus — it falls straight out of shared neighbours in the
   pairbank. Edge weight is the Jaccard overlap of neighbour sets. (47,299 such
   edges emerge from the current pairbank.)

3. **Connecting chains.** From a *glossed anchor* (a node whose meaning we
   know), walk `sound → synonym → sound`: cross a homophone bridge into the
   other language, optionally slide along a synonym, then cross back. The chain
   *transports the anchor's sound*; we then ask a **meaning oracle** whether the
   anchor's *sense* survived the trip. A chain where sound **and** sense both
   hold is a thread of the map. Score = `sound_strength × meaning_consistency`.

4. **Whittling — consensus over time.** Seed confidence on every bridge any
   chain touched (start broad), then iterate a reinforcement update where each
   correspondence is amplified by the confidence of the *other* bridges on the
   same chain (independent agreement). The pruning threshold **grows each
   round**, so weakly supported correspondences fall away progressively and the
   map condenses toward the meanings many chains agree on. A real run condenses
   `219 → 219 → 219 → 219 → 215 → 166 → 122`.

## 3. The meaning oracle (the honest part)

Meaning is pluggable and tiered, best-available first — and the active tier is
always printed, because **the project's golden rule is honesty about what
works**:

| Tier | Source | Trust |
|---|---|---|
| `embedding` | `sentence_transformers` cosine (if installed) | cross-lingual, coarse |
| `gloss` | curated `data/seed_glosses.tsv` EN↔FR translations | exact where seeded |
| `cognate` | orthographic Jaccard | weak proxy, never empty-handed |

Everything runs on the **standard library**. Heavy models only sharpen the
sense layer; they are never required.

## 4. What the first hunt found — and the frontier

Run it:

```sh
python hunt_meaning.py            # writes out/meaning_map.json, .tsv, MEANING_REPORT.md
```

The chains are already beautiful — e.g. `fr:nom → en:maw → fr:mon`,
`fr:nantes → en:knot → fr:tantes`. The whittling condenses as designed.

But **grail matches = 0**: not one homophone bridge in the current pairbank
coincided with a *seeded true translation*. That is not a bug — it is *the
finding*. The pairbank links sound-shapes, not lemmas, so a generic bilingual
dictionary barely intersects it. To make sound and sense meet, we must first
**project each phonetic shape onto the real lexicon** (it already ships:
`data/lexique.tsv`, `data/lexique_en.tsv` give surface→IPA for both languages),
and only then attach meaning. That projection is the heart of the next routine.

See **`NEXT_ROUTINE.md`** for the footsteps from here.

---

*Sound is data; meaning is the frontier. Walk it honestly.* 🗺️🔉➡️🧠
