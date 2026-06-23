# Next Routine — Footsteps Toward the Map of Meaning

*For the next Claude (or human) to pick up. Read `MEANING_MAP.md` first, then
this. Each step is small, testable, and pushes the grand goal: the potentiality
for perfection in all language to one another.*

---

## State at handoff

- **Done:** `src/meaning_map.py` (engine) + `hunt_meaning.py` (driver) +
  `data/seed_glosses.tsv` (50 curated EN↔FR translations) + `out/` artifacts.
  Sound graph → synonym layer → connecting chains → condensing whittle, all on
  the standard library. Run `python hunt_meaning.py`.
- **Key finding:** the pairbank links *sound-shapes*, not dictionary lemmas, so
  true meaning barely intersects it. **Grail matches (sound ∧ sense) = 0.**
  Closing that gap is the whole game now.

## The frontier, in priority order

### 1. Project phonetic shapes onto the real lexicon  ← do this first
The blocker is that EN `src` forms (`thuy`, `dew pee`) are not words. But
`data/lexique_en.tsv` and `data/lexique.tsv` give **surface→IPA** for ~65k EN
and ~246k FR real words. Build `src/lexicon_projection.py`:

- For each pairbank node, compute its IPA (`orchestrator.g2p`, already present).
- Find the nearest **real lemma(s)** in the same language by phonetic distance
  (the BK-tree in `src/lexicon.py` + `phone_metric.similarity` already exist —
  reuse them, don't rebuild).
- Emit `node → [(lemma, phon_sim), ...]`. Now every sound-shape has a *lexical
  shadow* that can carry meaning.

**Acceptance:** `thuy` projects to real EN words near /tuːi/; `tante` is its own
lemma. Add a smoke assert.

### 2. Attach real bilingual meaning
With lemmas in hand, meaning can finally land. Cheapest → richest:

- **(a)** Expand `data/seed_glosses.tsv` to a few hundred high-frequency EN↔FR
  pairs (one true translation per line). Immediate, zero-dependency lift to the
  `gloss` oracle tier.
- **(b)** If network/deps allow, wire a real bilingual source (Wiktionary
  EN↔FR, OPUS dictionaries, or `sentence_transformers` multilingual MiniLM) into
  `MeaningOracle`. Keep the tiered fallback intact — never make a model
  mandatory.

**Acceptance:** `oracle_tier` reported as `gloss` (large) or `embedding`, and
**grail_count > 0** in `out/meaning_map.json`.

### 3. Make chains meaning-led, not just sound-led
Right now chains seed from glossed anchors and *check* meaning at the end. Flip
it: enumerate chains that must stay within a **meaning band** at every hop
(beam search over `sound × running_meaning`). Add `--meaning-floor` wiring
through `hunt_chains` (the parameter already exists; expose it on the CLI and
prune mid-walk). This is the literal "semantic meaning chains" the vision asks
for.

### 4. Bidirectional banks
The vision names *both* directions: FR→EN and EN→FR. The pairbank is currently
all `en→fr`. Generate the mirror (or load FR-sourced rows as `fr→en`) and let
`build_sound_graph` ingest both directions, so chains can start on either
shore. Watch for the symmetry already half-present in `source_file`
(`french_phrase_to_english.csv`).

### 5. Whittle, evaluated
The condensation (`219→…→122`) is currently unvalidated. Add a tiny held-out
set of *known* cross-lingual sound+sense matches and measure precision@k of the
surviving correspondences across whittle rounds. Tune `alpha`, `prune_below`,
`prune_growth` against it. Honesty rule: **publish the curve even if it's flat.**

### 6. Cross-pollinate the sibling projects
`Lingua-Sound-Wave` already has a live `phoneme-chain` judge, a reservoir miner,
and Flit Lab (see its `replit.md`). The meaning-map's correspondences are exactly
the kind of EN↔FR pairs its reservoir wants — and its phoneme-chain scorer is a
far better `sound_strength` than our coarse multiplicity weight. Consider:
export `out/meaning_correspondences.tsv` as reservoir seeds; import
phoneme-chain as the bridge metric.

## Guardrails (inherited house rules)

- **Honesty > magic.** If the numbers lie or a tier is weak, surface it.
- **Stdlib-first.** Optional deps may sharpen, never gate.
- **Reuse before rebuild.** `phone_metric`, `lexicon` BK-tree, `embedding`,
  `g2p` are all already here.
- **One language to another, perfectly** is the north star — approach it in
  small, measured, testable steps.

## Quick commands

```sh
cd homophone/homophone-agent-audio
python hunt_meaning.py --top 30                 # full hunt
python hunt_meaning.py --limit 5000 --iterations 8   # fast experiment
cat out/MEANING_REPORT.md                       # read the narrative
```

---

*Pick up the sound; carry the sense; whittle to the truth. — previous Claude* 🔉➡️🧠
