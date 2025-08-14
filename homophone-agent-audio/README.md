# Homophonic–Literal Dual Translation Agent

This repository contains a minimal skeleton for building a dual‑track translation
agent that produces both a *literal* translation (Track A) and a
*homophonic* paraphrase (Track B) for a given source passage.  The goal is to
provide a starting point for experimenting with phonetic similarity and
semantic preservation as described in the accompanying specification.

## Directory Layout

- `data/` — Placeholder for external resources such as
  [Lexique 3](https://lexique.org/lexique/) and [WikiPron](https://github.com/kylebgorman/wikipron).
  Populate this directory with your dictionaries and phone tables before running
  the agent.  None of these resources are included here due to licensing and
  size constraints.
- `src/` — Core Python modules implementing the building blocks described in
  the specification.  Each module is self contained and designed to be easy to
  replace or extend.
- `main.py` — A simple command line interface demonstrating how the modules
  could be wired together.  It accepts an English string as input and prints
  a JSON structure containing the literal and homophonic outputs along with
  placeholder scores.

## Installation

This repository is a skeleton and does **not** vendor any third‑party
dependencies.  Some of the modules (e.g. `phone_distance.py`) reference
packages such as `panphon` and `epitran` which are not available in this
environment by default.  To run the complete system you will need to
install these packages yourself:

```sh
pip install panphon epitran rapidfuzz unidecode
```

If you cannot install `panphon` or `epitran`, you may implement your own
phonetic distance and grapheme‑to‑phoneme conversion routines or use
alternative libraries.  The code has been written so that failure to import
these packages results in clear errors rather than silent failures.

## Running the CLI

The `main.py` script exposes a very simple interface:

```sh
python main.py \
  --src-text "Hello world" \
  --src-lang en \
  --tgt-lang fr
```

Because this skeleton does not bundle a machine translation system or a
paraphrasing model, the current implementation simply echoes the source text
as the literal translation and uses a naive homophonic candidate based on
character distance.  You are expected to replace the stubs in
`src/orchestrator.py` with real calls to translation and paraphrase models
(e.g. via OpenAI, Hugging Face, or other APIs).

## Extending the Skeleton

The modules under `src/` correspond to different parts of the pipeline:

* `phone_distance.py` implements a feature‑aware phonetic similarity measure
  with onset/nucleus/coda bonuses.  If `panphon` is unavailable the
  similarity method will fall back to a simple Levenshtein ratio over
  character sequences and emit a warning.
* `elision.py` provides utilities for performing French elisions and
  determining whether liaison is permitted between adjacent words.  Feel
  free to extend these functions with a more comprehensive list of
  h‑aspiré words or additional contraction rules.
* `lexicon.py` contains a basic BK‑tree implementation for nearest
  neighbour lookup in phone space.  Loading of Lexique/WikiPron data is left
  to the user.
* `candidate_generation.py` defines a helper for suggesting near‑phonetic
  replacements for tokens in a sentence using the BK‑tree.  It does not
  perform any part‑of‑speech filtering; integrate with a tagger if needed.
* `scoring.py` combines phonetic, semantic and fluency scores into a single
  objective.  A simple CORT complexity metric can also be incorporated with
  a configurable weight.
* `co_optimization.py` sketches a simple A⇄B co‑optimization loop.  Both the
  candidate generator and the function that adjusts the literal translation
  are passed in as callables so you can control how co‑optimization works.
* `orchestrator.py` defines a set of placeholder functions representing the
  “tools” available to the agent (translation, paraphrase, grapheme‑to‑phoneme
  conversion, etc.).  Modify these functions to call into your own models or
  APIs.

This repository is intended as a starting point rather than a complete
solution.  The specification in the prompt above describes how to build a
full system; this code provides a scaffold on which to implement it.

### Audio‑assisted scoring and embedding semantics

An alternate entry point, `main_audio.py`, extends the basic CLI by adding
several optional capabilities:

1. **Embedding‑based semantics.**  If the optional
   [`sentence_transformers`](https://www.sbert.net/) package is
   installed, the script computes a cosine similarity between the
   literal translation and each homophonic candidate using the
   multilingual `all‑MiniLM‑L6‑v2` model.  When the package or model
   cannot be loaded, the system falls back to the original token
   overlap heuristic.  The embedding logic lives in
   `src/embedding.py`.

2. **Audio‑assisted “heard‑as” bonus.**  When invoked with
   `--use-audio-check`, the script attempts to synthesize each
   candidate’s text into audio, then transcribe it back via speech
   recognition.  If the transcript is sufficiently close to the source
   text (and optionally to its IPA), a small bonus (controlled by
   `--bonus-value`) is added to the combined score.  The default
   implementations of text‑to‑speech and automatic speech recognition
   in `src/audio_helpers.py` are stubs; you must override those
   functions to integrate your own services.  Without overriding,
   the bonus will always be zero.

3. **CORT complexity metric.**  Passing `--use-cort` enables computation of
   a lightweight complexity score based on token length variance.  The
   contribution of this score to the overall objective is controlled by
   `--cort-weight` (default `0.0`).

Example usage:

```sh
# Install optional dependency for embeddings
pip install sentence-transformers
# Run with audio check and CORT complexity enabled
python main_audio.py \
  --src-text "the night rate" \
  --src-lang en \
  --tgt-lang fr \
  --use-audio-check \
  --bonus-value 0.08 \
  --use-cort \
  --cort-weight 0.05
```

All other options from `main.py` (such as `--max-rounds` and
`--phonetic-threshold`) are still available.  This alternate script
otherwise behaves identically, emitting a JSON summary of the literal
and homophonic outputs along with component scores.