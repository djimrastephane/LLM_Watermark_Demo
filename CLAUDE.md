# Working on this project

**Primary audience: oil & gas professionals with little to no GenAI/LLM/programming
background.** This governs every notebook, README section, and example added to this project,
not just the original demo content.

- Explanatory text (Presentation Mode in the Mathematica notebook, the README) must be readable
  without prior AI/ML knowledge. Every technical term used in prose needs to already be in the
  notebook's glossary, or get a plain-language gloss inline.
- Never expose code-style identifiers (snake_case, camelCase) in a rendered table or chart a
  reader is meant to look at. Map them to plain English labels before display.
- Dense statistical detail (multiple parameter sweeps, raw entropy/bits, per-trial breakdowns)
  belongs in Technical Mode, not Presentation Mode. Presentation Mode gets one clear headline
  result plus a plain-language paragraph translating what it means for the reader's own work;
  Technical Mode gets the full rigor for anyone who wants to dig deeper.
- Ground new examples in oil & gas operations (drilling, completions, well intervention
  language) rather than generic ML examples, matching the existing prompt style in
  `notebooks/01_extract_next_token_probabilities.ipynb` and `experiments/`.
- This applies to new investigative work too (e.g. `experiments/`), not just the original two
  notebooks -- if a finding is worth presenting to this audience, it goes through the same
  plain-language bar before it reaches Presentation Mode.
