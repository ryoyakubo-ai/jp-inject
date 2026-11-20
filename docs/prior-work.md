# Prior work and how jp-inject relates to it

Checked in November 2026. Tools move quickly; if something here is out of
date, please open an issue.

## Prompt-injection attacks and benchmarks

- **Perez & Ribeiro (2022)**, *Ignore Previous Prompt: Attack Techniques For
  Language Models* ([arXiv:2211.09527](https://arxiv.org/abs/2211.09527)).
  Goal hijacking and prompt leaking; the origin of the `ignore` strategy and of
  the two goals used here.
- **Greshake et al. (2023)**, *Not what you've signed up for: Compromising
  Real-World LLM-Integrated Applications with Indirect Prompt Injection*
  ([arXiv:2302.12173](https://arxiv.org/abs/2302.12173)). Indirect injection
  through content the application reads; the summarizer and translator
  scenarios are instances of it.
- **Liu et al. (2024)**, *Formalizing and Benchmarking Prompt Injection Attacks
  and Defenses* ([arXiv:2310.12815](https://arxiv.org/abs/2310.12815)). The
  taxonomy (naive, escape characters, context ignoring, fake completion,
  combined) that the five strategies here follow.
- **Yi et al. (2023)**, BIPIA: *Benchmarking and Defending Against Indirect
  Prompt Injection Attacks on Large Language Models*
  ([arXiv:2312.14197](https://arxiv.org/abs/2312.14197)).
- **Miceli-Barone & Sun (2024)**, *A test suite of prompt injection attacks for
  LLM-based machine translation*
  ([arXiv:2410.05047](https://arxiv.org/abs/2410.05047)). Injection inside
  text to be translated, across language pairs.

These establish the attack patterns. They are overwhelmingly English, and
where other languages appear the attack is the same sentence in another
language.

## Other languages

- **Deng et al. (2023)**, *Multilingual Jailbreak Challenges in Large Language
  Models* ([arXiv:2310.06474](https://arxiv.org/abs/2310.06474)), and **Yong
  et al. (2023)**, *Low-Resource Languages Jailbreak GPT-4*
  ([arXiv:2310.02446](https://arxiv.org/abs/2310.02446)), show that safety
  behaviour varies by language, using translated prompts.
- **Wada, Sato & Yokoyama (JSAI 2026)**, *Building a Japanese Prompt-Injection
  Defense Model for LLMs and Its Challenges*. A detector for Japanese
  prompt injection, trained on translated English data plus samples reflecting
  Japanese-specific patterns (orthographic variation, honorifics, zero-width
  spaces). It approaches the same observation from the defence side: a
  classifier, and its training data. jp-inject measures what those patterns do
  to *target models*, with paired controls, and publishes the test cases.
- **Hayasaka, Koseki & Okahara (JSAI 2026)**, *Evaluation of Prompt Injection
  Attacks for Extracting Sensitive Data*. Japanese and image-rendered prompts
  for secret extraction.

## Character-level attacks

- **Boucher et al. (2021)**, *Bad Characters: Imperceptible NLP Attacks*
  ([arXiv:2106.09898](https://arxiv.org/abs/2106.09898)). Invisible
  characters, homoglyphs, reordering and deletion as a general attack class.
  `ja.zero_width` and `ja.lookalike` belong to this family. What is specific
  here is the look-alike inventory (katakana–kanji and hiragana–katakana
  pairs, which Latin-centred confusable lists do not cover) and the
  Japanese/English comparison.

Kana-only spelling, romaji, half-width katakana and kyujitai are not
"imperceptible" at all: a reader sees them and reads them without effort.
They are ordinary Japanese, which is what makes them interesting as inputs
that a filter may not have been built for.

## Tools

### garak

[garak](https://github.com/NVIDIA/garak) (Derczynski et al. 2024,
[arXiv:2406.11036](https://arxiv.org/abs/2406.11036)) is the natural home for
broad scanning. Overlap was checked against its reference documentation:

| garak feature | What it does | Overlap with jp-inject |
| --- | --- | --- |
| Translation support (`langservice`) | machine-translates probe and detector text into a target language | Produces translated attacks. Does not produce register, framing or orthographic variants. |
| `badchars` probes | invisible Unicode, homoglyphs, bidi reordering, deletions, after Boucher et al. | Overlaps in spirit with `ja.zero_width`. Homoglyph sets are not Japanese cross-script pairs. |
| `smuggling` probes, ASCII smuggling | Latin homoglyph substitution; Unicode tag / variation-selector encodings | Different mechanism (hiding ASCII), not Japanese orthography. |
| Buffs (`encoding`, `lowercase`, `paraphrase`, `low_resource_languages`) | transform any probe's prompts | Same *shape* as the orthographic transforms here; none is Japanese-specific. |

So the kana/romaji/half-width/kyujitai/look-alike transforms and the
register and framing variants are, as far as this check found, not available
in garak, and neither is a full-width transform. The transforms have the same
shape as garak's buffs. garak loads plugins only from its own package, so a
transform is usable there only once it is accepted upstream. After the
measurements, only the full-width transform has a case for that (see the
report and the README roadmap). A community package of Turkish-specific probes
for garak (`garak-multilingual`) is a precedent for language-specific
additions.

garak describes itself as a discovery tool rather than a benchmark. jp-inject
is narrower and makes the opposite trade: few attacks, a controlled design,
and intervals on every number.

### PyRIT and promptfoo

General red-teaming and evaluation frameworks with converters or strategies
for encodings, homoglyphs and translation. They are complementary: the frozen
cases in `cases.jsonl` are plain JSON and can be fed to any of them. An export
for promptfoo is on the roadmap.

## What jp-inject adds

1. Attack variation that is native to Japanese, written out in each register
   rather than translated.
2. A paired design: every technique is compared with the plain form of the
   same attack, and with its English equivalent where one exists.
3. Comprehension controls that separate robustness from unreadability.
4. A judge whose error is meant to be measured, plus bounds that do not
   depend on it.
5. Benign payloads throughout, so the whole thing can be public.
