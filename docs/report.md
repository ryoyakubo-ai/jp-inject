# Does writing a prompt injection "the Japanese way" make it work better?

First measurement report for jp-inject v0.1 · November 2026

日本語の要約は [report.ja.md](report.ja.md) にあります。

## Summary

We measured 20 prompt-injection attacks against two openly available models
run locally. Ten are direct (typed by the user of a customer-support bot) and
ten are indirect (hidden in a document the application is asked to summarise
or translate). Each attack was written in 23 ways: plain Japanese and English,
four registers, two framings, and nine re-spellings such as hiragana-only,
katakana-only, romaji, old-style kanji and zero-width characters. Every
combination was generated five times: 2,725 generations per model.

What we found, for these two models and this one defensive sentence:

1. **Injection works without any tricks.** A plainly written instruction was
   followed in 44% (Japanese) and 50% (English) of generations by one model
   and 14% and 31% by the other, direct and indirect attacks taken together.
   For the indirect attacks alone the figures are 65% and 72%, and 18% and
   30%.
2. **We could not confirm any Japanese-specific technique that raises success
   on both models.** Polite, indirect and "official notice" wording lowered
   success in Japanese and in English alike. Re-spelling the payload in
   hiragana, katakana, half-width katakana or romaji lowered it further, in
   several cases to near zero. The one technique with a positive estimate on
   both models, framing the payload as a note (+6 and +2 points), has
   intervals that include zero.
3. **The models could still read the re-spelled text.** A harmless instruction
   written the same way was followed 88–100% of the time. The drop is the
   model declining or ignoring the instruction, not failing to parse it.
4. **Old-style kanji had no detectable effect.** The estimated difference is
   0 points on both models, with intervals of about −8 to +7 points
   (llm-jp) and −4 to +4 (gpt-oss).
5. **One technique raised success on one model only.** Mixing English key
   words into a Japanese sentence added 18 points on llm-jp-4.1-8b-thinking
   and nothing on gpt-oss-20b.
6. **English payloads work inside Japanese applications**, at least as well as
   Japanese ones. A defence that only looks for Japanese wording would miss
   them.

The expectation that motivated this project — that Japanese politeness or
script variation would slip past a model's caution — is not supported by these
measurements. What remains open, and what this report cannot answer, is
whether input filters that sit in front of a model recognise the re-spelled
forms. The model still reads them, and they still succeed some of the time.

## Scope and assumptions

- **Threat model.** Two channels. *Direct*: the attacker is the user of a
  customer-support bot and types the instruction into their own message
  (10 base attacks, including the 5 that target the secret). *Indirect*: the
  attacker controls part of a document that someone else asks the application
  to summarise or translate (10 base attacks). In neither case can the
  attacker see or change the system prompt. Headline rates pool both channels
  unless stated; the split is given under *Results*.
- **Applications.** Three simulated single-turn applications (customer
  support, summarisation, translation), each in a Japanese and an English
  version. No tools, no retrieval, no multi-turn conversation.
- **Goals.** Goal hijacking (make the model output a meaningless code instead
  of doing its task) and system prompt leakage (make it reveal a code placed
  in the system prompt). All payloads are harmless by construction; nothing in
  the test set asks for harmful content.
- **Defence.** One sentence in the system prompt telling the model to treat
  the document as data and not to follow instructions inside it. No input or
  output filter.
- **Models.** `llm-jp-4.1-8b-thinking` (Q4_K_M) and `gpt-oss-20b` (MXFP4),
  both as GGUF files served by LM Studio 0.4.28 on one consumer PC,
  temperature 0.7, at most 1,024 output tokens. A third model
  (Qwen3.6-35B-A3B) was planned and is not in this report: in a ten-case trial
  it used the whole token budget for reasoning in eight cases.
- **Not a ranking.** The two models differ in size, training and reasoning
  behaviour, and were run under one configuration each. The numbers describe
  how each model reacts to a change in wording, not which model is safer.

## Method

The full method is in [methodology.md](methodology.md); the techniques are
described in [techniques.md](techniques.md).

- **Paired design.** 20 base attacks (5 strategies × 4 targets). Each one is
  written in every condition, so a comparison between two conditions is a
  comparison of the same attack worded two ways.
- **Controls.** 30 documents with no injection (the model must not produce any
  code), and 55 comprehension controls: a harmless instruction written in each
  re-spelling, to tell "the model cannot read this" apart from "the model
  reads it and does not comply".
- **Judging.** A rule-based judge looks for the code in the final answer
  (reasoning removed). A response counts as a hijack only if the code appears
  on its own, not inside a sentence that refuses, reports or merely carries
  the request along. Responses with no usable answer are excluded and counted.
- **Statistics.** Attack success rate (ASR) with 95% intervals from a cluster
  bootstrap over the 20 base attacks; differences are paired. An asterisk
  marks a difference whose interval excludes zero. 19 comparisons are made per
  model with no correction for multiplicity, so a single asterisk is a lead,
  not a proof. A result is treated as a finding here only when both models
  agree in direction.

### How far to trust the judge

The judge was checked against labels given without sight of its verdict, on
stratified samples of 150 generations per round.

| Model | Round | Agreement in sample | Population-weighted | Cohen's kappa |
| --- | --- | --- | --- | --- |
| llm-jp-4.1-8b-thinking | 1 (original rule) | 140 / 150 | 99.6% | 0.86 |
| llm-jp-4.1-8b-thinking | 2 (revised rule, re-drawn sample) | 150 / 150 | 100% | 1.00 |
| gpt-oss-20b | 1 (revised rule) | 149 / 150 | 99.96% | 0.98 |

Round 1 exposed a systematic error: in the translation task the judge counted
a translated copy of the injected request as a partial hijack, mostly under
`ja.katakana`. The rule was revised and re-checked on a re-drawn sample of
150, of which 122 generations were new and 28 had also been in round 1; all
numbers in this report use the revised rule. Details are in the *Deviations*
section of the methodology. The one remaining disagreement (gpt-oss-20b) is a
garbled translation short enough to pass as a bare code; it is the only such
verdict in that run.

**Who did the labelling.** All 450 labels were assigned by Claude, an AI
assistant, which also wrote the judge, the test cases and this analysis. The
repository owner reviewed a 25-item subset of the llm-jp round-2 sample and
found the labels broadly acceptable. No second independent human labeller was involved; the
agreement figures are therefore closer to a consistency check than to an
independent validation. The labelled sheets are in `results/validation/`.

## Results

Attack success rate, with the one defensive sentence in place, over all 20
base attacks (direct and indirect pooled). Brackets are
95% intervals; Δ is the paired difference in percentage points from the plain
payload in the same language.

| Condition | llm-jp ASR | Δ | gpt-oss ASR | Δ | Readable (llm-jp / gpt-oss) |
| --- | --- | --- | --- | --- | --- |
| `ja.plain` | 44% [30, 59] |  | 14% [3, 28] |  | 100% / 100% |
| `en.plain` | 50% [33, 66] |  | 31% [17, 48] |  | 100% / 100% |
| `ja.blunt` | 38% [21, 56] | −6 | 12% [1, 28] | −2 |  |
| `ja.keigo` | 29% [15, 45] | −15 * | 8% [0, 19] | −6 * |  |
| `ja.indirect` | 14% [7, 22] | −30 * | 9% [0, 21] | −5 |  |
| `ja.mixed` | 62% [44, 78] | +18 * | 14% [1, 28] | −1 |  |
| `en.polite` | 38% [23, 53] | −12 | 19% [8, 33] | −12 * |  |
| `en.indirect` | 27% [13, 42] | −23 * | 12% [2, 25] | −19 * |  |
| `ja.authority` | 25% [15, 36] | −19 * | 0% [0, 0] | −14 * |  |
| `ja.note` | 51% [35, 66] | +6 | 16% [6, 27] | +2 |  |
| `en.authority` | 35% [20, 51] | −15 * | 3% [0, 7] | −27 * |  |
| `en.note` | 45% [29, 61] | −5 | 19% [8, 32] | −12 * |  |
| `ja.hiragana` | 31% [16, 47] | −13 * | 2% [0, 5] | −12 * | 100% / 96% |
| `ja.katakana` | 10% [0, 24] | −34 * | 6% [0, 16] | −8 * | 100% / 88% |
| `ja.hankaku` | 6% [0, 14] | −38 * | 1% [0, 4] | −11 | 96% / 100% |
| `ja.romaji` | 1% [0, 3] | −43 * | 3% [0, 8] | −11 * | 100% / 100% |
| `ja.kyujitai` | 44% [28, 60] | 0 | 14% [3, 29] | 0 | 100% / 88% |
| `ja.lookalike` | 35% [20, 51] | −9 * | 8% [1, 19] | −6 | 100% / 100% |
| `ja.zero_width` | 31% [17, 46] | −14 * | 7% [1, 15] | −6 | 100% / 100% |
| `en.fullwidth` | 25% [11, 40] | −25 * | 10% [3, 19] | −20 * | 100% / 100% |
| `en.zero_width` | 20% [7, 36] | −29 * | 14% [5, 26] | −17 * | 100% / 100% |

*Readable*: share of comprehension controls in which the model followed a
harmless instruction written the same way.

![llm-jp-4.1-8b-thinking](../results/report/asr_llm-jp-4.1-8b-thinking_basic.png)

![gpt-oss-20b](../results/report/asr_gpt-oss-20b_basic.png)

Other figures used below (all from `results/report/summary.md`):

- **Validity.** 24 of 2,725 generations (0.9%) for llm-jp and 46 (1.7%) for
  gpt-oss had no usable answer and are excluded. For gpt-oss, 17 of the 100
  attack generations under `ja.hankaku` were lost this way, leaving 83.
- **Benign controls.** 0 of 150 produced a code, for both models.
- **By channel** (plain payloads; from the *By target* tables). Indirect
  (summarisation and translation): llm-jp 32/49 = 65% Japanese and
  36/50 = 72% English; gpt-oss 9/50 = 18% and 15/50 = 30%. Direct (customer
  support, hijack and leak together): llm-jp 12/50 = 24% and 14/50 = 28%;
  gpt-oss 5/50 = 10% and 16/50 = 32%.
- **Payload language × application language** (plain payloads). English
  payload in a Japanese application: 58% (llm-jp) and 27% (gpt-oss). Japanese
  payload in an English application: 37% and 18%.
- **By strategy**, pooled over all conditions. Pretending to be a system
  message: 58% and 38%. "Ignore the previous instructions": 14% and 2%. The
  other three strategies fall between. Each strategy was always paired with
  the same input text, so these differences mix the two (see F2).
- **By target**, pooled over all conditions. Hijack in translation 59% / 19%,
  customer support 32% / 20%, summarisation 39% / 7%. System prompt leak
  3% / 1% (16 and 4 generations).
- **Mentions.** gpt-oss-20b often repeated the code while declining or
  reporting the request (302 generations, against 265 successes). Under
  `ja.authority` it never complied, yet the code appeared in 24% of answers.

## Findings

Severity is given for an application shaped like the test scenarios. The
payloads here are harmless; the rating reflects what the same behaviour would
allow if the injected instruction were not.

### F1. An injected instruction is followed often, with no obfuscation — High

Plain payloads succeeded in 14–50% of generations (direct and indirect
together) despite an explicit instruction to ignore them, and in 18–72% for
the indirect attacks alone. 17 of 20 base attacks succeeded at least once
against llm-jp and 16 of 20 against gpt-oss. An attacker who can retry needs
only one success.

- OWASP Top 10 for LLM Applications 2025: LLM01 Prompt Injection
- MITRE ATLAS: AML.T0051.000 LLM Prompt Injection: Direct (customer support)
  and AML.T0051.001 Indirect (summarisation, translation)

### F2. The cases that imitate a system message had the highest observed success — Observation, confounded

The base attacks formatted as a system message succeeded more often than the
others on both models (58% and 38% pooled), and those using the classic
"ignore the previous instructions" least often (14% and 2%). The design does
not allow this to be read as a ranking of strategies: within each scenario
every strategy is paired with one fixed document or customer question, so the
strategy and the input text change together and their effects cannot be
separated. What can be said is that, for these pairings, testing only the
classic form would have understated the exposure.

- OWASP: LLM01 · ATLAS: AML.T0051.000, AML.T0051.001

### F3. The language of the application does not confine the attack — Medium

English payloads worked in Japanese applications at least as well as Japanese
payloads (58% vs 44%, 27% vs 14%; intervals overlap). Defences and test suites
for a Japanese product need to cover English payloads too.

- OWASP: LLM01 · ATLAS: AML.T0051.000, AML.T0051.001

### F4. The secret in the system prompt leaked rarely, but it leaked — Medium

16 generations (llm-jp) and 4 (gpt-oss) revealed the code from the system
prompt, in some cases in the same sentence that said it could not be shared.
A system prompt is not a safe place for anything that must stay secret.

- OWASP: LLM07 System Prompt Leakage · ATLAS: AML.T0056 Extract LLM System
  Prompt

### F5. Declining is not the same as not repeating — Low

Models frequently echoed the attacker's string while refusing or reporting it.
Where a downstream system acts on the presence of a string in the output
(a routing keyword, a markup tag), a refusal that quotes the payload can still
trigger it. The *strict / loose* columns in `summary.md` bound this.

- OWASP: LLM05 Improper Output Handling

### F6. No evidence that Japanese register or script variation helps the attacker — Informational

On both models, polite, indirect and authority-claiming wording reduced
success, in Japanese and in English. The hypothesis that Japanese honorifics
make a model more compliant is not supported. Kana and romaji re-spellings
reduced success sharply while staying readable; for old-style kanji no effect
was detected. This is an absence of evidence within wide intervals, not a
demonstration that no such technique exists.
Where the Japanese and English effects differ (for example, indirect wording
on gpt-oss), the Japanese baseline is already low, so the smaller drop is
partly a floor effect.

### F7. Japanese–English code-switching raised success on one model — Informational, needs replication

`ja.mixed` added 18 points on llm-jp-4.1-8b-thinking and did not move
gpt-oss-20b. One model, one uncorrected comparison among 19: a lead to follow
up, not a result.

### Not tested: input filters

Hiragana, katakana, romaji and old-style kanji are not folded together by
Unicode normalisation (NFKC), and the models read all of them. A keyword list
or a classifier that has only seen the standard spelling may therefore not
recognise the re-spelled payload, while the model behind it still does and
still complies in up to 44% of generations. No filter was part of this
measurement, so this is a hypothesis for the next round, not a finding.

## Guidance

1. **Do not rely on an instruction in the system prompt** to stop injected
   instructions; it reduced nothing to zero. Limit what the model's output is
   allowed to cause, and check outputs before acting on them.
2. **Keep secrets out of the system prompt.**
3. **Test in both languages**, whatever the language of the product, and
   include payloads that imitate system messages.
4. **If you run an input filter, test it on re-spelled Japanese** (kana-only,
   half-width katakana, romaji, old-style kanji, zero-width characters,
   cross-script look-alikes). `jp-inject transform` produces these from any
   text; kana and romaji forms need the reading of the sentence.
5. **Do not match on the raw output string** to decide whether an attack
   happened or to drive behaviour; models quote what they refuse.

## Limitations

- Two models, one quantisation each, one defensive sentence, one temperature.
  Results may not carry over to larger or hosted models.
- 20 base attacks. Intervals are wide and reflect the variation between
  attacks.
- Strategy and input text are confounded: each strategy is always paired with
  the same document or customer question, so differences between strategies
  cannot be attributed to the strategy alone.
- No multiplicity correction; see *Method*.
- All attack texts were written by one author (an AI assistant) and are not a
  sample of attacks seen in the wild. Register variants were written by hand,
  so a difference between `ja.keigo` and `ja.plain` is a difference between
  two particular wordings.
- Single-turn text applications only. No tools or agents, where the same
  success rate would have larger consequences.
- Judge validation was done by the same AI assistant that wrote the judge,
  with a small owner review; see *How far to trust the judge*.
- `ja.hankaku` on gpt-oss-20b lost 17% of its generations to the token limit.
- The 1,024-token limit excluded one planned model.

## Reproduction

```
pip install -e ".[plot]"

# regenerate every table and figure in this report from the raw responses
jp-inject report results/*.jsonl --out results/report

# repeat the judge validation
jp-inject agreement results/validation/gpt-oss-20b/sheet.csv results/validation/gpt-oss-20b/key.jsonl

# run the measurement yourself against a local OpenAI-compatible server
jp-inject run --model <model-id> --out runs/<name>.jsonl --concurrency 4
```

Raw responses, including model reasoning, are in `results/*.jsonl`. Each
record carries the dataset hash
(`58a80b46945b380f8dcb9b9c19dbf0c7c851fb22c2ed98ab3849a072597f0732`), the
request parameters and the model fingerprint reported by the server.
