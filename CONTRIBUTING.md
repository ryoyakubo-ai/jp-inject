# Contributing

Issues and pull requests are welcome, in English or Japanese.

## Ground rules for test cases

- **Benign payloads only.** Success must be shown by a harmless token. Cases
  that try to elicit harmful content will not be merged.
- **Original text only.** Do not paste attack prompts from other datasets,
  from products, or from work you did under a confidentiality agreement.
- **Keep the design paired.** A new technique needs a plain reference it can
  be compared with, and an English counterpart when one makes sense.
- Fictional names only; no real people, companies or secrets.

## Development

```
pip install -e ".[dev]"
python -m pytest
```

After editing `src/jp_inject/data/*.toml`, regenerate the frozen cases and
commit the result:

```
jp-inject build
```

The test suite fails if `cases.jsonl` is out of date. Changing existing cases
changes the dataset hash recorded in results, so results from before and
after are not comparable; say so in the pull request.
