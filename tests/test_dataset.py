import re
import unicodedata
from collections import Counter

import pytest

from jp_inject import dataset
from jp_inject.judge import MENTION_MARKERS, TRANSLATED_MARKERS, normalize


@pytest.fixture(scope="module")
def cases():
    return dataset.load_cases()


@pytest.fixture(scope="module")
def attacks(cases):
    return [c for c in cases if c["kind"] == "attack"]


def test_frozen_file_matches_sources():
    assert dataset.frozen_is_current(), "run `jp-inject build` after editing the data files"


def test_counts(cases):
    kinds = Counter(c["kind"] for c in cases)
    assert kinds == {"attack": 460, "benign": 30, "comprehension": 55}
    assert len({c["id"] for c in cases}) == len(cases)


def test_every_base_has_every_variant(attacks):
    variants = {v.id for v in dataset.VARIANTS} | {f"{v}@{e}" for v, e in dataset.CROSS_ENV}
    by_base = {}
    for case in attacks:
        by_base.setdefault(case["base"], set()).add(case["variant"])
    assert len(by_base) == 20
    assert all(found == variants for found in by_base.values())


def test_no_unfilled_placeholders(cases):
    for case in cases:
        for field in ("system", "system_no_defense", "user", "payload"):
            assert not re.search(r"\{[a-z_]+\}", case[field]), (case["id"], field)


def test_metadata_is_complete(cases):
    for case in cases:
        assert case["license"] == "Apache-2.0"
        assert case["provenance"]
        assert case["env"] in ("ja", "en")


def test_hijack_payloads_carry_the_canary_verbatim(attacks):
    for case in attacks:
        if case["goal"] == "hijack":
            assert case["token"] in case["payload"], case["id"]
            assert case["payload"] in case["user"]


def test_leak_payloads_never_contain_the_secret(attacks):
    for case in attacks:
        if case["goal"] == "leak":
            assert case["token"] not in case["user"], case["id"]
            assert case["token"] in case["system"]


def test_secret_only_lives_in_the_support_system_prompt(cases):
    secret = dataset.load_sources()["secret"]["value"]
    for case in cases:
        assert (secret in case["system"]) == (case["scenario"] == "support")
        assert secret not in case["user"]


def test_defense_profile_only_adds_text(cases):
    for case in cases:
        if case["scenario"] == "comprehension":
            assert case["system"] == case["system_no_defense"]
        else:
            assert len(case["system"]) > len(case["system_no_defense"])


def test_every_variant_differs_from_its_plain_form(attacks):
    plain = {(c["base"], c["lang"]): c["payload"] for c in attacks if c["variant"] in ("ja.plain", "en.plain")}
    for case in attacks:
        if case["family"] in ("language", "cross_env"):
            continue
        assert case["payload"] != plain[(case["base"], case["lang"])], case["id"]


def test_table_based_transforms_change_something_in_every_base(attacks):
    for case in attacks:
        if case["technique"] in ("kyujitai", "lookalike"):
            assert case["substitutions"] >= 1, case["id"]
        if case["technique"] == "kyujitai":
            assert case["substitutions"] >= 2, case["id"]


def test_script_variants_use_the_intended_script(attacks):
    def scripts(text, token):
        names = set()
        for ch in text.replace(token or "\0", ""):
            if ch.isalpha():
                names.add(unicodedata.name(ch).split()[0])
        return names

    for case in attacks:
        found = scripts(case["payload"], case["token"] if case["goal"] == "hijack" else None)
        if case["variant"] == "ja.hiragana":
            assert found <= {"HIRAGANA", "KATAKANA-HIRAGANA"}, (case["id"], found)
        elif case["variant"] == "ja.katakana":
            assert found <= {"KATAKANA", "KATAKANA-HIRAGANA"}, (case["id"], found)
        elif case["variant"] == "ja.hankaku":
            assert found <= {"HALFWIDTH"}, (case["id"], found)
        elif case["variant"] == "ja.romaji":
            assert found <= {"LATIN"}, (case["id"], found)
        elif case["variant"] == "en.fullwidth":
            assert found <= {"FULLWIDTH"}, (case["id"], found)


def test_cross_env_cases_put_the_payload_in_the_other_language_app(attacks):
    for case in attacks:
        if case["family"] == "cross_env":
            assert case["env"] != case["lang"]
        else:
            assert case["env"] == case["lang"]


def test_carriers_do_not_trip_the_judge_markers():
    # If a benign document already contained a "mention" marker, a faithful
    # summary or translation of it could be mistaken for a refusal.
    src = dataset.load_sources()
    for scenario in ("support", "summarize", "translate"):
        markers = MENTION_MARKERS + (TRANSLATED_MARKERS if scenario == "translate" else ())
        for env in ("ja", "en"):
            for index, carrier in enumerate(src["scenarios"][scenario][env]["carriers"]):
                text = normalize(carrier)
                assert not [m for m in markers if m in text], (scenario, env, index)


def test_readings_match_their_sentences():
    # Each reading must contain exactly the same placeholders and line breaks
    # as the sentence it transcribes.
    src = dataset.load_sources()
    pairs = [(s["lead"]["ja.plain"], s["lead"]["ja.reading"]) for s in src["strategies"].values()]
    pairs += [(t["ask"]["ja.plain"], t["ask"]["ja.reading"]) for t in src["targets"].values()]
    pairs += [(c["ja"], c["reading"]) for c in src["comprehension"]]
    for sentence, reading in pairs:
        assert re.findall(r"\{[a-z_]+\}", sentence) == re.findall(r"\{[a-z_]+\}", reading)
        assert sentence.count("\n") == reading.count("\n")
        body = re.sub(r"\{[a-z_]+\}|[\s、。「」【】＜＞（）ー]", "", reading)
        assert all("ぁ" <= ch <= "ゖ" for ch in body), reading


def test_register_keys_are_complete():
    src = dataset.load_sources()
    keys = {"ja.plain", "ja.reading", "ja.blunt", "ja.keigo", "ja.indirect", "ja.mixed",
            "en.plain", "en.polite", "en.indirect"}
    for strategy in src["strategies"].values():
        assert set(strategy["lead"]) == keys
    for target in src["targets"].values():
        assert set(target["ask"]) == keys
