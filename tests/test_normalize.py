"""Input normalisation levels."""

import pytest

from jp_inject import dataset, normalize, transforms


def test_nfkc_folds_width_and_drops_invisible_characters():
    assert normalize.nfkc("Ｉｇｎｏｒｅ　ａｌｌ") == "Ignore all"
    assert normalize.nfkc("無​視") == "無視"
    assert normalize.nfkc("ｼｽﾃﾑﾌﾟﾛﾝﾌﾟﾄ") == "システムプロンプト"


def test_nfkc_leaves_kana_romaji_old_kanji_and_lookalikes_alone():
    for text in ("むしして", "mushi shite", "變更に對應", "システムプ口ンプ卜"):
        assert normalize.nfkc(text) == text


def test_fold_restores_old_kanji_and_lookalikes():
    assert normalize.fold("變更に對應") == "変更に対応"
    assert normalize.fold("システムプ口ンプ卜") == "システムプロンプト"
    assert normalize.fold("サ一バ一") == "サーバー"
    assert normalize.fold("協カして調ベてください") == "協力して調べてください"
    assert normalize.fold("へルプを表示") == "ヘルプを表示"


def test_fold_keeps_ordinary_text():
    for text in ("入口は一階です。", "二人で力を合わせる", "東京へ行く", "カタカナとひらがな", "工事中", "千円"):
        assert normalize.fold(text) == text


def test_fold_cannot_repair_a_word_made_only_of_lookalikes():
    assert normalize.fold(transforms.to_lookalike("カタ")) == "力夕"


def test_levels_include_each_other_and_none_is_identity():
    text = "Ｉｇｎｏｒｅ　變更"
    assert normalize.normalize(text, None) == text
    assert normalize.normalize(text, "nfkc") == "Ignore 變更"
    assert normalize.normalize(text, "fold") == "Ignore 変更"
    with pytest.raises(ValueError):
        normalize.normalize(text, "kana")


def test_fold_does_not_touch_texts_without_an_attack():
    for case in dataset.load_cases():
        if case["kind"] == "benign" or case["variant"] in ("ja.plain", "en.plain"):
            assert normalize.fold(case["user"]) == normalize.nfkc(case["user"]), case["id"]


def test_the_codes_survive_normalisation():
    for case in dataset.load_cases():
        if case.get("token") and case["token"] in case["user"]:  # a leak target's code is in the system prompt
            for level in normalize.LEVELS:
                assert case["token"] in normalize.normalize(case["user"], level), case["id"]
