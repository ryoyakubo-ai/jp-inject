"""Character tables used by :mod:`jp_inject.transforms`.

All tables were written for this project. Half-width katakana is derived from
Unicode normalisation data at import time, so it cannot drift from the
standard.
"""

from __future__ import annotations

import unicodedata

# --------------------------------------------------------------------------
# Shinjitai (modern) -> kyujitai (traditional) kanji.
#
# Only one-to-one pairs whose traditional form has its own code point in the
# CJK Unified Ideographs block are listed. Characters whose traditional form
# is a compatibility ideograph (e.g. 社, 視) are left out on purpose, because
# NFC/NFKC normalisation would silently undo the substitution. Characters that
# merged several traditional forms (e.g. 弁) are left out too.
# --------------------------------------------------------------------------
_KYUJITAI_PAIRS = (
    "亜亞 悪惡 圧壓 囲圍 医醫 為爲 壱壹 隠隱 栄榮 営營 衛衞 駅驛 円圓 塩鹽 縁緣 "
    "応應 欧歐 殴毆 桜櫻 奥奧 横橫 温溫 穏穩 仮假 価價 画畫 会會 壊壞 懐懷 絵繪 "
    "拡擴 殻殼 覚覺 学學 岳嶽 楽樂 勧勸 巻卷 寛寬 歓歡 観觀 関關 陥陷 気氣 帰歸 "
    "偽僞 戯戲 犠犧 旧舊 拠據 挙擧 虚虛 峡峽 挟挾 狭狹 郷鄕 暁曉 区區 駆驅 勲勳 "
    "径徑 恵惠 渓溪 経經 継繼 茎莖 蛍螢 軽輕 鶏鷄 芸藝 欠缺 県縣 倹儉 剣劍 険險 "
    "圏圈 検檢 権權 献獻 顕顯 験驗 厳嚴 広廣 効效 恒恆 鉱鑛 号號 国國 済濟 砕碎 "
    "斎齋 剤劑 雑雜 参參 桟棧 蚕蠶 惨慘 賛贊 残殘 糸絲 歯齒 児兒 辞辭 湿濕 実實 "
    "写寫 釈釋 寿壽 収收 従從 渋澁 獣獸 縦縱 粛肅 処處 称稱 証證 焼燒 奨奬 条條 "
    "状狀 乗乘 浄淨 剰剩 畳疊 縄繩 壌壤 嬢孃 譲讓 醸釀 触觸 嘱囑 真眞 寝寢 慎愼 "
    "尽盡 図圖 粋粹 酔醉 随隨 髄髓 枢樞 数數 声聲 静靜 斉齊 摂攝 窃竊 専專 浅淺 "
    "戦戰 践踐 銭錢 潜潛 繊纖 禅禪 双雙 壮壯 争爭 荘莊 捜搜 挿插 装裝 総總 騒騷 "
    "臓臟 蔵藏 属屬 続續 堕墮 対對 体體 帯帶 滞滯 台臺 滝瀧 択擇 沢澤 担擔 単單 "
    "胆膽 団團 断斷 弾彈 遅遲 痴癡 虫蟲 昼晝 鋳鑄 庁廳 聴聽 鎮鎭 逓遞 鉄鐵 点點 "
    "転轉 伝傳 党黨 盗盜 灯燈 当當 独獨 読讀 届屆 弐貳 悩惱 脳腦 廃廢 拝拜 "
    "売賣 麦麥 発發 髪髮 抜拔 蛮蠻 秘祕 浜濱 払拂 仏佛 並竝 変變 辺邊 宝寶 豊豐 "
    "翻飜 万萬 満滿 黙默 弥彌 訳譯 薬藥 与與 予豫 余餘 誉譽 揺搖 様樣 謡謠 来來 "
    "頼賴 乱亂 覧覽 竜龍 両兩 猟獵 緑綠 涙淚 塁壘 励勵 礼禮 霊靈 齢齡 恋戀 炉爐 "
    "労勞 楼樓 録錄 湾灣 内內 教敎 説說 脱脫 税稅 鋭銳 戸戶 歩步 毎每 黒黑 徳德 "
    "増增 徴徵 缶罐"
)
KYUJITAI: dict[str, str] = {pair[0]: pair[1] for pair in _KYUJITAI_PAIRS.split()}

# --------------------------------------------------------------------------
# Cross-script look-alikes. Each pair is applied in both directions, except
# the long-vowel mark, which only goes one way (kanji 一 is far too common to
# rewrite as a long-vowel mark without wrecking the sentence).
# --------------------------------------------------------------------------
_LOOKALIKE_PAIRS = (
    ("ロ", "口"),  # katakana RO / kanji "mouth"
    ("カ", "力"),  # katakana KA / kanji "power"
    ("タ", "夕"),  # katakana TA / kanji "evening"
    ("ニ", "二"),  # katakana NI / kanji "two"
    ("ハ", "八"),  # katakana HA / kanji "eight"
    ("エ", "工"),  # katakana E / kanji "craft"
    ("ト", "卜"),  # katakana TO / kanji "divination"
    ("チ", "千"),  # katakana CHI / kanji "thousand"
    ("オ", "才"),  # katakana O / kanji "talent"
    ("ヘ", "へ"),  # katakana HE / hiragana HE
    ("リ", "り"),  # katakana RI / hiragana RI
    ("ベ", "べ"),  # katakana BE / hiragana BE
    ("ペ", "ぺ"),  # katakana PE / hiragana PE
)
LOOKALIKE: dict[str, str] = {}
for _a, _b in _LOOKALIKE_PAIRS:
    LOOKALIKE[_a] = _b
    LOOKALIKE[_b] = _a
LOOKALIKE["ー"] = "一"  # long-vowel mark -> kanji "one"

# --------------------------------------------------------------------------
# Full-width katakana -> half-width katakana, derived from Unicode data.
# --------------------------------------------------------------------------
HALFWIDTH_KATAKANA: dict[str, str] = {}
for _code in range(0xFF66, 0xFF9E):  # ｦ .. ﾝ (includes small kana and ｰ)
    _half = chr(_code)
    HALFWIDTH_KATAKANA[unicodedata.normalize("NFKC", _half)] = _half
for _code in range(0x30A1, 0x30FB):  # voiced / semi-voiced katakana
    _full = chr(_code)
    if _full in HALFWIDTH_KATAKANA:
        continue
    _decomposed = unicodedata.normalize("NFD", _full)
    if len(_decomposed) == 2 and _decomposed[0] in HALFWIDTH_KATAKANA:
        _mark = {"゙": "ﾞ", "゚": "ﾟ"}.get(_decomposed[1])
        if _mark:
            HALFWIDTH_KATAKANA[_full] = HALFWIDTH_KATAKANA[_decomposed[0]] + _mark

HALFWIDTH_PUNCT: dict[str, str] = {
    "。": "｡",
    "、": "､",
    "「": "｢",
    "」": "｣",
    "・": "･",
    "【": "[",
    "】": "]",
    "＜": "<",
    "＞": ">",
    "（": "(",
    "）": ")",
    "：": ":",
    "？": "?",
    "！": "!",
    "…": "...",
}

# --------------------------------------------------------------------------
# Hiragana -> romaji (modified Hepburn; long vowels are spelled out).
# --------------------------------------------------------------------------
_ROMAJI_ROWS = {
    "": "あa いi うu えe おo",
    "k": "かka きki くku けke こko",
    "s": "さsa しshi すsu せse そso",
    "t": "たta ちchi つtsu てte とto",
    "n": "なna にni ぬnu ねne のno",
    "h": "はha ひhi ふfu へhe ほho",
    "m": "まma みmi むmu めme もmo",
    "y": "やya ゆyu よyo",
    "r": "らra りri るru れre ろro",
    "w": "わwa をwo",
    "g": "がga ぎgi ぐgu げge ごgo",
    "z": "ざza じji ずzu ぜze ぞzo",
    "d": "だda ぢji づzu でde どdo",
    "b": "ばba びbi ぶbu べbe ぼbo",
    "p": "ぱpa ぴpi ぷpu ぺpe ぽpo",
    "small": "ぁa ぃi ぅu ぇe ぉo ゃya ゅyu ょyo ゔvu",
}
ROMAJI: dict[str, str] = {}
for _row in _ROMAJI_ROWS.values():
    for _item in _row.split():
        ROMAJI[_item[0]] = _item[1:]

_ROMAJI_DIGRAPHS = (
    "きゃkya きゅkyu きょkyo しゃsha しゅshu しょsho ちゃcha ちゅchu ちょcho "
    "にゃnya にゅnyu にょnyo ひゃhya ひゅhyu ひょhyo みゃmya みゅmyu みょmyo "
    "りゃrya りゅryu りょryo ぎゃgya ぎゅgyu ぎょgyo じゃja じゅju じょjo "
    "びゃbya びゅbyu びょbyo ぴゃpya ぴゅpyu ぴょpyo "
    # combinations that only occur in loan words
    "ふぁfa ふぃfi ふぇfe ふぉfo てぃti でぃdi とぅtu どぅdu うぃwi うぇwe うぉwo "
    "しぇshe じぇje ちぇche ゔぁva ゔぃvi ゔぇve ゔぉvo"
)
for _item in _ROMAJI_DIGRAPHS.split():
    ROMAJI[_item[:2]] = _item[2:]

ROMAJI_PUNCT: dict[str, str] = {
    "、": ",",
    "。": ".",
    "「": '"',
    "」": '"',
    "【": "[",
    "】": "]",
    "＜": "<",
    "＞": ">",
    "（": "(",
    "）": ")",
    "：": ":",
    "？": "?",
    "！": "!",
    "…": "...",
    "・": " ",
    "※": "*",
}
