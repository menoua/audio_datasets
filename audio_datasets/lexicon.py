import re

import pandas as pd
from syllabify import syllabify

from . import ROOT_VOCAB_DIR


def ELP(basepath=ROOT_VOCAB_DIR + "/ELP"):
    table = pd.read_csv(
        basepath + "/ELP_Items.csv",
        usecols=["Word", "Freq_HAL"],
        thousands=",",
        dtype={"Word": str, "Freq_HAL": int},
    )
    words = table.sort_values(by="Freq_HAL", ascending=False).Word.str.upper()
    return list(words)


def SubtlexUS(basepath=ROOT_VOCAB_DIR + "/OpenLexicon"):
    table = pd.read_excel(
        basepath + "/Lexique-SubtlexUS.xlsx", usecols=["Word", "SUBTLWF"]
    )
    words = table.sort_values(by="SUBTLWF", ascending=False).Word.str.upper()
    return list(words)


def WorldLex(basepath=ROOT_VOCAB_DIR + "/OpenLexicon"):
    table = pd.read_excel(
        basepath + "/Lexique-WorldLex.xlsx", usecols=["Word", "NewsFreq"]
    )
    words = table.sort_values(by="NewsFreq", ascending=False).Word.str.upper()
    return list(words)


def Switchboard(basepath=ROOT_VOCAB_DIR + "/ISIP"):
    with open(basepath + "/switchboard_lexicon", "r") as f:
        lines = f.readlines()
    words = [line.strip().split(maxsplit=2)[0].upper() for line in lines]
    return words


def Google(basepath=ROOT_VOCAB_DIR + "/Google"):
    table = pd.read_csv(
        basepath + "/vocab_common.txt",
        sep="\t",
        names=["Word", "Freq"],
        dtype={"Word": str, "Freq": int},
    )
    words = table.sort_values(by="Freq", ascending=False).Word.str.upper()
    return list(words)


def Google10K(basepath=ROOT_VOCAB_DIR + "/Google/google-10000-english-master"):
    words = pd.read_csv(
        basepath + "/google-10000-english.txt", names=["Word"], dtype={"Word": str}
    ).Word.str.upper()
    return list(words)


def Google20K(basepath=ROOT_VOCAB_DIR + "/Google/google-10000-english-master"):
    words = pd.read_csv(
        basepath + "/20k.txt", names=["Word"], dtype={"Word": str}
    ).Word.str.upper()
    return list(words)


def VoxForge(basepath=ROOT_VOCAB_DIR + "/VoxForge"):
    with open(basepath + "/VoxForgeDict", "r") as f:
        lines = f.readlines()
    words = [line.strip().split(maxsplit=2)[0].upper() for line in lines]
    return words


def Wictionary(basepath=ROOT_VOCAB_DIR + "/Wictionary"):
    words = pd.read_csv(
        basepath + "/wiki-100k.txt", names=["Word"], dtype={"Word": str}, comment="#"
    ).Word.str.upper()
    return list(words)


def CMU():
    return sorted(
        set([re.sub(r"\([0-9]+\)$", "", word) for word in PRONUN_DICT_CMU.keys()])
    )


def Montreal():
    return sorted(
        set([re.sub(r"\([0-9]+\)$", "", word) for word in PRONUN_DICT_MFA.keys()])
    )


def Prosodylab():
    return sorted(
        set([re.sub(r"\([0-9]+\)$", "", word) for word in PRONUN_DICT_PLA.keys()])
    )


def LibriSpeech(
    basepath=ROOT_VOCAB_DIR + "/LibriSpeech", return_count=False, compatibility=False
):
    table = pd.read_csv(
        basepath + "/vocabulary.csv" + (".compat" if compatibility else ""),
        usecols=["Word", "Count"],
        dtype={"Word": str, "Count": int},
        na_filter=False,
    )
    table = table.sort_values(by="Count", ascending=False).dropna()
    words, counts = table.Word.str.upper(), table.Count
    return (list(words), list(counts)) if return_count else list(words)


def Words(
    train=None,
    reference=None,
    vocab_size=None,
    insert_blank=True,
    insert_na=False,
    normalize=False,
    compatibility=False,
):
    train, count = (
        LibriSpeech(return_count=True, compatibility=compatibility)
        if train is None
        else train
    )
    if reference is None:
        reference = Montreal() if normalize else ELP()
    reference = _variations(reference) if normalize else set(reference)

    count_na = 0
    count_dict = dict()
    for w, c in zip(train, count):
        tokens = (
            _normalize(w, reference)
            if normalize
            else [w if w in reference else "[UNK]"]
        )
        for t in tokens:
            if t == "[UNK]":
                count_na += c
            elif t in count_dict:
                count_dict[t] += c
            else:
                count_dict[t] = c

    table = pd.DataFrame(count_dict.items(), columns=["Word", "Count"])
    table = table.sort_values(by="Count", ascending=False).dropna()
    vocab = list(table.Word.str.upper())

    if vocab_size:
        vocab = vocab[:vocab_size]

    if insert_na:
        # add unknown symbol
        vocab.insert(0, "[UNK]")

    if insert_blank:
        # add blank symbol
        vocab.insert(0, "_")

    return vocab


def Phones(
    basepath=ROOT_VOCAB_DIR + "/CMU",
    stressed=False,
    insert_blank=True,
    insert_space=False,
):
    if stressed:
        with open(basepath + "/cmudict/cmudict-0.7b.symbols", "r") as f:
            vocab = [line.strip() for line in f.readlines()]
        vocab = [s for s in vocab if f"{s}0" not in vocab]
    else:
        with open(basepath + "/cmudict/cmudict-0.7b.phones", "r") as f:
            vocab = [line.strip().split()[0] for line in f.readlines()]

    if insert_blank:
        # add blank symbol
        vocab.insert(0, "_")

    if insert_space:
        # add space symbol
        vocab.append(" ")

    return vocab


def Syllables(
    words=None,
    counts=None,
    stressed=False,
    vocab_size=None,
    insert_blank=True,
    insert_space=False,
    compatibility=False,
):
    if words is None:
        words, counts = LibriSpeech(return_count=True, compatibility=compatibility)
    if counts is None:
        counts = [1] * len(words)

    syll_count = dict()
    for word, count in zip(words, counts):
        for syl in syllabize(pronounce(word, stressed=True), stressed=stressed):
            if syl in syll_count:
                syll_count[syl] += count
            else:
                syll_count[syl] = count

    table = pd.DataFrame(syll_count.items(), columns=["Syllable", "Count"])
    table = table.sort_values(by="Count", ascending=False).dropna()
    vocab = list(table.Syllable[:vocab_size] if vocab_size else table.Syllable)

    if insert_blank:
        # add blank symbol
        vocab.insert(0, "_")

    if insert_space:
        # add space symbol
        vocab.append(" ")

    return vocab


def Characters():
    return list("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")


def pronounce(word, stressed=False):
    pronun = PRONUN_DICT.get(word.upper(), "")
    return pronun if stressed else re.sub(r"([A-Z]+)[0-9]", r"\g<1>", pronun)


def syllabize(phones=None, word=None, stressed=False):
    if word:  # ignore phones
        phones = pronounce(word, stressed=True)

    if phones is None or len(phones) == 0:
        return []

    try:
        syllables = syllabify(phones.split() if isinstance(phones, str) else phones)
    except ValueError:
        return []

    syllables = ["-".join(onset + nucleus + coda) for onset, nucleus, coda in syllables]
    return (
        syllables
        if stressed
        else [re.sub(r"([A-Z]+)[0-9]", r"\g<1>", syl) for syl in syllables]
    )


# Demi-syllabize
# m = re.match('^(.+-)?(A.|I.|U.|E.|O.)(-.+)?$', syl)
# o, n, c = m.groups()
# if o:
#     a.add(o + n)
# if c:
#     a.add(n + c)
# if o is None and c is None:
#     a.add(n)


def _variations(vocab):
    expansion = set(vocab)
    for token in vocab:
        # add plural
        if token.endswith("Y") and not token.endswith(("AY", "EY", "IY", "OY", "UY")):
            expansion.add(token[:-1] + "IES")
        elif token.endswith(("S", "SH", "CH", "Z", "X")):
            expansion.add(token + "ES")
        else:
            expansion.add(token + "S")

        # add tenses
        expansion.add(token + "ED")
        expansion.add(token + "ING")

    return expansion


def _normalize(token, reference=None):
    # split 's at end of word
    if token.endswith("S'"):
        possesive, token = True, token[:-1]
    elif token.endswith("'S"):
        possesive, token = True, token[:-2]
    else:
        possesive = False

    # complete ending abbreviations
    if token.endswith("IN'"):
        token = token[:-1] + "G"

    # map alternate spellings to common variants
    if token in COMMON_SPELLING:
        token = COMMON_SPELLING[token]

    # replace unknown word with [UNK]
    if reference and token not in reference:
        token = "[UNK]"

    # modifiers
    if token in ["A", "AN", "THE"]:
        token = token + "|"

    token_list = [token]
    if possesive:
        token_list.append("|('S)")
    return token_list


def _is_prefix(token):
    return token.endswith("|")


def _is_postfix(token):
    return token.startswith("|")


def _is_subtoken(token):
    return bool(re.match(r"^\|\(.+\)$", token)) or bool(re.match(r"^\(.+\)\|$", token))


def _next_available(word, dictionary):
    word = re.sub(r"\([0-9]+\)$", "", word)
    if word not in dictionary:
        return word

    alt = 1
    while f"{word}({alt})" in dictionary:
        alt += 1

    return f"{word}({alt})"


def _merge_pronunciations(dict1, dict2):
    for word in sorted(dict2.keys()):
        word, pron = re.sub(r"\([0-9]+\)$", "", word), dict2[word]

        if word not in dict1:
            dict1[word] = pron
            continue

        prons, alt = [dict1[word]], 1
        while f"{word}({alt})" in dict1:
            prons.append(dict1[f"{word}({alt})"])
            alt += 1

        if pron not in prons:
            dict1[_next_available(word, dict1)] = pron

    return dict1


PRONUN_DICT_CMU = dict()
with open(ROOT_VOCAB_DIR + "/CMU/cmudict/cmudict-0.7b", "r", encoding="latin-1") as f:
    lines = [
        line.strip().split(maxsplit=1)
        for line in f.readlines()
        if not line.startswith(";;;")
    ]
    for word, pronun in lines:
        PRONUN_DICT_CMU[_next_available(word, PRONUN_DICT_CMU)] = pronun

PRONUN_DICT_MFA = dict()
with open(
    ROOT_VOCAB_DIR + "/Montreal-Aligner/mfa-models/dictionary/english.dict", "r"
) as f:
    lines = [line.strip().split(maxsplit=1) for line in f.readlines()]
    for word, pronun in lines:
        PRONUN_DICT_MFA[_next_available(word, PRONUN_DICT_MFA)] = pronun

PRONUN_DICT_PLA = dict()
with open(ROOT_VOCAB_DIR + "/Prosodylab-Aligner/eng.dict", "r") as f:
    lines = [
        line.strip().split(maxsplit=1)
        for line in f.readlines()
        if not line.startswith(";;;")
    ]
    for word, pronun in lines:
        PRONUN_DICT_PLA[_next_available(word, PRONUN_DICT_PLA)] = pronun

PRONUN_DICT = PRONUN_DICT_MFA.copy()
PRONUN_DICT = _merge_pronunciations(PRONUN_DICT, PRONUN_DICT_CMU)
PRONUN_DICT = _merge_pronunciations(PRONUN_DICT, PRONUN_DICT_PLA)

COMMON_SPELLING = pd.read_csv(
    ROOT_VOCAB_DIR + "/common-spelling.txt", sep="\t", header=0, comment="#"
)
COMMON_SPELLING = dict(zip(COMMON_SPELLING["alternate"], COMMON_SPELLING["common"]))

PER_WORD = dict(
    chars=None,
    phones=3.6,
    syllables=1.4,
    words=1.0,
    words_lite=1.0,
)

LABELS = dict(
    chars=Characters(),
    phones=Phones(stressed=True),
    syllables=Syllables(stressed=True),
    words=Words(insert_na=True, vocab_size=44_000, normalize=True),
    words_22k=Words(insert_na=True, vocab_size=22_000, normalize=True),
    words_lite=Words(insert_na=True, vocab_size=16_384, compatibility=True),
)
