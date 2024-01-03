from dataclasses import dataclass


@dataclass
class Limits:
    time: float
    chars: int
    phones: int
    syllables: int
    words: int

    def tokens(self, targets):
        return {
            "chars": self.chars,
            "phones": self.phones,
            "syllables": self.syllables,
            "words": self.words,
        }[targets]


LIMITS = {
    "librispeech": {
        # 100th percentile
        "max": Limits(time=30.0, chars=530, phones=400, syllables=160, words=90),
        # 99.99th percentile
        "great": Limits(time=22.0, chars=380, phones=260, syllables=100, words=75),
        # 99.95th percentile
        "good": Limits(time=18.0, chars=330, phones=220, syllables=85, words=65),
        # 99th percentile
        "ok": Limits(time=17.0, chars=290, phones=200, syllables=75, words=60),
    }
}
