from dataclasses import dataclass
from typing import Union


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

    def with_limit(self, key: str, value: Union[int, float]):
        if key not in ["time", "chars", "phones", "syllables", "words"]:
            raise ValueError(
                "key has to be one of time, chars, phones, syllables, words"
            )

        return Limits(
            **{
                "time": self.time,
                "chars": self.chars,
                "phones": self.phones,
                "syllables": self.syllables,
                "words": self.words,
                key: value,
            }
        )


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
