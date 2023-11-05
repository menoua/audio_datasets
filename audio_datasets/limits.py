from dataclasses import dataclass


@dataclass
class Limits:
    time: float
    tokens: int


LIMITS_CHAR = {
    "librispeech": {
        "max": Limits(time=30.0, tokens=530),  # 100th percentile
        "great": Limits(time=22.0, tokens=380),  # 99.99th percentile
        "good": Limits(time=18.0, tokens=330),  # 99.95th percentile
        "ok": Limits(time=17.0, tokens=290),  # 99th percentile
    }
}

LIMITS_PHONE = {
    "librispeech": {
        "max": Limits(time=30.0, tokens=400),
        "great": Limits(time=22.0, tokens=260),
        "good": Limits(time=18.0, tokens=220),
        "ok": Limits(time=17.0, tokens=200),
    }
}

LIMITS_SYLLABLE = {
    "librispeech": {
        "max": Limits(time=30.0, tokens=160),
        "great": Limits(time=22.0, tokens=100),
        "good": Limits(time=18.0, tokens=85),
        "ok": Limits(time=17.0, tokens=75),
    }
}

LIMITS_WORD = {
    "librispeech": {
        "max": Limits(time=30.0, tokens=90),
        "great": Limits(time=22.0, tokens=75),
        "good": Limits(time=18.0, tokens=65),
        "ok": Limits(time=17.0, tokens=60),
    }
}
