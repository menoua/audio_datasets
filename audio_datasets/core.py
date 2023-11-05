import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy as sp
import textgrids
import torch
import torchaudio
from torchaudio import load as load_audio
from torchaudio.sox_effects import apply_effects_tensor as apply_sox_effects
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm

from .data import NonSpeech
from .limits import Limits
from .lexicon import (is_postfix, is_prefix, is_stressed, is_subtoken,
                      normalize_token, syllabize)

torchaudio.set_audio_backend("sox_io")

CONFIG_MOD_LO = {
    "BG_SNR": (10, 15),
    "PITCH_SHIFT": (-2, 2),
    "SPEED_RATE": (0.9, 1.1),
    "TEMPO_RATE": (0.9, 1.2),
    "CHORUS_N": (1, 3),
    "ECHO_N": (1, 3),
    "REVERB": (10, 40),
    "LOWPASS_F": (6000, 7500),
    "HIGHPASS_F": (100, 500),
    "BANDPASS_F": (100, 500),
    "BANDPASS_W": (12, 16),
    "BANDSTOP_F": (300, 4000),
    "BANDSTOP_W": (1, 2),
}

CONFIG_MOD_MID = {
    "BG_SNR": (0, 15),
    "PITCH_SHIFT": (-4, 4),
    "SPEED_RATE": (0.7, 1.3),
    "TEMPO_RATE": (0.8, 1.4),
    "CHORUS_N": (1, 4),
    "ECHO_N": (1, 4),
    "REVERB": (20, 70),
    "LOWPASS_F": (4000, 7000),
    "HIGHPASS_F": (300, 1000),
    "BANDPASS_F": (200, 1000),
    "BANDPASS_W": (6, 8),
    "BANDSTOP_F": (300, 2500),
    "BANDSTOP_W": (2, 3),
}

CONFIG_MOD_HI = {
    "BG_SNR": (-10, 15),
    "PITCH_SHIFT": (-6, 6),
    "SPEED_RATE": (0.5, 1.5),
    "TEMPO_RATE": (0.7, 1.6),
    "CHORUS_N": (1, 6),
    "ECHO_N": (1, 5),
    "REVERB": (30, 100),
    "LOWPASS_F": (2000, 6000),
    "HIGHPASS_F": (500, 2000),
    "BANDPASS_F": (300, 1500),
    "BANDPASS_W": (3, 5),
    "BANDSTOP_F": (300, 1500),
    "BANDSTOP_W": (3, 5),
}


@dataclass
class Sample:
    sound: torch.Tensor
    source: torch.Tensor
    rate: int
    labels: Optional[torch.Tensor] = None
    label_locs: Optional[torch.Tensor] = None
    intervals: Optional[list[tuple[float, float]]] = None
    skew: float = 1


@dataclass
class SampleBatch:
    sounds: torch.Tensor
    sound_lens: torch.Tensor
    # sound_locs: Optional[torch.Tensor]
    sources: torch.Tensor
    source_lens: torch.Tensor
    # source_locs: Optional[torch.Tensor]
    rate: int
    labels: Optional[torch.Tensor] = None
    label_lens: Optional[torch.Tensor] = None
    label_locs: Optional[torch.Tensor] = None
    intervals: Optional[list[tuple[float, float]]] = None


class SoundDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sounds: list[str],
        in_sr: int = 16_000,
        out_sr: int = 100,
        freqbins: int = 128,
        audio_proc: Optional[Callable] = None,
        noise_reduce: bool = False,
        mod_speech: bool = False,
        mod_room: bool = False,
        mod_channel: bool = False,
        mod_scene: list[str] = [],
        mod_custom: Optional[Callable] = None,
        mix_augments: int = 1,
        mod_intensity: str = "low",
        top_db: float = 70,
        batch_first: bool = True,
    ):
        self.sounds = sounds
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.freqbins = freqbins
        self.noise_reduce = noise_reduce
        self.mod_speech = mod_speech
        self.mod_room = mod_room
        self.mod_channel = mod_channel
        self.mod_scene = mod_scene
        self.mod_custom = mod_custom
        self.mix_augments = mix_augments
        self.set_intensity(mod_intensity)
        self.top_db = top_db
        self.batch_dim = 0 if batch_first else 1

        if audio_proc == "default":
            self.audio_proc = torch.nn.Sequential(
                MelSpectrogram(
                    in_sr,
                    n_fft=1024,
                    hop_length=int(in_sr / out_sr),
                    f_min=20,
                    f_max=8_000,
                    n_mels=freqbins,
                    power=2.0,
                ),
                AmplitudeToDB("power", top_db=top_db),
                type(
                    "Normalize",
                    (torch.nn.Module,),
                    dict(
                        forward=lambda _, x: (x - x.max()).squeeze(0).T.float() / top_db
                        + 1
                    ),
                )(),
            )
        else:
            self.audio_proc = audio_proc

    def __len__(self):
        return len(self.sounds)

    @torch.no_grad()
    def __getitem__(self, idx: int, waveform: bool = False) -> Sample:
        in_path = self.sounds[idx]

        xforms = []
        if self.mod_custom:
            xforms.append("mod_custom")
        if self.mod_speech:
            xforms.append("mod_speech")
        if self.mod_room:
            xforms.append("mod_room")
        if self.mod_channel:
            xforms.append("mod_channel")
        if self.mod_scene:
            xforms.append("mod_scene")
        if xforms and self.mix_augments >= 0:
            xforms = np.random.choice(xforms, size=self.mix_augments, replace=False)

        tfm = [["rate", str(self.in_sr)], ["channels", "1"]]
        # custom modification
        if "mod_custom" in xforms and self.mod_custom:
            self.mod_custom(tfm)
        # speech modification
        if "mod_speech" in xforms:
            self._speech_modifier(tfm)
        # room modification
        if "mod_room" in xforms:
            self._room_modifier(tfm)
        # channel modification
        if "mod_channel" in xforms:
            self._channel_modifier(tfm)
        # set volume
        tfm.append(["norm", "-3"])

        # process audio
        in_audio, in_sr = load_audio(in_path)
        mix_audio, mix_sr = apply_sox_effects(in_audio, in_sr, tfm)

        # scene modification
        if "mod_scene" in xforms:
            noise_path = np.random.choice(self.mod_scene)
            noise_audio, noise_sr = load_audio(noise_path)

            tfm = [["rate", str(self.in_sr)], ["channels", "1"]]
            # repeat to cover full speech
            dur_speech = mix_audio.shape[1] / mix_sr
            dur_noise = noise_audio.shape[1] / noise_sr
            count = math.ceil(dur_speech / dur_noise)
            tfm.append(["repeat", str(count)])
            # trim to same length as speech
            tfm.append(["trim", "0", str(dur_speech)])
            # set volume
            snr_db = np.random.uniform(*self.mod_config["BG_SNR"])
            tfm.append(["norm", str(-3 - snr_db)])
            # process audio
            noise_audio, noise_sr = apply_sox_effects(noise_audio, noise_sr, tfm)

            mix_audio = (mix_audio + noise_audio[:, : mix_audio.shape[1]]) / np.sqrt(2)
            mix_audio, mix_sr = apply_sox_effects(mix_audio, mix_sr, [["norm", "-3"]])

        # calculate skew
        skew = mix_audio.shape[1] / in_audio.shape[1]

        # calculate spectrograms
        if waveform or self.audio_proc is None:
            x = mix_audio.T
        else:
            x = self.audio_proc(mix_audio)

        # calculate spectrograms for clean input
        if waveform or self.audio_proc is None:
            x0 = in_audio.T
        else:
            x0 = self.audio_proc(in_audio)

        return Sample(
            sound=x,
            source=x0,
            rate=self.in_sr,
            skew=skew,
        )

    def augment(
        self,
        speech: bool = True,
        room: bool = True,
        channel: bool = True,
        scene=None,
        mix_n: int = 1,
        mod_intensity: str = "low",
    ):
        self.mod_speech = speech
        self.mod_room = room
        self.mod_channel = channel
        self.mod_scene = NonSpeech() if scene is None else scene
        self.mix_augments = mix_n
        self.set_intensity(mod_intensity)

        return self

    def speed_up(self, factor: float):
        def modifier(tfm):
            # if 0.9 < factor < 1.1:
            #    tfm.append(['stretch', str(1/factor)])
            # else:
            #    tfm.append(['tempo', '-s', str(factor)])
            tfm.append(["tempo", "-s", str(factor)])

            tfm.append(["rate", str(self.in_sr)])
            return tfm

        self.mod_custom = modifier

        return self

    def set_intensity(self, level: str):
        if level in ["low"]:
            self.mod_config = CONFIG_MOD_LO
        elif level in ["mid", "medium"]:
            self.mod_config = CONFIG_MOD_MID
        elif level in ["high"]:
            self.mod_config = CONFIG_MOD_HI
        else:
            raise ValueError(
                "Modification intensity should be one of low, mid/medium, high."
            )

        self.mod_intensity = level

        return self

    def annotate(
        self,
        annotations: list[str],
        vocabulary: list[str],
        target: str,
        limits: Limits,
        value_nil: int = 0,
        ignore_silence: bool = True,
        normalize: bool = False,
    ):
        return AnnotatedDataset(
            self.sounds,
            annotations,
            vocabulary,
            target,
            limits=limits,
            value_nil=value_nil,
            ignore_silence=ignore_silence,
            normalize=normalize,
            in_sr=self.in_sr,
            out_sr=self.out_sr,
            freqbins=self.freqbins,
            noise_reduce=self.noise_reduce,
            mod_speech=self.mod_speech,
            mod_room=self.mod_room,
            mod_channel=self.mod_channel,
            mod_scene=self.mod_scene,
        )

    def iterator(self, shuffle=False, num_workers=0):
        return torch.utils.data.DataLoader(
            self, shuffle=shuffle, num_workers=num_workers
        )

    def _speech_modifier(self, tfm):
        fx = np.random.choice(("pitch", "speed", "tempo"))

        if fx == "pitch":
            n_semitones = np.random.uniform(*self.mod_config["PITCH_SHIFT"])
            tfm.append(["pitch", str(n_semitones)])
            tfm.append(["rate", str(self.in_sr)])
        elif fx == "speed":
            factor = np.random.uniform(*self.mod_config["SPEED_RATE"])
            tfm.append(["speed", str(factor)])
            tfm.append(["rate", str(self.in_sr)])
        elif fx == "tempo":
            factor = np.random.uniform(*self.mod_config["TEMPO_RATE"])
            # if 0.9 < factor < 1.1:
            #    tfm.append(['stretch', str(1/factor)])
            # else:
            #    tfm.append(['tempo', '-s', str(factor)])
            tfm.append(["tempo", "-s", str(factor)])
            tfm.append(["rate", str(self.in_sr)])

        return tfm

    def _room_modifier(self, tfm):
        # fx = np.random.choice(('chorus', 'echo', 'reverb'))
        fx = np.random.choice(("reverb",))

        if fx == "chorus":
            n_voices = np.random.randint(*self.mod_config["CHORUS_N"])
            delays = np.random.uniform(40, 60, n_voices)
            decays = np.random.uniform(0.25, 0.45, n_voices)
            speeds = np.random.uniform(0.2, 0.4, n_voices)
            depths = np.random.uniform(1.0, 3.0, n_voices)
            modulations = np.random.choice(("s", "t"), n_voices)
            tfm.append(
                [
                    "chorus",
                    str(0.8 - n_voices * 0.1),
                    "0.9",
                    *sum(
                        [
                            [str(_) for _ in i[:-1]] + ["-" + i[-1]]
                            for i in zip(delays, decays, speeds, depths, modulations)
                        ],
                        [],
                    ),
                ]
            )
        elif fx == "echo":
            n_echos = np.random.randint(*self.mod_config["ECHO_N"])
            max_delay = np.random.randint(50, 300)
            delays = list(np.linspace(0, max_delay, n_echos + 1))[1:]
            min_decay = np.random.uniform(0.1, 0.7)
            decays = list(np.linspace(1, min_decay, n_echos + 1))[1:]
            tfm.append(
                [
                    "echo",
                    "0.8",
                    "0.9",
                    *sum([[str(_) for _ in i] for i in zip(delays, decays)], []),
                ]
            )
        elif fx == "reverb":
            reverberance = np.random.uniform(*self.mod_config["REVERB"])
            room_scale = np.random.uniform(50, 100)
            tfm.append(
                ["reverb", "--wet-only", str(reverberance), "50", str(room_scale)]
            )
            tfm.append(["channels", "1"])

        return tfm

    def _channel_modifier(self, tfm):
        fx = np.random.choice(("lowpass", "highpass", "bandpass", "bandstop"))

        if fx == "lowpass":
            cutoff_freq = np.random.uniform(*self.mod_config["LOWPASS_F"])
            tfm.append(["sinc", "-" + str(cutoff_freq)])
        elif fx == "highpass":
            cutoff_freq = np.random.uniform(*self.mod_config["HIGHPASS_F"])
            tfm.append(["sinc", str(cutoff_freq)])
        elif fx == "bandpass":
            cutoff_low = np.random.uniform(*self.mod_config["BANDPASS_F"])
            cutoff_high = cutoff_low * np.random.uniform(*self.mod_config["BANDPASS_W"])
            tfm.append(["sinc", str(cutoff_low) + "-" + str(cutoff_high)])
        elif fx == "bandstop":
            cutoff_low = np.random.uniform(*self.mod_config["BANDSTOP_F"])
            cutoff_high = cutoff_low * np.random.uniform(*self.mod_config["BANDSTOP_W"])
            tfm.append(["sinc", str(cutoff_high) + "-" + str(cutoff_low)])

        return tfm


class AnnotatedDataset(SoundDataset):
    def __init__(
        self,
        sounds: list[str],
        annotations: list[str],
        vocabulary: list[str],
        target: str,
        limits: Limits,
        value_nil: int = 0,
        ignore_silence: bool = True,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.annotations = annotations
        self.vocabulary = vocabulary
        self.target = target
        self.stressed = target in ("phones", "syllables") and any(
            is_stressed(w) for w in vocabulary
        )
        self.value_nil = value_nil
        self.limits = limits
        self.normalize = normalize
        self.spaced = " " in vocabulary
        self.include_na = "[UNK]" in vocabulary
        self.ignore_silence = ignore_silence
        assert self.target in ["chars", "phones", "words", "syllables"]

        self.key = dict([(key, i) for i, key in enumerate(vocabulary)])

    @torch.no_grad()
    def __getitem__(
        self,
        idx: int,
        waveform: bool = False,
    ) -> Optional[Sample]:
        sample = super().__getitem__(idx, waveform=waveform)
        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr

        y, intervals = self._annotation(self.annotations[idx])
        if len(y) == 0:
            return None

        intervals = [
            (start * sample.skew, stop * sample.skew) for start, stop in intervals
        ]
        sample.labels = y
        sample.intervals = intervals

        limits = self._get_limits(intervals)
        if limits:
            limits.time = int(limits.time * out_sr)
            sample.sound = sample.sound[:limits.time]
            sample.source = sample.source[:limits.time]
            sample.labels = sample.labels[:limits.tokens]
            sample.intervals = sample.intervals[:limits.tokens]

        return Sample(
            sound=sample.sound,
            source=sample.source,
            rate=out_sr,
            labels=y,
            intervals=intervals,
        )

    @property
    def num_classes(self):
        return len(self.vocabulary)

    def limit(self, limits: Limits):
        self.limits = limits
        return self

    def iterator(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        full_tensor=True,
        flat_labels=False,
    ):
        def collate_fn(samples: list[Sample]) -> Optional[SampleBatch]:
            if not samples or any(s is None for s in samples):
                return None

            xs, x0s, ys = zip(
                *[(s.sound, s.source, s.labels) for s in samples if s.labels is not None]
            )
            xlens = torch.tensor([len(x) for x in xs], dtype=torch.int)
            ylens = torch.tensor([len(y) for y in ys], dtype=torch.int)

            out_sr = self.in_sr if self.audio_proc is None else self.out_sr
            max_xlen = int(self.limits.time * out_sr if full_tensor else xlens.max())
            max_ylen = int(self.limits.tokens if full_tensor else ylens.max())

            xs = [_pad_axis(x, 0, max_xlen - len(x), axis=0) for x in xs]
            x0s = [_pad_axis(x0, 0, max_xlen - len(x0), axis=0) for x0 in x0s]
            if not flat_labels:
                ys = [_pad_axis(y, 0, max_ylen - len(y), axis=0) for y in ys]

            xs = torch.stack(xs, dim=self.batch_dim)
            x0s = torch.stack(x0s, dim=self.batch_dim)
            if flat_labels:
                ys = torch.cat(ys)
            else:
                ys = torch.stack(ys, dim=self.batch_dim)

            return SampleBatch(
                sounds=xs,
                sound_lens=xlens,
                sources=x0s,
                source_lens=xlens,  # TODO
                rate=samples[0].rate,
                labels=ys,
                label_lens=ylens,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def _spaced_textgrid(self, textgrid):
        it_phone = iter(textgrid["phones"])
        phone = next(it_phone)

        phones = []
        for word in textgrid["words"]:
            if word.text in ["", "sp", "spn", "sil", "<unk>"]:
                continue

            try:
                while phone.xmin < word.xmin - 1e-3:
                    phone = next(it_phone)
            except StopIteration:
                break

            if phone.xmin >= word.xmax + 1e-3:
                continue

            while phone.xmin < word.xmax - 1e-3:
                phones.append(phone)
                phone = next(it_phone)

            if self.spaced and len(phones) > 0:
                phones.append(textgrids.Interval(" ", phones[-1].xmax, phones[-1].xmax))

        return textgrids.Tier(phones)

    def _syllabized_textgrid(self, textgrid):
        it_phone = iter(textgrid["phones"])
        phone = next(it_phone)

        syllables = []
        for word in textgrid["words"]:
            if word.text in ["", "sp", "spn", "sil", "<unk>"]:
                continue

            try:
                while phone.xmin < word.xmin - 1e-3:
                    phone = next(it_phone)
            except StopIteration:
                break

            if phone.xmin >= word.xmax + 1e-3:
                continue

            phones = []
            while phone.xmin < word.xmax - 1e-3:
                phones.append(phone)
                phone = next(it_phone)

            syllbs = syllabize([p.text for p in phones], stressed=True)
            for syll in syllbs:
                nsyll = len(syll.split("-"))
                syllables.append(
                    textgrids.Interval(syll, phones[0].xmin, phones[nsyll - 1].xmax)
                )
                phones = phones[nsyll:]

            if self.spaced and len(syllbs) > 0:
                syllables.append(
                    textgrids.Interval(" ", syllables[-1].xmax, syllables[-1].xmax)
                )

        return textgrids.Tier(syllables)

    def _character_textgrid(self, textgrid):
        characters = []
        for word in textgrid["words"]:
            if word.text in ["", "sp", "spn", "sil", "<unk>"]:
                continue

            for char in word.text:
                characters.append(textgrids.Interval(char, word.xmin, word.xmax))

            if self.spaced:
                characters.append(textgrids.Interval(" ", word.xmax, word.xmax))

        return textgrids.Tier(characters)

    def _annotation(
        self,
        annotation: str,
        ignore_silence: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[float, float]]]:
        fmt, filepath = annotation.split(":")

        if fmt == "libri":
            # read annotation file
            textgrid = textgrids.TextGrid(filepath)
            if self.target == "phones":
                textgrid = self._spaced_textgrid(textgrid)
            elif self.target == "syllables":
                textgrid = self._syllabized_textgrid(textgrid)
            elif self.target == "chars":
                textgrid = self._character_textgrid(textgrid)
            else:
                textgrid = textgrid[self.target]
            # drop silence tokens
            textgrid = (
                [item for item in textgrid if item.text not in ["", "sp", "spn", "sil"]]
                if ignore_silence
                else textgrid
            )
            # transform to standard labels
            for item in textgrid:
                item.text = item.text.upper()

            target = [item.text for item in textgrid]
            interv = [(item.xmin, item.xmax) for item in textgrid]
        elif fmt == "swc":
            raise NotImplementedError(
                "Spoken Wikipedia annotation not yet implemented!"
            )
        elif fmt == "tedlium":
            raise NotImplementedError("TED-LIUM r3 annotation not yet implemented!")
        else:
            raise RuntimeError("Unknown annotation format:", fmt)

        if self.target in ["phones", "syllables"] and not self.stressed:
            target = [re.sub(r"([A-Z]+)[0-9]", r"\g<1>", token) for token in target]

        if self.target == "words" and self.normalize:
            expanded_target, expanded_interv = [], []
            for token, intv in zip(target, interv):
                subtokens = normalize_token(token)  # , self.vocabulary)
                expanded_target += subtokens
                expanded_interv += [
                    intv
                    if not is_subtoken(t)
                    else (intv[:1] * 2 if is_prefix(t) else intv[1:] * 2)
                    for t in subtokens
                ]
            target, interv = expanded_target, expanded_interv

        if self.key:
            encoded_target, encoded_interv = [], []
            for token, intv in zip(target, interv):
                if token in self.key:
                    pass
                elif self.include_na:
                    token = "[UNK]"
                else:
                    continue

                encoded_target.append(self.key[token])
                encoded_interv.append(intv)
            target, interv = encoded_target, encoded_interv

        return torch.tensor(target), interv

    def _get_limits(self, intervals: list[tuple[float, float]]) -> Limits:
        if self.limits.time < intervals[-1][1]:
            max_time = [end for _, end in intervals if end <= self.limits.time][-1]
        else:
            max_time = self.limits.time
        max_tokens = self.limits.tokens

        if len(intervals) > max_tokens:
            time_limit_equiv = intervals[max_tokens][0]
        else:
            time_limit_equiv = np.inf
        token_limit_equiv = len([1 for _, end in intervals if end <= max_time])

        max_time = min(max_time, time_limit_equiv)
        max_tokens = min(max_tokens, token_limit_equiv)

        return Limits(time=max_time, tokens=max_tokens)


class AlignedDataset(AnnotatedDataset):
    @torch.no_grad()
    def __getitem__(self, idx: int, waveform: bool = False) -> Optional[Sample]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None or sample.intervals is None:
            return None

        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        sample.label_locs = torch.as_tensor(
            [int(intv[0] * out_sr) for intv in sample.intervals], dtype=torch.long
        )

        return sample

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate_fn(samples: list[Sample]) -> Optional[SampleBatch]:
            samples = [s for s in samples if s is not None and s.label_locs is not None]
            if not samples:
                return None

            xs, ys, ylocs = zip(*[(s.sound, s.labels, s.label_locs) for s in samples])
            xlens = torch.tensor([len(x) for x in xs], dtype=torch.int)
            ylens = torch.tensor([len(y) for y in ys], dtype=torch.int)

            max_xlen = int(
                self.limits.time * self.out_sr if self.limits.time else xlens.max()
            )
            max_ylen = int(self.limits.tokens if self.limits.tokens else ylens.max())
            xs = torch.stack(
                [_pad_axis(x, 0, max_xlen - len(x), axis=0) for x in xs],
                dim=self.batch_dim,
            )
            ys = torch.stack(
                [_pad_axis(y, 0, max_ylen - len(y), axis=0) for y in ys],
                dim=self.batch_dim,
            )
            ylocs = torch.stack(
                [_pad_axis(loc, 0, max_ylen - len(loc), axis=0) for loc in ylocs],
                dim=self.batch_dim,
            )

            return SampleBatch(
                sounds=xs,
                sound_lens=xlens,
                sources=xs,
                source_lens=xlens,
                rate=samples[0].rate,
                labels=ys,
                label_locs=ylocs,
                label_lens=ylens,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


class TokenizedDataset(AnnotatedDataset):
    def __init__(
        self,
        sounds: list[str],
        annotations: list[str],
        vocabulary: list[str],
        target: str,
        duration: float = 1,
        scale: bool = False,
        context: tuple[int, int] = (0, 0),
        alignment: str = "left",
        drop_modifiers: bool = True,
        **kwargs,
    ):
        super().__init__(sounds, annotations, vocabulary, target, **kwargs)
        self.duration = duration
        self.scale = scale
        self.context = context
        self.alignment = alignment
        self.drop_modifiers = drop_modifiers
        assert self.alignment in ["left", "center", "right"]
        assert context[0] >= 0 and context[1] >= 0

    def __getitem__(self, idx: int, waveform: bool = False) -> Optional[SampleBatch]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None or sample.labels is None or sample.intervals is None:
            return None

        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr

        xs, xlens = [], []
        pre_ctx, post_ctx = self.context
        for start, stop in sample.intervals:
            length = stop - start
            ctx_start = int((start - length * pre_ctx) * out_sr)
            ctx_end = int((stop + length * post_ctx) * out_sr)
            x = sample.sound[max(ctx_start, 0) : ctx_end]
            x = _pad_axis(x, -ctx_start, ctx_end - len(sample.sound))
            xs.append(x)
            xlens.append(len(x))

        fix_t = int(self.duration * (1 + pre_ctx + post_ctx) * out_sr)
        freqs = np.linspace(1, self.freqbins, self.freqbins)

        ys = sample.labels.int()
        if not self.scale:
            will_fit = [i for i, length in enumerate(xlens) if length <= fix_t]
            xs = [xs[i] for i in will_fit]
            xlens = [xlens[i] for i in will_fit]
            ys = ys[will_fit]
        if len(xs) == 0:
            return None

        for i, x in enumerate(xs):
            if self.scale or len(x) > fix_t:
                t0 = np.linspace(1, len(x), len(x))
                t1 = np.linspace(1, len(x), fix_t)

                if waveform or self.audio_proc is None:
                    # TODO
                    # x[i] = librosa.effects.time_stretch(xi, xi.shape[0]/fix_t)[:fix_t]
                    raise NotImplementedError()
                else:
                    xs[i] = sp.interpolate.RectBivariateSpline(t0, freqs, x)(t1, freqs)
                xlens[i] = fix_t
            else:
                diff = fix_t - len(x)

                if self.alignment == "left":
                    pre_t, post_t = 0, diff
                elif self.alignment == "right":
                    pre_t, post_t = diff, 0
                else:
                    pre_t, post_t = math.floor(diff / 2), math.ceil(diff / 2)

                xs[i] = _pad_axis(x, pre_t, post_t, axis=0)
                xlens[i] = len(x) - post_t

        xs = torch.stack(xs, dim=self.batch_dim).float()
        xlens = torch.tensor(xlens, dtype=torch.int)

        # TODO I don't remember what this was supposed to do
        if self.drop_modifiers:
            xs = xs[xlens != 0]
            ys = ys[xlens != 0]
            xlens = xlens[xlens != 0]

        return SampleBatch(
            sounds=xs,
            sound_lens=xlens,
            sources=xs,
            source_lens=xlens,
            rate=sample.rate,
            labels=ys,
        )

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate_fn(samples: list[SampleBatch]) -> Optional[SampleBatch]:
            samples = [s for s in samples if s is not None and s.labels is not None]
            if not samples:
                return None

            xs, ys, xlens = zip(
                *[(s.sounds, s.labels, s.sound_lens) for s in samples if s.labels is not None]
            )

            xs = torch.cat(xs, dim=self.batch_dim)
            ys = torch.cat(ys, dim=self.batch_dim)
            xlens = torch.cat(xlens)

            return SampleBatch(
                sounds=xs,
                sound_lens=xlens,
                sources=xs,
                source_lens=xlens,
                rate=samples[0].rate,
                labels=ys,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def take(self, n: int, shuffle: bool = False, num_workers: int = 0):
        count = 0
        xs, ys = [], []

        pbar = tqdm(range(n))
        it = iter(self.iterator(shuffle=shuffle, num_workers=num_workers))
        while count < n:
            try:
                x, y = next(it)
            except StopIteration:
                break

            if x is None or y is None:
                continue

            xs.append(x)
            ys.append(y)
            count += len(x)
            pbar.update(len(x))

        xs = torch.cat(xs, dim=0)[:n]
        ys = torch.cat(ys, dim=0)[:n]

        return xs, ys

    def take_each(self, n: int, shuffle: bool = False, num_workers: int = 0):
        count = [0] * len(self.vocabulary)
        xs, ys = [], []

        min_count = 0
        pbar = tqdm(range(n))
        it = iter(self.iterator(shuffle=shuffle, num_workers=num_workers))
        while min_count < n:
            try:
                x, y = next(it)
            except StopIteration:
                break

            if x is None or y is None:
                continue

            for xi, yi in zip(x, y):
                if count[yi] >= n:
                    continue

                xs.append(xi)
                ys.append(yi)
                count[yi] += 1

            if min(count) > min_count:
                min_count = min(count)
                pbar.n = min_count
                pbar.refresh()

        xs = torch.stack(xs, dim=self.batch_dim)
        ys = torch.stack(ys, dim=self.batch_dim)

        return xs, ys

    def take_statistics(
        self,
        n: int = int(1e9),
        shuffle: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> dict:
        count = 0
        stats = {"length": [], "identity": []}

        pbar = tqdm(range(n))
        it = iter(
            self.iterator(
                shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
            )
        )
        out_sr = self.in_sr if self.audio_proc is None else self.out_sr

        while count < n:
            try:
                x, y, length = next(it)
            except StopIteration:
                break

            if x is None or y is None:
                continue

            stats["identity"].append(
                np.array([self.vocabulary[_] for _ in y], dtype=object)
            )
            stats["length"].append(length.numpy() / out_sr)
            count += len(x)
            pbar.update(len(x))

        for k in stats:
            stats[k] = np.concatenate(stats[k], axis=0)[:n]

        return stats


class BlockDataset(SoundDataset):
    def __init__(
        self,
        sounds: list[str],
        max_time: float = 2.5,
        block_min: float = 0.01,
        max_step: Optional[float] = None,
        batch_first: bool = False,
        prepad: bool = False,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.limits.time = max_time
        self.block_min = block_min
        self.batch_first = batch_first
        self.max_step = max_time if max_step is None else max_step
        self.prepad = prepad

    def __getitem__(self, idx: int, waveform: bool = False) -> Optional[SampleBatch]:
        sample = super().__getitem__(idx, waveform=waveform)
        maxblk = int(self.limits.time * sample.rate)
        minblk = int(self.block_min * sample.rate)
        maxstp = int(self.max_step * sample.rate)

        if self.prepad:
            sample.sound = _pad_axis(sample.sound, maxblk - 1, 0, axis=0)
            sample.source = _pad_axis(sample.source, maxblk - 1, 0, axis=0)

        index = 0
        xs, x0s, xlens = [], [], []
        while index < len(sample.sound):
            length = np.random.randint(minblk, maxblk + 1)

            if index + length > len(sample.sound):
                length = len(sample.sound) - index
            if length < minblk:
                break

            xs.append(sample.sound[index : index + length])
            x0s.append(sample.source[index : index + length])
            xlens.append(length)
            index += min(length, maxstp)

        if len(xs) == 0:
            return None

        xs = torch.stack(
            [_pad_axis(x, 0, maxblk - len(x), axis=self.batch_dim) for x in xs]
        )
        x0s = torch.stack(
            [_pad_axis(x0, 0, maxblk - len(x0), axis=self.batch_dim) for x0 in x0s]
        )
        xlens = torch.tensor(xlens, dtype=torch.int)

        return SampleBatch(
            sounds=xs,
            sound_lens=xlens,
            sources=x0s,
            source_lens=xlens,
            rate=sample.rate,
        )

    def iterator(
        self,
        batch_size=1,
        batch_max=None,
        shuffle=False,
        num_workers=0,
    ):
        def collate_fn(samples: list[SampleBatch]) -> Optional[SampleBatch]:
            samples = [s for s in samples if s is not None]
            if not samples:
                return None

            xs, xlens, x0s = zip(
                *[(s.sounds, s.sound_lens, s.sources) for s in samples]
            )
            xs = torch.cat(xs, dim=self.batch_dim)
            x0s = torch.cat(x0s, dim=self.batch_dim)
            xlens = torch.cat(xlens)

            if batch_max is not None:
                xs = xs[:batch_max] if self.batch_dim == 0 else xs[:, :batch_max]
                x0s = x0s[:batch_max] if self.batch_dim == 0 else x0s[:, :batch_max]
                xlens = xlens[:batch_max]

            return SampleBatch(
                sounds=xs,
                sound_lens=xlens,
                sources=x0s,
                source_lens=xlens,
                rate=samples[0].rate,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


class SequenceDataset(AnnotatedDataset):
    def __init__(
        self,
        sounds,
        annotations,
        vocabulary=[],
        target="words",
        seq_size=20,
        seq_min=1,
        seq_time=8.0,
        seq_overlap=False,
        check_boundaries=True,
        **kwargs,
    ):
        super().__init__(sounds, annotations, vocabulary, target=target, **kwargs)
        self.seq_size = seq_size
        self.seq_min = seq_min if seq_min else seq_size
        self.seq_time = seq_time
        self.seq_overlap = seq_overlap
        self.check_boundaries = check_boundaries

    @torch.no_grad()
    def __getitem__(self, idx: int, waveform: bool = False) -> Optional[SampleBatch]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None or sample.labels is None or sample.intervals is None:
            return None

        maxblk = int(self.seq_time * sample.rate)

        index = 0
        xs, ys, xlens, ylens, x0s, xpos, ypos = [], [], [], [], [], [], []
        while index < len(sample.intervals):
            if self.check_boundaries and is_postfix(
                self.vocabulary[sample.labels[index]]
            ):
                index += 1
                continue

            length = np.random.randint(self.seq_min, self.seq_size + 1)

            if index + length > len(sample.intervals):
                length = len(sample.intervals) - index
            if length < self.seq_min:
                break

            if self.check_boundaries:
                init_length = length

                while (
                    index + length <= len(sample.intervals)
                    and length <= self.seq_size
                    and is_prefix(self.vocabulary[sample.labels[index + length - 1]])
                ):
                    length += 1
                if length > self.seq_size or index + length > len(sample.intervals):
                    length = init_length

                while length >= self.seq_min and is_prefix(
                    self.vocabulary[sample.labels[index + length - 1]]
                ):
                    length -= 1
                if length < self.seq_min:
                    index += 1
                    continue

            start = max(int(sample.intervals[index][0] * sample.rate) - 1, 0)
            stop = int(sample.intervals[index + length - 1][1] * sample.rate) + 2
            while stop - start > maxblk and length > self.seq_min:
                length -= 1
                stop = int(sample.intervals[index + length - 1][1] * sample.rate) + 2
            if stop - start > maxblk:
                index += 1
                continue

            xs.append(sample.sound[start:stop])
            ys.append(sample.labels[index : index + length])
            xlens.append(stop - start)
            ylens.append(length)
            x0s.append(sample.source[start:stop])
            xpos.append((start, stop))
            ypos.append((index, index + length))
            index += 1 if self.seq_overlap else length

        if len(xs) == 0:
            return None

        xs = torch.stack(
            [_pad_axis(x, 0, maxblk - len(x), axis=0) for x in xs], dim=self.batch_dim
        )
        x0s = torch.stack(
            [_pad_axis(x, 0, maxblk - len(x), axis=0) for x in x0s], dim=self.batch_dim
        )
        ys = torch.stack(
            [_pad_axis(y, 0, self.seq_size - len(y), axis=0) for y in ys],
            dim=self.batch_dim,
        )
        xlens = torch.tensor(xlens, dtype=torch.int)
        ylens = torch.tensor(ylens, dtype=torch.int)
        xpos = torch.tensor(xpos, dtype=torch.int)
        ypos = torch.tensor(ypos, dtype=torch.int)

        return SampleBatch(
            sounds=xs,
            sound_lens=xlens,
            # sound_locs=xpos,
            sources=x0s,
            source_lens=xlens,
            # source_locs=xpos,
            rate=sample.rate,
            labels=ys,
            label_locs=ypos,
            label_lens=ylens,
        )

    def iterator(
        self,
        batch_size=1,
        batch_max=None,
        shuffle=False,
        num_workers=0,
    ):
        def collate_fn(samples: list[SampleBatch]) -> Optional[SampleBatch]:
            samples = [s for s in samples if s is not None]
            if not samples:
                return None

            # xs, ys, xlens, ylens, x0s, xpos, ypos = zip(*[
            xs, xlens, x0s, ys, ylens, ypos = zip(
                *[
                    (
                        s.sounds,
                        s.sound_lens,
                        s.sources,
                        s.labels,
                        s.label_lens,
                        s.label_locs,
                    )
                    for s in samples
                ]
            )

            xs = torch.cat(xs, dim=self.batch_dim)
            x0s = torch.cat(x0s, dim=self.batch_dim)
            ys = torch.cat(ys, dim=self.batch_dim)
            xlens = torch.cat(xlens)
            ylens = torch.cat(ylens)
            # xpos = torch.cat(xpos)
            ypos = torch.cat(ypos)

            # batch_size = xs.shape[self.batch_dim]
            if batch_max is not None:
                xs = xs[:batch_max] if self.batch_dim == 0 else xs[:, :batch_max]
                x0s = x0s[:batch_max] if self.batch_dim == 0 else x0s[:, :batch_max]
                ys = ys[:batch_max] if self.batch_dim == 0 else ys[:, :batch_max]
                xlens = xlens[:batch_max]
                ylens = ylens[:batch_max]
                # xpos = xpos[:batch_max]
                ypos = ypos[:batch_max]

            return SampleBatch(
                sounds=xs,
                sound_lens=xlens,
                # sound_locs=xpos,
                sources=x0s,
                source_lens=xlens,
                rate=samples[0].rate,
                labels=ys,
                label_lens=ylens,
                label_locs=ypos,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )  # , pin_memory=True)


class SymmetricTokenDataset(AnnotatedDataset):
    def __init__(
        self,
        sounds,
        annotations,
        vocabulary,
        target="words",
        base_rate="moderate",
        context=6,
        alignment="center",
        scale_factor=np.sqrt(2),
        # filter_vocab=True,
        rate="base",
        **kwargs,
    ):
        assert target in ["words", "syllables", "phones"]
        assert base_rate in [
            "fast",
            "moderate",
            "slow",
            "no_ext",
            "any",
            "c10",
            "c20",
            "c30",
            "c50",
            "c90",
            "c95",
        ]
        assert alignment in ["left", "center", "right"]
        assert rate in ["base", "faster", "slower"]

        filename = f"librispeech-statistics-{target}.npy"
        if os.path.exists(filename):
            stats = np.load(filename, allow_pickle=True).item()
        elif os.path.exists(os.path.join("stats", filename)):
            stats = np.load(os.path.join("stats", filename), allow_pickle=True).item()
        else:
            raise RuntimeError(f'Unit statistic file "{filename}" not found.')

        lengths = {}
        for k, length in zip(stats["identity"], stats["length"]):
            if k in lengths:
                lengths[k].append(length)
            else:
                lengths[k] = [length]

        alpha = 0.9
        max_stretch = scale_factor**2
        var_range = {
            "fast": (0.05, 0.15),
            "moderate": (0.45, 0.55),
            "slow": (0.85, 0.95),
            "no_ext": (0.05, 0.95),
            "any": (0.00, 1.00),
            "c10": (0.45, 0.55),
            "c20": (0.4, 0.6),
            "c30": (0.35, 0.65),
            "c50": (0.25, 0.75),
            "c90": (0.05, 0.95),
            "c95": (0.025, 0.975),
        }[base_rate]
        vocabulary = [
            k
            for k, length in lengths.items()
            if (vocabulary is None or k in vocabulary)
            and len(length) >= 100
            and (
                np.quantile(length, 0.5 + alpha / 2)
                / np.quantile(length, 0.5 - alpha / 2)
                >= max_stretch
            )
        ]
        acc_range = dict(
            [
                (
                    k,
                    (
                        np.quantile(lengths[k], var_range[0]),
                        np.quantile(lengths[k], var_range[1]),
                    ),
                )
                for k in vocabulary
            ]
        )

        super().__init__(sounds, annotations, vocabulary, target, **kwargs)
        self.base_rate = base_rate
        self.context = context
        self.alignment = alignment
        self.scale_factor = scale_factor
        self.var_range = var_range
        self.acc_range = acc_range
        self.rate = rate

    def __getitem__(self, idx: int, waveform: bool = False) -> Optional[SampleBatch]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None or sample.labels is None or sample.intervals is None:
            return None

        yint_filt = [
            (token, intv)
            for token, intv in zip(sample.labels, sample.intervals)
            if (self.acc_range[self.vocabulary[token]][0] <= intv[1] - intv[0])
            and (intv[1] - intv[0] <= self.acc_range[self.vocabulary[token]][1])
        ]
        if len(yint_filt) == 0:
            return None
        ys, intervals = zip(*yint_filt)

        intervals = [
            (int(start * sample.rate), int(stop * sample.rate))
            for start, stop in intervals
        ]
        # durations = [stop - start for start, stop in intervals]
        centers = [(start + stop) // 2 for start, stop in intervals]

        context = int(self.context * sample.rate)
        pre_context, post_context = math.floor(context / 2), math.ceil(context / 2)
        intervals = [
            (center - pre_context, center + post_context) for center in centers
        ]
        yint_filt = [
            (token, intv)
            for token, intv in zip(ys, intervals)
            if intv[0] >= 0 and intv[1] <= len(sample.sound)
        ]
        if len(yint_filt) == 0:
            return None
        ys, intervals = zip(*yint_filt)

        xs = [sample.sound[start:stop] for start, stop in intervals]
        scale_factor = {
            "faster": 1.0 / self.scale_factor,
            "base": 1.0,
            "slower": self.scale_factor,
        }[self.rate]
        target_t = int(context * scale_factor)

        if target_t != context:
            t0 = np.linspace(1, context, context)
            t1 = np.linspace(1, context, target_t)
            freqs = np.linspace(1, self.freqbins, self.freqbins)

            for i, xi in enumerate(xs):
                if waveform or self.audio_proc is None:
                    # x[i] = librosa.effects.time_stretch(xi, context/target_t)
                    pass
                else:
                    xs[i] = sp.interpolate.RectBivariateSpline(t0, freqs, xi)(t1, freqs)

        xlens = torch.tensor([len(x) for x in xs], dtype=torch.int)
        xs = torch.stack(xs, dim=0).float()
        ys = torch.stack(ys, dim=0).int()

        xs = xs[xlens != 0]
        ys = ys[xlens != 0]
        xlens = xlens[xlens != 0]

        return SampleBatch(
            sounds=xs,
            sound_lens=xlens,
            sources=xs,
            source_lens=xlens,
            rate=sample.rate,
            labels=ys,
        )

    def iterator(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    ):
        def collate_fn(samples: list[SampleBatch]) -> Optional[SampleBatch]:
            if any(s is None or s.labels is None for s in samples):
                return None

            xs, ys, xlens = zip(*[(s.sounds, s.labels, s.sound_lens) for s in samples])
            xs = torch.cat(xs, dim=self.batch_dim)
            ys = torch.cat(ys, dim=self.batch_dim)
            xlens = torch.cat(xlens)

            return SampleBatch(
                sounds=xs,
                sound_lens=xlens,
                sources=xs,
                source_lens=xlens,
                rate=samples[0].rate,
                labels=ys,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def take(
        self,
        n=int(1e9),
        shuffle=False,
        batch_size=1,
        num_workers=0,
        max_token_per_word=None,
        return_labels=True,
    ):
        token_per_word = dict()
        xs, ys = [], []

        pbar = tqdm(range(n))
        it = iter(
            self.iterator(
                shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
            )
        )
        while len(xs) < n:
            try:
                x, y = next(it)
            except StopIteration:
                break

            if x is None or y is None:
                continue

            batch_count = 0
            for xi, yi in zip(x, y):
                yi = yi.item()
                if yi not in token_per_word:
                    token_per_word[yi] = 0

                if (
                    max_token_per_word is None
                    or token_per_word[yi] < max_token_per_word
                ):
                    xs.append(xi)
                    ys.append(yi)
                    token_per_word[yi] += 1
                    batch_count += 1

            pbar.update(batch_count)

        xs = torch.stack(xs, dim=0)[:n]
        ys = (
            np.array([self.vocabulary[y] for y in ys])[:n]
            if return_labels
            else torch.tensor(ys)[:n]
        )

        return xs, ys


class MultiAnnotatedDataset(SoundDataset):
    def __init__(
        self,
        sounds,
        annotations,
        vocabulary,
        stressed=False,
        value_nil=0,
        ignore_silence=True,
        normalize=False,
        batch_first=True,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.annotations = annotations
        self.vocabulary = vocabulary
        self.stressed = stressed
        self.value_nil = value_nil
        self.normalize = normalize
        self.spaced = {k: (" " in v) for k, v in vocabulary.items()}
        self.include_na = {k: ("[UNK]" in v) for k, v in vocabulary.items()}
        self.ignore_silence = ignore_silence
        self.batch_first = batch_first
        self.targets = ["phones", "syllables", "words"]
        self.key = {
            k: {tok: i for i, tok in enumerate(v)} for k, v in vocabulary.items()
        }

    @torch.no_grad()
    def __getitem__(self, idx: int, waveform: bool = False):
        sample = super().__getitem__(idx, waveform=waveform)

        p = self._annotation(self.annotations[idx], "phones", skew=sample.skew)
        s = self._annotation(self.annotations[idx], "syllables", skew=sample.skew)
        w = self._annotation(self.annotations[idx], "words", skew=sample.skew)

        return sample.sound, p, s, w

    def _spaced_textgrid(self, textgrid):
        it_phone = iter(textgrid["phones"])
        phone = next(it_phone)

        phones = []
        for word in textgrid["words"]:
            if word.text in ["", "sp", "spn", "sil", "<unk>"]:
                continue

            try:
                while phone.xmin < word.xmin - 1e-3:
                    phone = next(it_phone)
            except StopIteration:
                break

            if phone.xmin >= word.xmax + 1e-3:
                continue

            while phone.xmin < word.xmax - 1e-3:
                phones.append(phone)
                phone = next(it_phone)

            if self.spaced["phones"] and len(phones) > 0:
                phones.append(textgrids.Interval(" ", phones[-1].xmax, phones[-1].xmax))

        return textgrids.Tier(phones)

    def _syllabized_textgrid(self, textgrid):
        it_phone = iter(textgrid["phones"])
        phone = next(it_phone)

        syllables = []
        for word in textgrid["words"]:
            if word.text in ["", "sp", "spn", "sil", "<unk>"]:
                continue

            try:
                while phone.xmin < word.xmin - 1e-3:
                    phone = next(it_phone)
            except StopIteration:
                break

            if phone.xmin >= word.xmax + 1e-3:
                continue

            phones = []
            while phone.xmin < word.xmax - 1e-3:
                phones.append(phone)
                phone = next(it_phone)

            syllbs = syllabize([p.text for p in phones], stressed=True)
            for syll in syllbs:
                nsyll = len(syll.split("-"))
                syllables.append(
                    textgrids.Interval(syll, phones[0].xmin, phones[nsyll - 1].xmax)
                )
                phones = phones[nsyll:]

            if self.spaced["syllables"] and len(syllbs) > 0:
                syllables.append(
                    textgrids.Interval(" ", syllables[-1].xmax, syllables[-1].xmax)
                )

        return textgrids.Tier(syllables)

    def _annotation(
        self, annotation, target_type, ignore_silence=True, skew=None
    ) -> tuple[torch.Tensor, list[tuple[float, float]]]:
        fmt, filepath = annotation.split(":")

        if fmt == "libri":
            # read annotation file
            textgrid = textgrids.TextGrid(filepath)
            if target_type == "phones":
                textgrid = self._spaced_textgrid(textgrid)
            elif target_type == "syllables":
                textgrid = self._syllabized_textgrid(textgrid)
            else:
                textgrid = textgrid[target_type]
            # drop silence tokens
            textgrid = (
                [item for item in textgrid if item.text not in ["", "sp", "spn", "sil"]]
                if ignore_silence
                else textgrid
            )
            # transform to standard labels
            for item in textgrid:
                item.text = item.text.upper()

            target = [item.text for item in textgrid]
            interv = [(item.xmin, item.xmax) for item in textgrid]
        elif fmt == "swc":
            raise NotImplementedError(
                "Spoken Wikipedia annotation not yet implemented!"
            )
        elif fmt == "tedlium":
            raise NotImplementedError("TED-LIUM r3 annotation not yet implemented!")
        else:
            raise RuntimeError("Unknown annotation format:", fmt)

        if target_type in ["phones", "syllables"] and not self.stressed:
            target = [re.sub(r"([A-Z]+)[0-9]", r"\g<1>", token) for token in target]

        if target_type == "words" and self.normalize:
            expanded_target, expanded_interv = [], []
            for token, intv in zip(target, interv):
                subtokens = normalize_token(token)  # , self.vocabulary)
                expanded_target += subtokens
                expanded_interv += [
                    intv
                    if not is_subtoken(t)
                    else (intv[:1] * 2 if is_prefix(t) else intv[1:] * 2)
                    for t in subtokens
                ]
            target, interv = expanded_target, expanded_interv

        if self.key[target_type]:
            encoded_target, encoded_interv = [], []
            for token, intv in zip(target, interv):
                if token in self.key[target_type]:
                    pass
                elif self.include_na[target_type]:
                    token = "[UNK]"
                else:
                    continue

                encoded_target.append(self.key[target_type][token])
                encoded_interv.append(intv)
            target, interv = encoded_target, encoded_interv

        if skew is not None:
            interv = [(start * skew, stop * skew) for start, stop in interv]

        return torch.tensor(target), interv


class MultiSymmetricTokenDataset(MultiAnnotatedDataset):
    def __init__(
        self,
        sounds,
        annotations,
        vocabulary,
        base_rate="moderate",
        context=6,
        alignment="center",
        scale_factor=np.sqrt(2),
        # filter_vocab=True,
        label_sr=100,
        **kwargs,
    ):
        assert base_rate in [
            "fast",
            "moderate",
            "slow",
            "no_ext",
            "any",
            "c10",
            "c20",
            "c30",
            "c50",
            "c90",
            "c95",
        ]
        assert alignment in ["left", "center", "right"]

        filename = "librispeech-statistics-words.npy"
        if os.path.exists(filename):
            stats = np.load(filename, allow_pickle=True).item()
        elif os.path.exists(os.path.join("stats", filename)):
            stats = np.load(os.path.join("stats", filename), allow_pickle=True).item()
        else:
            raise RuntimeError(f'Unit statistic file "{filename}" not found.')
        lengths = dict()
        for k, length in zip(stats["identity"], stats["length"]):
            if k in lengths:
                lengths[k].append(length)
            else:
                lengths[k] = [length]

        alpha = 0.9
        max_stretch = scale_factor**2
        var_range = {
            "fast": (0.05, 0.15),
            "moderate": (0.45, 0.55),
            "slow": (0.85, 0.95),
            "no_ext": (0.05, 0.95),
            "any": (0.00, 1.00),
            "c10": (0.45, 0.55),
            "c20": (0.4, 0.6),
            "c30": (0.35, 0.65),
            "c50": (0.25, 0.75),
            "c90": (0.05, 0.95),
            "c95": (0.025, 0.975),
        }[base_rate]
        anchor_vocabulary = [
            k
            for k, length in lengths.items()
            if (vocabulary["words"] is None or k in vocabulary["words"])
            and len(length) >= 100
            and (
                np.quantile(length, 0.5 + alpha / 2)
                / np.quantile(length, 0.5 - alpha / 2)
                >= max_stretch
            )
        ]
        acc_range = {
            k: (
                np.quantile(lengths[k], var_range[0]),
                np.quantile(lengths[k], var_range[1]),
            )
            for k in anchor_vocabulary
        }

        super().__init__(sounds, annotations, vocabulary, **kwargs)
        self.base_rate = base_rate
        self.context = context
        self.alignment = alignment
        self.scale_factor = scale_factor
        self.var_range = var_range
        self.acc_range = acc_range
        self.label_sr = label_sr
        self.anchor_vocabulary = anchor_vocabulary

    def __getitem__(self, idx: int, waveform: bool = False):
        x, p, s, w = super().__getitem__(idx, waveform=waveform)
        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        if x is None:
            return (None,) * 5

        anchors = [
            (tok, intv)
            for tok, intv in zip(*w)
            if self.vocabulary["words"][tok] in self.anchor_vocabulary
            and self.acc_range[self.vocabulary["words"][tok]][0] <= intv[1] - intv[0]
            and intv[1] - intv[0] <= self.acc_range[self.vocabulary["words"][tok]][1]
        ]
        if len(anchors) == 0:
            return (None,) * 5

        y, intervals = zip(*anchors)

        y_len = int(len(x) / out_sr * self.label_sr)
        p_vec = torch.zeros(y_len, dtype=torch.int)
        s_vec = torch.zeros(y_len, dtype=torch.int)
        w_vec = torch.zeros(y_len, dtype=torch.int)
        for tok, (start, stop) in zip(*p):
            p_vec[int(start * self.label_sr) : int(stop * self.label_sr)] = tok
        for tok, (start, stop) in zip(*s):
            s_vec[int(start * self.label_sr) : int(stop * self.label_sr)] = tok
        for tok, (start, stop) in zip(*w):
            w_vec[int(start * self.label_sr) : int(stop * self.label_sr)] = tok

        x_intervals = [
            (int(start * out_sr), int(stop * out_sr)) for start, stop in intervals
        ]
        y_intervals = [
            (int(start * self.label_sr), int(stop * self.label_sr))
            for start, stop in intervals
        ]

        # x_durations = [stop - start for start, stop in x_intervals]
        # y_durations = [stop - start for start, stop in y_intervals]

        x_centers = [(start + stop) // 2 for start, stop in x_intervals]
        y_centers = [(start + stop) // 2 for start, stop in y_intervals]

        x_context = int(self.context * out_sr)
        y_context = int(self.context * self.label_sr)

        x_pre_context, x_post_context = (
            math.floor(x_context / 2),
            math.ceil(x_context / 2),
        )
        y_pre_context, y_post_context = (
            math.floor(y_context / 2),
            math.ceil(y_context / 2),
        )

        x_intervals = [
            (center - x_pre_context, center + x_post_context) for center in x_centers
        ]
        y_intervals = [
            (center - y_pre_context, center + y_post_context) for center in y_centers
        ]

        anchors = [
            (token, x_intv, y_intv)
            for token, x_intv, y_intv in zip(y, x_intervals, y_intervals)
            if x_intv[0] >= 0
            and x_intv[1] <= len(x)
            and y_intv[0] >= 0
            and y_intv[1] <= y_len
        ]
        if len(anchors) == 0:
            return (None,) * 5
        y, x_intervals, y_intervals = zip(*anchors)

        x = [x[start:stop] for start, stop in x_intervals]
        p = [p_vec[start:stop] for start, stop in y_intervals]
        s = [s_vec[start:stop] for start, stop in y_intervals]
        w = [w_vec[start:stop] for start, stop in y_intervals]
        nonempty = [len(_) > 0 for _ in x]
        if len(nonempty) == 0:
            return (None,) * 5

        x = torch.stack(x)[nonempty]
        y = torch.tensor(y)[nonempty]
        p = torch.stack(p)[nonempty]
        s = torch.stack(s)[nonempty]
        w = torch.stack(w)[nonempty]

        return x, y, p, s, w

    def iterator(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # flat_labels=False
    ):
        def collate_fn(xpsw):
            xs, ys, ps, ss, ws = zip(*xpsw)
            if any(y is None for y in ys):
                return (None,) * 5

            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            ps = np.concatenate(ps, axis=0)
            ss = np.concatenate(ss, axis=0)
            ws = np.concatenate(ws, axis=0)

            xs = torch.as_tensor(xs)
            ys = torch.as_tensor(ys)
            ps = torch.as_tensor(ps)
            ss = torch.as_tensor(ss)
            ws = torch.as_tensor(ws)

            if not self.batch_first:
                xs = xs.transpose(0, 1).contiguous()
                ys = ys.transpose(0, 1).contiguous()
                ps = ps.transpose(0, 1).contiguous()
                ss = ss.transpose(0, 1).contiguous()
                ws = ws.transpose(0, 1).contiguous()

            return xs, ys, ps, ss, ws

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def take(
        self,
        n=int(1e9),
        shuffle=False,
        batch_size=1,
        num_workers=0,
        max_token_per_word=None,
        return_labels=True,
    ):
        token_per_word = dict()
        xs, ys, ps, ss, ws = [], [], [], [], []

        pbar = tqdm(range(n))
        it = iter(
            self.iterator(
                shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
            )
        )
        while len(xs) < n:
            try:
                x, y, p, s, w = next(it)
            except StopIteration:
                break

            if x is None or y is None:
                continue

            batch_count = 0
            for xi, yi, pi, si, wi in zip(x, y, p, s, w):
                yi = yi.item()
                if yi not in token_per_word:
                    token_per_word[yi] = 0

                if (
                    max_token_per_word is None
                    or token_per_word[yi] < max_token_per_word
                ):
                    xs.append(xi)
                    ys.append(yi)
                    ps.append(pi)
                    ss.append(si)
                    ws.append(wi)
                    token_per_word[yi] += 1
                    batch_count += 1

            pbar.update(batch_count)

        xs = torch.stack(xs, dim=0)[:n]
        ys = (
            np.array([self.vocabulary["words"][y] for y in ys])[:n]
            if return_labels
            else torch.tensor(ys)[:n]
        )
        ps = torch.stack(ps)[:n]
        ss = torch.stack(ss)[:n]
        ws = torch.stack(ws)[:n]

        return xs, ys, ps, ss, ws


def _pad_axis(
    array: torch.Tensor, pre: int = 0, post: int = 0, axis: int = 0
) -> torch.Tensor:
    pre, post = max(pre, 0), max(post, 0)
    if pre == 0 and post == 0:
        return array

    npad = [(pre, post) if i == axis else (0, 0) for i in range(array.ndim)]
    npad = [n for a, b in npad[::-1] for n in (a, b)]

    return torch.nn.functional.pad(array, npad)
