import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import textgrids
import torch
import torchaudio
from scipy.interpolate import RectBivariateSpline
from torch import Tensor
from torchaudio import load as load_audio
from torchaudio.sox_effects import apply_effects_tensor as apply_sox_effects
from tqdm import tqdm

from .lexicon import (
    is_postfix,
    is_prefix,
    is_stressed,
    is_subtoken,
    normalize_token,
    syllabize,
)
from .limits import Limits

torchaudio.set_audio_backend("sox_io")

Interval = list[tuple[float, float]]


@dataclass
class SoundSample:
    """
    sound: (audio: Tensor, sr: int)
    source: (audio: Tensor, sr: int)
    skew: float
    name: str
    """

    sound: tuple[Tensor, int]
    source: tuple[Tensor, int]
    skew: float
    name: str


@dataclass
class AnnotatedSample:
    """
    sound: (audio: Tensor, sr: int, intervals: Interval)
    source: (audio: Tensor, sr: int, intervals: Interval)
    label: Tensor
    skew: float
    name: str
    """

    sound: tuple[Tensor, int, Interval]
    source: tuple[Tensor, int, Interval]
    label: Tensor
    skew: float
    name: str


@dataclass
class TokenizedSample:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor)
    skew: float
    name: str
    """

    sound: tuple[Tensor, int, Tensor]
    source: tuple[Tensor, int, Tensor]
    skew: float
    name: str


@dataclass
class AnnotatedTokenizedSample:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor)
    label: Tensor
    skew: float
    name: str
    """

    sound: tuple[Tensor, int, Tensor]
    source: tuple[Tensor, int, Tensor]
    label: Tensor
    skew: float
    name: str


@dataclass
class MultiAnnotatedSample:
    """
    sound: (audio: Tensor, sr: int, intervals: dict[str, Interval])
    source: (audio: Tensor, sr: int, intervals: dict[str, Interval])
    label: dict[str, Tensor]
    skew: float
    name: str
    """

    sound: tuple[Tensor, int, dict[str, Interval]]
    source: tuple[Tensor, int, dict[str, Interval]]
    label: dict[str, Tensor]
    skew: float
    name: str


@dataclass
class SequenceSample:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor, spans: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor, spans: Tensor)
    label: (labels: Tensor, lengths: Tensor, spans: Tensor)
    skew: float
    name: str
    """

    sound: tuple[Tensor, int, Tensor, Tensor]
    source: tuple[Tensor, int, Tensor, Tensor]
    label: tuple[Tensor, Tensor, Tensor]
    skew: float
    name: str


@dataclass
class SoundBatch:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor)
    skew: Tensor
    name: list[str]
    """

    sound: tuple[Tensor, int, Tensor]
    source: tuple[Tensor, int, Tensor]
    skew: Tensor
    name: list[str]


@dataclass
class AnnotatedTokenizedBatch:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor)
    label: Tensor
    skew: Tensor
    name: list[str]
    """

    sound: tuple[Tensor, int, Tensor]
    source: tuple[Tensor, int, Tensor]
    label: Tensor
    skew: Tensor
    name: list[str]


@dataclass
class AnnotatedBatch:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor)
    label: (labels: Tensor, lengths: Tensor)
    skew: Tensor
    name: list[str]
    """

    sound: tuple[Tensor, int, Tensor]
    source: tuple[Tensor, int, Tensor]
    label: tuple[Tensor, Tensor]
    skew: Tensor
    name: list[str]


@dataclass
class MultiAnnotatedBatch:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor)
    label: dict[str, (labels: Tensor, lengths: Tensor)]
    skew: Tensor
    name: list[str]
    """

    sound: tuple[Tensor, int, Tensor]
    source: tuple[Tensor, int, Tensor]
    label: dict[str, tuple[Tensor, Tensor]]
    skew: Tensor
    name: list[str]


@dataclass
class SequenceBatch:
    """
    sound: (audio: Tensor, sr: int, lengths: Tensor, spans: Tensor)
    source: (audio: Tensor, sr: int, lengths: Tensor, spans: Tensor)
    label: (labels: Tensor, lengths: Tensor, spans: Tensor)
    skew: float
    name: list[str]
    """

    sound: tuple[Tensor, int, Tensor, Tensor]
    source: tuple[Tensor, int, Tensor, Tensor]
    label: tuple[Tensor, Tensor, Tensor]
    skew: Tensor
    name: list[str]


class SoundDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sounds: list[str],
        *,
        in_sr: int = 16_000,
        out_sr: Optional[int] = None,
        augmentation: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        limits: Optional[Limits] = None,
        batch_first: bool = True,
    ):
        self.sounds: list[str] = sounds
        self.in_sr: int = in_sr
        self.out_sr: Optional[int] = out_sr
        self.augmentation: Optional[Callable] = augmentation
        self.audio_transform: Optional[Callable] = audio_transform
        self.limits: Optional[Limits] = limits
        self.batch_dim = 0 if batch_first else 1

    def __len__(self):
        return len(self.sounds)

    @torch.no_grad()
    def __getitem__(self, idx: int, waveform: bool = False) -> SoundSample:
        in_path = self.sounds[idx]

        # load and resample audio
        in_audio, in_sr = load_audio(in_path)
        in_audio, in_sr = apply_sox_effects(
            in_audio,
            in_sr,
            [["rate", str(self.in_sr)], ["channels", "1"], ["norm", "-3"]],
        )

        # apply optional augmentation to source audio
        if self.augmentation:
            mix_audio, mix_sr = self.augmentation(in_audio, in_sr)
            mix_audio, mix_sr = apply_sox_effects(mix_audio, mix_sr, [["norm", "-3"]])
        else:
            mix_audio, mix_sr = in_audio, in_sr

        # make sure sampling rate wasn't changed during processing
        if mix_sr != in_sr:
            raise RuntimeError(
                "Changing the audio sampling rate during audio processing is not supported"
            )

        # calculate skew
        skew = mix_audio.shape[1] / in_audio.shape[1]

        # calculate spectrograms
        if waveform or self.audio_transform is None:
            mix_x = mix_audio.T
            in_x = in_audio.T
            out_sr = in_sr
        else:
            mix_x = self.audio_transform(mix_audio)
            in_x = self.audio_transform(in_audio)
            if self.out_sr:
                out_sr = self.out_sr
            else:
                out_sr = round(in_sr * mix_x.shape[0] / mix_audio.shape[1])

        return SoundSample(
            sound=(mix_x, out_sr),
            source=(in_x, out_sr),
            skew=skew,
            name=in_path,
        )

    def augment(self, augmentation: Optional[Callable]):
        self.augmentation = augmentation
        return self

    def limit(self, limits: Optional[Limits]):
        self.limits = limits
        return self

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        full_tensor: bool = True,
    ):
        if full_tensor and self.limits is None:
            raise RuntimeError("Argument full_tensor cannot be set without limits")

        def collate_fn(samples: list[SoundSample]) -> Optional[SoundBatch]:
            if not samples or any(s is None for s in samples):
                return None

            sounds, sound_sr = zip(*[s.sound for s in samples])
            sources, source_sr = zip(*[s.source for s in samples])
            names = [s.name for s in samples]

            sounds_lens = torch.tensor([len(x) for x in sounds], dtype=torch.int)
            source_lens = torch.tensor([len(x) for x in sources], dtype=torch.int)

            out_sr = samples[0].sound[1]
            if full_tensor and self.limits is not None:
                max_sounds_len = int(self.limits.time * sound_sr[0])
                max_sources_len = int(self.limits.time * source_sr[0])
            else:
                max_sounds_len = int(sounds_lens.max())
                max_sources_len = int(source_lens.max())

            sounds = [_pad_axis(x, 0, max_sounds_len - len(x), axis=0) for x in sounds]
            sources = [
                _pad_axis(x, 0, max_sources_len - len(x), axis=0) for x in sources
            ]

            sounds = torch.stack(sounds, dim=self.batch_dim)
            sources = torch.stack(sources, dim=self.batch_dim)
            skew = torch.tensor([s.skew for s in samples])

            return SoundBatch(
                sound=(sounds, out_sr, sounds_lens),
                source=(sources, out_sr, source_lens),
                skew=skew,
                name=names,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


class AnnotatedDataset(SoundDataset):
    def __init__(
        self,
        sounds: list[str],
        annotations: list[str],
        vocabulary: list[str],
        target: str,
        *,
        normalize: bool = False,
        ignore_silence: bool = True,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.annotations = annotations
        self.vocabulary = vocabulary
        self.target = target
        self.stressed = target in ("phones", "syllables") and any(
            is_stressed(w) for w in vocabulary
        )
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
    ) -> Optional[AnnotatedSample]:
        sample = super().__getitem__(idx, waveform=waveform)
        sound, sound_sr = sample.sound
        source, source_sr = sample.source
        skew = sample.skew

        y, source_intervals = self._annotation(self.annotations[idx])
        sound_intervals = [
            (start * skew, stop * skew) for start, stop in source_intervals
        ]
        if len(y) == 0:
            return None

        if skew >= 1:
            limits = self._get_limits(sound_intervals)
        else:
            limits = self._get_limits(source_intervals)
            if limits:
                limits.time *= skew

        if limits:
            sound = sound[: int(limits.time * sound_sr)]
            source = source[: int(limits.time / skew * source_sr)]
            y = y[: limits.tokens(self.target)]
            sound_intervals = sound_intervals[: limits.tokens(self.target)]
            source_intervals = source_intervals[: limits.tokens(self.target)]

        return AnnotatedSample(
            sound=(sound, sound_sr, sound_intervals),
            source=(source, source_sr, source_intervals),
            label=y,
            skew=sample.skew,
            name=sample.name,
        )

    @property
    def num_classes(self):
        return len(self.vocabulary)

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        full_tensor: bool = True,
        flat_labels: bool = False,
    ):
        if full_tensor and self.limits is None:
            raise RuntimeError("Argument full_tensor cannot be set without limits")

        def collate_fn(samples: list[AnnotatedSample]) -> Optional[AnnotatedBatch]:
            if not samples or any(s is None for s in samples):
                return None

            sounds = [s.sound[0] for s in samples]
            sources = [s.source[0] for s in samples]
            labels = [s.label for s in samples]
            _, sound_sr, _ = samples[0].sound
            _, source_sr, _ = samples[0].source
            names = [s.name for s in samples]

            sound_lens = torch.tensor([len(x) for x in sounds], dtype=torch.int)
            source_lens = torch.tensor([len(x) for x in sources], dtype=torch.int)
            label_lens = torch.tensor([len(y) for y in labels], dtype=torch.int)

            if full_tensor and self.limits is not None:
                max_sound_len = int(self.limits.time * sound_sr)
                max_source_len = int(self.limits.time * source_sr)
                max_label_len = int(self.limits.tokens(self.target))
            else:
                max_sound_len = int(sound_lens.max())
                max_source_len = int(source_lens.max())
                max_label_len = int(label_lens.max())

            sounds = [_pad_axis(x, 0, max_sound_len - len(x), axis=0) for x in sounds]
            sources = [
                _pad_axis(x, 0, max_source_len - len(x), axis=0) for x in sources
            ]
            if not flat_labels:
                labels = [
                    _pad_axis(y, 0, max_label_len - len(y), axis=0) for y in labels
                ]

            sounds = torch.stack(sounds, dim=self.batch_dim)
            sources = torch.stack(sources, dim=self.batch_dim)
            if flat_labels:
                labels = torch.cat(labels)
            else:
                labels = torch.stack(labels, dim=self.batch_dim)
            skew = torch.tensor([s.skew for s in samples])

            return AnnotatedBatch(
                sound=(sounds, sound_sr, sound_lens),
                source=(sources, source_sr, source_lens),
                label=(labels, label_lens),
                skew=skew,
                name=names,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def _annotation(
        self,
        annotation: str,
        ignore_silence: bool = True,
    ) -> tuple[Tensor, list[tuple[float, float]]]:
        fmt, filepath = annotation.split(":")

        if fmt == "libri":
            # read annotation file
            textgrid = textgrids.TextGrid(filepath)
            if self.target == "phones":
                textgrid = _spaced_textgrid(textgrid, self.spaced)
            elif self.target == "syllables":
                textgrid = _syllabized_textgrid(textgrid, self.spaced)
            elif self.target == "chars":
                textgrid = _character_textgrid(textgrid, self.spaced)
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

    def _get_limits(self, intervals: list[tuple[float, float]]) -> Optional[Limits]:
        limits = self.limits
        if limits is None:
            return None

        if limits.time < intervals[-1][1]:
            max_time = [end for _, end in intervals if end <= limits.time][-1]
        else:
            max_time = limits.time
        max_tokens = limits.tokens(self.target)

        if len(intervals) > max_tokens:
            time_limit_equiv = intervals[max_tokens][0]
        else:
            time_limit_equiv = np.inf
        token_limit_equiv = len([1 for _, end in intervals if end <= max_time])

        max_time = min(max_time, time_limit_equiv)
        max_tokens = min(max_tokens, token_limit_equiv)

        return limits.with_limit("time", max_time).with_limit(self.target, max_tokens)


class TokenizedDataset(AnnotatedDataset):
    def __init__(
        self,
        sounds: list[str],
        annotations: list[str],
        vocabulary: list[str],
        target: str,
        *,
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

    def __getitem__(
        self, idx: int, waveform: bool = False
    ) -> Optional[AnnotatedTokenizedSample]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None:
            return None

        pre_ctx, post_ctx = self.context
        sound, sound_sr, sound_interval = sample.sound
        source, source_sr, source_interval = sample.source
        label = sample.label.int()
        skew = sample.skew

        sounds, sounds_lens = [], []
        for start, stop in sound_interval:
            length = stop - start
            ctx_start = int((start - length * pre_ctx) * sound_sr)
            ctx_end = int((stop + length * post_ctx) * sound_sr)
            x = sound[max(ctx_start, 0) : ctx_end]
            x = _pad_axis(x, -ctx_start, ctx_end - len(sound))
            sounds.append(x)
            sounds_lens.append(len(x))

        sources, sources_lens = [], []
        for start, stop in source_interval:
            length = stop - start
            ctx_start = int((start - length * pre_ctx) * source_sr)
            ctx_end = int((stop + length * post_ctx) * source_sr)
            x = source[max(ctx_start, 0) : ctx_end]
            x = _pad_axis(x, -ctx_start, ctx_end - len(source))
            sources.append(x)
            sources_lens.append(len(x))

        freqbins = sound.shape[1]
        sound_fix_t = int(self.duration * (1 + pre_ctx + post_ctx) * sound_sr)
        source_fix_t = int(self.duration * (1 + pre_ctx + post_ctx) * source_sr)
        freqs = np.linspace(1, freqbins, freqbins)

        if not self.scale:
            will_fit = [
                i for i, length in enumerate(sounds_lens) if length <= sound_fix_t
            ]
            sounds = [sounds[i] for i in will_fit]
            sounds_lens = [sounds_lens[i] for i in will_fit]
            sources = [sources[i] for i in will_fit]
            sources_lens = [sources_lens[i] for i in will_fit]
            label = label[will_fit]
        if len(sounds) == 0:
            return None

        for i, x in enumerate(sounds):
            if self.scale or len(x) > sound_fix_t:
                if waveform or self.audio_transform is None:
                    # TODO
                    # x[i] = librosa.effects.time_stretch(xi, xi.shape[0]/fix_t)[:fix_t]
                    raise NotImplementedError()
                else:
                    t0 = np.linspace(1, len(x), len(x))
                    t1 = np.linspace(1, len(x), sound_fix_t)
                    sounds[i] = torch.from_numpy(
                        RectBivariateSpline(t0, freqs, x)(t1, freqs)
                    )
                    sounds_lens[i] = sound_fix_t
            else:
                diff = sound_fix_t - len(x)
                if self.alignment == "left":
                    pre_t, post_t = 0, diff
                elif self.alignment == "right":
                    pre_t, post_t = diff, 0
                else:
                    pre_t, post_t = math.floor(diff / 2), math.ceil(diff / 2)
                sounds[i] = _pad_axis(x, pre_t, post_t, axis=0)

        for i, x in enumerate(sources):
            if self.scale or len(x) > source_fix_t:
                if waveform or self.audio_transform is None:
                    # TODO
                    # x[i] = librosa.effects.time_stretch(xi, xi.shape[0]/fix_t)[:fix_t]
                    raise NotImplementedError()
                else:
                    t0 = np.linspace(1, len(x), len(x))
                    t1 = np.linspace(1, len(x), source_fix_t)
                    sources[i] = torch.from_numpy(
                        RectBivariateSpline(t0, freqs, x)(t1, freqs)
                    )
                    sources_lens[i] = source_fix_t
            else:
                diff = source_fix_t - len(x)
                if self.alignment == "left":
                    pre_t, post_t = 0, diff
                elif self.alignment == "right":
                    pre_t, post_t = diff, 0
                else:
                    pre_t, post_t = math.floor(diff / 2), math.ceil(diff / 2)
                sources[i] = _pad_axis(x, pre_t, post_t, axis=0)

        sounds = torch.stack(sounds, dim=self.batch_dim).float()
        sounds_lens = torch.tensor(sounds_lens, dtype=torch.int)
        sources = torch.stack(sources, dim=self.batch_dim).float()
        sources_lens = torch.tensor(sources_lens, dtype=torch.int)

        # TODO I don't remember what this was supposed to do
        if self.drop_modifiers:
            keep_mask = sounds_lens != 0
            sounds = sounds[keep_mask]
            sounds_lens = sounds_lens[keep_mask]
            sources = sources[keep_mask]
            sources_lens = sources_lens[keep_mask]
            label = label[keep_mask]

        # TODO clean sources not handled
        return AnnotatedTokenizedSample(
            sound=(sounds, sound_sr, sounds_lens),
            source=(sources, source_sr, sources_lens),
            label=label,
            skew=skew,
            name=sample.name,
        )

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate_fn(
            samples: list[AnnotatedTokenizedSample],
        ) -> Optional[AnnotatedTokenizedBatch]:
            samples = [s for s in samples if s is not None and s.label is not None]
            if not samples:
                return None

            sounds, sound_sr, sounds_lens = zip(*[s.sound for s in samples])
            sources, source_sr, sources_lens = zip(*[s.source for s in samples])
            labels = [s.label for s in samples]
            names = [s.name for s in samples]

            sounds = torch.cat(sounds, dim=self.batch_dim)
            sources = torch.cat(sources, dim=self.batch_dim)
            labels = torch.cat(labels, dim=self.batch_dim)
            sounds_lens = torch.cat(sounds_lens)
            sources_lens = torch.cat(sources_lens)
            skew = torch.tensor([s.skew for s in samples])

            return AnnotatedTokenizedBatch(
                sound=(sounds, sound_sr[0], sounds_lens),
                source=(sources, source_sr[0], sources_lens),
                label=labels,
                skew=skew,
                name=names,
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
                batch = next(it)
            except StopIteration:
                break

            if batch is None:
                continue

            xs.append(batch.sound)
            ys.append(batch.label)
            count += len(batch.sound)
            pbar.update(len(batch.sound))

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
                batch = next(it)
            except StopIteration:
                break

            if batch is None:
                continue

            for xi, yi in zip(batch.sound, batch.label):
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

        while count < n:
            try:
                batch = next(it)
            except StopIteration:
                break

            if batch is None:
                continue

            stats["identity"].append(
                np.array([self.vocabulary[_] for _ in batch.label], dtype=object)
            )
            stats["length"].append(batch.sound_length.numpy() / batch.rate)
            count += len(batch.sound)
            pbar.update(len(batch.sound))

        stats = {k: np.concatenate(stats[k], axis=0)[:n] for k in stats}

        return stats


class BlockDataset(SoundDataset):
    def __init__(
        self,
        sounds: list[str],
        *,
        max_time: float = 2.5,
        block_min: float = 0.01,
        max_step: Optional[float] = None,
        batch_first: bool = False,
        prepad: bool = False,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.max_time = max_time
        self.block_min = block_min
        self.batch_first = batch_first
        self.max_step = max_time if max_step is None else max_step
        self.prepad = prepad

    def __getitem__(
        self, idx: int, waveform: bool = False
    ) -> Optional[TokenizedSample]:
        sample = super().__getitem__(idx, waveform=waveform)
        sound, sound_sr = sample.sound
        source, source_sr = sample.source

        sound_maxblk = int(self.max_time * sound_sr)
        sound_minblk = int(self.block_min * sound_sr)
        sound_maxstp = int(self.max_step * sound_sr)
        source_maxblk = int(self.max_time * source_sr)
        source_minblk = int(self.block_min * source_sr)
        # source_maxstp = int(self.max_step * source_sr)

        if self.prepad:
            sound = _pad_axis(sound, sound_maxblk - 1, 0, axis=0)
            source = _pad_axis(source, source_maxblk - 1, 0, axis=0)

        index = 0
        sounds = []
        sources = []
        sounds_lens = []
        sources_lens = []
        while index < len(sound):
            length = np.random.randint(sound_minblk, sound_maxblk + 1)
            if index + length > len(sound):
                length = len(sound) - index
            if length < sound_minblk:
                break
            sounds.append(sound[index : index + length])
            sounds_lens.append(length)

            length = np.random.randint(source_minblk, source_maxblk + 1)
            if index + length > len(source):
                length = len(source) - index
            if length < source_minblk:
                break
            sources.append(source[index : index + length])
            sources_lens.append(length)

            index += min(length, sound_maxstp)

        if not sounds:
            return None

        sounds = torch.stack(
            [
                _pad_axis(x, 0, sound_maxblk - len(x), axis=self.batch_dim)
                for x in sounds
            ]
        )
        sources = torch.stack(
            [
                _pad_axis(x, 0, source_maxblk - len(x), axis=self.batch_dim)
                for x in sources
            ]
        )
        sounds_lens = torch.tensor(sounds_lens, dtype=torch.int)
        sources_lens = torch.tensor(sources_lens, dtype=torch.int)

        return TokenizedSample(
            sound=(sounds, sound_sr, sounds_lens),
            source=(sources, source_sr, sources_lens),
            skew=sample.skew,
            name=sample.name,
        )

    def iterator(
        self,
        batch_size: int = 1,
        batch_max: Optional[int] = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate_fn(samples: list[TokenizedSample]) -> Optional[SoundBatch]:
            samples = [s for s in samples if s is not None]
            if not samples:
                return None

            sounds, sound_sr, sounds_lens = zip(*[s.sound for s in samples])
            sources, source_sr, sources_lens = zip(*[s.source for s in samples])
            names = [s.name for s in samples]

            sounds = torch.cat(sounds, dim=self.batch_dim)
            sources = torch.cat(sources, dim=self.batch_dim)
            sounds_lens = torch.cat(sounds_lens)
            sources_lens = torch.cat(sources_lens)
            skew = torch.tensor([s.skew for s in samples])

            if batch_max is not None:
                sounds = (
                    sounds[:batch_max] if self.batch_dim == 0 else sounds[:, :batch_max]
                )
                sources = (
                    sources[:batch_max]
                    if self.batch_dim == 0
                    else sources[:, :batch_max]
                )
                sounds_lens = sounds_lens[:batch_max]
                sources_lens = sources_lens[:batch_max]

            return SoundBatch(
                sound=(sounds, sound_sr, sounds_lens),
                source=(sources, source_sr, sources_lens),
                skew=skew,
                name=names,
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
        sounds: list[str],
        annotations: list[str],
        vocabulary: list[str],
        target: str,
        *,
        seq_size: int = 20,
        seq_min: int = 1,
        seq_time: float = 8.0,
        seq_overlap: bool = False,
        check_boundaries: bool = True,
        **kwargs,
    ):
        super().__init__(sounds, annotations, vocabulary, target, **kwargs)
        self.seq_size = seq_size
        self.seq_min = seq_min if seq_min else seq_size
        self.seq_time = seq_time
        self.seq_overlap = seq_overlap
        self.check_boundaries = check_boundaries

    @torch.no_grad()
    def __getitem__(self, idx: int, waveform: bool = False) -> Optional[SequenceSample]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None:
            return None

        sound, sound_sr, sound_interval = sample.sound
        source, source_sr, source_interval = sample.source
        label = sample.label
        skew = sample.skew

        sound_maxblk = int(self.seq_time * sound_sr)
        source_maxblk = int(self.seq_time * source_sr)

        index = 0
        sounds, sounds_lens, sounds_span = [], [], []
        sources, sources_lens, sources_span = [], [], []
        labels, labels_lens, labels_span = [], [], []
        interval = sound_interval if skew >= 1 else source_interval
        while index < len(interval):
            if self.check_boundaries and is_postfix(self.vocabulary[label[index]]):
                index += 1
                continue

            length = np.random.randint(self.seq_min, self.seq_size + 1)

            if index + length > len(interval):
                length = len(interval) - index
            if length < self.seq_min:
                break

            if self.check_boundaries:
                init_length = length

                while (
                    index + length <= len(interval)
                    and length <= self.seq_size
                    and is_prefix(self.vocabulary[label[index + length - 1]])
                ):
                    length += 1
                if length > self.seq_size or index + length > len(interval):
                    length = init_length

                while length >= self.seq_min and is_prefix(
                    self.vocabulary[label[index + length - 1]]
                ):
                    length -= 1
                if length < self.seq_min:
                    index += 1
                    continue

            start = max(int(sound_interval[index][0] * sound_sr) - 1, 0)
            stop = int(sound_interval[index + length - 1][1] * sound_sr) + 2
            while stop - start > sound_maxblk and length > self.seq_min:
                length -= 1
                stop = int(sound_interval[index + length - 1][1] * sound_sr) + 2
            if stop - start > sound_maxblk:
                index += 1
                continue
            sounds.append(sound[start:stop])
            sounds_lens.append(stop - start)
            sounds_span.append((start, stop))

            start = max(int(source_interval[index][0] * source_sr) - 1, 0)
            stop = int(source_interval[index + length - 1][1] * source_sr) + 2
            sources.append(source[start:stop])
            sources_lens.append(stop - start)
            sources_span.append((start, stop))

            labels.append(label[index : index + length])
            labels_lens.append(length)
            labels_span.append((index, index + length))
            index += 1 if self.seq_overlap else length

        if not sounds:
            return None

        sounds = torch.stack(
            [_pad_axis(x, 0, sound_maxblk - len(x), axis=0) for x in sounds],
            dim=self.batch_dim,
        )
        sources = torch.stack(
            [_pad_axis(x, 0, source_maxblk - len(x), axis=0) for x in sources],
            dim=self.batch_dim,
        )
        labels = torch.stack(
            [_pad_axis(y, 0, self.seq_size - len(y), axis=0) for y in labels],
            dim=self.batch_dim,
        )
        sounds_lens = torch.tensor(sounds_lens, dtype=torch.int)
        sources_lens = torch.tensor(sources_lens, dtype=torch.int)
        labels_lens = torch.tensor(labels_lens, dtype=torch.int)
        sounds_span = torch.tensor(sounds_span, dtype=torch.int)
        sources_span = torch.tensor(sources_span, dtype=torch.int)
        labels_span = torch.tensor(labels_span, dtype=torch.int)

        # TODO check if clean sources properly handled
        return SequenceSample(
            sound=(sounds, sound_sr, sounds_lens, sounds_span),
            source=(sources, source_sr, sources_lens, sources_span),
            label=(labels, labels_lens, labels_span),
            skew=skew,
            name=sample.name,
        )

    def iterator(
        self,
        batch_size: int = 1,
        batch_max: Optional[int] = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate_fn(samples: list[SequenceSample]) -> Optional[SequenceBatch]:
            samples = [s for s in samples if s is not None]
            if not samples:
                return None

            sounds, sound_sr, sounds_lens, sounds_span = zip(
                *[s.sound for s in samples]
            )
            sources, source_sr, sources_lens, sources_span = zip(
                *[s.source for s in samples]
            )
            labels, labels_lens, labels_span = zip(*[s.label for s in samples])
            names = [s.name for s in samples]

            sounds = torch.cat(sounds, dim=self.batch_dim)
            sources = torch.cat(sources, dim=self.batch_dim)
            labels = torch.cat(labels, dim=self.batch_dim)
            sounds_lens = torch.cat(sounds_lens)
            sources_lens = torch.cat(sources_lens)
            labels_lens = torch.cat(labels_lens)
            sounds_span = torch.cat(sounds_span)
            sources_span = torch.cat(sources_span)
            labels_span = torch.cat(labels_span)
            skew = torch.tensor([s.skew for s in samples])

            # batch_size = xs.shape[self.batch_dim]
            if batch_max is not None:
                sounds = (
                    sounds[:batch_max] if self.batch_dim == 0 else sounds[:, :batch_max]
                )
                sources = (
                    sources[:batch_max]
                    if self.batch_dim == 0
                    else sources[:, :batch_max]
                )
                labels = (
                    labels[:batch_max] if self.batch_dim == 0 else labels[:, :batch_max]
                )
                sounds_lens = sounds_lens[:batch_max]
                sources_lens = sources_lens[:batch_max]
                labels_lens = labels_lens[:batch_max]
                sounds_span = sounds_span[:batch_max]
                sources_span = sources_span[:batch_max]
                labels_span = labels_span[:batch_max]

            return SequenceBatch(
                sound=(sounds, sound_sr, sounds_lens, sounds_span),
                source=(sources, source_sr, sources_lens, sources_span),
                label=(labels, labels_lens, labels_span),
                skew=skew,
                name=names,
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
        target,
        *,
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

    def __getitem__(
        self, idx: int, waveform: bool = False
    ) -> Optional[AnnotatedTokenizedSample]:
        sample = super().__getitem__(idx, waveform=waveform)
        if sample is None:
            return None

        sound, sound_sr, sound_interval = sample.sound
        source, source_sr, source_interval = sample.source
        label = sample.label
        skew = sample.skew

        yint_filt = [
            (token, snd_intv, src_intv)
            for token, snd_intv, src_intv in zip(label, sound_interval, source_interval)
            if (self.acc_range[self.vocabulary[token]][0] <= snd_intv[1] - snd_intv[0])
            and (snd_intv[1] - snd_intv[0] <= self.acc_range[self.vocabulary[token]][1])
        ]
        if not yint_filt:
            return None
        label, sound_interval, source_interval = zip(*yint_filt)

        sound_interval = [
            (int(start * sound_sr), int(stop * sound_sr))
            for start, stop in sound_interval
        ]
        source_interval = [
            (int(start * source_sr), int(stop * source_sr))
            for start, stop in source_interval
        ]
        # durations = [stop - start for start, stop in intervals]
        sound_centers = [(start + stop) // 2 for start, stop in sound_interval]
        source_centers = [(start + stop) // 2 for start, stop in source_interval]

        context = int(self.context * sound_sr)
        pre_context, post_context = math.floor(context / 2), math.ceil(context / 2)
        sound_interval = [
            (center - pre_context, center + post_context) for center in sound_centers
        ]
        context = int(self.context * source_sr)
        pre_context, post_context = math.floor(context / 2), math.ceil(context / 2)
        source_interval = [
            (center - pre_context, center + post_context) for center in source_centers
        ]

        yint_filt = [
            (token, snd_intv, src_intv)
            for token, snd_intv, src_intv in zip(label, sound_interval, source_interval)
            if snd_intv[0] >= 0 and snd_intv[1] <= len(sample.sound)
        ]
        if not yint_filt:
            return None
        label, sound_interval, source_interval = zip(*yint_filt)

        scale_factor = {
            "faster": 1.0 / self.scale_factor,
            "base": 1.0,
            "slower": self.scale_factor,
        }[self.rate]

        sounds = [sound[start:stop] for start, stop in sound_interval]
        target_t = int(self.context * sound_sr * scale_factor)
        freqbins = sound.shape[1]
        if target_t != context:
            t0 = np.linspace(1, context, context)
            t1 = np.linspace(1, context, target_t)
            freqs = np.linspace(1, freqbins, freqbins)

            for i, xi in enumerate(sounds):
                if waveform or self.audio_transform is None:
                    # x[i] = librosa.effects.time_stretch(xi, context/target_t)
                    raise NotImplementedError()
                else:
                    sounds[i] = torch.from_numpy(
                        RectBivariateSpline(t0, freqs, xi)(t1, freqs)
                    )

        sources = [source[start:stop] for start, stop in source_interval]
        target_t = int(self.context * source_sr * scale_factor)
        freqbins = source.shape[1]
        if target_t != context:
            t0 = np.linspace(1, context, context)
            t1 = np.linspace(1, context, target_t)
            freqs = np.linspace(1, freqbins, freqbins)

            for i, xi in enumerate(sources):
                if waveform or self.audio_transform is None:
                    # x[i] = librosa.effects.time_stretch(xi, context/target_t)
                    raise NotImplementedError()
                else:
                    sources[i] = torch.from_numpy(
                        RectBivariateSpline(t0, freqs, xi)(t1, freqs)
                    )

        sounds_lens = torch.tensor([len(x) for x in sounds], dtype=torch.int)
        sources_lens = torch.tensor([len(x) for x in sources], dtype=torch.int)
        sounds = torch.stack(sounds, dim=0).float()
        sources = torch.stack(sources, dim=0).float()
        label = torch.stack(label, dim=0).int()

        sounds = sounds[sounds_lens != 0]
        label = label[sounds_lens != 0]
        sounds_lens = sounds_lens[sounds_lens != 0]

        # TODO clean sources not handled
        return AnnotatedTokenizedSample(
            sound=(sounds, sound_sr, sounds_lens),
            source=(sources, source_sr, sources_lens),
            label=label,
            skew=skew,
            name=sample.name,
        )

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate_fn(
            samples: list[AnnotatedTokenizedSample],
        ) -> Optional[AnnotatedTokenizedBatch]:
            if any(s is None for s in samples):
                return None

            sounds, sound_sr, sounds_lens = zip(*[s.sound for s in samples])
            sources, source_sr, sources_lens = zip(*[s.source for s in samples])
            labels = [s.label for s in samples]
            names = [s.name for s in samples]

            sounds = torch.cat(sounds, dim=self.batch_dim)
            sources = torch.cat(sources, dim=self.batch_dim)
            labels = torch.cat(labels, dim=self.batch_dim)
            sounds_lens = torch.cat(sounds_lens)
            sources_lens = torch.cat(sources_lens)
            skew = torch.tensor([s.skew for s in samples])

            # clean sources not handled
            return AnnotatedTokenizedBatch(
                sound=(sounds, sound_sr, sounds_lens),
                source=(sources, source_sr, sources_lens),
                label=labels,
                skew=skew,
                name=names,
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
        raise NotImplementedError()
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
        sounds: list[str],
        annotations: list[str],
        vocabulary: dict[str, list[str]],
        *,
        normalize: bool = False,
        ignore_silence: bool = True,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.annotations = annotations
        self.vocabulary = vocabulary
        self.stressed = {
            k: k in ("phones", "syllables") and any(is_stressed(w) for w in v)
            for k, v in vocabulary.items()
        }
        self.normalize = normalize
        self.spaced = {k: " " in v for k, v in vocabulary.items()}
        self.include_na = {k: "[UNK]" in v for k, v in vocabulary.items()}
        self.ignore_silence = ignore_silence
        self.targets = sorted(vocabulary.keys())
        self.key = {
            k: {tok: i for i, tok in enumerate(v)} for k, v in vocabulary.items()
        }
        assert all(k in ["chars", "phones", "words", "syllables"] for k in vocabulary)

    @torch.no_grad()
    def __getitem__(
        self, idx: int, waveform: bool = False
    ) -> Optional[MultiAnnotatedSample]:
        sample = super().__getitem__(idx, waveform=waveform)
        sound, sound_sr = sample.sound
        source, source_sr = sample.source
        skew = sample.skew

        labels = {}
        sound_intervals = {}
        source_intervals = {}
        for k in self.targets:
            labels[k], source_intervals[k] = self._annotation(self.annotations[idx], k)
            sound_intervals[k] = [
                (start * skew, stop * skew) for start, stop in source_intervals[k]
            ]
            if len(labels[k]) == 0:
                return None

        if self.limits:
            if skew >= 1:
                limits = self._get_limits(sound_intervals)
            else:
                limits = self._get_limits(source_intervals)
                limits.time *= skew

            sound = sound[: int(limits.time * sound_sr)]
            source = source[: int(limits.time / skew * source_sr)]
            labels = {k: labels[k][: limits.tokens(k)] for k in self.targets}
            sound_intervals = {
                k: sound_intervals[k][: limits.tokens(k)] for k in self.targets
            }
            source_intervals = {
                k: source_intervals[k][: limits.tokens(k)] for k in self.targets
            }

        return MultiAnnotatedSample(
            sound=(sound, sound_sr, sound_intervals),
            source=(source, source_sr, source_intervals),
            label=labels,
            skew=sample.skew,
            name=sample.name,
        )

    @property
    def num_classes(self):
        return {k: len(v) for k, v in self.vocabulary.items()}

    def iterator(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        full_tensor: bool = True,
        flat_labels: bool = False,
    ):
        if full_tensor and self.limits is None:
            raise RuntimeError("Argument full_tensor cannot be set without limits")

        def collate_fn(
            samples: list[MultiAnnotatedSample],
        ) -> Optional[MultiAnnotatedBatch]:
            if not samples or any(s is None for s in samples):
                return None

            sounds = [s.sound[0] for s in samples]
            sources = [s.source[0] for s in samples]
            labels = {k: [s.label[k] for s in samples] for k in self.targets}
            _, sound_sr, _ = samples[0].sound
            _, source_sr, _ = samples[0].source
            names = [s.name for s in samples]

            sound_lens = torch.tensor([len(x) for x in sounds], dtype=torch.int)
            source_lens = torch.tensor([len(x) for x in sources], dtype=torch.int)
            label_lens = {
                k: torch.tensor([len(y) for y in labels[k]], dtype=torch.int)
                for k in self.targets
            }

            if full_tensor and self.limits is not None:
                max_sound_len = int(self.limits.time * sound_sr)
                max_source_len = int(self.limits.time * source_sr)
                max_label_len = {k: int(self.limits.tokens(k)) for k in self.targets}
            else:
                max_sound_len = int(sound_lens.max())
                max_source_len = int(source_lens.max())
                max_label_len = {k: int(label_lens[k].max()) for k in self.targets}

            sounds = [_pad_axis(x, 0, max_sound_len - len(x), axis=0) for x in sounds]
            sources = [
                _pad_axis(x, 0, max_source_len - len(x), axis=0) for x in sources
            ]
            if not flat_labels:
                for k in self.targets:
                    labels[k] = [
                        _pad_axis(y, 0, max_label_len[k] - len(y), axis=0)
                        for y in labels[k]
                    ]

            sounds = torch.stack(sounds, dim=self.batch_dim)
            sources = torch.stack(sources, dim=self.batch_dim)
            if flat_labels:
                labels = {k: torch.cat(labels[k]) for k in self.targets}
            else:
                labels = {
                    k: torch.stack(labels[k], dim=self.batch_dim) for k in self.targets
                }
            skew = torch.tensor([s.skew for s in samples])

            return MultiAnnotatedBatch(
                sound=(sounds, sound_sr, sound_lens),
                source=(sources, source_sr, source_lens),
                label={k: (labels[k], label_lens[k]) for k in self.targets},
                skew=skew,
                name=names,
            )

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def _annotation(
        self,
        annotation: str,
        target_type: str,
        ignore_silence: bool = True,
        # skew: Optional[float] = None,
    ) -> tuple[Tensor, list[tuple[float, float]]]:
        fmt, filepath = annotation.split(":")

        if fmt == "libri":
            # read annotation file
            textgrid = textgrids.TextGrid(filepath)
            if target_type == "phones":
                textgrid = _spaced_textgrid(textgrid, self.spaced["phones"])
            elif target_type == "syllables":
                textgrid = _syllabized_textgrid(textgrid, self.spaced["syllables"])
            elif target_type == "chars":
                textgrid = _character_textgrid(textgrid, self.spaced["chars"])
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

        # if skew is not None:
        #     interv = [(start * skew, stop * skew) for start, stop in interv]

        return torch.tensor(target), interv

    def _get_limits(self, intervals: dict[str, list[tuple[float, float]]]) -> Limits:
        if self.limits is None:
            raise ValueError("__get_limits requires self.limit to be set")

        max_time = self.limits.time
        for k in self.targets:
            if max_time < intervals[k][-1][1]:
                max_time = [end for _, end in intervals[k] if end <= max_time][-1]

        max_tokens = {}
        for k in self.targets:
            max_tokens[k] = self.limits.tokens(k)

            if len(intervals[k]) > max_tokens[k]:
                time_limit_equiv = intervals[k][max_tokens[k]][0]
                max_time = min(max_time, time_limit_equiv)

        for k in self.targets:
            token_limit_equiv = len([1 for _, end in intervals[k] if end <= max_time])
            max_tokens[k] = min(max_tokens[k], token_limit_equiv)

        return Limits(time=max_time, **max_tokens)


def _pad_axis(array: Tensor, pre: int = 0, post: int = 0, axis: int = 0) -> Tensor:
    pre, post = max(pre, 0), max(post, 0)
    if pre == 0 and post == 0:
        return array

    npad = [(pre, post) if i == axis else (0, 0) for i in range(array.ndim)]
    npad = [n for a, b in npad[::-1] for n in (a, b)]

    return torch.nn.functional.pad(array, npad)


def _spaced_textgrid(textgrid, spaced):
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

        if spaced and len(phones) > 0:
            phones.append(textgrids.Interval(" ", phones[-1].xmax, phones[-1].xmax))

    return textgrids.Tier(phones)


def _syllabized_textgrid(textgrid, spaced):
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

        if spaced and len(syllbs) > 0:
            syllables.append(
                textgrids.Interval(" ", syllables[-1].xmax, syllables[-1].xmax)
            )

    return textgrids.Tier(syllables)


def _character_textgrid(textgrid, spaced):
    characters = []
    for word in textgrid["words"]:
        if word.text in ["", "sp", "spn", "sil", "<unk>"]:
            continue

        for char in word.text:
            characters.append(textgrids.Interval(char, word.xmin, word.xmax))

        if spaced:
            characters.append(textgrids.Interval(" ", word.xmax, word.xmax))

    return textgrids.Tier(characters)
