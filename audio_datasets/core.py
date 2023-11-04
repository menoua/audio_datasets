import math
import os
import re

import numpy as np
import scipy as sp
import textgrids
import torch
import torchaudio
from tqdm import tqdm

from .data import NonSpeech
from .lexicon import (_is_postfix, _is_prefix, _is_subtoken, _normalize,
                      syllabize)

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


class EmptySequence(Exception):
    pass


class SoundDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sounds,
        in_sr=16_000,
        out_sr=100,
        freqbins=128,
        audio_proc="default",
        noise_reduce=False,
        mod_speech=False,
        mod_room=False,
        mod_channel=False,
        mod_scene=None,
        mod_custom=None,
        mix_augments=1,
        mod_intensity=0,
        top_db=70,
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

        if audio_proc == "default":
            self.audio_proc = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    in_sr,
                    n_fft=1024,
                    hop_length=int(in_sr / out_sr),
                    f_min=20,
                    f_max=8_000,
                    n_mels=freqbins,
                    power=2.0,
                ),
                torchaudio.transforms.AmplitudeToDB("power", top_db=top_db),
                type(
                    "Normalize",
                    (torch.nn.Module,),
                    dict(
                        forward=lambda _, x: (x - x.max()).squeeze(0).T.float() / top_db
                        + 1
                    ),
                )(),
            )
        elif audio_proc == "legacy":
            self.audio_proc = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    in_sr,
                    n_fft=1024,
                    hop_length=int(in_sr / out_sr),
                    f_min=20,
                    f_max=8_000,
                    n_mels=freqbins,
                    power=2.0,
                ),
                torchaudio.transforms.AmplitudeToDB("power", top_db=top_db),
                type(
                    "Normalize",
                    (torch.nn.Module,),
                    dict(forward=lambda _, x: (x - x.max()).squeeze(0).T.float() + 60),
                )(),
            )
        else:
            self.audio_proc = audio_proc

    def __len__(self):
        return len(self.sounds)

    @torch.no_grad()
    def __getitem__(self, idx, return_skew=False, waveform=False, return_clean=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

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

        # select scene sample
        noise_path = np.random.choice(self.mod_scene) if "mod_scene" in xforms else None

        # load files
        in_audio, in_sr = torchaudio.load(in_path)
        noise_audio, noise_sr = (
            torchaudio.load(noise_path) if noise_path else (None, None)
        )

        tfm = [["rate", str(self.in_sr)], ["channels", "1"]]
        # custom modification
        if "mod_custom" in xforms:
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
        mix_audio, mix_sr = torchaudio.sox_effects.apply_effects_tensor(
            in_audio, in_sr, tfm
        )

        # scene modification
        if noise_path:
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
            noise_audio, noise_sr = torchaudio.sox_effects.apply_effects_tensor(
                noise_audio, noise_sr, tfm
            )

            mix_audio = (mix_audio + noise_audio[:, : mix_audio.shape[1]]) / np.sqrt(2)
            mix_audio, mix_sr = torchaudio.sox_effects.apply_effects_tensor(
                mix_audio, mix_sr, [["norm", "-3"]]
            )

        # calculate skew
        skew = mix_audio.shape[1] / in_audio.shape[1]

        # calculate spectrograms
        if waveform or self.audio_proc is None:
            x = mix_audio.T
        else:
            x = self.audio_proc(mix_audio)

        # calculate spectrograms for clean input
        if return_clean:
            if waveform or self.audio_proc is None:
                x0 = in_audio.T
            else:
                x0 = self.audio_proc(in_audio)

        if return_clean:
            return (x, skew, x0) if return_skew else (x, x0)
        else:
            return (x, skew) if return_skew else x

    def augment(
        self, speech=True, room=True, channel=True, scene=None, mix_n=1, mod_intensity=0
    ):
        self.mod_speech = speech
        self.mod_room = room
        self.mod_channel = channel
        self.mod_scene = NonSpeech() if scene is None else scene
        self.mix_augments = mix_n
        self.set_intensity(mod_intensity)

        return self

    def speed_up(self, factor):
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

    def set_intensity(self, level):
        if level in [0, "low", "LOW"]:
            self.mod_config = CONFIG_MOD_LO
        elif level in [1, "mid", "medium", "MID", "MEDIUM"]:
            self.mod_config = CONFIG_MOD_MID
        elif level in [2, "high", "HIGH"]:
            self.mod_config = CONFIG_MOD_HI
        else:
            raise ValueError(
                "Modification intensity should be one of 0 (low), 1 (medium), or 2 (high)."
            )

        self.mod_intensity = level

        return self

    def annotate(
        self,
        annotations,
        vocabulary,
        target="words",
        value_nil=0,
        ignore_silence=True,
        normalize=False,
    ):
        return AnnotatedDataset(
            self.sounds,
            annotations,
            vocabulary,
            target,
            value_nil,
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
        sounds,
        annotations,
        vocabulary,
        target="words",
        stressed=False,
        value_nil=0,
        ignore_silence=True,
        max_time=None,
        max_tokens=None,
        normalize=False,
        batch_first=True,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.annotations = annotations
        self.vocabulary = vocabulary
        self.target = target
        self.stressed = stressed
        self.value_nil = value_nil
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.spaced = " " in vocabulary
        self.include_na = "[UNK]" in vocabulary
        self.ignore_silence = ignore_silence
        self.batch_first = batch_first
        assert self.target in ["chars", "phones", "words", "syllables"]

        self.key = dict([(key, i) for i, key in enumerate(vocabulary)])

    @torch.no_grad()
    def __getitem__(
        self, idx, return_intervals=False, waveform=False, return_clean=True
    ):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x = super().__getitem__(
            idx, return_skew=True, waveform=waveform, return_clean=return_clean
        )
        x, skew, x0 = x if return_clean else (*x, None)
        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr

        annotation = self.annotations[idx]

        try:
            y, intervals = self._annotation(annotation, return_intervals=True)
            intervals = [(start * skew, stop * skew) for start, stop in intervals]

            time_limit, token_limit = self._time_limit(annotation, skew)
        except EmptySequence:
            if return_clean and return_intervals:
                return None, None, None, None
            elif return_clean or return_intervals:
                return None, None, None
            else:
                return None, None

        if time_limit or token_limit:
            time_limit = int(time_limit * out_sr)
            x, y = x[:time_limit], y[:token_limit]
            intervals = intervals[:token_limit]
            x0 = x0[:time_limit] if return_clean else None

        if return_clean:
            return (x, y, intervals, x0) if return_intervals else (x, y, x0)
        else:
            return (x, y, intervals) if return_intervals else (x, y)

    @property
    def num_classes(self):
        return len(self.vocabulary)

    def limit(self, max_time=None, max_tokens=None):
        self.max_time = max_time
        self.max_tokens = max_tokens

        return self

    def iterator(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # return_len=True,
        full_tensor=True,
        flat_labels=False,
    ):
        def collate_fn(xys):
            xs, ys, x0s = zip(*xys)
            if any(x is None for x in xs):
                return {
                    "inputs": None,
                    "inputs_clean": None,
                    "labels": None,
                    "input_lengths": None,
                    "label_lengths": None,
                }

            xlens = torch.tensor([len(x) for x in xs], dtype=int)
            ylens = torch.tensor([len(y) for y in ys], dtype=int)

            out_sr = self.in_sr if self.audio_proc is None else self.out_sr
            max_xlen = (
                int(self.max_time * out_sr)
                if full_tensor and self.max_time
                else max(xlens)
            )
            max_ylen = (
                self.max_tokens if full_tensor and self.max_tokens else max(ylens)
            )

            xs = torch.stack(
                [torch.nn.functional.pad(x, (0, 0, 0, max_xlen - len(x))) for x in xs],
                dim=1,
            )
            x0s = torch.stack(
                [
                    torch.nn.functional.pad(x0, (0, 0, 0, max_xlen - len(x0)))
                    for x0 in x0s
                ],
                dim=1,
            )
            if flat_labels:
                ys = torch.cat(ys, dim=0)  # [np.newaxis]
            else:
                ys = torch.stack(
                    [torch.nn.functional.pad(y, (0, max_ylen - len(y))) for y in ys],
                    dim=1,
                )

            if self.batch_first:
                xs = xs.transpose(0, 1).contiguous()
                x0s = x0s.transpose(0, 1).contiguous()
                ys = ys.transpose(0, 1).contiguous() if not flat_labels else ys

            return {
                "inputs": xs,
                "inputs_clean": x0s,
                "labels": ys,
                "input_lengths": xlens,
                "label_lengths": ylens,
            }

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

    def _annotation(self, annotation, ignore_silence=True, return_intervals=False):
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
                subtokens = _normalize(token)  # , self.vocabulary)
                expanded_target += subtokens
                expanded_interv += [
                    intv
                    if not _is_subtoken(t)
                    else (intv[:1] * 2 if _is_prefix(t) else intv[1:] * 2)
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

        if len(target) == 0:
            raise EmptySequence()

        return (
            (torch.tensor(target), interv) if return_intervals else torch.tensor(target)
        )

    def _time_limit(self, annotation, skew=1):
        if self.max_time is None and self.max_tokens is None:
            return None, None

        _, intervals = self._annotation(annotation, return_intervals=True)
        intervals = [(start * skew, stop * skew) for start, stop in intervals]

        try:
            if self.max_time is None or self.max_time >= intervals[-1][1]:
                max_time = self.max_time
            else:
                max_time = [end for _, end in intervals if end <= self.max_time][-1]
        except IndexError:
            raise EmptySequence()

        max_tokens = (
            self.max_tokens
            if self.max_tokens and len(intervals) > self.max_tokens
            else None
        )

        limit_time = max_time
        limit_token = max_tokens

        if max_tokens:
            limit_equiv = intervals[limit_token][0]
            limit_time = min(limit_time, limit_equiv) if limit_time else limit_equiv

        if max_time:
            limit_equiv = len([1 for _, end in intervals if end <= limit_time])
            limit_token = min(limit_token, limit_equiv) if limit_token else limit_equiv

        return limit_time, limit_token


class AlignedDataset(AnnotatedDataset):
    @torch.no_grad()
    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x, y, intervals = super().__getitem__(
            idx, return_intervals=True, waveform=waveform, return_clean=False
        )

        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        ylocs = torch.as_tensor(
            [int(intervals[j][0] * out_sr) for j in range(len(y))], dtype=torch.long
        )

        return x, y, ylocs

    def iterator(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # flat_labels=False
    ):
        def collate_fn(xys):
            xys = [(x, y, ylocs) for x, y, ylocs in xys if x is not None]

            if len(xys) == 0:
                return {
                    "inputs": None,
                    "labels": None,
                    "label_locations": None,
                    "input_lengths": None,
                    "label_lengths": None,
                }

            xs, ys, ylocs = zip(*xys)
            xlens = [len(x) for x in xs]
            ylens = [len(y) for y in ys]

            max_xlen = int(self.max_time * self.out_sr) if self.max_time else max(xlens)
            max_ylen = self.max_tokens if self.max_tokens else max(ylens)
            xs = torch.stack(
                [torch.nn.functional.pad(x, (0, 0, 0, max_xlen - len(x))) for x in xs],
                dim=1,
            )
            ys = torch.stack(
                [torch.nn.functional.pad(y, (0, max_ylen - len(y))) for y in ys], dim=1
            )
            ylocs = torch.stack(
                [
                    torch.nn.functional.pad(loc, (0, max_ylen - len(loc)))
                    for loc in ylocs
                ],
                dim=1,
            )
            xlens = torch.as_tensor(xlens, dtype=int)
            ylens = torch.as_tensor(ylens, dtype=int)

            if self.batch_first:
                xs = xs.transpose(0, 1).contiguous()
                ys = ys.transpose(0, 1).contiguous()
                ylocs = ylocs.transpose(0, 1).contiguous()

            return {
                "inputs": xs,
                "labels": ys,
                "label_locations": ylocs,
                "input_lengths": xlens,
                "label_lengths": ylens,
            }

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
        sounds,
        annotations,
        vocabulary,
        target="words",
        duration=1,
        scale=False,
        context=None,
        alignment="left",
        drop_modifiers=True,
        **kwargs,
    ):
        super().__init__(sounds, annotations, vocabulary, target, **kwargs)
        self.duration = duration
        self.scale = scale
        self.context = (context, 0) if np.isscalar(context) else context
        self.alignment = alignment
        self.drop_modifiers = drop_modifiers
        assert self.alignment in ["left", "center", "right"]

    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x, y, intervals = super().__getitem__(
            idx, return_intervals=True, return_clean=False, waveform=waveform
        )
        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        if x is None or y is None or intervals is None:
            return None, None, None

        if self.context and sum(self.context) > 0:
            xs, (pre_ctx, post_ctx) = [], self.context
            for start, stop in intervals:
                length = stop - start
                xnew = x[
                    max(int((start - length * pre_ctx) * out_sr), 0) : int(
                        (stop + length * post_ctx) * out_sr
                    )
                ]
                xnew = self._pad_axis(
                    xnew,
                    int((length * pre_ctx - start) * out_sr),
                    int((stop + length * post_ctx) * out_sr) - len(x),
                )
                xs.append(xnew)
            x = xs
        else:
            pre_ctx, post_ctx = 0, 0
            x = [
                x[int(start * out_sr) : int(stop * out_sr)]
                for (start, stop) in intervals
            ]

        fix_t = (
            int(self.duration * (1 + pre_ctx + post_ctx) * out_sr)
            if self.context
            else int(self.duration * out_sr)
        )
        freqs = np.linspace(1, self.freqbins, self.freqbins)
        xlens = []

        xys = [
            list(_)
            for _ in zip(
                *[
                    xyi
                    for xyi in zip(x, y, intervals)
                    if (len(xyi[0]) <= fix_t or self.scale)
                ]
            )
        ]
        if len(xys) == 0:
            return None, None, None
        x, y, intervals = xys

        for i, xi in enumerate(x):
            if self.scale or xi.shape[0] > fix_t:
                t0 = np.linspace(1, xi.shape[0], xi.shape[0])
                t1 = np.linspace(1, xi.shape[0], fix_t)

                if waveform or self.audio_proc is None:
                    # x[i] = sp.interpolate.InterpolatedUnivariateSpline(t0, xi)(t1)
                    # x[i] = librosa.effects.time_stretch(xi, xi.shape[0]/fix_t)[:fix_t]
                    pass
                else:
                    x[i] = sp.interpolate.RectBivariateSpline(t0, freqs, xi)(t1, freqs)
                xlens.append(fix_t)
            else:
                diff = fix_t - xi.shape[0]

                if self.alignment == "left":
                    pre_t, post_t = 0, diff
                elif self.alignment == "right":
                    pre_t, post_t = diff, 0
                else:
                    pre_t, post_t = math.floor(diff / 2), math.ceil(diff / 2)

                # if waveform or self.audio_proc is None:
                #    x[i] = np.concatenate([np.zeros(pre_t), xi, np.zeros(post_t)], axis=0)
                # else:
                #    x[i] = np.concatenate([np.zeros((pre_t, xi.shape[1])), xi, np.zeros((post_t, xi.shape[1]))], axis=0)
                x[i] = np.concatenate(
                    [
                        np.zeros((pre_t, xi.shape[1])),
                        xi,
                        np.zeros((post_t, xi.shape[1])),
                    ],
                    axis=0,
                )
                xlens.append(len(xi) + pre_t)

        x = np.stack(x, axis=0).astype("single")
        y = np.stack(y, axis=0).astype("int")
        xlens = np.array(xlens).astype("int")

        if self.drop_modifiers:
            x = x[xlens != 0]
            y = y[xlens != 0]
            xlens = xlens[xlens != 0]

        return x, y, xlens

    @staticmethod
    def _pad_axis(array, pre=0, post=0, axis=0):
        pre, post = max(pre, 0), max(post, 0)
        if pre == 0 and post == 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (pre, post)

        return np.pad(array, pad_width=npad, mode="constant", constant_values=0)

    def iterator(self, batch_size=1, shuffle=False, num_workers=0, return_len=False):
        def collate_fn(xys):
            xs, ys, xlens = zip(*xys)
            if any(y is None for y in ys):
                return (None, None, None) if return_len else (None, None)

            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            xlens = np.concatenate(xlens, axis=0)

            xs, ys = torch.as_tensor(xs), torch.as_tensor(ys)
            xlens = torch.as_tensor(xlens)

            if not self.batch_first:
                xs = xs.transpose(0, 1).contiguous()

            return (xs, ys, xlens) if return_len else (xs, ys)

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def take(self, n, shuffle=False, num_workers=0):
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

    def take_each(self, n, shuffle=False, num_workers=0):
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

        xs = torch.stack(xs, axis=0)
        ys = torch.stack(ys, axis=0)

        return xs, ys

    def take_statistics(self, n=int(1e9), shuffle=False, batch_size=1, num_workers=0):
        count = 0
        stats = dict(length=[], identity=[])

        pbar = tqdm(range(n))
        it = iter(
            self.iterator(
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                return_len=True,
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

        for k in stats.keys():
            stats[k] = np.concatenate(stats[k], axis=0)[:n]

        return stats


class BlockDataset(SoundDataset):
    def __init__(
        self,
        sounds,
        max_time=2.5,
        block_min=0.01,
        max_step=None,
        batch_first=False,
        return_clean=False,
        prepad=False,
        **kwargs,
    ):
        super().__init__(sounds, **kwargs)
        self.max_time = max_time
        self.block_min = block_min
        self.batch_first = batch_first
        self.max_step = max_time if max_step is None else max_step
        self.return_clean = return_clean
        self.prepad = prepad

    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x = super().__getitem__(idx, waveform=waveform, return_clean=self.return_clean)
        x, x0 = x if self.return_clean else (x, None)
        samplerate = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        maxblk = int(self.max_time * samplerate)
        minblk = int(self.block_min * samplerate)
        maxstp = int(self.max_step * samplerate)

        if self.prepad:
            x = np.pad(x, ((maxblk - 1, 0), (0, 0)))
            x0 = np.pad(x, ((maxblk - 1, 0), (0, 0))) if self.return_clean else None

        index = 0
        xs, x0s, xlens = [], [], []
        while index < len(x):
            length = np.random.randint(minblk, maxblk + 1)

            if index + length > len(x):
                length = len(x) - index
            if length < minblk:
                break

            xs.append(x[index : index + length])
            x0s.append(x0[index : index + length] if self.return_clean else None)
            xlens.append(length)
            index += min(length, maxstp)

        if len(xs) == 0:
            return (None, None, None) if self.return_clean else (None, None)

        if waveform or self.audio_proc is None:
            xs = np.stack([np.pad(x, ((0, maxblk - len(x)),)) for x in xs], axis=1)
            x0s = (
                np.stack([np.pad(x0, ((0, maxblk - len(x0)),)) for x0 in x0s], axis=1)
                if self.return_clean
                else None
            )
        else:
            xs = np.stack(
                [np.pad(x, ((0, maxblk - len(x)), (0, 0))) for x in xs], axis=1
            )
            x0s = (
                np.stack(
                    [np.pad(x0, ((0, maxblk - len(x0)), (0, 0))) for x0 in x0s], axis=1
                )
                if self.return_clean
                else None
            )
        xlens = np.array(xlens, dtype=int)

        return (xs, xlens, x0s) if self.return_clean else (xs, xlens)

    def iterator(
        self,
        batch_size=1,
        batch_max=None,
        shuffle=False,
        num_workers=0,
        # flat_labels=False,
    ):
        def collate_fn(xys):
            if self.return_clean:
                xys = [(x, l, x0) for x, l, x0 in xys if x is not None]
            else:
                xys = [(x, l) for x, l in xys if x is not None]

            if len(xys) == 0:
                if self.return_clean:
                    return {"inputs": None, "input_lengths": None, "inputs_clean": None}
                else:
                    return {"inputs": None, "input_lengths": None}

            if self.return_clean:
                xs, xlens, x0s = zip(*xys)
            else:
                xs, xlens, x0s = (*zip(*xys), None)

            xs = np.concatenate(xs, axis=1)
            xlens = np.concatenate(xlens, axis=0)
            x0s = np.concatenate(x0s, axis=1) if self.return_clean else None

            xs = torch.as_tensor(xs)
            xlens = torch.as_tensor(xlens)
            x0s = torch.as_tensor(x0s) if self.return_clean else None

            if batch_max is not None:
                xs = xs[:batch_max] if self.batch_first else xs[:, :batch_max]
                xlens = xlens[:batch_max]
                x0s = (
                    (x0s[:batch_max] if self.batch_first else x0s[:, :batch_max])
                    if self.return_clean
                    else None
                )

            if self.batch_first:
                xs = xs.transpose(0, 1).contiguous()
                x0s = x0s.transpose(0, 1).contiguous() if self.return_clean else None

            if self.return_clean:
                return {"inputs": xs, "input_lengths": xlens, "inputs_clean": x0s}
            else:
                return {"inputs": xs, "input_lengths": xlens}

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
        return_clean=False,
        **kwargs,
    ):
        super().__init__(sounds, annotations, vocabulary, target=target, **kwargs)
        self.seq_size = seq_size
        self.seq_min = seq_min if seq_min else seq_size
        self.seq_time = seq_time
        self.seq_overlap = seq_overlap
        self.check_boundaries = check_boundaries
        self.return_clean = return_clean

    @torch.no_grad()
    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x = super().__getitem__(
            idx,
            return_intervals=True,
            waveform=waveform,
            return_clean=self.return_clean,
        )
        x, y, intervals, x0 = x if self.return_clean else (*x, None)

        if intervals is None:
            return (
                (None, None, None, None, None)
                if self.return_clean
                else (None, None, None, None)
            )

        samplerate = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        maxblk = int(self.seq_time * samplerate)

        index = 0
        xs, ys, xlens, ylens, x0s, xpos, ypos = [], [], [], [], [], [], []
        while index < len(intervals):
            if self.check_boundaries and _is_postfix(self.vocabulary[y[index]]):
                index += 1
                continue

            length = np.random.randint(self.seq_min, self.seq_size + 1)

            if index + length > len(intervals):
                length = len(intervals) - index
            if length < self.seq_min:
                break

            if self.check_boundaries:
                init_length = length

                while (
                    index + length <= len(intervals)
                    and length <= self.seq_size
                    and _is_prefix(self.vocabulary[y[index + length - 1]])
                ):
                    length += 1
                if length > self.seq_size or index + length > len(intervals):
                    length = init_length

                while length >= self.seq_min and _is_prefix(
                    self.vocabulary[y[index + length - 1]]
                ):
                    length -= 1
                if length < self.seq_min:
                    index += 1
                    continue

            start = max(int(intervals[index][0] * samplerate) - 1, 0)
            stop = int(intervals[index + length - 1][1] * samplerate) + 2
            while stop - start > maxblk and length > self.seq_min:
                length -= 1
                stop = int(intervals[index + length - 1][1] * samplerate) + 2
            if stop - start > maxblk:
                index += 1
                continue

            xs.append(x[start:stop])
            ys.append(y[index : index + length])
            xlens.append(stop - start)
            ylens.append(length)
            x0s.append(x0[start:stop] if self.return_clean else None)
            xpos.append((start, stop))
            ypos.append((index, index + length))
            index += 1 if self.seq_overlap else length

        if len(xs) == 0:
            return (
                (None, None, None, None, None, None, None, None)
                if self.return_clean
                else (None, None, None, None, None, None, None)
            )

        if waveform or self.audio_proc is None:
            xs = torch.stack(
                [torch.nn.functional.pad(x, (0, maxblk - len(x))) for x in xs], dim=1
            )
            x0s = (
                torch.stack(
                    [torch.nn.functional.pad(x, (0, maxblk - len(x))) for x in x0s],
                    dim=1,
                )
                if self.return_clean
                else None
            )
        else:
            xs = torch.stack(
                [torch.nn.functional.pad(x, (0, 0, 0, maxblk - len(x))) for x in xs],
                dim=1,
            )
            x0s = (
                torch.stack(
                    [
                        torch.nn.functional.pad(x, (0, 0, 0, maxblk - len(x)))
                        for x in x0s
                    ],
                    dim=1,
                )
                if self.return_clean
                else None
            )
        ys = torch.stack(
            [torch.nn.functional.pad(y, (0, self.seq_size - len(y))) for y in ys], dim=1
        )
        xlens = torch.tensor(xlens, dtype=int)
        ylens = torch.tensor(ylens, dtype=int)
        xpos = torch.tensor(xpos, dtype=int)
        ypos = torch.tensor(ypos, dtype=int)

        return (
            (xs, ys, xlens, ylens, x0s, xpos, ypos)
            if self.return_clean
            else (xs, ys, xlens, ylens, xpos, ypos)
        )

    def iterator(
        self,
        batch_size=1,
        batch_max=None,
        shuffle=False,
        num_workers=0,
        # flat_labels=False,
    ):
        def collate_fn(xys):
            if self.return_clean:
                xys = [
                    (x, y, xl, yl, x0, xpos, ypos)
                    for x, y, xl, yl, x0, xpos, ypos in xys
                    if x is not None
                ]
            else:
                xys = [
                    (x, y, xl, yl, xpos, ypos)
                    for x, y, xl, yl, xpos, ypos in xys
                    if x is not None
                ]

            if len(xys) == 0:
                if self.return_clean:
                    return {
                        "inputs": None,
                        "inputs_clean": None,
                        "labels": None,
                        "input_lengths": None,
                        "label_lengths": None,
                        "xpositions": None,
                        "ypositions": None,
                    }
                else:
                    return {
                        "inputs": None,
                        "labels": None,
                        "input_lengths": None,
                        "label_lengths": None,
                        "xpositions": None,
                        "ypositions": None,
                    }

            if self.return_clean:
                xs, ys, xlens, ylens, x0s, xpos, ypos = zip(*xys)
            else:
                xs, ys, xlens, ylens, xpos, ypos = zip(*xys)

            xs = torch.cat(xs, dim=1)
            ys = torch.cat(ys, dim=1)  # sum(ys, start=[])
            xlens = torch.cat(xlens, dim=0)
            ylens = torch.cat(ylens, dim=0)
            x0s = torch.cat(x0s, dim=1) if self.return_clean else None
            xpos = torch.cat(xpos, dim=0)
            ypos = torch.cat(ypos, dim=0)

            batch_size = xs.shape[1]
            if batch_max is not None:
                xs = (
                    xs[:, :batch_max]
                    if batch_size >= batch_max
                    else torch.nn.functional.pad(xs, (0, 0, 0, batch_max - batch_size))
                )
                ys = (
                    ys[:, :batch_max]
                    if batch_size >= batch_max
                    else torch.nn.functional.pad(ys, (0, batch_max - batch_size))
                )
                xlens = (
                    xlens[:batch_max]
                    if batch_size >= batch_max
                    else torch.nn.functional.pad(xlens, (0, batch_max - batch_size))
                )
                ylens = (
                    ylens[:batch_max]
                    if batch_size >= batch_max
                    else torch.nn.functional.pad(ylens, (0, batch_max - batch_size))
                )
                x0s = (
                    (
                        x0s[:, :batch_max]
                        if batch_size >= batch_max
                        else torch.nn.functional.pad(
                            x0s, (0, 0, 0, batch_max - batch_size)
                        )
                    )
                    if self.return_clean
                    else None
                )
                xpos = (
                    xpos[:batch_max]
                    if batch_size >= batch_max
                    else torch.nn.functional.pad(
                        xpos, (0, 0, 0, batch_max - batch_size)
                    )
                )
                ypos = (
                    ypos[:batch_max]
                    if batch_size >= batch_max
                    else torch.nn.functional.pad(
                        ypos, (0, 0, 0, batch_max - batch_size)
                    )
                )

            if self.batch_first:
                xs = xs.transpose(0, 1).contiguous()
                x0s = x0s.transpose(0, 1).contiguous() if self.return_clean else None
                ys = ys.transpose(0, 1).contiguous()

            if self.return_clean:
                return {
                    "inputs": xs,
                    "inputs_clean": x0s,
                    "labels": ys,
                    "input_lengths": xlens,
                    "label_lengths": ylens,
                    "xpositions": xpos,
                    "ypositions": ypos,
                }
            else:
                return {
                    "inputs": xs,
                    "labels": ys,
                    "input_lengths": xlens,
                    "label_lengths": ylens,
                    "xpositions": xpos,
                    "ypositions": ypos,
                }

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
            for k, l in lengths.items()
            if (vocabulary is None or k in vocabulary)
            and len(l) >= 100
            and np.quantile(l, 0.5 + alpha / 2) / np.quantile(l, 0.5 - alpha / 2)
            >= max_stretch
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

    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x, y, intervals = super().__getitem__(
            idx, return_intervals=True, waveform=waveform, return_clean=False
        )
        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr
        if x is None or y is None or intervals is None:
            return None, None, None

        yint_filt = [
            (token, intv)
            for token, intv in zip(y, intervals)
            if self.acc_range[self.vocabulary[token]][0]
            <= intv[1] - intv[0]
            <= self.acc_range[self.vocabulary[token]][1]
        ]
        if len(yint_filt) == 0:
            return None, None, None
        y, intervals = zip(*yint_filt)

        intervals = [
            (int(start * out_sr), int(stop * out_sr)) for start, stop in intervals
        ]
        # durations = [stop - start for start, stop in intervals]
        centers = [(start + stop) // 2 for start, stop in intervals]

        context = int(self.context * out_sr)
        pre_context, post_context = math.floor(context / 2), math.ceil(context / 2)
        intervals = [
            (center - pre_context, center + post_context) for center in centers
        ]
        yint_filt = [
            (token, intv)
            for token, intv in zip(y, intervals)
            if intv[0] >= 0 and intv[1] <= len(x)
        ]
        if len(yint_filt) == 0:
            return None, None, None
        y, intervals = zip(*yint_filt)

        x = [x[start:stop] for start, stop in intervals]
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

            for i, xi in enumerate(x):
                if waveform or self.audio_proc is None:
                    # x[i] = librosa.effects.time_stretch(xi, context/target_t)
                    pass
                else:
                    x[i] = sp.interpolate.RectBivariateSpline(t0, freqs, xi)(t1, freqs)

        xlens = [len(_) for _ in x]

        x = np.stack(x, axis=0).astype("single")
        y = np.stack(y, axis=0).astype("int")
        xlens = np.array(xlens).astype("int")

        x = x[xlens != 0]
        y = y[xlens != 0]
        xlens = xlens[xlens != 0]

        return x, y, xlens

    @staticmethod
    def _pad_axis(array, pre=0, post=0, axis=0):
        pre, post = max(pre, 0), max(post, 0)
        if pre == 0 and post == 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (pre, post)

        return np.pad(array, pad_width=npad, mode="constant", constant_values=0)

    def iterator(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        return_len=False,
        # flat_labels=False,
    ):
        def collate_fn(xys):
            xs, ys, xlens = zip(*xys)
            if any(y is None for y in ys):
                return (None, None, None) if return_len else (None, None)

            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            xlens = np.concatenate(xlens, axis=0)

            xs, ys = torch.as_tensor(xs), torch.as_tensor(ys)
            xlens = torch.as_tensor(xlens)

            if not self.batch_first:
                xs = xs.transpose(0, 1).contiguous()

            return (xs, ys, xlens) if return_len else (xs, ys)

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
    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

        x, skew = super().__getitem__(idx, return_skew=True, waveform=waveform)
        out_sr = self.in_sr if waveform or self.audio_proc is None else self.out_sr

        annotation = self.annotations[idx]

        try:
            p = self._annotation(annotation, "phones", skew=skew)
            s = self._annotation(annotation, "syllables", skew=skew)
            w = self._annotation(annotation, "words", skew=skew)
        except EmptySequence:
            return (None,) * 4

        return x, p, s, w

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

    def _annotation(self, annotation, target_type, ignore_silence=True, skew=None):
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
            target = [re.sub("([A-Z]+)[0-9]", "\g<1>", token) for token in target]

        if target_type == "words" and self.normalize:
            expanded_target, expanded_interv = [], []
            for token, intv in zip(target, interv):
                subtokens = _normalize(token)  # , self.vocabulary)
                expanded_target += subtokens
                expanded_interv += [
                    intv
                    if not _is_subtoken(t)
                    else (intv[:1] * 2 if _is_prefix(t) else intv[1:] * 2)
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

        if len(target) == 0:
            raise EmptySequence()

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
        filter_vocab=True,
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
        for k, l in zip(stats["identity"], stats["length"]):
            if k in lengths:
                lengths[k].append(l)
            else:
                lengths[k] = [l]

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
            for k, l in lengths.items()
            if (vocabulary["words"] is None or k in vocabulary["words"])
            and len(l) >= 100
            and np.quantile(l, 0.5 + alpha / 2) / np.quantile(l, 0.5 - alpha / 2)
            >= max_stretch
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

    def __getitem__(self, idx, waveform=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return (self[i] for i in range(start, stop, step))

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
        p_vec = np.zeros(y_len, dtype=int)
        s_vec = np.zeros(y_len, dtype=int)
        w_vec = np.zeros(y_len, dtype=int)
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

        x_durations = [stop - start for start, stop in x_intervals]
        y_durations = [stop - start for start, stop in y_intervals]

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

        x = np.stack(x)[nonempty]
        y = np.array(y)[nonempty]
        p = np.stack(p)[nonempty]
        s = np.stack(s)[nonempty]
        w = np.stack(w)[nonempty]

        return x, y, p, s, w

    @staticmethod
    def _pad_axis(array, pre=0, post=0, axis=0):
        pre, post = max(pre, 0), max(post, 0)
        if pre == 0 and post == 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (pre, post)

        return np.pad(array, pad_width=npad, mode="constant", constant_values=0)

    def iterator(self, batch_size=1, shuffle=False, num_workers=0, flat_labels=False):
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

        xs = torch.stack(xs, axis=0)[:n]
        ys = (
            np.array([self.vocabulary["words"][y] for y in ys])[:n]
            if return_labels
            else torch.tensor(ys)[:n]
        )
        ps = torch.stack(ps)[:n]
        ss = torch.stack(ss)[:n]
        ws = torch.stack(ws)[:n]

        return xs, ys, ps, ss, ws
