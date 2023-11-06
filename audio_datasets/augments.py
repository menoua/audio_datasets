import math
from typing import Callable

import numpy as np
from torch import Tensor
from torchaudio import load as load_audio
from torchaudio.sox_effects import apply_effects_tensor as apply_sox_effects

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


def apply_speech_modifier(
    audio: Tensor, sr: int, config: dict = CONFIG_MOD_MID
) -> tuple[Tensor, int]:
    fx = np.random.choice(("pitch", "speed", "tempo"))

    if fx == "pitch":
        n_semitones = np.random.uniform(*config["PITCH_SHIFT"])
        tfm = [["pitch", str(n_semitones)], ["rate", str(sr)]]
    elif fx == "speed":
        factor = np.random.uniform(*config["SPEED_RATE"])
        tfm = [["speed", str(factor)], ["rate", str(sr)]]
    elif fx == "tempo":
        factor = np.random.uniform(*config["TEMPO_RATE"])
        tfm = [["tempo", "-s", str(factor)], ["rate", str(sr)]]
    else:
        raise RuntimeError("Reached unreachable state")

    return apply_sox_effects(audio, sr, tfm)


def apply_room_modifier(
    audio: Tensor, sr: int, config: dict = CONFIG_MOD_MID
) -> tuple[Tensor, int]:
    # fx = np.random.choice(('chorus', 'echo', 'reverb'))
    fx = np.random.choice(("reverb",))

    if fx == "chorus":
        n_voices = np.random.randint(*config["CHORUS_N"])
        delays = np.random.uniform(40, 60, n_voices)
        decays = np.random.uniform(0.25, 0.45, n_voices)
        speeds = np.random.uniform(0.2, 0.4, n_voices)
        depths = np.random.uniform(1.0, 3.0, n_voices)
        modulations = np.random.choice(("s", "t"), n_voices)
        tfm = [
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
        ]
    elif fx == "echo":
        n_echos = np.random.randint(*config["ECHO_N"])
        max_delay = np.random.randint(50, 300)
        delays = list(np.linspace(0, max_delay, n_echos + 1))[1:]
        min_decay = np.random.uniform(0.1, 0.7)
        decays = list(np.linspace(1, min_decay, n_echos + 1))[1:]
        tfm = [
            [
                "echo",
                "0.8",
                "0.9",
                *sum([[str(_) for _ in i] for i in zip(delays, decays)], []),
            ]
        ]
    elif fx == "reverb":
        reverberance = np.random.uniform(*config["REVERB"])
        room_scale = np.random.uniform(50, 100)
        tfm = [
            ["reverb", "--wet-only", str(reverberance), "50", str(room_scale)],
            ["channels", "1"],
        ]
    else:
        raise RuntimeError("Reached unreachable state")

    return apply_sox_effects(audio, sr, tfm)


def apply_channel_modifier(
    audio: Tensor, sr: int, config: dict = CONFIG_MOD_MID
) -> tuple[Tensor, int]:
    fx = np.random.choice(("lowpass", "highpass", "bandpass", "bandstop"))

    if fx == "lowpass":
        cutoff_freq = np.random.uniform(*config["LOWPASS_F"])
        tfm = [["sinc", "-" + str(cutoff_freq)]]
    elif fx == "highpass":
        cutoff_freq = np.random.uniform(*config["HIGHPASS_F"])
        tfm = [["sinc", str(cutoff_freq)]]
    elif fx == "bandpass":
        cutoff_low = np.random.uniform(*config["BANDPASS_F"])
        cutoff_high = cutoff_low * np.random.uniform(*config["BANDPASS_W"])
        tfm = [["sinc", str(cutoff_low) + "-" + str(cutoff_high)]]
    elif fx == "bandstop":
        cutoff_low = np.random.uniform(*config["BANDSTOP_F"])
        cutoff_high = cutoff_low * np.random.uniform(*config["BANDSTOP_W"])
        tfm = [["sinc", str(cutoff_high) + "-" + str(cutoff_low)]]
    else:
        raise RuntimeError("Reached unreachable state")

    return apply_sox_effects(audio, sr, tfm)


def apply_noise_modifier(
    audio: Tensor, sr: int, noise_sounds: list[str], config: dict = CONFIG_MOD_MID
) -> tuple[Tensor, int]:
    if not noise_sounds:
        return audio, sr

    noise_id = np.random.choice(len(noise_sounds))
    noise_audio, noise_sr = load_audio(noise_sounds[noise_id])

    tfm = [["rate", str(sr)], ["channels", "1"]]
    # repeat to cover full speech
    dur_speech = audio.shape[1] / sr
    dur_noise = noise_audio.shape[1] / noise_sr
    count = math.ceil(dur_speech / dur_noise)
    tfm.append(["repeat", str(count)])
    # trim to same length as speech
    tfm.append(["trim", "0", str(dur_speech)])
    # set volume
    snr_db = np.random.uniform(*config["BG_SNR"])
    tfm.append(["norm", str(-3 - snr_db)])
    # process noise and add to foreground
    noise_audio, noise_sr = apply_sox_effects(noise_audio, noise_sr, tfm)
    audio = (audio + noise_audio[:, : audio.shape[1]]) / np.sqrt(2)

    return audio, sr


def apply_fixed_tempo(
    audio: Tensor, sr: int, _: dict, factor: float
) -> tuple[Tensor, int]:
    tfm = [["tempo", "-s", str(factor)], ["rate", str(sr)]]
    return apply_sox_effects(audio, sr, tfm)
