import numpy as np

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


def add_speech_modifier(
    tfm: list[list[str]], sr: int, config: dict = CONFIG_MOD_MID
) -> list[list[str]]:
    fx = np.random.choice(("pitch", "speed", "tempo"))

    if fx == "pitch":
        n_semitones = np.random.uniform(*config["PITCH_SHIFT"])
        tfm.append(["pitch", str(n_semitones)])
        tfm.append(["rate", str(sr)])
    elif fx == "speed":
        factor = np.random.uniform(*config["SPEED_RATE"])
        tfm.append(["speed", str(factor)])
        tfm.append(["rate", str(sr)])
    elif fx == "tempo":
        factor = np.random.uniform(*config["TEMPO_RATE"])
        # if 0.9 < factor < 1.1:
        #    tfm.append(['stretch', str(1/factor)])
        # else:
        #    tfm.append(['tempo', '-s', str(factor)])
        tfm.append(["tempo", "-s", str(factor)])
        tfm.append(["rate", str(sr)])

    return tfm


def add_room_modifier(
    tfm: list[list[str]], config: dict = CONFIG_MOD_MID
) -> list[list[str]]:
    # fx = np.random.choice(('chorus', 'echo', 'reverb'))
    fx = np.random.choice(("reverb",))

    if fx == "chorus":
        n_voices = np.random.randint(*config["CHORUS_N"])
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
        n_echos = np.random.randint(*config["ECHO_N"])
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
        reverberance = np.random.uniform(*config["REVERB"])
        room_scale = np.random.uniform(50, 100)
        tfm.append(["reverb", "--wet-only", str(reverberance), "50", str(room_scale)])
        tfm.append(["channels", "1"])

    return tfm


def add_channel_modifier(
    tfm: list[list[str]], config: dict = CONFIG_MOD_MID
) -> list[list[str]]:
    fx = np.random.choice(("lowpass", "highpass", "bandpass", "bandstop"))

    if fx == "lowpass":
        cutoff_freq = np.random.uniform(*config["LOWPASS_F"])
        tfm.append(["sinc", "-" + str(cutoff_freq)])
    elif fx == "highpass":
        cutoff_freq = np.random.uniform(*config["HIGHPASS_F"])
        tfm.append(["sinc", str(cutoff_freq)])
    elif fx == "bandpass":
        cutoff_low = np.random.uniform(*config["BANDPASS_F"])
        cutoff_high = cutoff_low * np.random.uniform(*config["BANDPASS_W"])
        tfm.append(["sinc", str(cutoff_low) + "-" + str(cutoff_high)])
    elif fx == "bandstop":
        cutoff_low = np.random.uniform(*config["BANDSTOP_F"])
        cutoff_high = cutoff_low * np.random.uniform(*config["BANDSTOP_W"])
        tfm.append(["sinc", str(cutoff_high) + "-" + str(cutoff_low)])

    return tfm
