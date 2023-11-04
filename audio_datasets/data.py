import glob
import math
import re

import pandas as pd

from . import ROOT_DATA_DIR

DEFAULT_PARAMS = {}
UNIFORM_FORMAT = True


def _ufmt(filename):
    base, fmt = filename.rsplit(".", maxsplit=1)
    return ".".join((base, "flac" if UNIFORM_FORMAT else fmt))


def LibriSpeech(basepath=ROOT_DATA_DIR + "/LibriSpeech", subset="train-*"):
    annotations = glob.glob(basepath + "/" + subset + "/*/*/*.TextGrid")
    sounds = [_ufmt(path.replace(".TextGrid", ".flac")) for path in annotations]
    annotations = ["libri:" + path for path in annotations]
    return sounds, annotations


def SpokenWikiEnglish(basepath=ROOT_DATA_DIR + "/SWC_English"):
    sounds = glob.glob(_ufmt(basepath + "/*/audio.ogg"))
    annotations = ["swc:" + path.replace("audio.ogg", "aligned.swc") for path in sounds]
    return sounds, annotations


def TedLium3(basepath=ROOT_DATA_DIR + "/TEDLIUM_release-3"):
    sounds = glob.glob(_ufmt(basepath + "/data/sph/*.sph"))
    annotations = [
        "tedlium:" + re.sub("/sph/(.*).sph", "/stm/\\1.stm", path) for path in sounds
    ]
    return sounds, annotations


def SynthNoise(basepath=ROOT_DATA_DIR + "/SynthNoise"):
    return glob.glob(_ufmt(basepath + "/pink_*.wav"))


def FMA(basepath=ROOT_DATA_DIR + "/FMA"):
    # Remove [ 080391 , 106628 ] -- corrupt?
    blacklist = [80391, 106628]

    table = pd.read_csv(
        basepath + "/fma_metadata/tracks.csv",
        skiprows=[0, 1, 2],
        names=["id", "genre_top"],
        usecols=[0, 40],
        index_col=0,
    ).groupby("genre_top")
    tracks = list(
        pd.concat(
            [table.get_group("Classical")]  # table.get_group('Instrumental'),
        ).index
    )
    return [
        _ufmt(f"{basepath}/fma_large/{math.floor(i/1000):03d}/{i:06d}.mp3")
        for i in tracks
        if i not in blacklist
    ]


def ESC50(basepath=ROOT_DATA_DIR + "/ESC-50-master"):
    return glob.glob(_ufmt(basepath + "/audio/*.wav"))


def FSD2019(basepath=ROOT_DATA_DIR + "/FSDKaggle2019"):
    groups = [
        "Accelerating_and_revving_and_vroom",
        "Accordion",
        "Acoustic_guitar",
        "Applause",
        "Bark",
        "Bass_drum",
        "Bass_guitar",
        "Bathtub_(filling_or_washing)",
        "Bicycle_bell",
        "Bus",
        "Buzz",
        "Car_passing_by",
        "Cheering",
        "Chewing_and_mastication",
        "Chink_and_clink",
        "Chirp_and_tweet",
        "Church_bell",
        "Clapping",
        "Computer_keyboard",
        "Crackle",
        "Cricket",
        "Crowd",
        "Cupboard_open_or_close",
        "Cutlery_and_silverware",
        "Dishes_and_pots_and_pans",
        "Drawer_open_or_close",
        "Drip",
        "Electric_guitar",
        "Fill_(with_liquid)",
        "Finger_snapping",
        "Frying_(food)",
        "Gasp",
        "Glockenspiel",
        "Gong",
        "Gurgling",
        "Harmonica",
        "Hi-hat",
        "Hiss",
        "Keys_jangling",
        "Knock",
        "Marimba_and_xylophone",
        "Mechanical_fan",
        "Meow",
        "Microwave_oven",
        "Motorcycle",
        "Printer",
        "Purr",
        "Raindrop",
        "Run",
        "Scissors",
        "Screaming",
        "Shatter",
        "Sigh",
        "Sink_(filling_or_washing)",
        "Skateboard",
        "Slam",
        "Sneeze",
        "Squeak",
        "Stream",
        "Strum",
        "Tap",
        "Tick-tock",
        "Traffic_noise_and_roadway_noise",
        "Trickle_and_dribble",
        "Walk_and_footsteps",
        "Water_tap_and_faucet",
        "Waves_and_surf",
        "Writing",
        "Zipper_(clothing)",
    ]

    table = pd.read_csv(
        basepath + "/FSDKaggle2019.meta/train_curated_post_competition.csv",
        usecols=["fname", "labels"],
    )
    tracks = [
        table["fname"][i]
        for i in range(len(table))
        if all(label in groups for label in table["labels"][i].split(","))
    ]
    return [
        _ufmt(f"{basepath}/FSDKaggle2019.audio_train_curated/{track}")
        for track in tracks
    ]


def UrbanSound(basepath=ROOT_DATA_DIR + "/UrbanSound"):
    return glob.glob(_ufmt(basepath + "/data/*/*.wav")) + glob.glob(
        _ufmt(basepath + "/data/*/*.mp3")
    )


def Rouen(basepath=ROOT_DATA_DIR + "/Rouen"):
    return glob.glob(_ufmt(basepath + "/*.wav"))


def TAU2019(basepath=ROOT_DATA_DIR + "/TAU-urban-acoustic-scenes-2019-development"):
    return glob.glob(_ufmt(basepath + "/audio/*.wav"))


# def MedleyDB(basepath=ROOT_DATA_DIR + "/RWCP"):
#     return []


# def MixingSecrets(basepath=ROOT_DATA_DIR + "/RWCP"):
#     return []


# def NSynth(basepath=ROOT_DATA_DIR + "/RWCP"):
#     return []


def Speech(nested=False):
    if nested:
        return [LibriSpeech(), SpokenWikiEnglish(), TedLium3()]
    else:
        lib = LibriSpeech()
        swc = SpokenWikiEnglish()
        ted = TedLium3()
        return lib[0] + swc[0] + ted[0], lib[1] + swc[1] + ted[1]


def NonSpeech(nested=False):
    if nested:
        return [
            SynthNoise(),
            FMA(),
            ESC50(),
            FSD2019(),
            UrbanSound(),
            Rouen(),
            TAU2019(),
        ]
    else:
        return (
            SynthNoise()
            + FMA()
            + ESC50()
            + FSD2019()
            + UrbanSound()
            + Rouen()
            + TAU2019()
        )


# MedleyDB problematics
# > Audio/AimeeNorwich_Child/AimeeNorwich_Child_MIX.wav
# > Audio/TrevorAndTheSoundflaces_AloneAndSad/TrevorAndTheSoundwaves_AloneAndSad_MIX.wav

# SWC_English problematics
# > Gas_reinjection/audio.ogg
# > Pacific_Southwest_Airlines_Flight_1771/audio.ogg
