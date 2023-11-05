import os

if "ROOT_DATA_DIR" in os.environ:
    ROOT_DATA_DIR = os.environ["ROOT_DATA_DIR"]
else:
    raise RuntimeError("Environment variable `ROOT_DATA_DIR` needs to be set.")

if "ROOT_VOCAB_DIR" in os.environ:
    ROOT_VOCAB_DIR = os.environ["ROOT_VOCAB_DIR"]
else:
    ROOT_VOCAB_DIR = os.path.join(ROOT_DATA_DIR, "Vocabulary")

import audio_datasets.core
import audio_datasets.data
import audio_datasets.dataloaders
import audio_datasets.lexicon
import audio_datasets.limits
import audio_datasets.transforms
import audio_datasets.utils
