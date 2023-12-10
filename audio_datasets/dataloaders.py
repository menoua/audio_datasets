from typing import Callable, Optional

from .core import AnnotatedDataset, SequenceDataset, TokenizedDataset
from .data import LibriSpeech, NonSpeech
from .lexicon import LABELS
from .limits import LIMITS_WORD, Limits
from .transforms import mel_spectrogram


class LibriSpeechDataloader:
    def __init__(
        self,
        dataset_type=AnnotatedDataset,
        target: str = "words",
        labels: Optional[list[str]] = None,
        limits: Optional[Limits] = LIMITS_WORD["librispeech"]["max"],
        batch_size: int = 12,
        num_workers: int = 4,
        flat_labels: bool = False,
        batch_first: bool = True,
        audio_transform: Optional[Callable] = mel_spectrogram(),
        mod_speech: bool = False,
        mod_room: bool = False,
        mod_channel: bool = False,
        mod_scene: list[str] = [],  # NonSpeech(),
        mod_foreground: Optional[Callable] = None,
        mod_final: Optional[Callable] = None,
        mod_intensity: str = "mid",
    ):
        if labels is None:
            labels = LABELS[target]

        self.dataset_type = dataset_type

        sounds, annots = {}, {}
        sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
        sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
        sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
        self.sounds, self.annots = sounds, annots

        self.data_config = {
            "batch_first": batch_first,
            "target": target,
            "vocabulary": labels,
            "limits": limits,
            "audio_transform": audio_transform,
            "normalize": len([w for w in labels if "|" in w]) > 0,
        }

        self.mod_config = {
            "mod_speech": mod_speech,
            "mod_room": mod_room,
            "mod_channel": mod_channel,
            "mod_scene": mod_scene,
            "mod_foreground": mod_foreground,
            "mod_final": mod_final,
            "mod_intensity": mod_intensity,
        }

        self.dataloader_config = {
            "flat_labels": flat_labels,
            "batch_size": batch_size,
            "num_workers": num_workers,
        }

    def train_dataloader(
        self,
        data_config: dict = {},
        mod_config: dict = {},
        dataloader_config: dict = {},
    ):
        data_config = {**self.data_config, **data_config}
        dataset = self.dataset_type(
            self.sounds["train"], self.annots["train"], **data_config
        )

        mod_config = {**self.mod_config, **mod_config}
        dataset.augment(**mod_config)

        dataloader_config = {
            **self.dataloader_config,
            "shuffle": True,
            **dataloader_config,
        }
        return dataset.iterator(**dataloader_config)

    def val_dataloader(
        self,
        data_config: dict = {},
        mod_config: dict = {},
        dataloader_config: dict = {},
    ):
        data_config = {**self.data_config, **data_config}
        dataset = self.dataset_type(
            self.sounds["val"], self.annots["val"], **data_config
        )

        dataset.augment(**mod_config)

        dataloader_config = {
            **self.dataloader_config,
            "shuffle": False,
            **dataloader_config,
        }
        dataloader_config = {
            **dataloader_config,
            "batch_size": max(dataloader_config["batch_size"] // 2, 1),
        }
        return dataset.iterator(**dataloader_config)

    def test_dataloader(
        self,
        data_config: dict = {},
        mod_config: dict = {},
        dataloader_config: dict = {},
    ):
        data_config = {**self.data_config, **data_config}
        dataset = self.dataset_type(
            self.sounds["test"], self.annots["test"], **data_config
        )

        dataset.augment(**mod_config)

        dataloader_config = {
            **self.dataloader_config,
            "shuffle": False,
            **dataloader_config,
        }
        dataloader_config = {
            **dataloader_config,
            "batch_size": max(dataloader_config["batch_size"] // 2, 1),
        }
        return dataset.iterator(**dataloader_config)


class LibriSpeechSequenceDataloader(LibriSpeechDataloader):
    def __init__(
        self,
        dataset_type=SequenceDataset,
        seq_size: int = 20,
        seq_min: int = 1,
        seq_time: float = 8.0,
        seq_per_sample: float = 4.0,
        seq_overlap: bool = False,
        check_boundaries: bool = True,
        **kwargs,
    ):
        super().__init__(dataset_type=dataset_type, **kwargs)
        self.seq_per_sample = seq_per_sample

        self.data_config = {
            **self.data_config,
            "seq_size": seq_size,
            "seq_min": seq_min,
            "seq_time": seq_time,
            "seq_overlap": seq_overlap,
            "check_boundaries": check_boundaries,
        }

    def train_dataloader(
        self,
        data_config: dict = {},
        mod_config: dict = {},
        dataloader_config: dict = {},
    ):
        dataloader_config = {**self.dataloader_config, **dataloader_config}
        dataloader_config = {
            **dataloader_config,
            "batch_max": int(dataloader_config["batch_size"] * self.seq_per_sample),
        }
        return super().train_dataloader(
            data_config=data_config,
            mod_config=mod_config,
            dataloader_config=dataloader_config,
        )

    def val_dataloader(
        self,
        data_config: dict = {},
        mod_config: dict = {},
        dataloader_config: dict = {},
    ):
        dataloader_config = {**self.dataloader_config, **dataloader_config}
        dataloader_config = {
            **dataloader_config,
            "batch_max": int(dataloader_config["batch_size"] * self.seq_per_sample),
        }
        return super().val_dataloader(
            data_config=data_config,
            mod_config=mod_config,
            dataloader_config=dataloader_config,
        )

    def test_dataloader(
        self,
        data_config: dict = {},
        mod_config: dict = {},
        dataloader_config: dict = {},
    ):
        dataloader_config = {**self.dataloader_config, **dataloader_config}
        dataloader_config = {
            **dataloader_config,
            "batch_max": int(dataloader_config["batch_size"] * self.seq_per_sample),
        }
        return super().test_dataloader(
            data_config=data_config,
            mod_config=mod_config,
            dataloader_config=dataloader_config,
        )


class LibriSpeechTokenDataloader(LibriSpeechDataloader):
    def __init__(self, dataset_type=TokenizedDataset, **kwargs):
        super().__init__(dataset_type=dataset_type, **kwargs)
        self.data_config = {**self.data_config}
        del self.dataloader_config["flat_labels"]


def librispeech(
    target: str = "words",
    vocabulary: Optional[list[str]] = None,
    limits: Optional[Limits] = LIMITS_WORD["librispeech"]["max"],
    batch_size: int = 12,
    num_workers: int = 4,
    flat_labels: bool = False,
    audio_transform: Optional[Callable] = mel_spectrogram(),
    split: str = "val",
):
    sounds, annots = {}, {}
    sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
    sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
    sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
    vocabulary = LABELS[target] if vocabulary is None else vocabulary

    data_config = {
        "batch_first": True,
        "target": target,
        "vocabulary": vocabulary,
        "limits": limits,
        "audio_transform": audio_transform,
        "normalize": len([w for w in vocabulary if "|" in w]) > 0,
    }

    mod_config = {
        "mod_speech": True,
        "mod_room": False,
        "mod_channel": True,
        "mod_scene": NonSpeech(),
        "mod_intensity": "mid",
    }

    if split == "train":
        dataset = AnnotatedDataset(sounds["train"], annots["train"], **data_config)
        dataset.augment(**mod_config)
        return dataset.iterator(
            shuffle=True,
            flat_labels=flat_labels,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif split == "val":
        dataset = AnnotatedDataset(sounds["val"], annots["val"], **data_config)
        return dataset.iterator(
            shuffle=False,
            flat_labels=flat_labels,
            batch_size=max(batch_size // 2, 1),
            num_workers=num_workers,
        )
    elif split == "test":
        dataset = AnnotatedDataset(sounds["test"], annots["test"], **data_config)
        return dataset.iterator(
            shuffle=False,
            flat_labels=flat_labels,
            batch_size=max(batch_size // 2, 1),
            num_workers=num_workers,
        )
    else:
        raise ValueError()


def librispeech_sequence(
    target: str = "words",
    vocabulary: Optional[list[str]] = None,
    seq_size: int = 20,
    seq_min: int = 1,
    block_size: float = 8.0,
    batch_size: int = 12,
    seq_per_sample: float = 4.0,
    num_workers: int = 4,
    audio_transform: Optional[Callable] = mel_spectrogram(),
    split: str = "val",
):
    sounds, annots = {}, {}
    sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
    sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
    sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
    vocabulary = LABELS[target] if vocabulary is None else vocabulary

    data_config = {
        "batch_first": False,
        "target": target,
        "vocabulary": vocabulary,
        "seq_size": seq_size,
        "seq_min": seq_min,
        "block_size": block_size,
        "audio_transform": audio_transform,
        "normalize": len([w for w in vocabulary if "|" in w]) > 0,
    }

    mod_config = {
        "mod_speech": False,
        "mod_room": False,
        "mod_channel": True,
        "mod_scene": NonSpeech(),
        "mod_intensity": "mid",
    }

    batch_max = int(batch_size * seq_per_sample)

    if split == "train":
        dataset = SequenceDataset(
            sounds["train"], annots["train"], return_clean=True, **data_config
        )
        dataset.augment(**mod_config)
        return dataset.iterator(
            shuffle=True,
            batch_size=batch_size,
            batch_max=batch_max,
            num_workers=num_workers,
        )
    elif split == "val":
        dataset = SequenceDataset(sounds["val"], annots["val"], **data_config)
        return dataset.iterator(
            shuffle=False,
            batch_size=max(batch_size // 2, 1),
            batch_max=batch_max,
            num_workers=num_workers,
        )
    elif split == "test":
        dataset = SequenceDataset(sounds["test"], annots["test"], **data_config)
        return dataset.iterator(
            shuffle=False,
            batch_size=max(batch_size // 2, 1),
            batch_max=batch_max,
            num_workers=num_workers,
        )
    else:
        raise ValueError()


def librispeech_token(
    target: str = "words",
    vocabulary: Optional[list[str]] = None,
    limits: Limits = LIMITS_WORD["librispeech"]["max"],
    batch_size: int = 12,
    num_workers: int = 4,
    audio_transform: Optional[Callable] = mel_spectrogram(),
    split: str = "val",
):
    sounds, annots = {}, {}
    sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
    sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
    sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
    vocabulary = LABELS[target] if vocabulary is None else vocabulary

    data_config = {
        "target": target,
        "vocabulary": vocabulary,
        "batch_first": True,
        "limits": limits,
        "audio_transform": audio_transform,
        "normalize": len([w for w in vocabulary if "|" in w]) > 0,
    }

    mod_config = {
        "mod_speech": True,
        "mod_room": False,
        "mod_channel": True,
        "mod_scene": NonSpeech(),
        "mod_intensity": "mid",
    }

    if split == "train":
        dataset = TokenizedDataset(sounds["train"], annots["train"], **data_config)
        dataset.augment(**mod_config)
        return dataset.iterator(
            shuffle=True, batch_size=batch_size, num_workers=num_workers
        )
    elif split == "val":
        dataset = TokenizedDataset(sounds["val"], annots["val"], **data_config)
        return dataset.iterator(
            shuffle=False, batch_size=max(batch_size // 2, 1), num_workers=num_workers
        )
    elif split == "test":
        dataset = TokenizedDataset(sounds["test"], annots["test"], **data_config)
        return dataset.iterator(
            shuffle=False, batch_size=max(batch_size // 2, 1), num_workers=num_workers
        )
    else:
        raise ValueError()
