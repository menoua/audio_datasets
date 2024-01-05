from typing import Callable, Optional

from .core import (AnnotatedDataset, MultiAnnotatedDataset, SequenceDataset,
                   TokenizedDataset)
from .data import LibriSpeech
from .lexicon import LABELS
from .limits import LIMITS, Limits
from .transforms import mel_spectrogram


class LibriSpeechDataloader:
    def __init__(
        self,
        dataset_type=AnnotatedDataset,
        *,
        target: str = "words",
        labels: Optional[list[str]] = None,
        limits: Optional[Limits] = LIMITS["librispeech"]["max"],
        batch_size: int = 16,
        num_workers: int = 4,
        flat_labels: bool = False,
        batch_first: bool = True,
        audio_transform: Optional[Callable] = mel_spectrogram(),
        in_sr: int = 16_000,
        out_sr: Optional[int] = None,
        train_subset: str = "train-*",
        dev_subset: str = "dev-other",
        test_subset: str = "test-clean",
        **kwargs,
    ):
        if labels is None:
            labels = LABELS[target]

        self.dataset_type = dataset_type

        sounds, annots = {}, {}
        sounds["train"], annots["train"] = LibriSpeech(subset=train_subset)
        sounds["dev"], annots["dev"] = LibriSpeech(subset=dev_subset)
        sounds["test"], annots["test"] = LibriSpeech(subset=test_subset)
        self.sounds, self.annots = sounds, annots

        self.data_config = {
            "batch_first": batch_first,
            "target": target,
            "vocabulary": labels,
            "limits": limits,
            "audio_transform": audio_transform,
            "in_sr": in_sr,
            "out_sr": out_sr,
            "normalize": len([w for w in labels if "|" in w]) > 0,
            **kwargs,
        }

        self.dataloader_config = {
            "flat_labels": flat_labels,
            "batch_size": batch_size,
            "num_workers": num_workers,
        }

    def training(
        self,
        shuffle: bool = True,
        **kwargs,
    ):
        dataset = self.dataset_type(
            self.sounds["train"],
            self.annots["train"],
            **self.data_config,
            **kwargs,
        )

        return dataset.iterator(shuffle=shuffle, **self.dataloader_config)

    def validation(
        self,
        shuffle: bool = False,
        **kwargs,
    ):
        dataset = self.dataset_type(
            self.sounds["dev"],
            self.annots["dev"],
            **self.data_config,
            **kwargs,
        )

        return dataset.iterator(shuffle=shuffle, **self.dataloader_config)

    def test(
        self,
        shuffle: bool = False,
        **kwargs,
    ):
        dataset = self.dataset_type(
            self.sounds["test"],
            self.annots["test"],
            **self.data_config,
            **kwargs,
        )

        return dataset.iterator(shuffle=shuffle, **self.dataloader_config)


class LibriSpeechSequenceDataloader(LibriSpeechDataloader):
    def __init__(
        self,
        *,
        seq_size: int = 20,
        seq_min: int = 1,
        seq_time: float = 8.0,
        seq_per_sample: float = 4.0,
        seq_overlap: bool = False,
        check_boundaries: bool = True,
        **kwargs,
    ):
        super().__init__(dataset_type=SequenceDataset, **kwargs)
        self.seq_per_sample = seq_per_sample

        self.data_config = {
            **self.data_config,
            "seq_size": seq_size,
            "seq_min": seq_min,
            "seq_time": seq_time,
            "seq_overlap": seq_overlap,
            "check_boundaries": check_boundaries,
        }

        self.dataloader_config = {
            **self.dataloader_config,
            "batch_max": int(
                self.dataloader_config["batch_size"] * self.seq_per_sample
            ),
        }


class LibriSpeechTokenDataloader(LibriSpeechDataloader):
    def __init__(
        self,
        *,
        duration: float = 1,
        scale: bool = False,
        context: tuple[int, int] = (0, 0),
        alignment: str = "left",
        **kwargs,
    ):
        super().__init__(dataset_type=TokenizedDataset, **kwargs)

        if "flat_labels" in kwargs:
            raise ValueError(
                "LibriSpeechTokenDataloader does not support `flat_labels`"
            )

        self.data_config = {
            **self.data_config,
            "duration": duration,
            "scale": scale,
            "context": context,
            "alignment": alignment,
        }

        del self.dataloader_config["flat_labels"]


class LibriSpeechMultiDataloader(LibriSpeechDataloader):
    def __init__(
        self,
        *,
        labels: Optional[dict[str, list[str]]] = None,
        **kwargs,
    ):
        super().__init__(dataset_type=MultiAnnotatedDataset, **kwargs)

        if "target" in kwargs:
            raise ValueError("LibriSpeechMultiDataloader does not support `target`")

        if labels is None:
            labels = {k: LABELS[k] for k in ("chars", "phones", "syllables", "words")}

        self.data_config = {
            **self.data_config,
            "vocabulary": labels,
        }
        del self.data_config["target"]
