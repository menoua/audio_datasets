from typing import Callable, Optional

from .core import AnnotatedDataset, SequenceDataset, TokenizedDataset
from .data import LibriSpeech
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
        batch_size: int = 16,
        num_workers: int = 4,
        flat_labels: bool = False,
        batch_first: bool = True,
        audio_transform: Optional[Callable] = mel_spectrogram(),
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
        duration: float = 1,
        scale: bool = False,
        context: tuple[int, int] = (0, 0),
        alignment: str = "left",
        **kwargs,
    ):
        super().__init__(dataset_type=TokenizedDataset, **kwargs)

        self.data_config = {
            **self.data_config,
            "duration": duration,
            "scale": scale,
            "context": context,
            "alignment": alignment,
        }

        del self.dataloader_config["flat_labels"]


#
# def librispeech(
#     target: str = "words",
#     vocabulary: Optional[list[str]] = None,
#     limits: Optional[Limits] = LIMITS_WORD["librispeech"]["max"],
#     batch_size: int = 12,
#     num_workers: int = 4,
#     flat_labels: bool = False,
#     audio_transform: Optional[Callable] = mel_spectrogram(),
#     split: str = "dev",
# ):
#     sounds, annots = {}, {}
#     sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
#     sounds["dev"], annots["dev"] = LibriSpeech(subset="dev-other")
#     sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
#     vocabulary = LABELS[target] if vocabulary is None else vocabulary
#
#     data_config = {
#         "batch_first": True,
#         "target": target,
#         "vocabulary": vocabulary,
#         "limits": limits,
#         "audio_transform": audio_transform,
#         "normalize": len([w for w in vocabulary if "|" in w]) > 0,
#     }
#
#     mod_config = {
#         "mod_speech": True,
#         "mod_room": False,
#         "mod_channel": True,
#         "mod_scene": NonSpeech(),
#         "mod_intensity": "mid",
#     }
#
#     if split == "train":
#         dataset = AnnotatedDataset(sounds["train"], annots["train"], **data_config)
#         dataset.augment(**mod_config)
#         return dataset.iterator(
#             shuffle=True,
#             flat_labels=flat_labels,
#             batch_size=batch_size,
#             num_workers=num_workers,
#         )
#     elif split == "dev":
#         dataset = AnnotatedDataset(sounds["dev"], annots["dev"], **data_config)
#         return dataset.iterator(
#             shuffle=False,
#             flat_labels=flat_labels,
#             batch_size=max(batch_size // 2, 1),
#             num_workers=num_workers,
#         )
#     elif split == "test":
#         dataset = AnnotatedDataset(sounds["test"], annots["test"], **data_config)
#         return dataset.iterator(
#             shuffle=False,
#             flat_labels=flat_labels,
#             batch_size=max(batch_size // 2, 1),
#             num_workers=num_workers,
#         )
#     else:
#         raise ValueError()
#
#
# def librispeech_sequence(
#     target: str = "words",
#     vocabulary: Optional[list[str]] = None,
#     seq_size: int = 20,
#     seq_min: int = 1,
#     block_size: float = 8.0,
#     batch_size: int = 12,
#     seq_per_sample: float = 4.0,
#     num_workers: int = 4,
#     audio_transform: Optional[Callable] = mel_spectrogram(),
#     split: str = "dev",
# ):
#     sounds, annots = {}, {}
#     sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
#     sounds["dev"], annots["dev"] = LibriSpeech(subset="dev-other")
#     sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
#     vocabulary = LABELS[target] if vocabulary is None else vocabulary
#
#     data_config = {
#         "batch_first": False,
#         "target": target,
#         "vocabulary": vocabulary,
#         "seq_size": seq_size,
#         "seq_min": seq_min,
#         "block_size": block_size,
#         "audio_transform": audio_transform,
#         "normalize": len([w for w in vocabulary if "|" in w]) > 0,
#     }
#
#     mod_config = {
#         "mod_speech": False,
#         "mod_room": False,
#         "mod_channel": True,
#         "mod_scene": NonSpeech(),
#         "mod_intensity": "mid",
#     }
#
#     batch_max = int(batch_size * seq_per_sample)
#
#     if split == "train":
#         dataset = SequenceDataset(
#             sounds["train"], annots["train"], return_clean=True, **data_config
#         )
#         dataset.augment(**mod_config)
#         return dataset.iterator(
#             shuffle=True,
#             batch_size=batch_size,
#             batch_max=batch_max,
#             num_workers=num_workers,
#         )
#     elif split == "dev":
#         dataset = SequenceDataset(sounds["dev"], annots["dev"], **data_config)
#         return dataset.iterator(
#             shuffle=False,
#             batch_size=max(batch_size // 2, 1),
#             batch_max=batch_max,
#             num_workers=num_workers,
#         )
#     elif split == "test":
#         dataset = SequenceDataset(sounds["test"], annots["test"], **data_config)
#         return dataset.iterator(
#             shuffle=False,
#             batch_size=max(batch_size // 2, 1),
#             batch_max=batch_max,
#             num_workers=num_workers,
#         )
#     else:
#         raise ValueError()
#
#
# def librispeech_token(
#     target: str = "words",
#     vocabulary: Optional[list[str]] = None,
#     limits: Limits = LIMITS_WORD["librispeech"]["max"],
#     batch_size: int = 12,
#     num_workers: int = 4,
#     audio_transform: Optional[Callable] = mel_spectrogram(),
#     split: str = "dev",
# ):
#     sounds, annots = {}, {}
#     sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
#     sounds["dev"], annots["dev"] = LibriSpeech(subset="dev-other")
#     sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
#     vocabulary = LABELS[target] if vocabulary is None else vocabulary
#
#     data_config = {
#         "target": target,
#         "vocabulary": vocabulary,
#         "batch_first": True,
#         "limits": limits,
#         "audio_transform": audio_transform,
#         "normalize": len([w for w in vocabulary if "|" in w]) > 0,
#     }
#
#     mod_config = {
#         "mod_speech": True,
#         "mod_room": False,
#         "mod_channel": True,
#         "mod_scene": NonSpeech(),
#         "mod_intensity": "mid",
#     }
#
#     if split == "train":
#         dataset = TokenizedDataset(sounds["train"], annots["train"], **data_config)
#         dataset.augment(**mod_config)
#         return dataset.iterator(
#             shuffle=True, batch_size=batch_size, num_workers=num_workers
#         )
#     elif split == "dev":
#         dataset = TokenizedDataset(sounds["dev"], annots["dev"], **data_config)
#         return dataset.iterator(
#             shuffle=False, batch_size=max(batch_size // 2, 1), num_workers=num_workers
#         )
#     elif split == "test":
#         dataset = TokenizedDataset(sounds["test"], annots["test"], **data_config)
#         return dataset.iterator(
#             shuffle=False, batch_size=max(batch_size // 2, 1), num_workers=num_workers
#         )
#     else:
#         raise ValueError()
