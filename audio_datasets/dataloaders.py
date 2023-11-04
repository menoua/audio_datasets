from .core import AlignedDataset, AnnotatedDataset, SequenceDataset
from .data import LibriSpeech, NonSpeech
from .lexicon import LABELS


class LibriSpeechDataloader:
    def __init__(
        self,
        dataset_type=AnnotatedDataset,
        target="words",
        stressed=True,
        labels=None,
        freqbins=128,
        max_time=20.0,
        batch_size=12,
        num_workers=4,
        flat_labels=False,
        batch_first=True,
        audio_proc="default",
        augment_speech=False,
        augment_room=False,
        augment_channel=True,
        augment_scene=[],  # NonSpeech(),
        augment_mix_n=1,
        mod_intensity="mid",
    ):
        if labels is None:
            labels = LABELS[target]

        self.dataset_type = dataset_type

        sounds, annots = {}, {}
        sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
        sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
        sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
        self.sounds, self.annots = sounds, annots

        self.data_cfg = dict(
            freqbins=freqbins,
            batch_first=batch_first,
            target=target,
            stressed=stressed,
            vocabulary=labels,
            max_time=max_time,
            audio_proc=audio_proc,
            normalize=len([w for w in labels if "|" in w]) > 0,
        )

        self.augment_cfg = dict(
            speech=augment_speech,
            room=augment_room,
            channel=augment_channel,
            scene=augment_scene,
            mix_n=augment_mix_n,
            mod_intensity=mod_intensity,
        )

        self.dataloader_cfg = dict(
            flat_labels=flat_labels, batch_size=batch_size, num_workers=num_workers
        )

    def train_dataloader(
        self, data_cfg: dict = {}, augment_cfg: dict = {}, dataloader_cfg: dict = {}
    ):
        data_cfg = {**self.data_cfg, **data_cfg}
        dataset = self.dataset_type(
            self.sounds["train"], self.annots["train"], **data_cfg
        )

        augment_cfg = {**self.augment_cfg, **augment_cfg}
        dataset.augment(**augment_cfg)

        dataloader_cfg = {**self.dataloader_cfg, "shuffle": True, **dataloader_cfg}
        return dataset.iterator(**dataloader_cfg)

    def val_dataloader(self, data_cfg: dict = {}, dataloader_cfg: dict = {}):
        data_cfg = {**self.data_cfg, **data_cfg}
        dataset = self.dataset_type(self.sounds["val"], self.annots["val"], **data_cfg)

        dataloader_cfg = {**self.dataloader_cfg, "shuffle": False, **dataloader_cfg}
        dataloader_cfg = {
            **dataloader_cfg,
            "batch_size": max(dataloader_cfg["batch_size"] // 2, 1),
        }
        return dataset.iterator(**dataloader_cfg)

    def test_dataloader(self, data_cfg: dict = {}, dataloader_cfg: dict = {}):
        data_cfg = {**self.data_cfg, **data_cfg}
        dataset = self.dataset_type(
            self.sounds["test"], self.annots["test"], **data_cfg
        )

        dataloader_cfg = {**self.dataloader_cfg, "shuffle": False, **dataloader_cfg}
        dataloader_cfg = {
            **dataloader_cfg,
            "batch_size": max(dataloader_cfg["batch_size"] // 2, 1),
        }
        return dataset.iterator(**dataloader_cfg)


class LibriSpeechSequenceDataloader(LibriSpeechDataloader):
    def __init__(
        self,
        dataset_type=SequenceDataset,
        seq_size=20,
        seq_min=1,
        seq_time=8.0,
        seq_per_sample=4.0,
        seq_overlap=False,
        check_boundaries=True,
        **kwargs,
    ):
        super().__init__(dataset_type=dataset_type, **kwargs)
        self.seq_per_sample = seq_per_sample

        self.data_cfg = {
            **self.data_cfg,
            "seq_size": seq_size,
            "seq_min": seq_min,
            "seq_time": seq_time,
            "seq_overlap": seq_overlap,
            "check_boundaries": check_boundaries,
        }

    def train_dataloader(
        self, data_cfg: dict = {}, augment_cfg: dict = {}, dataloader_cfg: dict = {}
    ):
        dataloader_cfg = {**self.dataloader_cfg, **dataloader_cfg}
        dataloader_cfg = {
            **dataloader_cfg,
            "batch_max": int(dataloader_cfg["batch_size"] * self.seq_per_sample),
        }
        return super().train_dataloader(
            data_cfg=data_cfg, augment_cfg=augment_cfg, dataloader_cfg=dataloader_cfg
        )

    def val_dataloader(self, data_cfg: dict = {}, dataloader_cfg: dict = {}):
        dataloader_cfg = {**self.dataloader_cfg, **dataloader_cfg}
        dataloader_cfg = {
            **dataloader_cfg,
            "batch_max": int(dataloader_cfg["batch_size"] * self.seq_per_sample),
        }
        return super().val_dataloader(data_cfg=data_cfg, dataloader_cfg=dataloader_cfg)

    def test_dataloader(self, data_cfg: dict = {}, dataloader_cfg: dict = {}):
        dataloader_cfg = {**self.dataloader_cfg, **dataloader_cfg}
        dataloader_cfg = {
            **dataloader_cfg,
            "batch_max": int(dataloader_cfg["batch_size"] * self.seq_per_sample),
        }
        return super().test_dataloader(data_cfg=data_cfg, dataloader_cfg=dataloader_cfg)


class LibriSpeechTokenDataloader(LibriSpeechDataloader):
    def __init__(self, dataset_type=AlignedDataset, max_tokens=80, **kwargs):
        super().__init__(dataset_type=dataset_type, **kwargs)

        self.data_cfg = {**self.data_cfg, "max_tokens": max_tokens}


def librispeech(
    target="words",
    vocabulary=None,
    freqbins=128,
    max_time=20.0,
    batch_size=12,
    num_workers=4,
    flat_labels=False,
    audio_proc="default",
    split="val",
):
    sounds, annots = {}, {}
    sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
    sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
    sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
    vocabulary = LABELS[target] if vocabulary is None else vocabulary

    data_config = {
        "freqbins": freqbins,
        "batch_first": True,
        "target": target,
        "vocabulary": vocabulary,
        "max_time": max_time,
        "audio_proc": audio_proc,
        "normalize": len([w for w in vocabulary if "|" in w]) > 0,
    }

    augment_config = {
        "speech": True,
        "room": False,
        "channel": True,
        "scene": NonSpeech(),
        "mix_n": 1,
        "mod_intensity": "mid",
    }

    if split == "train":
        dataset = AnnotatedDataset(sounds["train"], annots["train"], **data_config)
        dataset.augment(**augment_config)
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
    target="words",
    vocabulary=None,
    freqbins=128,
    seq_size=20,
    seq_min=1,
    block_size=8.0,
    batch_size=12,
    seq_per_sample=4.0,
    num_workers=4,
    audio_proc="default",
    split="val",
):
    sounds, annots = {}, {}
    sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
    sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
    sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
    vocabulary = LABELS[target] if vocabulary is None else vocabulary

    data_config = {
        "freqbins": freqbins,
        "batch_first": False,
        "target": target,
        "vocabulary": vocabulary,
        "seq_size": seq_size,
        "seq_min": seq_min,
        "block_size": block_size,
        "audio_proc": audio_proc,
        "normalize": len([w for w in vocabulary if "|" in w]) > 0,
    }

    augment_config = {
        "speech": False,
        "room": False,
        "channel": True,
        "scene": NonSpeech(),
        "mix_n": 1,
        "mod_intensity": "mid",
    }

    batch_max = int(batch_size * seq_per_sample)

    if split == "train":
        dataset = SequenceDataset(
            sounds["train"], annots["train"], return_clean=True, **data_config
        )
        dataset.augment(**augment_config)
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
    target="words",
    vocabulary=None,
    freqbins=128,
    max_time=18.0,
    max_tokens=80,
    batch_size=12,
    num_workers=4,
    audio_proc="default",
    split="val",
):
    sounds, annots = {}, {}
    sounds["train"], annots["train"] = LibriSpeech(subset="train-*")
    sounds["val"], annots["val"] = LibriSpeech(subset="dev-other")
    sounds["test"], annots["test"] = LibriSpeech(subset="test-clean")
    vocabulary = LABELS[target] if vocabulary is None else vocabulary

    data_config = {
        "target": target,
        "vocabulary": vocabulary,
        "freqbins": freqbins,
        "batch_first": True,
        "max_time": max_time,
        "max_tokens": max_tokens,
        "audio_proc": audio_proc,
        "normalize": len([w for w in vocabulary if "|" in w]) > 0,
    }

    augment_config = {
        "speech": True,
        "room": False,
        "channel": True,
        "scene": NonSpeech(),
        "mix_n": 1,
        "mod_intensity": "mid",
    }

    if split == "train":
        dataset = AlignedDataset(sounds["train"], annots["train"], **data_config)
        dataset.augment(**augment_config)
        return dataset.iterator(
            shuffle=True, batch_size=batch_size, num_workers=num_workers
        )
    elif split == "val":
        dataset = AlignedDataset(sounds["val"], annots["val"], **data_config)
        return dataset.iterator(
            shuffle=False, batch_size=max(batch_size // 2, 1), num_workers=num_workers
        )
    elif split == "test":
        dataset = AlignedDataset(sounds["test"], annots["test"], **data_config)
        return dataset.iterator(
            shuffle=False, batch_size=max(batch_size // 2, 1), num_workers=num_workers
        )
    else:
        raise ValueError()
