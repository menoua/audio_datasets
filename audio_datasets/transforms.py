from torch.nn import Module, Sequential
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


def mel_spectrogram(
    in_sr: int, out_sr: int = 100, n_mels: int = 128, top_db: float = 70
):
    return Sequential(
        MelSpectrogram(
            in_sr,
            n_fft=1024,
            hop_length=int(in_sr / out_sr),
            f_min=20,
            f_max=8_000,
            n_mels=n_mels,
            power=2.0,
        ),
        AmplitudeToDB("power", top_db=top_db),
        type(
            "Normalize",
            (Module,),
            dict(forward=lambda _, x: (x - x.max()).squeeze(0).T.float() / top_db + 1),
        )(),
    )
