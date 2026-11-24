import librosa
import numpy as np

from .config import AudioConfig, FeatureConfig



def load_audio(filepath: str, cfg: AudioConfig = AudioConfig()) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a mono waveform.

    Parameters
    ----------
    filepath : str
        Path to the audio file. Any format supported by librosa/soundfile
        or audioread can be used.
    cfg : AudioConfig, optional (default = AudioConfig())
        Parameters controlling resampling and which excerpt is loaded.

    Returns
    -------
    wave : np.ndarray, shape (n_samples,)
        Mono waveform, as floating-point values in [-1, 1].
    sr : int
        Sample rate of `wave` in Hz. Equals `cfg.sample_rate` if it was given,
        otherwise the native sample rate of the file.

    Raises
    ------
    ValueError
        If `cfg.start` or `cfg.duration` are out of range.
    """

    if cfg.start < 0 or cfg.start > 30:
        raise ValueError(f'Invalid start time: {cfg.start}')

    if cfg.duration is not None and (cfg.duration < 0 or cfg.duration > 30):
        raise ValueError(f'Invalid duration time: {cfg.duration}')

    return librosa.load(filepath, sr=cfg.sr, offset=cfg.start, duration=cfg.duration)


def compute_spectrogram(wave: np.ndarray, sr: int, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """
    Compute a non-negative time-frequency representation of a waveform.

    Parameters
    ----------
    wave : np.ndarray, shape (n_samples,)
        Mono waveform.
    sr : int
        Sample rate of `wave` in Hz.
    cfg : FeatureConfig, optional (default = FeatureConfig())
        Parameters defining the kind of representation and its resolution.

    Returns
    -------
    spectrogram : np.ndarray, shape (num_bins, n_frames)
        Magnitude spectrogram. `num_bins` is `cfg.num_bins` for 'stft', 84 for 'cqt',
        and 12 for 'chroma'. Values are on the scale of the raw STFT magnitude
        for 'stft' and normalized for 'chroma', so thresholds are not directly
        comparable between the two.
    """

    stft_args = dict(n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)

    if cfg.kind == "stft":
        return np.abs(librosa.stft(y=wave, **stft_args))[:cfg.num_bins]
    if cfg.kind == "cqt":
        return np.abs(librosa.cqt(y=wave, sr=sr, hop_length=cfg.hop_length))
    if cfg.kind == "chroma":
        return librosa.feature.chroma_stft(y=wave, sr=sr, **stft_args)
