from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(frozen=True)
class AudioConfig:
    """
    Configuration parameters to load an audio file.

    Parameters
    ----------
    sr : int, optional (default = None)
        Target sample rate to resample the audio. If None, the native sample rate is used.
    start : float, optional (default = 0.0)
        Time offset in seconds to start loading the audio. Must be between 0 and 30 seconds.
    duration : float, optional (default = None)
        Duration of the audio to load in seconds. Start + duration must not exceed 30 seconds.
        If None, the whole audio starting from `start` is loaded.
    """

    sr: Optional[int] = None
    start: float = 0.0
    duration: Optional[float] = None

    def __post_init__(self):
        if self.sr is not None and self.sr <= 0:
            raise ValueError("Sample rate must be positive.")

        if self.start < 0 or self.start > 30:
            raise ValueError("start must be between 0 and 30 seconds.")

        if self.duration is not None:
            if self.duration <= 0:
                raise ValueError("duration must be positive.")
            if self.start + self.duration > 30:
                raise ValueError("start + duration must not exceed 30 seconds.")


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration parameters used to compute the time-frequency representation
    from which the peaks are extracted.

    Parameters
    ----------
    kind : {'stft', 'cqt', 'chroma'}, optional (default = 'stft')
        Type of representation to compute.
        - 'stft': magnitude of the Short-Time Fourier Transform, with linear
          frequency resolution.
        - 'cqt': magnitude of the Constant-Q Transform, with logarithmic
          frequency resolution of one bin per semitone, starting at C1
          (~32.7 Hz), having 7 octaves (84 bins) by default.
        - 'chroma': 12-bin pitch-class profile (one bin per semitone, octaves
          folded together), computed from the STFT.
    n_fft : int, optional (default = 2048)
        Length of the analysis window in samples. Larger values give finer
        frequency resolution but coarser time resolution. Used by 'stft' and
        'chroma'; ignored by 'cqt'.
    hop_length : int, optional (default = 512)
        Number of samples between consecutive frames. Smaller values give
        finer time resolution (and more frames). For 'cqt' it must be a
        multiple of 64, i.e. 2**(7-1) = 64 (where 7 is the number of octaves)
        for the default 84 bins.
    win_length : int, optional (default = None)
        Size of the window function in samples. Must not exceed `n_fft`.
        If None, `n_fft` is used. Used by 'stft' and 'chroma'; ignored by 'cqt'.
    num_bins : int, optional (default = 128)
        Number of frequency bins to keep when using the 'stft' feature kind.
        Ignored by 'cqt' and 'chroma', which always have a fixed number of bins.
    """

    kind: Literal["stft", "cqt", "chroma"] = "stft"
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    num_bins: Optional[int] = 128

    def __post_init__(self):
        if self.kind not in {"stft", "cqt", "chroma"}:
            raise ValueError("kind must be one of 'stft', 'cqt', or 'chroma'.")

        if self.win_length is not None:
            if self.win_length <= 0:
                raise ValueError("win_length must be positive.")
            if self.kind == 'stft' and self.win_length > self.n_fft:
                raise ValueError("win_length must not exceed n_fft.")

        if self.kind == "cqt" and self.hop_length % 64 != 0:
            raise ValueError(
                "For 'cqt', hop_length must be a multiple of 64."
            )

        if self.num_bins is not None and self.num_bins <= 0:
            raise ValueError("num_bins must be positive.")


@dataclass(frozen=True)
class PeakConfig:
    """
    Configuration parameters used to build the constellation map with peak peaking.

    Parameters
    ----------
    freq_neighborhood : int, optional (default = 11)
        Height of the local window, in frequency bins (centered on the point,
        so a value of 11 looks 5 bins up and 5 bins down). Should be odd.
    time_neighborhood : int, optional (default = 5)
        Width of the local window, in time frames (centered on the point,
        so a value of 5 looks 2 frames left and 2 frames right). Should be odd.
    threshold : float, optional (default = 0.01)
        Minimum magnitude required to be considered a peak.
    """

    freq_neighborhood: int = 11
    time_neighborhood: int = 5
    threshold: float = 0.01

    def __post_init__(self):
        if self.freq_neighborhood % 2 == 0 or self.time_neighborhood % 2 == 0:
            raise ValueError("freq_neighborhood and time_neighborhood must be odd.")

        if self.freq_neighborhood <= 0 or self.time_neighborhood <= 0:
            raise ValueError("freq_neighborhood and time_neighborhood must be positive.")

        if self.threshold < 0:
            raise ValueError("threshold must be non-negative.")


@dataclass(frozen=True)
class HashConfig:
    """
    Configuration parameters used to build the hashes based on the constellation maps.

    Parameters
    ----------
    target_zone_freq_size : int, optional (default = 31)
        Height of the target zone, in frequency bins (centered on the anchor's
        frequency, so a value of 31 looks 15 bins up and 15 bins down). Must be odd.
    target_zone_time_size : int, optional (default = 19)
        Width of the target zone, in time frames (centered `target_zone_time_delay`
        frames after the anchor, so a value of 19 spans 9 frames on each side of
        that center). Must be odd.
    target_zone_time_delay : int, optional (default = 20)
        Number of frames between the anchor and the center of the target zone.
        Must be larger than half of `target_zone_time_size`, so that every
        target lies strictly after the anchor.
    """

    target_zone_freq_size: int = 31
    target_zone_time_size: int = 19
    target_zone_time_delay: int = 20

    def __post_init__(self):
        if self.target_zone_freq_size % 2 == 0 or self.target_zone_time_size % 2 == 0:
            raise ValueError("target_zone_freq_size and target_zone_time_size must be odd.")

        if self.target_zone_freq_size <= 0 or self.target_zone_time_size <= 0:
            raise ValueError("target_zone_freq_size and target_zone_time_size must be positive.")

        if self.target_zone_time_delay <= self.target_zone_time_size // 2:
            raise ValueError(
                "target_zone_time_delay must exceed half of target_zone_time_size, "
                "otherwise targets can fall on or before the anchor."
            )
