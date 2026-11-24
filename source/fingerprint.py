import bisect
import numpy as np

from scipy import ndimage
from typing import Any, Optional
from collections import defaultdict

from .utils import load_audio, compute_spectrogram
from .config import AudioConfig, FeatureConfig, PeakConfig, HashConfig


def compute_constellation_map(
    filepath: str,
    audio_cfg: AudioConfig = AudioConfig(),
    features_cfg: FeatureConfig = FeatureConfig(),
    peaks_cfg: PeakConfig = PeakConfig(),
) -> np.ndarray:
    """
    Compute the constellation map of an audio file.

    The constellation map is a sparse binary time-frequency image that keeps
    only the most prominent spectral peaks. The pipeline is:
    load audio -> compute spectrogram -> pick local peaks.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    audio_cfg : AudioConfig, optional (default = AudioConfig())
        Parameters controlling how the audio is loaded.
    features_cfg : FeatureConfig, optional (default = FeatureConfig())
        Parameters controlling the time-frequency representation.
    peaks_cfg : PeakConfig, optional (default = PeakConfig())
        Parameters controlling peak picking.

    Returns
    -------
    constellation_map : np.ndarray of bool, shape (n_freq_bins, n_frames)
        True where a spectral peak was found. See `compute_spectrogram` for the
        meaning of the axes, and `_find_peaks` for the peak criterion.

    See Also
    --------
    load_audio, compute_spectrogram, _find_peaks :
        The individual steps, useful for inspecting intermediate results.
    """

    wave, sr = load_audio(filepath, audio_cfg)
    spectrogram = compute_spectrogram(wave, sr, features_cfg)
    return _find_peaks(spectrogram, peaks_cfg)


def compute_hashes(
    constellation_map: np.ndarray,
    sr: int,
    hop_length: Optional[int] = 512,
    metadata: Optional[dict[str, Any]] = None,
    cfg: HashConfig = HashConfig()
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate fingerprint hashes from a constellation map using anchor-target pairs.

    Every peak in the constellation map is treated as an anchor. For each anchor,
    a rectangular target zone is defined in the time-frequency plane, centered
    `cfg.target_zone_time_delay` frames after the anchor in time and on the anchor's
    frequency bin. Each peak that falls inside this zone is paired with the
    anchor to form a hash of the form ``"{anchor_freq}_{target_freq}_{timedelta}"``.

    Parameters
    ----------
    constellation_map : numpy.ndarray
        Boolean 2D array of shape ``(num_freq_bins, num_time_frames)``, where
        True marks a constellation peak.
    sr : int
        Sample rate in Hz of the audio the constellation map was computed from.
    hop_length : int
        Hop length in samples used to compute the spectrogram. Together with `sr`
        it converts frame indices to seconds (frames are assumed centered, as in
        librosa's defaults).
    metadata : dict of str to Any, optional (default = None)
        Additional information (e.g. song name or ID) stored alongside every hash
        occurrence. If None, an empty dictionary is used.
    cfg : HashConfig, optional (default = HashConfig())
        Parameters defining the target zone for hashing.

    Returns
    -------
    collections.defaultdict of str to list of dict
        Mapping from each hash string to the list of its occurrences. Each
        occurrence is a dictionary containing the entries from `metadata` plus
        ``'offset_in_seconds'`` (float), the time of the anchor peak.

    Notes
    -----
    Hashes are only comparable between constellation maps computed with the same
    `FeatureConfig`, since the meaning of a frequency bin depends on it.

    ``'offset_in_seconds'`` is relative to the start of the loaded excerpt
    (see `AudioConfig.start`).

    Peaks are sorted by time so that the candidate targets for each anchor can
    be located efficiently with binary search instead of scanning every peak.
    """

    if metadata is None:
        metadata = {}

    peak_freqs, peak_times = np.where(constellation_map)
    peaks = sorted(zip(peak_freqs, peak_times), key=lambda peak: peak[1])
    sorted_peak_times = [peak_time for _, peak_time in peaks]

    seconds_per_frame = hop_length / sr
    half_freq = cfg.target_zone_freq_size // 2
    half_time = cfg.target_zone_time_size // 2

    hashes = defaultdict(list)
    for anchor_freq, anchor_time in peaks:
        zone_min_freq = anchor_freq - half_freq
        zone_max_freq = anchor_freq + half_freq

        zone_center_time = anchor_time + cfg.target_zone_time_delay
        zone_min_time = zone_center_time - half_time
        zone_max_time = zone_center_time + half_time

        anchor_info = {**metadata, "offset_in_seconds": float(anchor_time * seconds_per_frame)}

        first_candidate_idx = bisect.bisect_left(sorted_peak_times, zone_min_time)
        last_candidate_idx = bisect.bisect_right(sorted_peak_times, zone_max_time)

        for target_freq, target_time in peaks[first_candidate_idx:last_candidate_idx]:
            if zone_min_freq <= target_freq <= zone_max_freq:
                time_delta = target_time - anchor_time
                hashes[f"{anchor_freq}_{target_freq}_{time_delta}"].append(anchor_info)

    return hashes


def _find_peaks(spectrogram: np.ndarray, cfg: PeakConfig = PeakConfig()) -> np.ndarray:
    """
    Find the local maxima of a spectrogram that make up the constellation map.

    A point is a peak if it is the maximum of its local window (of size
    `freq_neighborhood` x `time_neighborhood`) and its magnitude is above
    `threshold`.

    Parameters
    ----------
    spectrogram : np.ndarray, shape (n_freq_bins, n_frames)
        Non-negative time-frequency representation, e.g. the output of
        `compute_spectrogram`.
    cfg : PeakConfig, optional (default = PeakConfig())
        Parameters defining the local window and the minimum peak magnitude.

    Returns
    -------
    peaks : np.ndarray of bool, shape (n_freq_bins, n_frames)
        Binary mask with the same shape as `spectrogram`, True where a peak
        was found. Use `np.argwhere(peaks)` to get (freq_bin, frame) pairs.
    """
    local_max = ndimage.maximum_filter(
        spectrogram, size=(cfg.freq_neighborhood, cfg.time_neighborhood)
    )
    return (spectrogram == local_max) & (spectrogram > cfg.threshold)
