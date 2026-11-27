import bisect
import librosa
import numpy as np

from scipy import ndimage
from typing import Any, Optional
from collections import defaultdict


_LIBROSA_DEFAULT_HOP_LENGTH = 512


def load_audio(
    filepath: str,
    sr: int = None,
    start: float = 0.0,
    duration: float = None
) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a mono waveform.

    Parameters
    ----------
    filepath : str
        Path to the audio file. Any format supported by librosa/soundfile
        or audioread can be used.
    sr : int, optional (default = None)
        Target sample rate to resample the audio. If None, the native sample rate is used.
    start : float, optional (default = 0.0)
        Time offset in seconds to start loading the audio. Must be between 0 and 30 seconds.
    duration : float, optional (default = None)
        Duration of the audio to load in seconds. Start + duration must not exceed 30 seconds.
        If None, the whole audio starting from `start` is loaded.

    Returns
    -------
    wave : np.ndarray, shape (n_samples,)
        Mono waveform, as floating-point values in [-1, 1].
    sr : int
        Sample rate of `wave` in Hz. Equals `sr` if it was given,
        otherwise the native sample rate of the file.

    Raises
    ------
    ValueError
        If `start` or `duration` are out of range.
    """

    if sr is not None and sr <= 0:
        raise ValueError("Sample rate must be positive.")

    if start < 0 or start > 30:
        raise ValueError("start must be between 0 and 30 seconds.")

    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive.")
        if start + duration > 30:
            raise ValueError("start + duration must not exceed 30 seconds.")

    return librosa.load(filepath, sr=sr, offset=start, duration=duration)


def compute_spectral_feature(
    wave: np.ndarray,
    sr: int, kind:
    str = 'stft'
) -> np.ndarray:
    """
    Compute a non-negative time-frequency representation of a waveform.

    Parameters
    ----------
    wave : np.ndarray, shape (n_samples,)
        Mono waveform.
    sr : int
        Sample rate of `wave` in Hz.
    kind : {'stft', 'cqt', 'chroma'}, optional (default = 'stft')
        Type of representation to compute.
        - 'stft': magnitude of the Short-Time Fourier Transform, with linear
          frequency resolution. Only the first 128 frequency bins are used.
        - 'cqt': magnitude of the Constant-Q Transform, with logarithmic
          frequency resolution of one bin per semitone, starting at C1
          (~32.7 Hz), having 7 octaves (84 bins) by default.
        - 'chroma': 12-bin pitch-class profile (one bin per semitone, octaves
          folded together), computed from the STFT.

    Returns
    -------
    spectrogram : np.ndarray, shape (num_bins, n_frames)
        Magnitude spectrogram. `num_bins` is 128 for 'stft', 84 for 'cqt',
        and 12 for 'chroma'.

    Notes
    -----
    Values are on the scale of the raw STFT magnitude for 'stft' and normalized
    between [0, 1] for 'chroma', so thresholds are not directly comparable between
    the two.
    """

    if kind == "stft":
        num_bins = 128
        return np.abs(librosa.stft(y=wave, hop_length=_LIBROSA_DEFAULT_HOP_LENGTH))[:num_bins]
    elif kind == "cqt":
        return np.abs(librosa.cqt(y=wave, sr=sr, hop_length=_LIBROSA_DEFAULT_HOP_LENGTH))
    elif kind == "chroma":
        return librosa.feature.chroma_stft(y=wave, sr=sr, hop_length=_LIBROSA_DEFAULT_HOP_LENGTH)
    else:
        raise ValueError("kind must be one of 'stft', 'cqt', or 'chroma'.")


def compute_constellation_map(
    spectral_feature: np.ndarray,
    freq_neighborhood: int = 11,
    time_neighborhood: int = 5,
    threshold: float = 0.01
) -> np.ndarray:
    """
    Find the local maxima of a spectral feature that make up the constellation map.

    A point is a peak if it is the maximum of its local window (of size
    `freq_neighborhood` x `time_neighborhood`) and its magnitude is above
    `threshold`.

    Parameters
    ----------
    spectral_feature : np.ndarray, shape (n_freq_bins, n_frames)
        Non-negative time-frequency representation, e.g. the output of
        `compute_spectral_feature`.
    freq_neighborhood : int, optional (default = 11)
        Height of the local window, in frequency bins (centered on the point,
        so a value of 11 looks 5 bins up and 5 bins down). Should be odd.
    time_neighborhood : int, optional (default = 5)
        Width of the local window, in time frames (centered on the point,
        so a value of 5 looks 2 frames left and 2 frames right). Should be odd.
    threshold : float, optional (default = 0.01)
        Minimum magnitude required to be considered a peak.

    Returns
    -------
    constellation_map : np.ndarray of bool, shape (n_freq_bins, n_frames)
        Binary mask with the same shape as `spectral_feature`, True where a peak
        was found. Use `np.argwhere(constellation_map)` to get (freq_bin, frame) pairs.
    """

    if freq_neighborhood % 2 == 0 or time_neighborhood % 2 == 0:
        raise ValueError("freq_neighborhood and time_neighborhood must be odd.")
    if freq_neighborhood <= 0 or time_neighborhood <= 0:
        raise ValueError("freq_neighborhood and time_neighborhood must be positive.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    size = (freq_neighborhood, time_neighborhood)
    local_max = ndimage.maximum_filter(spectral_feature, size=size)

    constellation_map = (spectral_feature == local_max) & (spectral_feature > threshold)
    return constellation_map


def compute_hashes(
    constellation_map: np.ndarray,
    sr: int,
    metadata: Optional[dict[str, Any]] = None,
    target_zone_freq_size: int = 31,
    target_zone_time_size: int = 19,
    target_zone_time_delay: int = 20
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate fingerprint hashes from a constellation map using anchor-target pairs.

    Every peak in the constellation map is treated as an anchor. For each anchor,
    a rectangular target zone is defined in the time-frequency plane, centered
    `target_zone_time_delay` frames after the anchor in time and on the anchor's
    frequency bin. Each peak that falls inside this zone is paired with the
    anchor to form a hash of the form ``"{anchor_freq}_{target_freq}_{timedelta}"``.

    Parameters
    ----------
    constellation_map : numpy.ndarray
        Boolean 2D array of shape ``(num_freq_bins, num_time_frames)``, where
        True marks a constellation peak.
    sr : int
        Sample rate in Hz of the audio the constellation map was computed from.
    metadata : dict of str to Any, optional (default = None)
        Additional information (e.g. song name or ID) stored alongside every hash
        occurrence. If None, an empty dictionary is used.
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

    Returns
    -------
    collections.defaultdict of str to list of dict
        Mapping from each hash string to the list of its occurrences. Each
        occurrence is a dictionary containing the entries from `metadata` plus
        ``'offset_in_seconds'`` (float), the time of the anchor peak.

    Notes
    -----
    Hashes are only comparable between constellation maps computed with the same
    feature kind, since the meaning of a frequency bin depends on it.

    ``'offset_in_seconds'`` is relative to the start of the loaded excerpt
    (see `start` parameter of the `load_audio` function).

    Peaks are sorted by time so that the candidate targets for each anchor can
    be located efficiently with binary search instead of scanning every peak.
    """

    if metadata is None:
        metadata = {}

    peak_freqs, peak_times = np.where(constellation_map)
    peaks = sorted(zip(peak_freqs, peak_times), key=lambda peak: peak[1])
    sorted_peak_times = [peak_time for _, peak_time in peaks]

    seconds_per_frame = _LIBROSA_DEFAULT_HOP_LENGTH / sr
    half_freq = target_zone_freq_size // 2
    half_time = target_zone_time_size // 2

    hashes = defaultdict(list)
    for anchor_freq, anchor_time in peaks:
        zone_min_freq = anchor_freq - half_freq
        zone_max_freq = anchor_freq + half_freq

        zone_center_time = anchor_time + target_zone_time_delay
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



def find_match(
    query_hashes: dict[str, list[dict]],
    db_hashes: dict[str, list[dict]]
) -> tuple[str, float, int]:
    """Find the best match for a query constellation map in the database.

    Parameters
    ----------
    query_hashes : np.ndarray
        The hashes of the query audio.
    db_hashes : dict[str, list[dict]]
        The hashes of the database.

    Returns
    -------
    matched_song : str
        The name of the matched song.
    matched_offset : float
        The offset of the match in seconds.
    counter : int
        The number of matching peaks.
    """

    match_counter = defaultdict(int)

    for hash, query_anchors in query_hashes.items():
        if hash not in db_hashes:
            continue

        for query_anchor in query_anchors:
            query_anchor_time = query_anchor['offset_in_seconds']

            for db_anchor in db_hashes[hash]:
                alignment_offset = db_anchor['offset_in_seconds'] - query_anchor_time

                match_key = (db_anchor['track_id'], round(alignment_offset, 2))
                match_counter[match_key] += 1

    (matched_track_id, matched_offset), counter = max(
        match_counter.items(),
        key=lambda item: item[1]
    )

    return matched_track_id, matched_offset, counter
