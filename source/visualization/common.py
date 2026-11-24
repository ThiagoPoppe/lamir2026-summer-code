import IPython.display as ipd
from ..utils import load_audio


def play_song(filepath: str) -> None:
    """Load an audio file and display a player for it."""
    
    print("Loading file:", filepath)
    wave, sr = load_audio(filepath)
    print("Loaded sampling rate:", sr)
    ipd.display(ipd.Audio(wave, rate=sr))
