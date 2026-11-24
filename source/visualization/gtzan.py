import pandas as pd
import ipywidgets as widgets
import IPython.display as ipd
import matplotlib.pyplot as plt

from typing import Callable
from .common import play_song
from ..fingerprint import load_audio

_SONG_DROPDOWN_PLACEHOLDER = ("Select a song...", None)


def plot_top10_artists_per_genre(catalog_df, genre):
    if genre == 'All':
        top10_artists = catalog_df['artist'].value_counts().head(10)
    else:
        genre_catalog_df = catalog_df[catalog_df.genre == genre]
        top10_artists = genre_catalog_df['artist'].value_counts().head(10)

    plt.title(f'Top 10 most common artists in the "{genre}" genre')
    plt.barh(top10_artists.index, top10_artists.values)
    plt.xlabel('Number of songs')
    plt.ylabel('Artist')
    plt.show()


def song_selector(
    catalog_df: pd.DataFrame,
    on_select: Callable[[str], None] = play_song,
    width: str = "400px",
) -> widgets.VBox:
    """
    Build an interactive artist -> song selector from a catalog.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Catalog with the columns ``artist``, ``title`` and ``filepath``.
    on_select : callable, optional (default = play_song)
        Function called with the filepath whenever a song is selected. Anything
        it prints or displays appears below the selector.
    width : str, optional (default = "400px")
        CSS width of the widgets.

    Returns
    -------
    widgets.VBox
        The selector, ready to be displayed in a notebook.
    """
    artists = sorted(catalog_df.artist.dropna().unique())
    layout = widgets.Layout(width=width)

    artist_box = widgets.Combobox(
        options=artists, placeholder="Type here...", description="Artist:",
        ensure_option=True, layout=layout,
    )
    song_dropdown = widgets.Dropdown(
        options=[_SONG_DROPDOWN_PLACEHOLDER], description="Song:", disabled=True, layout=layout,
    )
    out = widgets.Output()

    def on_artist_change(change):
        out.clear_output()
        artist = change["new"]
        if artist in artists:
            songs = catalog_df[catalog_df.artist == artist]
            song_dropdown.options = [_SONG_DROPDOWN_PLACEHOLDER] + list(zip(songs.title, songs.filepath))
            song_dropdown.disabled = False
        else:
            song_dropdown.options = [_SONG_DROPDOWN_PLACEHOLDER]
            song_dropdown.disabled = True

    def on_song_change(change):
        out.clear_output()
        filepath = change["new"]
        if filepath is not None:
            with out:
                on_select(filepath)

    artist_box.observe(on_artist_change, names="value")
    song_dropdown.observe(on_song_change, names="value")

    return widgets.VBox([artist_box, song_dropdown, out])
