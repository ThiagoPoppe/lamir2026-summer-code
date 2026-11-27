import pandas as pd
import ipywidgets as widgets
import IPython.display as ipd

from ..fingerprint import load_audio

_SONG_DROPDOWN_PLACEHOLDER = ("Select a song...", None)


def song_selector(
    catalog_df: pd.DataFrame,
    width: str = "400px",
) -> widgets.VBox:
    """
    Build an interactive artist -> song selector from a catalog.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Catalog with the columns ``artist``, ``title`` and ``filepath``.
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
                print(f'Loading track from {filepath}')
                wave, sr = load_audio(filepath)
                print(f'Loaded track with sampling rate = {sr}')
                ipd.display(ipd.Audio(wave, rate=sr))

    artist_box.observe(on_artist_change, names="value")
    song_dropdown.observe(on_song_change, names="value")

    return widgets.VBox([artist_box, song_dropdown, out])
