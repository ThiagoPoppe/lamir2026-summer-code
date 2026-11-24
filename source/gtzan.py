import os
import kagglehub
import urllib.request

import pandas as pd

_CORRUPTED_FILES = {"jazz.00054.wav"}


def download(output_dir: str = './gtzan_dataset') -> tuple[str, str]:
    dataset_path = kagglehub.dataset_download(
        handle='andradaolteanu/gtzan-dataset-music-genre-classification',
        output_dir=output_dir
    )

    audio_dir = os.path.join(dataset_path, "Data", "genres_original")
    return dataset_path, audio_dir


def build_catalog_dataframe(audio_dir: str) -> pd.DataFrame:
    records = []
    metadata = _fetch_metadata_information()

    for genre in sorted(os.listdir(audio_dir)):
        genre_path = os.path.join(audio_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        for filename in sorted(os.listdir(genre_path)):
            if not filename.endswith(".wav") or filename in _CORRUPTED_FILES:
                continue

            meta = metadata.get(filename, {"artist": "Unknown", "title": "Unknown", "fullname": filename})
            records.append({
                "track_id": os.path.splitext(filename)[0],  # e.g., "blues.00000"
                "filename": filename,
                "genre": genre,
                "artist": meta["artist"],
                "title": meta["title"],
                "display_name": meta["fullname"],
                "filepath": os.path.join(genre_path, filename)
            })

    catalog_df = pd.DataFrame(records)
    return catalog_df


def _fetch_metadata_information():
    index_url = "https://raw.githubusercontent.com/boblsturm/GTZAN/main/index.txt"
    metadata = {}

    with urllib.request.urlopen(index_url) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith("##"):
                continue

            parts = [p.strip() for p in line.split(":::")]
            filename = parts[0]
            artist = parts[1] if len(parts) > 1 and parts[1] else "Unknown"
            title = parts[2] if len(parts) > 2 and parts[2] else "Unknown"

            metadata[filename] = {
                "artist": artist,
                "title": title,
                "fullname": f"{artist} - {title}" if artist != "Unknown" else filename
            }

    return metadata
