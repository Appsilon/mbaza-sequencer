import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pandarallel import pandarallel
from pathlib import Path
from pydantic import BaseModel, FilePath, DirectoryPath
from time import time
from mbaza_sequencer import utils

pandarallel.initialize(progress_bar=True)


class Settings(BaseModel):
    csv_path: FilePath
    image_path: DirectoryPath
    max_images: float = float("inf")
    max_delay: float = 3.


def process(settings: Settings):

    start_time = time()

    df = pd.read_csv(settings.csv_path)
    print(f"Total images to process: {df.shape[0]}")
    csv_columns = list(df.columns)

    df["timestamp"] = df["timestamp"].replace(" ", np.nan)

    if df["timestamp"].isna().any():
        print("\nExtracting timestamps from images ...")
        full_paths = df["location"].map(lambda x: settings.image_path / x)
        exif_data = full_paths.parallel_map(lambda x: utils.get_exif_data(x))
        exif_df = exif_data.apply(pd.Series)

        df["timestamp"] = exif_df["timestamp"]
        df["dir_path"] = full_paths.map(lambda x: x.parent)

    df["timestamp"] = df["timestamp"].map(lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S"))
    df = utils.sequenize(df, max_seq_len=settings.max_images, max_image_delay=settings.max_delay)
    df = utils.combine_sequence_predictions(df)

    # Output original columns + new sequence columns
    df = df.filter(regex="sequence*|" + "|".join(csv_columns))

    print(
        "--------------------------------",
        "Sequence output snippet",
        "--------------------------------",
        df.filter(regex="timestamp|sequence*|pred_1|score_1").head(20),
        "--------------------------------",
        "Sequence statistics",
        "--------------------------------",
        f"Total sequences: {df['sequence'].nunique()}", 
        f"Mean images per sequence: {df['sequence'].value_counts().mean():.2f}", 
        f"Max images per sequence {df['sequence'].value_counts().max()}",
        "--------------------------------",
        f"Time taken: {time() - start_time:.2f}s",
        "--------------------------------",
        sep="\n"
    )

    df.to_csv((settings.csv_path.parent / (settings.csv_path.stem + "_sequenced")).with_suffix(".csv"))

def main():
    parser = argparse.ArgumentParser(description="Convert Mbaza image predictions to sequence predictions")
    parser.add_argument("csv_path", type=Path, help="Path to csv output from Mbaza")
    parser.add_argument("image_path", type=Path, help="Path to images (same as supplied to Mbaza)")
    parser.add_argument("--max_images", type=int, help="Maximum number of images per sequence", default=float("inf"))
    parser.add_argument("--max_delay", type=float, help="Maximum image-to-image delay (seconds) before a new sequence is defined", default=3.)

    args = parser.parse_args()

    settings = Settings(
        csv_path=args.csv_path,
        image_path=args.image_path,
        max_images=args.max_images,
        max_delay=args.max_delay
    )

    process(settings=settings)


if __name__ == "__main__":
    
    main()
