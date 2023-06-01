import pandas as pd
from datetime import datetime
from pandarallel import pandarallel
from pathlib import Path
from time import time
from src.mbaza_sequence import utils

pandarallel.initialize(progress_bar=True)


start_time = time()
base_data_path = Path("/home/andrew/data/2018/01_10_2018/Corridor 1/A")
csv_path = Path("/home/andrew/mbaza-sequence/classification_result_2023-06-01_11-51-33.csv")

df = pd.read_csv(csv_path)
print(f"Total images to process: {df.shape[0]}")
csv_columns = list(df.columns)

full_paths = df["location"].map(lambda x: base_data_path / x)
exif_data = full_paths.map(lambda x: utils.get_exif_data(x))
exif_df = exif_data.apply(pd.Series)

df["timestamp"] = exif_df["timestamp"].map(lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S"))
df["dir_path"] = full_paths.map(lambda x: x.parent)

df = utils.sequenize(df)
df = utils.combine_sequence_predictions(df)

# Output original columns + new sequence columns
df = df.filter(regex="sequence*|" + "|".join(csv_columns))

print(
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
    sep="\n"
)

df.to_csv((csv_path.parent / (csv_path.stem + "_sequenced")).with_suffix(".csv"))