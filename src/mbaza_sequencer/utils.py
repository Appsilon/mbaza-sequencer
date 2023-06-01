import numpy as np
import pandas as pd
from pathlib import Path
from pandarallel import pandarallel
from PIL import Image
from tqdm import tqdm
from scipy.stats import gmean


def get_pred_cols(df: pd.DataFrame):
    # Final "score" column
    last_score_column = list(filter(lambda col: col.startswith('score'), df.columns))[-1]

    # Get the ID (position) of the final "score" column
    first_class_id = df.columns.get_loc(last_score_column) + 1
    label_id = df.columns.get_loc('label')

    # Last column is "label"
    return df.columns[first_class_id:label_id]


def get_top_k_predictions(df: pd.DataFrame, k: int = 3):

    existing_top_k = df.columns[df.columns.str.startswith("score")].tolist()
    if existing_top_k:
        k = max(k, len(existing_top_k))
        update = True
    else:
        update = False

    pred_cols = get_pred_cols(df)
    preds = df[pred_cols].to_numpy()

    top_k = []
    for row in preds:
        inds = np.argpartition(row, -k)[-k:]
        inds = inds[np.argsort(-row[inds])]
        top_k.append(pred_cols[inds].tolist() + row[inds].tolist())

    top_cols = [f"pred_{i+1}" for i in range(k)] + [f"score_{i+1}" for i in range(k)]
    top_df = pd.DataFrame(top_k, columns=top_cols, index=df.index)

    if update:
        df.update(top_df)
    else:
        df = top_df.join(df)

    return df


def combine_sequence_predictions(df: pd.DataFrame):

    pred_cols = get_pred_cols(df)
    # One-liner throws a strange error
    # combined_preds = df.groupby("sequence")[pred_cols].apply(gmean)

    combined_preds = {}
    for n, group in df.groupby("sequence"):
        combined_preds[n] = gmean(group[pred_cols])

    combined_preds = pd.DataFrame(combined_preds, index=pred_cols).transpose()
    
    # Merging and indexing to directly replace new prediction columns and preserve order
    df[pred_cols] = df.merge(combined_preds, left_on="sequence", right_index=True, suffixes=("_DROP", ""))[pred_cols]
    
    # Need to reset top k scores
    df = get_top_k_predictions(df)

    return df


def get_exif_data(path: Path):

    exif = Image.open(path)._getexif()

    data = {
        "width": exif[40962],
        "height": exif[40963],
        "timestamp": exif[36867]
    }

    return data

from uuid import uuid4

def sequenize(df: pd.DataFrame, max_seq_len: float = float("inf"), max_image_delay: float = 5.):

    def group_sequence(group):

        seq, seq_id = None, 0
        for index, row in group.iterrows():

            if (
                index == group.index[0]
                or seq_id == max_seq_len - 1
                or abs(row["time_diff"]) > max_image_delay
                ):

                seq = uuid4()
                seq_id = 0
            else:
                seq_id += 1

            group.loc[index, "sequence"] = seq
            group.loc[index, "sequence_id"] = seq_id

        return group

    # Sort by folder and timestamp, then calculate time between images
    df = df.sort_values(by=["dir_path", "timestamp"])
    df["time_diff"] = df.groupby("dir_path")["timestamp"].diff().dt.total_seconds()

    seq_df = df.groupby("dir_path", group_keys=False).parallel_apply(group_sequence)
    seq_df = seq_df.sort_index()
    seq_df["sequence"] = pd.factorize(seq_df["sequence"])[0]
    
    return seq_df

    # reconstruct = []
    # seq_num, seq_id = 0, 0
    # for _, group in tqdm(df.groupby("dir_path"), desc="Sequencing groups"):
        
    #     for index, row in group.iterrows():

    #         if (
    #             index == group.index[0]
    #             or seq_id == max_seq_len - 1
    #             or abs(row["time_diff"]) > max_image_delay
    #             ):

    #             seq_id = 0
    #             seq_num += 1
    #         else:
    #             seq_id += 1

    #         group.loc[index, "sequence_num"] = seq_num
    #         group.loc[index, "sequence_id"] = seq_id

    #     reconstruct.append(group)

    return pd.concat(reconstruct)
