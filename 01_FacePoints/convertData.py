import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tqdm


TRAIN_SIZE = 0.8
CHUNK_SIZE = 10000


def convert_data(root, split="train", part=1.0):
    landmark_file_name = os.path.join(root, 'landmarks.csv') if split is not "test" \
        else os.path.join(root, "test_points.csv")

    col_names = pd.read_csv(landmark_file_name, nrows=0, sep='\t').columns
    n_points = len(col_names) - 1

    with open(landmark_file_name, "rt") as fp:
        num_lines = sum(1 for line in fp)
    num_lines -= 1  # header
    num_lines = int(num_lines * part)

    n_samples = {
        "train": int(TRAIN_SIZE * num_lines),
        "val": num_lines - int(TRAIN_SIZE * num_lines),
        "test": num_lines
    }

    borders = {"train": (0, int(TRAIN_SIZE * num_lines)),
               "val": (int(TRAIN_SIZE * num_lines), num_lines),
               "test": (0, num_lines)}
    border = borders[split]
    if split in ("train", "val"):
        chunks = pd.read_csv(landmark_file_name, skiprows=border[0], nrows=(border[1] - border[0]), delimiter='\t',
                             dtype={name: np.uint16 for name in col_names[1:]},
                             chunksize=CHUNK_SIZE)
    else:
        chunks = pd.read_csv(landmark_file_name, delimiter='\t',
                             chunksize=CHUNK_SIZE)

    if split in ("train", "val"):
        X = np.zeros((n_samples[split], n_points), dtype=np.int16)
    names = []
    beg_line = 0
    for chunk in tqdm.tqdm(chunks, total=(n_samples[split] // CHUNK_SIZE + 1)):
        names.extend(os.path.join(root, "images/") + chunk.iloc[:, 0].values)
        if split in ("train", "val"):
            X[beg_line:beg_line + chunk.shape[0]] = chunk.iloc[:, 1:].values
        beg_line += chunk.shape[0]

    if split in ("train", "val"):
        return names, X
    else:
        return names


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--data", "-d", help="Path to dir with dataset (Contain folders test and train)", default=None)
    parser.add_argument(
        "--part", "-p", help="Part of dataset (create small dataset for experiments)", default="1.0")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    logging.info("convert train data")
    names_train, X_train = convert_data(
        os.path.join(args.data, 'train'), split="train", part=float(args.part))
    logging.info("convert val data")
    names_val, X_val = convert_data(
        os.path.join(args.data, 'train'), split="val", part=float(args.part))
    logging.info("convert test data")
    names_test = convert_data(os.path.join(args.data, 'test'), split="test")

    if (float(args.part)==1.0):
        folderSave = "prepare_data"
    else:
        folderSave = "prepare_data_{}PERCENT".format(float(args.part)*100.0)

    if not os.path.exists(os.path.join(args.data, folderSave)):
        os.makedirs(os.path.join(args.data, folderSave))

    np.save(os.path.join(args.data, folderSave, 'X_train.npy'), X_train)
    np.save(os.path.join(args.data, folderSave, 'X_val.npy'), X_val)

    with open(os.path.join(args.data, folderSave, 'names_train'), "w") as f:
        for name in names_train:
            f.write(f"{name}\n")

    with open(os.path.join(args.data, folderSave, 'names_val'), "w") as f:
        for name in names_val:
            f.write(f"{name}\n")

    with open(os.path.join(args.data, folderSave, 'names_test'), "w") as f:
        for name in names_test:
            f.write(f"{name}\n")


if __name__ == '__main__':
    main()

