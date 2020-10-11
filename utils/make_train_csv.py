import argparse
import glob
import pandas as pd
import os


def get_arguments():
    parser = argparse.ArgumentParser(description="make csv files")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="input the PATH of the directry where the datasets are saved",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv",
        help="input the PATH of the directry where the csv files will be saved",
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    img_paths = []
    cls_ids = []
    cls_labels = []

    for data in range(10):
        img_dir = os.path.join(args.dataset_dir, "normal" + str(data + 1))
        paths = glob.glob(os.path.join(img_dir, "*.npy"))
        paths.sort()
        img_paths += paths

    cls_ids += [0 for _ in range(len(img_paths))]
    cls_labels += ["normal" for _ in range(len(img_paths))]

    df = pd.DataFrame(
        {"img_path": img_paths, "cls_id": cls_ids, "cls_label": cls_labels},
        columns=["img_path", "cls_id", "cls_label"],
    )

    df.to_csv(os.path.join(args.save_dir, "{}.csv").format("train_npy"), index=None)

    print("Done")


if __name__ == "__main__":
    main()
