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
    class2ids = {"normal": 0, "anomal": 1}

    for c in ["anomal", "normal"]:
        img_dir = os.path.join(args.dataset_dir)
        if c == "anomal":
            for sample in range(2):
                paths = glob.glob(
                    os.path.join(img_dir, c, c + str(sample + 1), "*.npy")
                )

                img_numbers = len(paths)
                for i in range(img_numbers):
                    img_paths.append(
                        os.path.join(
                            "/mana/test/eg{}".format(img_dir[-1]),
                            c,
                            c + str(sample + 1),
                            "%06d.npy" % i,
                        )
                    )

                cls_ids += [class2ids[c] for _ in range(img_numbers)]
                cls_labels += [c for _ in range(img_numbers)]
        else:
            for sample in range(3):
                paths = glob.glob(
                    os.path.join(img_dir, c, c + str(sample + 1), "*.npy")
                )

                img_numbers = len(paths)
                for i in range(img_numbers):
                    img_paths.append(
                        os.path.join(
                            "/mana/test/eg{}".format(img_dir[-1]),
                            c,
                            c + str(sample + 1),
                            "%06d.npy" % i,
                        )
                    )

                cls_ids += [class2ids[c] for _ in range(img_numbers)]
                cls_labels += [c for _ in range(img_numbers)]

    df = pd.DataFrame(
        {"img_path": img_paths, "cls_id": cls_ids, "cls_label": cls_labels},
        columns=["img_path", "cls_id", "cls_label"],
    )

    df.to_csv(
        os.path.join(args.save_dir, "{}.csv").format(
            "test_npy_eg{}".format(img_dir[-1])
        ),
        index=None,
    )

    print("Done")


if __name__ == "__main__":
    main()
