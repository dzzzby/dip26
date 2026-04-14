import glob
import os
import tarfile
import argparse
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description='Download pix2pix dataset.')
    parser.add_argument('--dataset', default='facades', help='Dataset name, e.g. facades, maps, edges2shoes, night2day')
    return parser.parse_args()


def main():
    args = parse_args()
    file_name = args.dataset
    url = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{file_name}.tar.gz"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    tar_path = os.path.join(datasets_dir, f"{file_name}.tar.gz")
    target_dir = os.path.join(datasets_dir, file_name)

    print(f"Downloading dataset from: {url}")
    urllib.request.urlretrieve(url, tar_path)

    print(f"Extracting to: {datasets_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=datasets_dir)

    if os.path.exists(tar_path):
        os.remove(tar_path)

    train_list = sorted(glob.glob(os.path.join(target_dir, "train", "*.jpg")))
    val_list = sorted(glob.glob(os.path.join(target_dir, "val", "*.jpg")))

    train_list_path = os.path.join(script_dir, "train_list.txt")
    val_list_path = os.path.join(script_dir, "val_list.txt")

    with open(train_list_path, "w", encoding="utf-8") as f:
        for p in train_list:
            f.write(p + "\n")

    with open(val_list_path, "w", encoding="utf-8") as f:
        for p in val_list:
            f.write(p + "\n")

    print(f"Done. dataset={file_name}, train images: {len(train_list)}, val images: {len(val_list)}")
    print(f"Generated: {train_list_path}")
    print(f"Generated: {val_list_path}")


if __name__ == "__main__":
    main()
