import pandas as pd
import argparse
import os


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Prepare movie reviews for testing models"
    )
    parser.add_argument("imdb_file", type=str, help="Path to imdb dataset")
    return parser.parse_args()


def generate_test_data():
    df = pd.read_csv(os.environ["IMDB_PATH"]).iloc[:int(os.environ["N_TEST_REVIEWS"])]

    texts_pos = df[df["sentiment"] == "positive"]['review'].tolist()
    texts_neg = df[df["sentiment"] == "negative"]['review'].tolist()

    os.makedirs(os.environ["POSITIVE_DIR"], exist_ok=True)
    os.makedirs(os.environ["NEGATIVE_DIR"], exist_ok=True)

    for i, review in enumerate(texts_pos):
        filename = os.path.join(os.environ["POSITIVE_DIR"] , f"pos_{i+1}.txt")
        with open(filename, "w") as file:
            file.writelines(review)

    for i, review in enumerate(texts_neg):
        filename = os.path.join(os.environ["NEGATIVE_DIR"], f"neg_{i+1}.txt")
        with open(filename, "w") as file:
            file.writelines(review)


if __name__ =="__main__":
    args = argument_parser()
    generate_test_data(args.imdb_file)