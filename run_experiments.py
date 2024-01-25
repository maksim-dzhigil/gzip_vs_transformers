import argparse
from model_predict import gzip_predict
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
from create_request_data import generate_test_data


os.environ["POSITIVE_DIR"] = "request_data/pos_reviews"
os.environ["NEGATIVE_DIR"] = "request_data/neg_reviews"
os.environ["IMDB_PATH"] = "data/IMDB.csv"


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Predict text class using the NCD"
    )
    parser.add_argument("-f", "--file_name", type=str, help="Path to single request file")
    parser.add_argument("-d", "--dir_name", type=str, help="Path to request files directory")
    parser.add_argument("-k", "--k_neighbours", default=1, type=int, help="Number of nearest neighbors for knn")
    parser.add_argument("-n", "--n_test_reviews", default=100, type=str, help="Number of imdb reviews for test")
    parser.add_argument("-c", "--console_input", type=bool, action=argparse.BooleanOptionalAction, help="Will the request be entered into the console? [True/False]")
    parser.add_argument("--regenerate_requests", default=False, action=argparse.BooleanOptionalAction, help="Whether the test data should be regenerated [True/False]")
    parser.add_argument("--plots_path", default="charts/", type=str, help="Folder for plots")
    
    return parser.parse_args()
    # return parser.parse_args(["-d", "request_data/", "-n", "2", "--regenerate_requests"])


def get_single_request(file_name):
    if file_name:
        with open(file_name, "r") as file:
            request = "".join([f for f in file.readlines()])
    else:
        request = input("Enter a movie review for classification: ")
    return request


def run_gzip_experiments(filenames, k):

    times = []
    preds = []
    for pos_file in tqdm(filenames):
        with open(pos_file, "r") as file:
            request = file.readlines()
        pred, time = gzip_predict(request, k)
        
        preds.append(pred)
        times.append(time)

    return np.array(preds), np.array(times)


def get_gzip_experiments_results(args):
    pos_filenames = sorted(os.listdir(os.path.join(args.dir_name, "pos_reviews")))
    neg_filenames = sorted(os.listdir(os.path.join(args.dir_name, "neg_reviews")))

    pos_filenames = [os.path.join(args.dir_name, "pos_reviews", filename) for filename in pos_filenames]
    neg_filenames = [os.path.join(args.dir_name, "neg_reviews", filename) for filename in neg_filenames]

    pos_preds, pos_times = run_gzip_experiments(pos_filenames, args.k_neighbours)
    neg_preds, neg_times = run_gzip_experiments(neg_filenames, args.k_neighbours)

    return {"pred": pos_preds, "time": pos_times}, {"pred": neg_preds, "time": neg_times}


def make_gzip_report(args):
    pos_res, neg_res = get_gzip_experiments_results(args)

    report = {}
    time = np.concatenate((pos_res["time"], neg_res["time"]))
    report["time_ndarray"] = time
    report["avg_time"] = np.mean(time)

    labels = np.concatenate((np.ones(pos_res["pred"].shape[0]), np.zeros(neg_res["pred"].shape[0])))
    preds = np.concatenate((pos_res["pred"], neg_res["pred"]))

    f1 = f1_score(labels, preds)
    report["f1_score"] = f1

    return report


def clear_request_data():
    for filename in os.listdir(os.environ["POSITIVE_DIR"]):
        os.remove(os.path.join(os.environ["POSITIVE_DIR"], filename))
    for filename in os.listdir(os.environ["NEGATIVE_DIR"]):
        os.remove(os.path.join(os.environ["NEGATIVE_DIR"], filename))
    os.rmdir(os.environ["POSITIVE_DIR"])
    os.rmdir(os.environ["NEGATIVE_DIR"])


def main():
    args = argument_parser()
    os.environ["N_TEST_REVIEWS"] = args.n_test_reviews

    if args.regenerate_requests:
        clear_request_data()
        generate_test_data()

    if not args.dir_name:
        request = get_single_request(args)
        pred, time = gzip_predict(request, args.k_neighbours)
        print(f"{time:.2f} sec | predicted class: {pred}")
        return
    
    gzip_report = make_gzip_report(args)
    print("***********************************")
    print(f"GZIP | avg time: {gzip_report['avg_time']:.2f} | f1: {gzip_report['f1_score']:.5f}")
    print("***********************************")


if __name__ == "__main__":
    main()