import argparse
from model_predict import gzip_predict

def argument_parser():
    parser = argparse.ArgumentParser(
        description="Predict text class using the NCD"
    )
    parser.add_argument("-f", "--file_name", type=str, help="Path to single request file")
    parser.add_argument("-d", "--dir_name", type=str, help="Path to request files directory")
    parser.add_argument("-c", "--console_input", type=bool, help="Will the request be entered into the console? [True/False]")
    parser.add_argument("-k", "--k_neighbours", default=1, type=int, help="Number of nearest neighbors for knn")
    parser.add_argument("--plots_path", default="charts/", type=str, help="Folder for plots")
    
    return parser.parse_args()


def get_single_request(file_name):
    if file_name:
        with open(file_name, "r") as file:
            request = "".join([f for f in file.readlines()])
    else:
        request = input("Enter a movie review for classification: ")
    return request


def main():
    args = argument_parser()
    request = get_single_request(args.file_name)

    response = gzip_predict(request, args.k_neighbours)
    print(f"time: {response['time']:.2f} | predicted class: {response['class']}")


if __name__ == "__main__":
    main()