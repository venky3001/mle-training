import argparse
import logging
import os
import pickle as pkl

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    help="Please give the appropriate input path for the CSV file",
    # choices=["datasets"],
    default="datasets",
)
parser.add_argument(
    "-p",
    "--pickle",
    help="Please give the appropriate output path for the pickle file",
    # choices=["artifacts"],
    default="artifacts",
)
parser.add_argument(
    "--loglevel",
    help="Please specify the log level",
    choices=["INFO", "DEBUG", "ERROR", "CRITICAL", "WARNING"],
    default="INFO",
)
parser.add_argument(
    "--logpath",
    help="Please specify the directory for the log file ",
    # choices=["", "logs"],
    default="logs",
)
args = parser.parse_args()
os.makedirs("../" + args.logpath, exist_ok=True)
log_dict = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
}
if args.logpath == "":
    file_name = ""
else:
    file_name = "../" + args.logpath + "/score.log"
logging.basicConfig(
    level=log_dict[args.loglevel],
    filename=file_name,
    format="%(asctime)s-%(levelname)s-[%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
ARTIFACTS_PATH = os.path.join("../" + args.pickle)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "lin_rmse.pkl")
with open(PICKLE_PATH, "rb") as file:
    lin_rmse = pkl.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "lin_mae.pkl")
with open(PICKLE_PATH, "rb") as file:
    lin_mae = pkl.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "tree_rmse.pkl")
with open(PICKLE_PATH, "rb") as file:
    tree_rmse = pkl.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "cvres.pkl")
with open(PICKLE_PATH, "rb") as file:
    cvres = pkl.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "cvres2.pkl")
with open(PICKLE_PATH, "rb") as file:
    cvres2 = pkl.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "final_rmse.pkl")
with open(PICKLE_PATH, "rb") as file:
    final_rmse = pkl.load(file)

print("Linear Regression mae  : ", lin_mae)
print("Linear Regression rmse : ", lin_rmse)
print("Decision Tree          : ", tree_rmse)
print("Random Forest ")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("Modified Random Forest ")
for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
    print(np.sqrt(-mean_score), params)


print("Final Model rmse : ", final_rmse)

logging.info("scores are displayed")
