
import boto3
import sagemaker
import sys
from pathlib import Path

BUCKET = "mlschool-cda"
MNIST_FOLDER = "mnist"


S3_FILEPATH = f"s3://{BUCKET}/{MNIST_FOLDER}"

# DATASET_FOLDER = Path("dataset")

DATA_FILEPATH_TRAIN = Path().resolve() / "dataset" / "mnist_train.csv" #estas variables estan definidas aqui
DATA_FILEPATH_TEST = Path().resolve() / "dataset" / "mnist_test.csv"

#*************************************************************
#TRAIN_SET_S3_URI = sagemaker.s3.S3Uploader.upload(
#    local_path=str(DATASET_FOLDER / "mnist_train.csv"), 
#    desired_s3_uri=S3_FILEPATH,
#)

#TEST_SET_S3_URI = sagemaker.s3.S3Uploader.upload(
#    local_path=str(DATASET_FOLDER / "mnist_test.csv"), 
#    desired_s3_uri=S3_FILEPATH,
#)
#*************************************************************

#BUCKET = "mlschool-cda"
#S3_LOCATION = f"s3://{BUCKET}/penguins"
#DATA_FILEPATH = Path().resolve() / "data.csv"


sagemaker_client = boto3.client("sagemaker")
iam_client = boto3.client("iam")
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
