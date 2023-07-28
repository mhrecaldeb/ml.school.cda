
import os
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pickle import dump


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIRECTORY = "/opt/ml/processing"
DATA_FILEPATH_TRAIN = Path(BASE_DIRECTORY) / "input_train" / "mnist_train.csv" #input independiente para cada una
DATA_FILEPATH_TEST = Path(BASE_DIRECTORY) / "input_test" / "mnist_test.csv"



def _save_splits(base_directory, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def _save_pipeline(base_directory, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_directory) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", "wb"))


def _save_classes(base_directory, classes):
    """
    Saves the list of classes from the dataset.
    """
    path = Path(base_directory) / "classes"
    path.mkdir(parents=True, exist_ok=True)

    np.asarray(classes).tofile(path / "classes.csv", sep=",")


def _save_baseline(base_directory, df_train, df_test):
    """
    During the data and quality monitoring steps, we will need a baseline
    to compute constraints and statistics. This function will save that
    baseline to the disk.
    """

    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()
        df.to_json(
            baseline_path / f"{split}-baseline.json", orient="records", lines=True
        )


def preprocess(base_directory, data_filepath_train, data_filepath_test):  # comparado con pinguins este solicita el test file porque no lo genera en la funcion
    """
    Preprocesses the supplied raw dataset and splits it into a train,
    validation, and a test set.
    """

    df_train = pd.read_csv(data_filepath_train)
    df_test = pd.read_csv(data_filepath_test)

    numeric_features = df_train.select_dtypes(include=['int']).columns.tolist()
    numeric_features.remove("label")
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

# No hay valores categoricos todos son int incluso label 
    
#    categorical_transformer = Pipeline(
#        steps=[
#            ("imputer", SimpleImputer(strategy="most_frequent")),
#            ("encoder", OneHotEncoder()),
#        ]
#    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            # ("categorical", categorical_transformer, ["island"]),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor)
        ]
    )

#    df.drop(["sex"], axis=1, inplace=True)
#    df = df.sample(frac=1, random_state=42)
    
# no se pueden ordenar randomicamente porque son imagenes

#    df_train, temp = train_test_split(df, test_size=0.3)
#    df_validation, df_test = train_test_split(temp, test_size=0.5)

# se dividira el set de train en train y validation en 80/20

    df_train, df_validation = train_test_split(df_train, test_size=0.2)

# el set de test se mantiene como viene en la data original
    
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train.label)
    y_validation = label_encoder.transform(df_validation.label)
    y_test = label_encoder.transform(df_test.label)
    
    _save_baseline(base_directory, df_train, df_test)

    df_train = df_train.drop(["label"], axis=1)
    df_validation = df_validation.drop(["label"], axis=1)
    df_test = df_test.drop(["label"], axis=1)

    X_train = pipeline.fit_transform(df_train)
    X_validation = pipeline.transform(df_validation)
    X_test = pipeline.transform(df_test)

    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate((X_validation, np.expand_dims(y_validation, axis=1)), axis=1)
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

    _save_splits(base_directory, train, validation, test)
    _save_pipeline(base_directory, pipeline=pipeline)
    _save_classes(base_directory, label_encoder.classes_)  # hay que tomar en cuenta que las classes no estan representadas todas en df_train


if __name__ == "__main__":
    preprocess(BASE_DIRECTORY, DATA_FILEPATH_TRAIN, DATA_FILEPATH_TEST)
