import os
import urllib.request as request
from mlProject import logger
import pandas as pd

from mlProject.entity.config_entity import DataCleaningConfig


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def cleaning_data(self):
        df = pd.read_csv(self.config.data_path)
        # dropping id column
        df.drop("id", axis=1, inplace=True)
        # Renommer les colonnes qui commencent par "Var Signalitiques."
        df.columns = [
            "age",
            "blood_pressure",
            "specific_gravity",
            "albumin",
            "sugar",
            "red_blood_cells",
            "pus_cell",
            "pus_cell_clumps",
            "bacteria",
            "blood_glucose_random",
            "blood_urea",
            "serum_creatinine",
            "sodium",
            "potassium",
            "haemoglobin",
            "packed_cell_volume",
            "white_blood_cell_count",
            "red_blood_cell_count",
            "hypertension",
            "diabetes_mellitus",
            "coronary_artery_disease",
            "appetite",
            "peda_edema",
            "aanemia",
            "classification",
        ]

        # Replace incorrect values in the 'diabetes_mellitus' column
        df["diabetes_mellitus"] = df["diabetes_mellitus"].replace(
            {"\tno": "no", "\tyes": "yes", " yes": "yes"}
        )

        # Replace incorrect values in the 'coronary_artery_disease' column
        df["coronary_artery_disease"] = df["coronary_artery_disease"].replace(
            {"\tno": "no"}
        )

        # Replace incorrect values in the 'class' column
        df["classification"] = df["classification"].replace(
            {"ckd\t": "ckd", "notckd": "not ckd"}
        )

        # Séparation des colonnes numériques et catégorielles
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Remplir les valeurs manquantes pour les colonnes numériques et catégorielles
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )  # Pour les colonnes numériques
        df[categorical_cols] = df[categorical_cols].fillna(
            df[categorical_cols].mode().iloc[0]
        )

        df.to_csv(os.path.join(self.config.root_dir, "clean_data.csv"), index=False)
        logger.info("Cleaning the data")
        logger.info(df.shape)
