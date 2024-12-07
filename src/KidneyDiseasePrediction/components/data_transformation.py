import os
from mlProject import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transforming_data(self):
        df = pd.read_csv(self.config.data_path)
        cat_cols = [col for col in df.columns if df[col].dtype == "object"]
        num_cols = [col for col in df.columns if df[col].dtype != "object"]
        le = LabelEncoder()

        for col in cat_cols:
            df[col] = le.fit_transform(df[col])

        scaler = StandardScaler()
        for col in num_cols:
            df[col] = scaler.fit_transform(df[num_cols])

        # Enregistrer le dataset final dans le répertoire configuré
        df.to_csv(
            os.path.join(self.config.root_dir, "transforming_data.csv"), index=False
        )

        # Log information
        logger.info("Data transformation complete")
        logger.info(f"Data shape: {df.shape}")
