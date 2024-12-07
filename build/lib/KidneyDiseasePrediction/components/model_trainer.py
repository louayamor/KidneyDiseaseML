import pandas as pd
import os
from mlProject import logger
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
)
import mlflow

from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Prepare features (X) and target (y)
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Initialize Logistic Regression with hyperparameters from config
        rf = RandomForestClassifier (
            n_estimators=self.config.n_estimators,
            criterion=self.config.criterion,
            min_samples_split=self.config.min_samples_split,
    
        )

        # Train the model
        rf.fit(train_x, train_y)

        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))
