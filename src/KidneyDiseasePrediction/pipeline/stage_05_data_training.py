from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from mlProject.components.data_training import DataTraining

from pathlib import Path


STAGE_NAME = "Data Spliting stage"


class DataSplitingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_training_config = config.get_data_training_config()
                data_training = DataTraining(config=data_training_config)
                data_training.train_test_spliting()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataSplitingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
