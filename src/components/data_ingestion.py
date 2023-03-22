import os
import sys
sys.path.insert(0, 'D:\Storage\mlproject\src') #sys.path.append('D:\Storage\mlproject\src') 
from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig
from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:  
# three paths for data saving:          
#                                        #folder name  #file name
    train_data_path: str = os.path.join('artifacts', "train.csv") #1
    test_data_path: str = os.path.join('artifacts', "test.csv") #2
    raw_data_path: str = os.path.join('artifacts', "data.csv") #3

class DataIngestion: #when we initialize it
    def __init__(self):
        self.ingestion_config = DataIngestionConfig #the above 3 paths gets saved in this class variable

    def initiate_data_ingestion(self):
        logging.info("Executing data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Dataset imported/ loaded")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header= True)

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)

            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
# combining data tranformation and ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))