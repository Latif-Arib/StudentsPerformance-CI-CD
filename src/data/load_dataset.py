import sys
import os
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from preprocess import Preprocess
from src.models.train_model import TrainModel
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class LoadDataConfig:
    data_path: str = os.path.join('data','processed','data.csv')
    train_data_path: str = os.path.join('data','processed','train.csv')
    test_data_path: str = os.path.join('data','processed','test.csv')
    


class LoadData:
    def __init__(self):
        self.loading_config = LoadDataConfig()    
        
    def start_loading(self):
        """
        This method loads the raw dataset. Rename it as data.csv ( for convenient).
        Split it into train.csv and test.csv and then place then in data/processed directory. 
        
        """    
        
        logging.info('Entered into start loading method.')
        
        try:
            df = pd.read_csv(r"data\raw\StudentsPerformance.csv")
            logging.info('Dataset is read as DataFrame')
            
            os.makedirs(os.path.dirname(self.loading_config.data_path), exist_ok=True)
            
            df.to_csv(self.loading_config.data_path, index=False,header=True) 
            logging.info('Dataset is renamed and replaced.') 
            
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42) 
            
            train_set.to_csv(self.loading_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.loading_config.test_data_path, index=False, header=True)
    
            logging.info('Dataset split is done.')
            
            
            return (
                self.loading_config.train_data_path,
                self.loading_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == '__main__':
       load_data = LoadData()
       train_data, test_data = load_data.start_loading()     
       
       preprocess = Preprocess()
       X_train, X_test, y_train, y_test = preprocess.transform(train_data,test_data)
       
       train_model = TrainModel()
       print(train_model.train(X_train, X_test, y_train, y_test))