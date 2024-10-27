import os
import sys
import pandas as pd
from src.utils.utils import load_model_and_data
from src.utils.exceptions import CustomException
from src.utils.logger import logging


class Prediction:
    def __init__(self):
        pass
    
    def predict(self,new_data) -> float:
        
        try:
            preprocessor = load_model_and_data('data/processed/processed.pkl')
            ripe_data = preprocessor.transform(new_data)
            
            model = load_model_and_data('models/model.pkl')
            prediction = model.predict(ripe_data)
            print(prediction)
            
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
        
    def make_df(self,raw_data):
        """
        This method turn the raw_data to Pandas DataFrame.
        
        Args:
            raw_data(dict): raw data to be converted to DataFrame.
        
        Return:
            pandas DataFrame of the raw_data.    
        """    
        
        try:
            df = pd.DataFrame(raw_data)
            df.columns =  ['gender','race/ethnicity','parental level of education','lunch','test preparation course','reading score','writing score']
            logging.info('Raw Data converted to DataFrame.')
            return df
        
        except Exception as e:
            raise CustomException(e,sys)