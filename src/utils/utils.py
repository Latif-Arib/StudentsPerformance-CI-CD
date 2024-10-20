import sys
import os
from typing import Union
from src.utils.exceptions import CustomException
from src.utils.logger import logging
import pickle
import dill
from sklearn.metrics import r2_score


        
def save_model_or_data(object_to_save:Union[dict,object], filepath:str) -> None:
    """
    This function Save the model and preprocessed data to the specified file path.
    
    Args:
        object_to_save(Union[dict,object]): The model or preprocessed data, either in object or dictionary forms.
        filepath (str): The file path, where the model or data will be saved.
        
    """
    try:
        with open(filepath,'wb') as f:
            dill.dump(object_to_save,f)
            
            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_model_and_data(filepath:str) -> object:
      """
      This functions loads the serialized model or data from given path.
      
      Args:
        filepath (str): the location of serialized object.
        
      Return:
        the serialized object of model or data.   
      """  
      
      try:
          with open(filepath, 'rb') as file:
              return dill.loads(file)
          
          
          
      except Exception as e:
          raise CustomException(e, sys)    