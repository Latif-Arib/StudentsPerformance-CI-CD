import sys
import os
import pickle
import dill
import numpy as np 
from typing import Union
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


        
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
            
        logging.info('Model or data has been saved.')    
            
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
              return dill.load(file)
          
          logging.info('Model or data has been loaded.')
                    
      except Exception as e:
          raise CustomException(e, sys)    

def evaluate(true, predicted):
    """
    This method evaluates each model performance metrics( mae, mse , rmse and r-square)
    
    args:
    true: array of true values
    predicted: array of predicted values of the model.
    
    output:
    mae, rmse and r-square in the form of tuple (mae, rmse, r-square)
    
    """
    try:
        logging.info('Model is being evaluated.')
        
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r_square = r2_score(true, predicted)
        
        logging.info('Model evaluation is finished.')
        
        return mae, rmse, r_square
    
    except Exception as e:
        raise CustomException(e,sys)
    

def pick_best_model(models_report:dict, model_path:str) -> dict:
    """
    This function select the best model out of the passed models.
    
    Args:
        models_report (dict): a dict of models with the name and its accuracy.
        model_path (str): the path to save the trained model.
    Return:
        model with highest accuracy.
    """   
    try:
        logging.info('Searching for the best model.')
        
        models_name = list(models_report.keys())
        models =  [model['model'] for model in models_report.values()]
        accuracies  = [model['accuracy'] for model in models_report.values()]
        highest_accuracy_index = accuracies.index(max(accuracies))
                
        logging.info('Best model is found.')
        
        save_model_or_data(models[highest_accuracy_index],model_path)
        
        return {models_name[highest_accuracy_index]:accuracies[highest_accuracy_index]} 
     
    except Exception as e:
        raise CustomException(e,sys)    
    
      
def gridsearch(X_train, X_test, y_train, y_test, models:dict, params:dict,filepath:str) -> dict:
    """
    This function does the gridsearch.
    
    Args:
        X_train : X_train array.
        X_test : X_test array.
        y_train : y_train array.
        y_test : y_test array.
        models (dict) : dict of models name and definitions.
        params (dict) : dict of params for each model.
        
    Return:
        report (dict) : a dict of models evaluation.    

    """     
    try:
        logging.info('Preparing for Hyperparameter tunning.')
        
        models_report = {}
        
        for (name, model),param in zip(models.items(),params.values()):
            model = model
            param = param
            
            
            grid_search = GridSearchCV(model,param,cv=5)
            grid_search.fit(X_train,y_train)
            
            logging.info(f'Best parameters of {name} is found.')
            
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)
            logging.info(f'{name} is trained on best parameters.')
            
            y_test_pred = model.predict(X_test)
            
            _,_,accuracy = evaluate(y_test, y_test_pred)
            
            models_report[name] = {'model':model,'accuracy':accuracy}
            logging.info(f'{name} accuracy is stored.')
            
        best_model = pick_best_model(models_report,filepath)
            
        return best_model
        
    except Exception as e:
        raise CustomException(e,sys)