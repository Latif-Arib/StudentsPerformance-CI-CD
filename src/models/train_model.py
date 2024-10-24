import sys
import os
from src.utils.exceptions import CustomException
from src.utils.logger import logging
from src.utils.utils import save_model_or_data, load_model_and_data, gridsearch
from dataclasses import dataclass

# Model building libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

@dataclass
class TrainConfig:
    save_model_path: str = os.path.join('models','model.pkl')
    

class TrainModel:
    def __init__(self):
        self.train_config = TrainConfig()
        
    
    def train(self,X_train,X_test, y_train, y_test):
        """
        This method train the models.
        
        Args:
            X_train: array of independent features for training.
            X_test: array of independent features for testing.
            y_train: array of training dependent variable
            y_test: array of testing dependent variable
            
        output:
            r2 score of best model.    
        """       
        try:
            
            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "GradientBoostingRegressor":GradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            param_grid = {
            "linear_regression": {
                "fit_intercept": [True, False]
            },
            "lasso": {
                "alpha": [0.01, 0.1, 1, 10],
                "fit_intercept": [True, False],
                "max_iter": [1000, 5000, 10000]
            },
            "ridge": {
                "alpha": [0.01, 0.1, 1, 10],
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
                "tol": [1e-3, 1e-4, 1e-5]
            },
            "k_neighbors_regressor": {
                "n_neighbors": [3, 5, 10, 20],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],
                "n_jobs": [ 1,-1]
            },
            "decision_tree": {
                "criterion": ["squared_error", "friedman_mse","absolute_error", "poisson"],
                "splitter": ["best", "random"],
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],      
                "max_features": ["sqrt", "log2"]
            },
            "random_forest_regressor": {
                "n_estimators": [8,16,32,64,128,256],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False]
            },
            "gb_regressor": {
                "loss":['squared_error', 'absolute_error', 'huber', 'quantile'],
                "n_estimators": [8,16,32,64,128,256],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 6, 9, 12],
                "subsample": [0.6, 0.8, 1.0],
                "alpha": [0.1, 0.5, 0.9]
            },
            "xgb_regressor": {
                "n_estimators": [8,16,32,64,128,256],
                "eta": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 6, 9, 12],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.5, 1]
            },
            "catboost_regressor": {
                "iterations": [30, 50, 100],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
                "bagging_temperature": [0, 1, 2, 3]
            },
            "ada_boost_regressor": {
                "n_estimators": [8,16,32,64,128,256],
                "learning_rate": [0.01, 0.1, 0.5, 1],
                "loss": ["linear", "square", "exponential"],
                "random_state": [42]
            }
            }
            
            logging.info('Models and Parameters are being passed to gridsearchcv')
            
            best_model = gridsearch(
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                models,
                                param_grid,
                                self.train_config.save_model_path
                                )
            
            logging.info(f'The model is {best_model}')
            
            return best_model 
            
        except Exception as e:
            raise CustomException(e, sys)