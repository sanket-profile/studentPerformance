import sys
import os
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

@dataclass
class modelTrainingConfig():
    modelPicklePath : str = os.path.join(os.getcwd(),"Artifacts","model.pkl")

class modelTraining():
    def __init__(self):
        self.modelTrainingConfig = modelTrainingConfig()

    def initiateModelTraining(self,X_train,Y_train,X_test,Y_test):
        try:
            logger.info("Loading the data into Dataframes")

            X_train = pd.read_csv(X_train)
            Y_train = pd.read_csv(Y_train)
            X_test = pd.read_csv(X_test)
            Y_test = pd.read_csv(Y_test)

            logger.info("Loading of data finished")

            models = {
                        'Linear Regression': (LinearRegression(), {}),
                        'Support Vector Machine': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}),
                        'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100, 150]}),
                        'Decision Tree': (DecisionTreeRegressor(), {'max_depth': [None, 10, 20]})
                    }
            
            bestModel  = None
            r2prev = -1

            logger.info("Starting search for best model")

            for model_name, (model, param_grid) in models.items():
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
                grid_search.fit(X_train, Y_train)
                
                # Get the best model
                best_model = grid_search.best_estimator_
                
                # Predict on the test set
                y_pred = best_model.predict(X_test)
                
                # Calculate R^2 score
                r2 = r2_score(Y_test, y_pred)
                if(r2 > r2prev):
                    bestModel = best_model
                    r2prev = r2
                
            logger.info(f"Best model is - {bestModel}")
            logger.info("Saving bestModel into Artifact folder")

            save_object(self.modelTrainingConfig.modelPicklePath,bestModel)

            logger.info("bestModel saved into Artifact folder")

            return (
                self.modelTrainingConfig.modelPicklePath
            )
        except Exception as e:
            raise CustomException("Something wrong in modelTraining File")