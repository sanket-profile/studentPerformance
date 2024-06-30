import os
import sys

import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

@dataclass
class dataTransformationConfig():
    preprocessorPath : str = os.path.join(os.getcwd(),"Artifacts","preprocessor.pkl")
    X_train_transformed : str = os.path.join(os.getcwd(),"Artifacts","X_train_transformed.csv")
    X_test_transformed : str = os.path.join(os.getcwd(),"Artifacts","X_test_transformed.csv")

class dataTransformation():
    def __init__(self):
        self.dataTransformationConfig = dataTransformationConfig()

    def initiateDataTransformation(self,X_train,X_test):
        try:
            logger.info("Reading the Data Provided by Ingestion")

            X_train = pd.read_csv(X_train)
            X_test = pd.read_csv(X_test)

            logger.info("Reading of Data Provided by Ingestion COMPLETED")
            logger.info("Checking numerical and categorical features in data")

            num_features = [features for features in X_train.columns if(X_train[features].dtype != 'O')]
            cat_features = [features for features in X_train.columns if(X_train[features].dtype == 'O')]

            logger.info(f"NUMERICAL FEATURES ARE - {num_features}. And CATEGORICAL FEATURES ARE - {cat_features}")

            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder()

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_features),
                    ('cat', categorical_transformer, cat_features)
                ]
                )
            logger.info("Transformation beginning")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            X_train_transformed = pd.DataFrame(X_train_transformed)
            X_test_transformed = pd.DataFrame(X_test_transformed)

            logger.info("Transformation completed")
            logger.info("Creating the pickle file for preprocessor")

            #pickle.dump(preprocessor,self.dataTransformationConfig.preprocessorPath,'wb')
            save_object(self.dataTransformationConfig.preprocessorPath,preprocessor)

            logger.info("Dumped the pickle file in Artifacts Folder")
            logger.info("Saving X_train_transformed X_test_transformed into Artifacts Folder")

            X_train_transformed.to_csv(self.dataTransformationConfig.X_train_transformed,index=False,header=True)
            X_test_transformed.to_csv(self.dataTransformationConfig.X_test_transformed,index=False,header=True)

            logger.info("Both CSV files Saved")

            return(
                self.dataTransformationConfig.preprocessorPath,
                self.dataTransformationConfig.X_train_transformed,
                self.dataTransformationConfig.X_test_transformed
            )
        
        except Exception as e:
            raise CustomException("Something is not wrong")