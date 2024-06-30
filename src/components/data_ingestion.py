import os
import warnings
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.components.data_transformation import dataTransformation
from src.components.model_training import modelTraining

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

@dataclass
class dataIngestionConfig():
    artifactsPath : str = os.path.join(os.getcwd(),"Artifacts")
    XtrainPath : str = os.path.join(artifactsPath,"X_train.csv")
    YtrainPath : str = os.path.join(artifactsPath,"Y_train.csv")
    XtestPath : str = os.path.join(artifactsPath,"X_test.csv")
    YtestPath : str = os.path.join(artifactsPath,"Y_test.csv")

class dataIngestion():
    def __init__(self):
        self.artifactsPaths = dataIngestionConfig()

    def initiateDataIngestion(self):
        try:
            logger.info("starting the Data Ingestion")
            df = pd.read_csv("src/NOTEBOOKS/data/stud.csv")
            X = df.drop("math_score",axis=1)
            Y = df['math_score']
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

            logger.info("Ingestion ended .... HURRAY")

            os.makedirs(self.artifactsPaths.artifactsPath ,exist_ok=True)

            logger.info("Saving the Data into Artifacts Folder")

            X_train.to_csv(self.artifactsPaths.XtrainPath, index = False ,header=True)
            Y_train.to_csv(self.artifactsPaths.YtrainPath, index = False ,header=True)
            X_test.to_csv(self.artifactsPaths.XtestPath, index = False ,header=True)
            Y_test.to_csv(self.artifactsPaths.YtestPath, index = False ,header=True)

            logger.info("Saving of Data Finished")

            return (
                self.artifactsPaths.XtrainPath,
                self.artifactsPaths.YtrainPath,
                self.artifactsPaths.XtestPath,
                self.artifactsPaths.YtestPath
                )
            
        except Exception as e:
            raise CustomException(e)
        



if __name__ == "__main__":
    data_ingestion = dataIngestion()
    X,Y,x,y = data_ingestion.initiateDataIngestion()
    data_transformation = dataTransformation()
    _,X_train_transformed,X_test_transformed=data_transformation.initiateDataTransformation(X_train=X,X_test=x)
    model_training = modelTraining()
    model_training.initiateModelTraining(X_train=X_train_transformed,Y_train=Y,X_test=X_test_transformed,Y_test=y)


