import pickle
import warnings
import pandas as pd

from src.logger import logger
from src.exception import CustomException

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

class predictData():
    def __init__(self):
        pass

    def predict(self,X):
        try:
            logger.info("loading preprocessor pickle file")

            preprocessor = pickle.load(open("/Users/sanketsaxena/Desktop/studentPerformance/Artifacts/preprocessor.pkl",'rb'))

            logger.info("Loaded the preprocessor pickle file")
            logger.info("Fitting preprocessor on input X")
            X = pd.DataFrame(X).T
            X.columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course","reading_score","writing_score"]
            X = preprocessor.transform(X)

            logger.info("Fitted the preprocessor on X")
            logger.info("Loading Model pickle file")

            model = pickle.load(open("/Users/sanketsaxena/Desktop/studentPerformance/Artifacts/model.pkl",'rb'))

            logger.info("Loaded the model pickle file")
            logger.info("Predicting using Model")

            mathScore = model.predict(X)

            logger.info(f"Prediction done. Prediction is - {mathScore[0]:.2f}")

            return f"{mathScore[0]:.2f}"
        
        except Exception as e:
            raise CustomException("Something is wrong in prediction pipeline")
        
