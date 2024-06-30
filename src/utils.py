import pickle
import sys
import os

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        with open(file_path,"wb") as file:
            pickle.dump(obj=obj,file=file)
    except Exception as e:
        raise CustomException("File not loaded properly")