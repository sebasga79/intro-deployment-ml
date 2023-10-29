from joblib import load
from scipy.sparse import data 
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO

def get_model() -> Pipeline:
    model = load("model/model.pkl")
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = {key:[value] for key, value in class_model.model_dump().items()}
    data_frame = DataFrame(transition_dictionary)
    return data_frame