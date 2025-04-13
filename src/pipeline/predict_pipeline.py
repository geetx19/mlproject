import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
from pathlib import Path

class PredictPipeline:
    def __init__(self):
        # Get the TRUE project root (ml1 directory)
        self.root_dir = Path(__file__).resolve().parent.parent.parent
    
    def predict(self, features):
        try:
            # Path to artifacts - directly in project root
            artifacts_dir = os.path.join(self.root_dir, "artifacts")
            model_path = os.path.join(artifacts_dir, "model.pkl")
            preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
            
            # Debug prints
            print(f"TRUE Project root: {self.root_dir}")
            print(f"Artifacts directory contents: {os.listdir(artifacts_dir)}")
            
            # Load objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            print(f"FULL ERROR DETAILS: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

