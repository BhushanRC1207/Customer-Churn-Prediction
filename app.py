import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline


def engg_features(X):
    X["BalanceSalaryRatio"] = X["Balance"] / X["EstimatedSalary"]
    X["TenureByAge"] = X["Tenure"] / X["Age"]
    X["CreditScoreGivenAge"] = X["CreditScore"] / X["Age"]
    X["HasBalance"] = np.where(X["Balance"] > 0, 1, 0)
    X["ActiveByAge"] = X["IsActiveMember"] * X["Age"]
    X['AgeCategory'] = pd.cut(X['Age'], bins=[0, 35, 55, np.inf], labels=['Young', 'MiddleAge', 'Senior'])
    return X

preprocess_pipeline = pickle.load(open("./models/pipeline.pkl", "rb"))
model_pipeline = pickle.load(open("./models/final_model_pipeline.pkl", "rb"))

model = Pipeline([
    ("preprocess", preprocess_pipeline),
    ("model", model_pipeline)
])

columns = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", 
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

input_data = pd.DataFrame([[619, "France", "Female", 42, 2, 0.00, 1, 1, 1, 101348.88]], columns=columns)

# print("Pipeline Steps:", preprocess_pipeline.steps)
# print("Pipeline Steps:", model_pipeline.steps)
print(model.predict(input_data))

