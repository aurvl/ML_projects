import pandas as pd
import pickle
import joblib
from sklearn.metrics import r2_score
from pipeline import preprocessor
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

data = pd.read_csv('data/data_eval.csv')

target = data['salary_in_usd']
features = data.drop(columns=['salary_in_usd'])

# Loading utils
with open('modules/preprocessors.pkl', 'rb') as f:
    pipelines = pickle.load(f)

models = joblib.load('modules/models.pkl')
lr, rf = models
lr_model, lr_features = lr
rf_model, rf_params = rf

# Preprocessing
valid = preprocessor(features, False)
print('\n' * 2)

# Linear regression
print('================== Linear Regression ==================')
print('Variables Used :', lr_features)
lr_prediction = lr_model.predict(valid[lr_features])
r2 = r2_score(target, lr_prediction)
print(f"R2 OUT SAMPLE: {r2:.3f}")

print('\n' * 2)

# Random Forest
print('================== Random Forest ==================')
print('Paramètres du model :', rf_params)
rf_prediction = rf_model.predict(valid)
r2 = r2_score(target, rf_prediction)
print(f"R2 OUT SAMPLE: {r2:.3f}")