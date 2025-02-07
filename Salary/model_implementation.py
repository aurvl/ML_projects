import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from pipeline import preprocessor
import joblib

target = 'salary_in_usd'
data = pd.read_csv('data/training.csv')

# ===============================================================================================
# Preprocessing data
train, test = train_test_split(data, test_size=0.3, random_state=42)
train = train.reset_index()
test = test.reset_index()

X_train = preprocessor(train, True)
y_train = train[target]
X_test = preprocessor(test, False)
y_test = test[target]

# =====================================================================================================
# Random Forest Regressor
rf_params = {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 400}

rf_model = RandomForestRegressor(
    n_estimators= rf_params['n_estimators'],
    max_depth= rf_params['max_depth'],
    min_samples_split= rf_params['min_samples_split'],
    random_state= 42
)

rf_model.fit(X_train, y_train)
rf_predit = rf_model.predict(X_test)
r2 = r2_score(y_test, rf_predit)
print(f"R-SQUARED OUT: {r2:.2f}")
print('Best parametres :', rf_params)

rf = (rf_model, rf_params)

# ==============================================================================
# Linear Regressor
lr_features = ['company_location' , 'work_year experience_level_SE' , 'company_location job_groups_Scientist' , 'employment_type_FT job_groups_Engineer']

lr_model = LinearRegression()
lr_model.fit(X_train[lr_features], y_train)

score = cross_val_score(lr_model, X_train[lr_features], y_train, cv= 10, scoring='r2')
print(f"CV R2 scores : {score}")
print(f"R-SQUARED moyenne : {score.mean()}  |  std : {score.std()}")

lr_predit = lr_model.predict(X_test[lr_features])
r2 = r2_score(y_test, lr_predit)
print(f"R-squared: {r2:.2f}")
print("Variable utilisées : ", lr_features)

lr = (lr_model, lr_features)

# ===============================================================================
# Compilation
models = (lr, rf)
joblib.dump(models, 'modules/models.pkl')