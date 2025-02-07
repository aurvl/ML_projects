import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from itertools import combinations
from sklearn.metrics import r2_score
from pipeline import preprocessor


target = 'salary_in_usd'
data = pd.read_csv('data.csv')

train, test = train_test_split(data, test_size=0.2, random_state=42)
train = train.reset_index()
test = test.reset_index()

X_train = preprocessor(train, True)
y_train = train[target]
X_test = preprocessor(test, False)
y_test = test[target]


# =====================================================================
# Linear Regression features selection

lasso_cv = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
selected_features = np.where(lasso_cv.coef_ != 0)[0]
selected_feature_names = X_train.columns[selected_features]
print("Variables sélectionnées :", selected_feature_names.tolist())
features = selected_feature_names.tolist()

all = []
elements = list(features)
combinations_4 = list(combinations(elements, 4))
for combi in combinations_4:
    L=list(combi)
    all.append(L)

print("Nombre de combianisons à tester :", len(all))
model = LinearRegression()
Scores_mean=[]
Scores_std=[]
Accuracy_test=[]
Nb=[]
Features=[]

for nb, combi in enumerate(all):
    print('\n=====================================================================')
    print(nb,combi)
    # cross validation
    scores = cross_val_score(model, X_train[combi], y_train, cv=5, scoring='r2')
    print("Validation scores (Classification): ", scores)
    print("Mean validation score: ", scores.mean())
    
    # prédiction "out-sample"
    model.fit(X_train[combi],y_train)
    y_test_prediction = model.predict(X_test[combi])
    accuracy_test = r2_score(y_test, y_test_prediction)
        
    if scores.mean()>0.3:
        Nb.append(nb)
        Scores_mean.append(scores.mean())
        Scores_std.append(scores.std())
        Accuracy_test.append(accuracy_test)
        Features.append(' - '.join(combi))


df_result=pd.DataFrame({'Numero':Nb,'Score_out_sample':Accuracy_test,
                        'Moy_score_cv':Scores_mean,'Std_score_cv':Scores_std,
                       'Features':Features})
b = df_result.sort_values(by='Score_out_sample', ascending=False).head(10)
b.to_csv('res.csv')

# =====================================================================
# Random Forest parameters
rf_tester = RandomForestRegressor(random_state=42)
rf_tester.fit(X_train, y_train)

params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(estimator= rf_tester, param_grid= params, cv= 5, n_jobs= -1, verbose= 2)
grid_search.fit(X_train, y_train)

print(f"Meilleure r² : {grid_search.best_score_}")
rf_params = grid_search.best_params_
print(rf_params)