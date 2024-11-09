# Classification Models for Family History of Overweight 🧮

## Description

This project explores the use of various classification models to predict whether an individual has a family history of overweight. Three different machine learning algorithms are implemented and evaluated: Logistic Regression, Random Forest Classifier, and XGBoost Classifier. The primary goal is to compare the performance of these models in terms of accuracy, precision, recall, F1-score, and AUC-ROC.

## Goals

- **Model Training:** Implement and train Logistic Regression, Random Forest, and XGBoost models on the dataset.
- **Performance Evaluation:** Assess each model’s performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
- **Feature Importance:** Determine the importance of various features in predicting the target variable.
- **Model Comparison:** Compare the effectiveness of different models to identify the best-performing classifier for the dataset.

## Dataset

The dataset used in this analysis ([source](https://github.com/Eben2020-hp/Obesity/raw/main/Obesity.csv)) contains various features related to individuals, including:

- `Weight`: Weight of the individual (numeric)
- `Age`: Age of the individual (numeric)
- `Gender`: Gender of the individual (categorical)
- `Height`: Height of the individual (numeric)
- `NObeyesdad`: Indicator of overweight status (categorical)
- `FAVC`: Consumption of high-calorie foods (categorical)
- `CAECSometimes`: Frequency of eating high-calorie foods (categorical)
- `CALCSometimes`: Frequency of consuming calories (categorical)
- `CALCno`: Frequency of not consuming calories (categorical)

## Tools

The analysis was performed using R with the following tools:

- **`glm()`**: For Logistic Regression model.
- **`randomForest()`**: For Random Forest Classifier model.
- **`xgboost()`**: For XGBoost Classifier model.
- **`pROC`**: For plotting AUC-ROC curves.
- **`caret`**: For confusion matrix and performance metrics.

## Key Functions Used

- `predict()`: To generate predictions from the trained models.
- `confusionMatrix()`: To compute the confusion matrix and derive performance metrics.
- `roc()`: To calculate and plot the AUC-ROC curve.
- `importance()`: To extract feature importance from the Random Forest model.
- `xgb.importance()`: To get feature importance from the XGBoost model.

## Results

### Logistic Regression
- **Accuracy:** 85.34%
- **Precision:** 81.67%
- **Recall:** 84.85%
- **F1-score:** 83.23%
- **AUC-ROC:** 85.28%

### Random Forest Classifier
- **Accuracy:** 94.43%
- **Precision:** 94.27%
- **Recall:** 92.64%
- **F1-score:** 93.45%
- **AUC-ROC:** 94.21%

### XGBoost Classifier
- **Accuracy:** 57.14%
- **Precision:** NaN (Not Applicable)
- **Recall:** 0
- **F1-score:** NaN (Not Applicable)
- **AUC-ROC:** 97.99%

## Conclusion

The Random Forest model demonstrates superior performance across all metrics, including accuracy, precision, recall, and F1-score. The XGBoost Classifier, while having a high AUC-ROC, shows poor performance in terms of accuracy and recall, making it less reliable for this dataset.

For detailed implementation and further analysis, refer to the Rmd scripts and results provided in this directory ([Overweight.Rmd](/Overweight/Overweight.Rmd) & download [Overweight.html](/Overweight/Overweight.html)).
