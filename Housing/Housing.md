# House Prices Prediction

### Project Description

This project focuses on predicting house prices using a dataset with various features related to housing and demographic characteristics. The goal is to build and evaluate machine learning models to determine how well they can estimate house prices based on these attributes. The analysis includes data exploration, visualization, and model implementation to better understand the relationships between variables and their impact on housing prices.

### Objectives
- **Data Analysis**: Explore the dataset to understand the distribution of features such as income, house age, and population, and how they relate to house prices.
- **Data Visualization**: Use visualizations (e.g., histograms, scatter plots, correlation matrices) to uncover patterns and correlations in the data.
- **Model Implementation**: Implement and compare machine learning models (e.g., Linear Regression, K-Nearest Neighbors) for price prediction.
- **Model Evaluation**: Assess model performance using metrics like MSE and R², providing insights into prediction accuracy and model fit.

### Dataset
The dataset (housing.csv) includes 5000 records, each representing a house with the following features:
- **Avg. Area Income**: Average income of area residents.
- **Avg. Area House Age**: Average age of houses in the area.
- **Avg. Area Number of Rooms**: Average number of rooms per house.
- **Avg. Area Number of Bedrooms**: Average number of bedrooms per house.
- **Area Population**: Population of the area.
- **Price**: Sale price of the house (target variable).

### Tools
The analysis and modeling were performed using Python and the following libraries:
- **pandas**: For data manipulation and exploration.
- **seaborn**: For data visualization.
- **matplotlib**: For plotting graphs and visualizations.
- **scikit-learn**: For machine learning tasks such as scaling data, handling missing values, splitting datasets, and building models.

### Key Functions Used
- `pd.read_csv()`: Load the dataset.
- `sns.histplot()`: Visualize the distribution of house prices and other features.
- `sns.heatmap()`: Create a correlation matrix to show relationships between variables.
- `train_test_split()`: Split the dataset into training and testing sets for model evaluation.
- `LinearRegression()`, `KNeighborsRegressor()`: Build and evaluate regression models.
- `cross_val_score()`: Perform cross-validation to assess model performance.
- `mean_squared_error()` and `r2_score()`: Evaluate model accuracy using MSE and R² metrics.

This project aims to produce reliable and interpretable models to aid stakeholders in making informed decisions about real estate pricing and investments.

For further details, please refer to the Jupyter Notebook provided in this repository ([housing.ipynb](https://github.com/aurvl/ML_projects/blob/main/Housing/housing.ipynb))
