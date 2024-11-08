# House Price Prediction 🏠

## Description
This project applies machine learning techniques to predict house prices based on a dataset containing various features related to housing and area demographics. The goal is to build a predictive model using linear regression and evaluate its performance in terms of how well it can estimate house prices. Additionally, we will explore the dataset through data visualization and analysis.

## Goals
- **Data Exploration**: Analyze the dataset to understand the distribution of house prices and the relationships between different variables (e.g., area income, number of rooms).
- **Data Visualization**: Create graphs such as histograms, scatter plots, and correlation heatmaps to visualize key insights.
- **Model Building**: Build a machine learning model (linear regression) to predict house prices.
- **Model Evaluation**: Evaluate the model's performance using metrics such as the R² score and compare predicted vs. actual house prices.

## Dataset
The dataset used for this analysis contains 5000 records with the following features:
- **Avg. Area Income**: Average income of residents in the area.
- **Avg. Area House Age**: Average age of the houses in the area.
- **Avg. Area Number of Rooms**: Average number of rooms in the houses.
- **Avg. Area Number of Bedrooms**: Average number of bedrooms in the houses.
- **Area Population**: Population of the area.
- **Price**: Price of the house (target variable).

## Tools
The analysis and modeling were performed using Python and the following libraries:
- **pandas**: For data manipulation and exploration.
- **seaborn**: For data visualization.
- **matplotlib**: For plotting graphs and visualizations.
- **scikit-learn**: For machine learning tasks such as scaling data, splitting datasets, and building the linear regression model.

## Key Functions Used
- `pd.read_csv()`: Load the dataset.
- `sns.displot()`: Visualize the distribution of house prices.
- `sns.scatterplot()`: Plot the relationships between variables.
- `sns.heatmap()`: Create a correlation matrix to show relationships between variables.
- `train_test_split()`: Split the dataset into training and testing sets.
- `LinearRegression()`: Build the linear regression model.
- `r2_score()`: Evaluate the model’s performance using the R² metric.

For further details, please refer to the Jupyter Notebook provided in this repository ([`housing.ipynb`](/Housing/housing.ipynb)).
