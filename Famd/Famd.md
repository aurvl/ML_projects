# [Factorial Analysis of Mixed Data (FAMD)](https://aurvl.github.io/NotebookSites/famd/Famd.html) 🔢

#### Structure of the Famd directory
```
Famd/
  ├── Famd.md : Documentation on the project
  └── famd.R : R script where analysis, visualisations and modelisations are realised.
  └── FamdNotebook.qmd : A Notebook version of the project.
```
Notebook link : [https://aurvl.github.io/NotebookSites/famd/Famd.html](https://aurvl.github.io/NotebookSites/famd/Famd.html)

## Project Description

This project focuses on applying **Factorial Analysis of Mixed Data (FAMD)** to the "adult" dataset from the UCI Machine Learning Repository. FAMD is a dimension reduction technique that allows for the simultaneous analysis of both quantitative and qualitative variables. The goal is to understand the relationships between variables and individuals by reducing the dataset's dimensionality while preserving as much of the variance as possible.

## Goals

- **Perform Dimensionality Reduction**: Utilize FAMD to reduce the number of dimensions in the dataset while retaining the most significant information.
- **Visualize the Data**: Create various visualizations to interpret the contributions of variables to the principal dimensions and observe the distribution of individuals.
- **Understand Variable Contributions**: Identify which variables (both quantitative and qualitative) most significantly impact each dimension.
- **Explore Relationships**: Examine the correlations between different variables and how they relate to the principal dimensions.

## Dataset

The dataset used in this analysis is the "adult" dataset from the UCI Machine Learning Repository ([dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)). This dataset contains socio-economic information about individuals and is commonly used for tasks such as income classification. The dataset includes the following columns:

- **age**: Age of the individual
- **workclass**: Type of work (categorical)
- **fnlwgt**: Final weight (numeric)
- **education_num**: Numeric representation of education (numeric)
- **marital_status**: Marital status (categorical)
- **occupation**: Type of occupation (categorical)
- **relationship**: Relationship status (categorical)
- **race**: Race (categorical)
- **sex**: Sex (categorical)
- **capital_gain**: Capital gain (numeric)
- **capital_loss**: Capital loss (numeric)
- **hours_per_week**: Hours worked per week (numeric)
- **native_country**: Country of origin (categorical)
- **income**: Income classification (categorical)

## Tools

The analysis was performed using R with the following tools:

- **`FactoMineR`**: A package for multivariate analysis, which provides functions for performing FAMD.
- **`factoextra`**: A package for visualizing the results of multivariate analyses, including FAMD.
- **`dplyr`**: A package for data manipulation, used here to preprocess and clean the dataset.

### Key Functions Used

- `FAMD()`: Executes the Factorial Analysis of Mixed Data.
- `get_eigenvalue()`: Extracts eigenvalues to understand the importance of each dimension.
- `fviz_screeplot()`: Plots the explained variance to help decide the number of dimensions to retain.
- `get_famd_var()`: Extracts results related to variable loadings and contributions.
- `fviz_famd_var()`: Visualizes the contributions of variables to the principal dimensions.
- `get_famd_ind()`: Provides coordinates of individuals in the reduced dimensional space.

For further details, please refer to the R script & the Notebook provided in this directory ([`famd.R`](/Famd/famd.R)).
