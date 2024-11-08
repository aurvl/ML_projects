# Factorial Analysis of Mixed Data (FAMD)
# This study focus on the dimension reduction method named FAMD

# Clearing the workspace
rm(list = ls())

# Installing & Loading necessary packages

# install.packages(c("FactoMineR", "factoextra", "dplyr", "ggplot2"))

library(FactoMineR)
library(factoextra)
library(dplyr)
library(ggplot2)


# The dataset ==================================================================
# We will use the "adult" dataset from the UCI (University of California, Irvine) archive,
# which is commonly used as a sample dataset in machine learning and statistics contexts.
# This dataset is often used for tasks such as income classification or prediction.

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

column_names <- c(
   "age", "workclass", "fnlwgt", "education", "education_num","marital_status", 
   "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
   "hours_per_week", "native_country", "income"
) # we are going to rename the variables

data_adult <- read.table(url, header = FALSE, sep = ",", col.names = column_names)
rm(column_names)
head(data_adult)

# We are going to create a subset dataframe named "df" of the overall dataset
df <- data_adult[, 1:15]
glimpse(df)

# Description:
# This dataset contains demographic and occupational information on 30,162 individuals, 
# divided into 14 variables. Data include age, work class (public/private), education 
# level, marital status, occupation and hours worked per week. Economic variables such 
# as capital gain/loss and country of origin are also included.


# Data cleaning ================================================================
# Removing the redundant "education" variable
# "education" is already numerically encoded under education-num, so we will remove it
df <- df %>%
  select(-education)

# Converting categorical variables to factors
df$workclass <- as.factor(df$workclass)
df$marital_status <- as.factor(df$marital_status)
df$occupation <- as.factor(df$occupation)
df$relationship <- as.factor(df$relationship)
df$race <- as.factor(df$race)
df$sex <- factor(df$sex)
df$native_country <- as.factor(df$native_country)
df$income <- factor(df$income)

# Deleting "?" values in "workclass"
# Some lines in 'workclass' variable contain that symbole '?'
df <- df[!(grepl("\\?", df$workclass) | grepl("\\?", df$native_country) | grepl("\\?", df$occupation)), ]

# Obtained dataset
head(df[, 1:14], 5)
summary(df)
glimpse(df)
# In the df dataset, we have:
# 8 categorical variables (workclass, marital_status, occupation, relationship, race, sex, native_country, income)
# 6 quantitative variables (age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week).



# FAMD Implementation ==========================================================
# FAMD (Factor Analysis of Mixed Data) is a dimensionality reduction technique designed 
# for datasets with both quantitative and qualitative variables. It combines elements 
# of PCA (for quantitative variables) and MCA (for qualitative variables) to capture 
# and preserve the information structure from both types in a lower-dimensional space.

## FAMD Execution ====
res.famd <- FAMD(df, graph = FALSE)
print(res.famd)

# Eigenvalues (explain variance by each dimension) and Screeplot (Plot of eigenvalues)
get_eigenvalue(res.famd)
fviz_screeplot(res.famd)


## Results ====
# Extracting the results from the loadings of all variables (quantitatives & qualitatives).
var <- get_famd_var(res.famd)
var
# "$coord" provides the coordinates of the loadings of each quantitative variable related to each dimension
# Contribution of the variables to the new factorial space
head(var$coord)

# Quality of representation of quantitative variables in each dimension
# Higher cos2 implies good variable representation in the dimension).
# through a trigonometric representation
head(var$cos2)

## Ploting results ====
# Graphical representation of quantitative and qualitative variables in the factorial plan
fviz_famd_var(res.famd, "quanti", repel = TRUE, col.var = "black") # Quanti variables
fviz_famd_var(res.famd, "quali", repel = TRUE, col.var = "red") # Quali variables

# Interpretation
# Quantitatives variables
# The graphic illustrates the importance and orientation of each numerical variable
# on the first two dimensions of the FAMD (Dim1 and Dim2). The arrows represent the 
# variables, with their lengths indicating their contribution to variance along each 
# dimension. For instance, education_num and age contribute primarily to Dim1, while 
# fnlwgt points in a different direction, reflecting a unique contribution. The 
# closeness of capital_gain and capital_loss suggests a potential relationship.
# 
# Qualitatives variables 
# For categorical variables, this graph displays the distribution of category levels
# across the same dimensions. Categories that are closer together share similar 
# profiles in the dataset. For example, Male and Husband are close, indicating an 
# association. Similarly, the `>50K` income category is near certain professions like 
# `Exec-managerial` and `Self-emp-inc`, suggesting a trend toward higher income in 
# these roles. The positions of countries and marital statuses also reveal interesting groupings.


## Contributions ====
# "contrib" provides the coordinates and the contribution (in terms of magnitude) of the loadings of
# each quantitative variable w.r.t each dimension
head(var$contrib)

# Contributions with colors by contribution level
fviz_famd_var(res.famd, "quanti", col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)
fviz_famd_var(res.famd, "quali", col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

# Contributions of variables to Dimension 1
fviz_contrib(res.famd, "var", axes = 1)
# Contributions of variables to Dimension 2
fviz_contrib(res.famd, "var", axes = 2)
# From the plots, we can deduce that:
# 1 - The variables which contribute the most to the 1st dimension are: relationship, marital_status & income.
# 2 - The variables which contribute the most to the 2nd dimension are: occupation, relationship & gender.


# Position of all variables related to the two dimensions
fviz_famd_var(res.famd, repel = TRUE)
# Comment:
# Overall, the plot suggests that occupation, relationship, and marital_status contribute
# significantly to distinguishing profiles in Dim2, while income and hours_per_week 
# align more with Dim1.


# Coordinates of the individuals projected into the factorial space following the FAMD analysis
ind <- get_famd_ind(res.famd)
head(ind$coord)
