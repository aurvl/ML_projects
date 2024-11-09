# Family History with Overweight Classification models

## Installing and loading packages
rm(list = ls())

# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("tidyr")
# install.packages("readr")
# install.packages("data.table")
# install.packages("stringr")
# install.packages("corrplot")
# install.packages("caret")
# install.packages("xgboost")
# install.packages("randomForest")
# install.packages("xgboost")

library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(data.table)
library(stringr)
library(corrplot)
library(caret)
library(xgboost)
library(randomForest)
library(pROC)

# 1. Exploratory Data Analysis =================================================
# Loading the data
data <- read.csv("https://github.com/Eben2020-hp/Obesity/raw/main/Obesity.csv")
head(data[, 1:8]) # Preview of the 8 first variables of the dataset

## 1.1. Variables of the dataframe:
glimpse(data)

## 1.2. Quality of the data:
colSums(is.na(data)) > 0

## 1.3. Data visualization
p1 <- ggplot(data, aes(x = Age, y = Weight)) + geom_point(col = "green") + 
   labs(title = "Age vs Weight") +
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p1)

p2 <- ggplot(data, aes(x = Gender, y = Weight, fill = Gender)) + geom_boxplot() +  
   labs(title = "Weight Distribution by Gender") +
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p2)

p3 <- ggplot(data, aes(x = Weight, y = Height)) + geom_point(col = "#8576FF") +  
   labs(title = "Weight vs Height") +
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p3)

p4 <- ggplot(data, aes(x = Age, y = FCVC)) + geom_point(col = "#4793AF") +  
   labs(title = "Age vs Fruits consumption") +
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p4)

p5 <- ggplot(data, aes(x = Weight, y = FCVC)) + geom_point(col = "#FF76CE") + 
   labs(title = "Weight vs Fruits consumption") +
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p5)

p6 <- ggplot(data, aes(x = Age, y = CH2O)) + geom_point(col = "#7ABA78") +  
   labs(title = "Age vs Water consumption") + 
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p6)

p7 <- ggplot(data, aes(x = SMOKE, y = Weight, fill = SMOKE)) + stat_summary(fun = "mean", geom = "col", color = "white") +
   labs(title = "Weight vs Smoke") +
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p7)

p8 <- ggplot(data, aes(x = CALC, y = Weight, fill = CALC)) + geom_boxplot() +  
   labs(title = "Weight distribution by CALC") + 
   theme(axis.text = element_text(size = 4), axis.title = element_text(size = 6))
print(p8)


data %>% ggplot(mapping = aes(x = Age, y = Weight, color = NObeyesdad)) +
   geom_point() + theme_bw() + theme(legend.position = "bottom") +
   labs(title = "Age vs Weight", subtitle = "By Obesity level") +
   theme_gray()

ggplot(data, aes(x = Weight, y = family_history_with_overweight)) + geom_point(col = "black") + 
   labs(title = "Weight vs family history with overweight") +
   theme(axis.text = element_text(size = 8), axis.title = element_text(size = 6))


## 1.4. Correlation of features:
X <- data %>%
select(Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE) # We select only numerical variables

cor_mat <- cor(X) # Computing the correlation matrix of X
corrplot(cor_mat, method = "color",col = COL2('PuOr'), tl.col ="purple", tl.cex = 0.5)



# 2. Data preprocessing ========================================================
## Numerical variables
X1 <- data %>%
select(Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE) # X1 contains numerical variables
head(X1)


## Categorical variables
X2 <- data.frame() # We create an empty df for storing categorical variables

cat_cols <- c()
for (col in names(data)) {
   if (is.character(data[[col]])) {
      cat_cols <- c(cat_cols, col) # loop for finding the categorical variables
   }
}

print(cat_cols) # Only CAEC, CALC, MTRANS, and NObeyesdad have more than 2 levels

X2 <- data %>%
select(CAEC, CALC, MTRANS, NObeyesdad) # We store them in X2

# Converting them into categoricals (factor)
X2$CAEC <- factor(X2$CAEC)
X2$CALC <- factor(X2$CALC)
X2$MTRANS <- factor(X2$MTRANS)
X2$NObeyesdad <- factor(X2$NObeyesdad)

# And we create dummies for each categorical variable
X2 <- model.matrix(~ . - 1, data = X2)
X2 <- as.data.frame(X2)
head(X2[, 1:6]) # Preview the 6 first columns


# Binary variables
# Renaming the target variable to simplify our task
data <- data %>% 
rename(fhwo = family_history_with_overweight)

X3 <- data %>%
select(fhwo, Gender, FAVC, SMOKE, SCC)

X3$Gender <- factor(X3$Gender)
X3$FAVC <- factor(X3$FAVC)
X3$SMOKE <- factor(X3$SMOKE)
X3$SCC <- factor(X3$SCC)

X3 <- X3 %>%
mutate(fhwo = ifelse(fhwo == "yes", 1, 0),
Gender = ifelse(Gender == "Female", 1, 0),
FAVC = ifelse(FAVC == "yes", 1, 0),
SMOKE = ifelse(SMOKE == "yes", 1, 0),
SCC = ifelse(SCC == "yes", 1, 0))
head(X3)


attach(data)
data_final <- cbind(X1, X3, X2) # Combining all df & the target variable to create the final one
head(data_final[, 1:14]) # preview the first observations of the 14 first columns of our final dataset
dim(data_final)

# Target variable
summary(data_final$fhwo) # Proportions in the target variable
# In average 81% of individuals have antecendent with overweight (`fhwo = 1`).

# Plotting the target variable
ggplot(data_final, aes(x = fhwo)) + geom_bar(stat = "count", fill = c("#F27BBD", "#10439F"), color = "white") +
   labs(title = "Distribution of family_history_with_overweight", y = "Number of people") +
   theme_minimal() + theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
   geom_text(aes(label = ..count.., y = ..count..), stat = "count", position = position_dodge(width = 0.9), 
             vjust = -0.25, size = 3, fontface = "bold", color = "black") +
   scale_fill_brewer(palette = "Set3") +
   theme_gray()


# 3. Data splitting ============================================================
## 3.1. Oversampled Dataset
remotes::install_github("cran/DMwR")
library(DMwR)

set.seed(123)
data_final$fhwo <- as.factor(data_final$fhwo) # We convert the target as factor for oversampling
ovs_data <- SMOTE(fhwo ~ ., data = data_final) # New oversampled dataset
attach(ovs_data)

# Observing our target var again:
ggplot(ovs_data, aes(x = fhwo)) + geom_bar(stat = "count", fill = c("#F27BBD", "#10439F"), color = "white") +
   labs(title = "Distribution of family_history_with_overweight", y = "Number of people") +
   theme_minimal() + theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
   geom_text(aes(label = ..count.., y = ..count..), stat = "count", position = position_dodge(width = 0.9), 
             vjust = -0.25, size = 3, fontface = "bold", color = "black") +
   scale_fill_brewer(palette = "Set3") +
   theme_gray()

## 3.2. Splitting into two sets
# 80% = training & 20%  = evaluation (test)
inTraining <- createDataPartition(ovs_data$fhwo, p = .80, list = FALSE)
train_set <- ovs_data[inTraining, ]
test_set  <- ovs_data[- inTraining, ]

dim(train_set)
table(train_set$fhwo)


# 4. Implementing machine learning algorithms ==================================

## 4.1. Logistic regression ====
### Training
set.seed(123)
model_logit <- train(fhwo ~ ., data = train_set, method = "glm", family = "binomial")
summary(model_logit)

### Predictions
reg_class <- predict(model_logit, test_set)
final_dataset <- cbind(Real = test_set$fhwo, Predicted = reg_class)
final_dataset <- as.data.frame(final_dataset)
head(final_dataset) # 1 = fwho : "no" and 2 = fwho : "yes"

### Performance Metrics
conf_matrix1 <- confusionMatrix(reg_class, test_set$fhwo)
print(conf_matrix1)

f1_score1 <- 2 * (precision1 * recall1) / (precision1 + recall1)
print(f1_score1)

library(pROC)
auc1 <- roc(test_set$fhwo, as.numeric(reg_class))
print(auc1$auc)


## 4.2. Random Forest classifier ====
target_train <- train_set$fhwo
features_train <- train_set[-9]

target_test <- test_set$fhwo
features_test <- test_set[-9]

### Training the model
set.seed(123)
classifier_RF = randomForest(x = features_train, y = target_train, ntree = 500) 
classifier_RF

plot(classifier_RF, main = 'RF Classifier')
legend("topright", legend = c("Global OOB error", "Error for class 0", "Error for class 1"),
       col = c("black", "red", "green"), lty = 1, cex = 0.8)

### Feaures Importance
importance(classifier_RF)
varImpPlot(classifier_RF, n.var = 15, main = "Random Forest Feature Importance", col = "#9365DB", pch = 16)
partialPlot(classifier_RF, train_set, Weight)


### Performance metrics
RF_class = predict(classifier_RF, newdata = features_test)
conf_matrix2 <- confusionMatrix(RF_class, target_test)
conf_matrix2

accuracy2 <- conf_matrix2$overall['Accuracy']
accuracy2

precision2 <- conf_matrix2$byClass["Pos Pred Value"]
recall2 <- conf_matrix2$byClass["Sensitivity"]
f1_score2 <- (2 * (precision2 * recall2)) / (precision2 + recall2)
f1_score2

auc2 <- roc(target_test, as.numeric(RF_class))
auc2 <- auc2$auc
auc2


## 4.3. XGBoost Classifier ====
### Training
# Convert training and test features to matrix
ft_train_mtx <- data.matrix(features_train)
ft_test_mtx <- data.matrix(features_test)

# Create DMatrix objects for training and testing sets
xgb_train <- xgb.DMatrix(data = ft_train_mtx, label = target_train)
xgb_test <- xgb.DMatrix(data = ft_test_mtx, label = target_test)

# Model training
xgb_model <- xgb.train(data = xgb_train, max.depth = 3, nrounds = 50)
summary(xgb_model)


# feature importances from the model
importance_matrix <- xgb.importance(feature_names = colnames(ft_train_mtx), model = xgb_model)
importance_matrix
xgb.plot.importance(importance_matrix, left_margin = 8, main = 'XGBoost Feature Importances', col='blue')


### Evaluation
# test set predictions
xgb_class <- predict(xgb_model, ft_test_mtx)
xgb_class_df <- data.frame(Predicted = ifelse(xgb_class >= 0.5, 1, 0), Real = target_test)
head(xgb_class_df)

conf_matrix3 <- confusionMatrix(as.factor(xgb_class_df$Predicted), as.factor(xgb_class_df$Real))
conf_matrix3

accuracy3 <- conf_matrix3$overall['Accuracy']
accuracy3

precision3 <- conf_matrix3$byClass["Pos Pred Value"]
recall3 <- conf_matrix3$byClass["Sensitivity"]
f1_score3 <- 2 * (precision3 * recall3) / (precision3 + recall3)
f1_score3

auc3 <- roc(as.numeric(xgb_class_df$Real), as.numeric(xgb_class))
auc3 <- auc3$auc
auc3


# 5. Results ===================================================================
# In conclusion, the RForest model stands out as the optimal choice among the 
# three evaluated models. Its outstanding overall performance, particularly in 
# terms of accuracy, precision, recall, and AUC-ROC, make it a reliable tool for 
# classifying this dataset. Its high F1-score also confirms its ability to 
# maintain a balance between precision and recall, which is crucial for effective 
# classification tasks. Thus, for this specific scenario, the Random Forest model 
# appears to be the most suitable solution for predicting the presence of a family 
# history of overweight.
