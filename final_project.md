# Coursera Final Project
#### Title: Practical Machine Learning Project - Quantified Self Movement Data Analysis Report
#### Author: Pierre Roux-Lafargue
 
## Code
 ``` {r}
 # Loading the required packages
library(caret)
library(rpart)
library(corrplot)
library(httr)
library(RCurl)
library(rattle)
```

 ``` {r}
# Getting the data
testing <- read.csv("https://raw.githubusercontent.com/pierrerl-py/Practical-Machine-Learning---Johns-Hopkins-University/master/pml-testing.csv")
training <- read.csv("https://raw.githubusercontent.com/pierrerl-py/Practical-Machine-Learning---Johns-Hopkins-University/master/pml-training.csv")
```

### Cleaning up the data.

 ``` {r}
# Removing predictors which are not useful (either have near 0 variance or are N/A or have little predictive power).
NZV <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, !NZV$nzv]
testing <- testing[, !NZV$nzv]
rm(NZV)
  
keep_not_NA <- (colSums(is.na(training)) == 0)
training <- training[, keep_not_NA]
testing <- testing[, keep_not_NA]
rm(keep_not_NA)

regex <- grepl("^X|timestamp|user_name", names(training))
training <- training[, !regex]
testing <- testing[, !regex]
rm(regex)
dim(training)
```

### Partitionaing the training data to have a validation set. 

 ``` {r}
set.seed(4321) # to replicate.

inTrain <- createDataPartition(training$classe, 
                               p = 0.70, 
                               list = FALSE)

validation <- training[-inTrain, ]
training <- training[inTrain, ]
rm(inTrain) 
```

### Modeling.
#### Classification Tree

 ``` {r}
set.seed(4321)
mod_ct <- train(classe ~ ., data=training, method = "rpart")

# Ploting the model.

fancyRpartPlot(mod_ct$finalModel)

# Testing the model.
pred_ct <- predict(mod_ct, validation)
CM_ct <- confusionMatrix(pred_ct, validation$classe) #statistics to see accuracy of model
CM_ct

varImp(mod_ct)

# Plot the model
plot(CM_ct$table, col = CM_ct$byClass, 
     main = paste("Decision Tree - Accuracy =", round(CM_ct$overall['Accuracy'], 4)))
```

#### Random Forest

 ``` {r}
Control_rf = trainControl(method = "cv", 5, verboseIter = FALSE)

set.seed(4321)
mtry <- sqrt(length(names(training)))
tunegrid <- expand.grid(.mtry=mtry)
mod_rf <- train(classe~., 
                    data=training, 
                    method='rf', 
                    metric='Accuracy', 
                    tuneGrid=tunegrid, 
                    trControl=Control_rf)

mod_rf
mod_rf$finalModel$ntree

  # Testing the model.
pred_rf <- predict(mod_rf, validation)
CM_rf <- confusionMatrix(pred_rf, validation$classe) #statistics to see accuracy of model
CM_rf

  # Ploting the model

plot(CM_rf$table, col = CM_rf$byClass, 
     main = paste("Decision Tree - Accuracy =", round(CM_rf$overall['Accuracy'], 4)))
```

#### Boosting 
 
 ``` {r}
Control_gbm <- trainControl(method = "cv", 5)

set.seed(4321)
mod_gbm <- train(classe~., data = training, trControl = Control_gbm, method = "gbm", verbose = FALSE)
print(mod_gbm)

# Testing the model.
pred_gbm <- predict(mod_gbm, validation)
CM_gbm <- confusionMatrix(pred_gbm, validation$classe) #statistics to see accuracy of model

trellis.par.set(caretTheme())
plot(mod_gbm)  

# Plot the model
plot(CM_gbm$table, col = CM_gbm$byClass, 
     main = paste("Decision Tree - Accuracy =", round(CM_gbm$overall['Accuracy'], 4)))
```

### Comparing the three alogrithms.
 ``` {r}
CM_ct$overall['Accuracy']
CM_rf$overall['Accuracy']
CM_gbm$overall['Accuracy']
```

### Testing best model
 ``` {r}
results <- predict(mod_rf, testing)
results
```
