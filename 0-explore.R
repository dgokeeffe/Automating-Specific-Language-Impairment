# Specific Language Impairment Data Set 
# 
# 0 <- explore.R 
#
# This is an R implementation of my SLI classifier
# Version: 15012017 

# Library Headers ------------------------------------------------
library(caret)
library(doParallel)
library(pROC)

# Functions ------------------------------------------------------

# Load data ------------------------------------------------------
conti4 <- read.csv("Conti4.csv")
eg <- read.csv("EG.csv")
enni <- read.csv("ENNI.csv")
gillam <- read.csv("Gillam.csv")
all3 <- read.csv("all3.csv")

# Print summary --------------------------------------------------
#str(conti4)
#str(eg)
#str(enni)
#str(gillam)
#str(all3)

# Convert class labels to factors --------------------------------
conti4$X[conti4$X==1] <- "SLI"
conti4$X[conti4$X==0] <- "TD"

# Remove zero length predictors ----------------------------------
nzvCols <- nearZeroVar(conti4)
if(length(nzvCols) > 0) conti4 <- conti4[, -nzvCols]

# Split the datasets ---------------------------------------------
trainIndex <- createDataPartition(conti4$X, p = .66, list = FALSE)
conti4.train <- conti4[ trainIndex,]
conti4.test  <- conti4[-trainIndex,]

# Train the models -----------------------------------------------

############################ CONTI4 ##############################
set.seed(98)
fitControl <- trainControl(## Leave One Out CV
                           method = "LOOCV",
                           repeats = 10,
                           p = 0.7,
                           classProbs = TRUE,
                           allowParallel = TRUE,
                           summaryFunction = twoClassSummary,
                           search = "random")

# Set up parallel processing
registerDoParallel(4)
getDoParWorkers()


# SVM ------------------------------------------------------------
svm.fit <- train(as.factor(X) ~ .,data = conti4, 
                method = "svmRadial", 
                trControl = fitControl,
                preProcess = c("nzv", "center", "scale"),
                metric = "ROC")

# Logistic Regression --------------------------------------------
lg.fit <- train(as.factor(X) ~ .,data = conti4, 
                method = "LogitBoost", 
                trControl = fitControl,
                preProcess = c("nzv", "center", "scale"),
                metric = "ROC")

# Neural Network with Selection -----------------------------------
pcnn.fit <- train(as.factor(X) ~ .,data = conti4, 
                  method = "pcaNNet", 
                  trControl = fitControl,
                  preProcess = c("nzv", "center", "scale"),
                  metric = "ROC",
                  verbose = FALSE)

# Deep Boost ------------------------------------------------------
fitControl <- trainControl(## Leave One Out CV
                           method = "LOOCV",
                           repeats = 10,
                           p = 0.7,
                           classProbs = FALSE,
                           allowParallel = TRUE,
                           search = "random")

db.fit <- train(as.factor(X) ~ .,data = conti4, 
                method = "deepboost", 
                trControl = fitControl,
                preProcess = c("nzv", "center", "scale"),
                metric = "Accuracy")

# Stacked Autoencoder Deep NN -------------------------------------
dnn.fit <- train(as.factor(X) ~ .,data = conti4, 
                method = "dnn", 
                trControl = fitControl,
                preProcess = c("nzv", "center", "scale"),
                metric = "Accuracy")

# ROC -------------------------------------------------------------
roc.fit <- train(as.factor(X) ~ .,data = conti4, 
                method = "rocc", 
                trControl = fitControl,
                preProcess = c("nzv", "center", "scale"))


## Cost sensitive learning ----------------------------------------
#c5.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "C5.0Cost", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy",
#                verbose = FALSE)
#
## AdaBoost M1 ----------------------------------------------------
#fitControl <- trainControl(## Leave One Out CV
#                           method = "LOOCV",
#                           repeats = 10,
#                           p = 0.7,
#                           classProbs = FALSE,
#                           allowParallel = TRUE)
#
#adaboost.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "adaboost", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy")
#
## Linear Discriminant Analysis -----------------------------------
#lda.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "lda", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy")
#
## Random Forest --------------------------------------------------
#rf.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "rf", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy")
#
## Oblique Random Forest ------------------------------------------
#orf.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "ORFlog", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy")
#
### Conditional Inference Random Forest ----------------------------
#cforest.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "cforest", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy")
#
## Random Forest with Additional Feature Selection ----------------
#boruta.fit <- train(as.factor(X) ~ .,data = conti4, 
#                method = "Boruta", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "Accuracy")
#
#

# Print model results --------------------------------------------
svm.fit
lg.fit
pcnn.fit
db.fit
dnn.fit
roc.fit
#c5.fit
#adaboost.fit
#lda.fit
#orf.fit
#rf.fit
#cforest.fit
#boruta.fit

# 
#plot(svm.fit)
#svm.test <- predict(svm.fit, conti4.test, type = "prob")
#conti4.test$SVMprob <- svm.test[,"SLI"]
#conti4.test$SVMclass <- predict(svm.fit, conti4.test)
#confusionMatrix(data = conti4.test$SVMclass, 
#                reference = conti4.test$X, 
#                positive = "SLI")
#svm.ROC <- roc(predictor=conti4.test$SVMprob,
#               response=conti4.test$X)
#svm.ROC$auc
##Area under the curve: 0.8731
#plot(svm.ROC,main="SVM ROC")

## Extreme Gradient Boosting --------------------------------------
#xgb.fit <- train(as.factor(X) ~ .,data = conti4.train, 
#                method = "xgbTree", 
#                trControl = fitControl,
#                preProcess = c("nzv", "center", "scale"),
#                metric = "ROC")
#xgb.fit
#xgb(xvg.fit)
#xgb.test <- predict(sgb.fit, conti4.test, type = "prob")
#conti4.test$xbgprob <- sgb.test[,"SLI"]
#conti4.test$xbgclass <- predict(sgb.fit, conti4.test)
#confusionMatrix(data = conti4.test$xbgclass, 
#                reference = conti4.test$X, 
#                positive = "SLI")
#xbg.ROC <- roc(predictor=conti4.test$xgbprob,
#               response=conti4.test$X)
#xbg.ROC$auc
##Area under the curve: 0.8731
#plot(xbg.ROC,main="xgb ROC")

# Stochastic Gradient Boosting -----------------------------------
#fitControl <- trainControl(## Leave One Out CV
#                           method = "LOOCV",
#                           classProbs = FALSE)
#set.seed(46)
#gbm.fit <- train(as.factor(X) ~ ., data = conti4.train, 
#                 method = "gbm", 
#                 trControl = fitControl,
#                 verbose = FALSE,
#                 preProcess = c("center", "scale"))
#
## Learning Vector Quantization -----------------------------------
#set.seed(46)
#lvq.fit <- train(as.factor(X) ~ ., data = conti4.train, 
#                 method = "lvq", 
#                 trControl = fitControl,
#                 preProcess = c("center", "scale"))
#
# Collect resamples ---------------------------------------------
# Summarize the distributions
