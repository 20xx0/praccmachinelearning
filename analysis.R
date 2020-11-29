library(caret)
library(randomForest)
set.seed(9916)

trainingRaw <- read.csv("pml-training.csv")
testingRaw <- read.csv("pml-testing.csv")

trainingEdit <- trainingRaw
testing <- testingRaw

trainset <- createDataPartition(trainingEdit$classe, p = 0.75, list = FALSE)
training <- trainingEdit[trainset, ]
validate <- trainingEdit[-trainset, ]

# Remove first 7 variables
training <- training[,8:dim(training)[2]]
testing <- testing[,8:dim(testing)[2]]
# Remove NAs
training <- training[,colSums(is.na(training))==0]
testing <- testing[,colSums(is.na(testing))==0]
# Remove near zero variables
#nzvcol <- nearZeroVar(training)
training <- training[, -nearZeroVar(training)]

# Create 2 models

ldaModel <- train(classe ~ .,data=training,method="lda")
rfModel <- randomForest(as.factor(classe) ~ ., data = training)

# Find accuracy of both
ldaPredTrain <- predict(ldaModel,training)
confusionMatrix(ldaPredTrain,as.factor(training$classe))

rfPred <- predict(rfModel,training)
confusionMatrix(rfPred,as.factor(training$classe))

# Find accuracy of validation set
ldaPredValid <- predict(ldaModel,validate)
confusionMatrix(ldaPred,as.factor(validate$classe))

rfPred <- predict(rfModel,validate)
confusionMatrix(rfPred,as.factor(validate$classe))

# The accuracy of the random forest model is very high.

finalPred <- predict(rfModel,testing)
confusionMatrix(finalPred,testing)
