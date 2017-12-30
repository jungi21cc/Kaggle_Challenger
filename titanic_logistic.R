#titanic shirinking prediction model#

#1.logistic regression#
library(caret)
library(nnet)
library(readr)
gender_submission <- read_csv("C:/Users/JK/Desktop/gender_submission.csv")
test <- read_csv("C:/Users/JK/Desktop/test.csv")
train <- read_csv("C:/Users/JK/Desktop/train.csv")
titanic.te<-cbind(gender_submission,test)
titanic.te<-titanic.te[,-3]
titanic.te<-titanic.te[,-4]
titanic.te<-titanic.te[,-8]
titanic.te<-titanic.te[,-9]
str(titanic.te)
titanic.tr<-train[,-4]
titanic.tr<-titanic.tr[,-8]
titanic.tr<-titanic.tr[,-9]
str(titanic.tr)
model <- multinom(Survived~., data = titanic.tr)
summary(model)
y <- titanic.te$Survived
pred <- predict(model, titanic.te)
pred
confusionMatrix(pred,y)
#accuracy=0.90
