# Naive Bayes Classification

#  Uses k-fold cross validation to prevent overfitting the model
#  Uses doSNOW for multithreading
#  Removes G1, and G2 from the source of predictors.

# Clear the console window
cat("\014") 

# Clear all environment variables
rm(list=ls())

library(e1071)
library(arules)

#########################
# Import the data
########################

set.seed(1254)
setwd('~/Dropbox/Data_Warehousing/project/student')

# Import the CSV files and store them as tables
# student.portuguese <- read.table("student-mat.csv",sep=";",header=TRUE)
student.portuguese <- read.table("student-por.csv",sep=";",header=TRUE)



##########################
#  DISCRETIZE THE COLUMNS
##########################
#students <- read.transactions(file.choose(), sep = ";")


## student.portuguese 
# The discretized data frame to be used
dStudent.portuguese <- student.portuguese

# school

# sex
dStudent.portuguese$sex <- as.factor(student.portuguese$sex)

# age
dStudent.portuguese$age <- discretize(student.portuguese$age, method = "fixed",
                                      categories = c(-Inf,18,21,Inf), 
                                      labels=c("under18", "18-21", "21+"))

# famsize
dStudent.portuguese$famsize <- as.factor(student.portuguese$famsize)

# Pstatus
dStudent.portuguese$Pstatus <- as.factor(student.portuguese$Pstatus)

# Mjob (remove)
dStudent.portuguese <- dStudent.portuguese[, -grep("^Mjob$", colnames(dStudent.portuguese))]

# Fjob (remove)
dStudent.portuguese <- dStudent.portuguese[, -grep("^Fjob$", colnames(dStudent.portuguese))]

# Medu
dStudent.portuguese$Medu <- as.factor(student.portuguese$Medu)
dStudent.portuguese$Medu <- mapvalues(dStudent.portuguese$Medu,
                                      from = c(0, 1, 2, 3, 4),
                                      to = c("none",
                                             "primary education (4th grade)",
                                             "5th to 9th grade","secondary education",
                                             "higher education"))

# Fedu
dStudent.portuguese$Fedu <- as.factor(student.portuguese$Fedu)
dStudent.portuguese$Fedu <- mapvalues(dStudent.portuguese$Fedu,
                                      from = c(0, 1, 2, 3, 4),
                                      to = c("none",
                                             "primary education (4th grade)",
                                             "5th to 9th grade","secondary education",
                                             "higher education"))

# traveltime 
dStudent.portuguese$traveltime <- as.factor(student.portuguese$traveltime)
dStudent.portuguese$traveltime <- mapvalues(dStudent.portuguese$traveltime,
                                            from = 1:4,
                                            to = c("0-15min.", "15-30min.", "30min-1hr", ">1hr"))

# higher
dStudent.portuguese$higher <- as.factor(student.portuguese$higher)


# studytime
dStudent.portuguese$studytime <- as.factor(student.portuguese$studytime)
dStudent.portuguese$studytime <- mapvalues(dStudent.portuguese$studytime,
                                           from = 1:4,
                                           to = c("0-2hrs", "2-5hrs ", "5-10hrs", ">10hrs"))

# failures
dStudent.portuguese$failures <- as.factor(student.portuguese$failures)
dStudent.portuguese$failures <- mapvalues(dStudent.portuguese$failures,
                                          from = c(0, 1, 2, 3),
                                          to = c("never","once","twice","three+"))

# famrel
dStudent.portuguese$famrel <- as.factor(student.portuguese$famrel)
dStudent.portuguese$famrel <- mapvalues(dStudent.portuguese$famrel,
                                        from = 1:5,
                                        to = c("Very Low", "Low", "Medium", "High", "Very High"))

# freetime
dStudent.portuguese$freetime <- as.factor(student.portuguese$freetime)
dStudent.portuguese$freetime <- mapvalues(dStudent.portuguese$freetime,
                                          from = 1:5,
                                          to = c("Very Low", "Low", "Medium", "High", "Very High"))

# Dalc
dStudent.portuguese$Dalc <- as.factor(student.portuguese$Dalc)
dStudent.portuguese$Dalc <- mapvalues(dStudent.portuguese$Dalc,
                                      from = 1:5,
                                      to = c("Very Low", "Low", "Medium", "High", "Very High"))
# Walc
dStudent.portuguese$Walc <- as.factor(student.portuguese$Walc)
dStudent.portuguese$Walc <- mapvalues(dStudent.portuguese$Walc,
                                      from = 1:5,
                                      to = c("Very Low", "Low", "Medium", "High", "Very High"))

# health
dStudent.portuguese$health <- as.factor(student.portuguese$health)
dStudent.portuguese$health <- mapvalues(dStudent.portuguese$health,
                                        from = 1:5,
                                        to = c("Very Low", "Low", "Medium", "High", "Very High"))

# absences 
# clean outliers outside 3 times std.dev.
#limit <- 3*sd(dStudent.portuguese$absences)

#dStudent.portuguese$absences[dStudent.portuguese$absences > limit] <- floor(limit)
dStudent.portuguese$absences <-discretize(dStudent.portuguese$absences, method = "frequency",
                                          categories = 3,
                                          labels = c("low", "med", "high"))


# goout
dStudent.portuguese$goout <- as.factor(student.portuguese$goout)
dStudent.portuguese$goout <- mapvalues(dStudent.portuguese$goout,
                                       from = 1:5,
                                       to = c("Very Low", "Low", "Medium", "High", "Very High"))
# Paid (remove)
dStudent.portuguese <- dStudent.portuguese[, -grep("^paid$", colnames(dStudent.portuguese))]

# G1 (remove)
dStudent.portuguese <- dStudent.portuguese[, -grep("^G1$", colnames(dStudent.portuguese))]

# G2 (remove)
dStudent.portuguese <- dStudent.portuguese[, -grep("^G2$", colnames(dStudent.portuguese))]

# G3
dStudent.portuguese$G3 <- discretize(student.portuguese$G3, method = "fixed",
                                     categories = c(-Inf,10,12,14,16,Inf), 
                                     labels=c("fail", "sufficient", "satisfactory", "good", "excellent"))



#####################################
# K-fold Cross validation
#####################################
library(caret)
library(doSNOW)

# Random selection for sampling data
set.seed(12345)
percentCover <- 70
s <- sample(dim(dStudent.portuguese)[1], floor(percentCover/100*dim(dStudent.portuguese)[1]))

student_train <- dStudent.portuguese[s,]
student_test <- dStudent.portuguese[-s,]

# Create k-folds multiplied bu n times, for stratification.
cv.10.folds <-createMultiFolds(student_train$G3, k = 10, times = 10)

# Caret's Training Control repeated Cross Validations
ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, index = cv.10.folds)

# Setup the multithreading cluster from the doSNOW package. (Makes training all the folds much quicker.)
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# Create the trained model with caret using the Naive Bayes (nb) algorithm method 
student_train.model <- train(x = student_train[, -which(colnames(student_train) == "G3")],
                             y = student_train$G3,
                             method = "nb",
                             trControl = ctrl.1)

summary(student_train.model$finalModel)

# shutdown the cluster
stopCluster(cl)

# results
student_train.model

# make the predictions
predictions <- predict(student_train.model, student_test)

# Confusion Matrix
conMat <- confusionMatrix( predictions, student_test[,'G3'] )
conMat


###############################################################
# Create the model without cross validation
#############################################################
percentCover <- 70
s <- sample(dim(dStudent.portuguese)[1], floor(percentCover/100*dim(dStudent.portuguese)[1]))

student_train <- dStudent.portuguese[s,]
student_test <- dStudent.portuguese[-s,]

# Can handle both categorical and numeric input variables, but output must be categorical
model <- naiveBayes(G3~., data=student_train)
prediction <- predict(model, student_test[,-grep("^G3$", colnames(student_test))])

# Confusion Matrix
conMat <- confusionMatrix(prediction, student_test[,"G3"])
conMat
