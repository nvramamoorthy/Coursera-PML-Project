# Coursera : Practical Machine Learning Project - Prediction Model
nvramamoorthy  
14 June 2015  
###Synopsis

The project is about HAR - Human Activity Recognition , where 6  participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways and the relevant data were collected using the devices like  Jawbone Up, Nike FuelBand, and Fitbit which were worn by the paricipants.

The participant  regularly  quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

The dataset contains data on  5 classes (sitting-down, standing-up, standing, walking, and sitting) collected on 8 hours of activities through those devices.

The aim  to train a model based on the data of various sensor values, which could later be used to predict the Classe variable, that is the manner in which the participants of HAR did the exercise.

###Initial Setup

####Enivironment

#####Hardware : 

                Macbook Pro with OSX Yosemite 10.10.4

#####Software : 

                RStudio 
                
                GitHub DeskTop / Web Version
                

###Data Processing

####Setting up working directory:


```r
setwd("~/Desktop/Machine Learning/Coursera-PML-Project")
```


The required R Packeges were installed :


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(corrplot)
```


####Download the Data    :

            The training and test data were downloaded as instructed.
            Trainig Data : pml-training.csv 
            Test data    : pml-testing.csv
            


```r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```

####Read the data into Dataframes and check for the content:


```r
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
dim(trainRaw)
```

```
## [1] 19622   160
```

```r
dim(testRaw)
```

```
## [1]  20 160
```

####Cleaning the Data :

On inspecting both traing and test the dataset, some missing data  and unwated data were found which were to be cleaned.


```r
sum(complete.cases(trainRaw))
```

```
## [1] 406
```

The columns containing NA were replaced with 0:


```r
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
```
Few columns where those  data do not have any meaning with the measurements data were removed.


```r
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```

The cleaned training data set now contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

####Slicing the data:

Then, we can split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps. 


```r
set.seed(22519) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

####Data Modeling:

We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use 5-fold cross validation when applying the algorithm. 


```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10989, 10989, 10991, 10990, 10989 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9910462  0.9886729  0.001301269  0.001647776
##   27    0.9914102  0.9891334  0.001717547  0.002174708
##   52    0.9850037  0.9810264  0.002718384  0.003439965
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
The performance of the model on the validation data set is estimated.


```r
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    0    0    1
##          B    7 1128    4    0    0
##          C    0    0 1021    5    0
##          D    0    0   14  949    1
##          E    0    0    1    7 1074
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2853         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9991   0.9817   0.9875   0.9981
## Specificity            0.9995   0.9977   0.9990   0.9970   0.9983
## Pos Pred Value         0.9988   0.9903   0.9951   0.9844   0.9926
## Neg Pred Value         0.9983   0.9998   0.9961   0.9976   0.9996
## Prevalence             0.2853   0.1918   0.1767   0.1633   0.1828
## Detection Rate         0.2841   0.1917   0.1735   0.1613   0.1825
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9977   0.9984   0.9903   0.9922   0.9982
```


```r
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9930331 0.9911872
```

```r
outSampErr <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
outSampErr
```

```
## [1] 0.006966865
```

The accuracy of the model is estimated as 99.30% and out of sample error of 0.69 % is estimated.

####Predicting for Test Data Set:

Finally  we apply the model to the original testing data set downloaded from the data source. We remove the problem_id column first. 


```r
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

###Conclusion

#####It is found that the accuracy of the model is  99.30% and out of sample error is  0.69% .

Plots are available in the folder 'writeup_files/figure-html' in th GitHub Repository and appended in Appendix.

I acknowledge my thanks to the Coursera team and course participants for giving this opporunity to enrich my knowledge by doing this interesting project.

I also thank HAR data provider for this excercise.

####Appendix : Plots

#####Decision Tree Visualization


```r
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) 
```

![](writeup_files/figure-html/unnamed-chunk-13-1.png) 

#####Correlation Matrix Visualization


```r
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```

![](writeup_files/figure-html/unnamed-chunk-14-1.png) 





