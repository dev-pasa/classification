


```{r}
library(ISLR)
library(MASS)
library(caret)
wine <- read.delim("https://github.com/dev-pasa/classification-using-R/blob/master/Wine-Quality-Training-File")

```


```{r}
unique(wine$type)
wine$type <-factor(wine$type)
table(wine$type)
```

```{r}
str(wine)
```

```{r}
inTrain <- createDataPartition(y = wine$type, p = 0.75, list = FALSE)
train_dat1 <- wine[inTrain,]
test_dat1 <- wine[-inTrain,]
```
```{r}
nzv <- nearZeroVar(train_dat1, saveMetrics = TRUE)
nzv
```
```{r}
cor(train_dat1[,-1])
```
```{r}
library(FNN)
knnGrid <- expand.grid(.k=c(2))
# Use k = 2, since we expect 2 classes
KNN1 <- train(x=train_dat1[,-1], method='knn',
             y=train_dat1$type, 
             preProcess=c('center', 'scale'), 
             tuneGrid = knnGrid)
KNN1
```

```{r}
knnConf1 <- confusionMatrix(predict(KNN1, test_dat1[,-1]), test_dat1$type)
```

```{r}
library(stats)
logit1 <- train(type~., data=train_dat1, 
               method='glm', family=binomial(link='logit'),
               preProcess=c('scale', 'center'))
logit1
```
```{r}
summary(logit1)
```
```{r}
glmConf1 <- confusionMatrix(predict(logit1, test_dat1), test_dat1$type)
```

```{r}
QDA1 <- train(type~., data=train_dat1,
             method='qda', 
             preProcess=c('scale', 'center'))
QDA1
```

```{r}
qdaConf1 <- confusionMatrix(test_dat1$type, predict(QDA1, test_dat1))
```

```{r}
library(rpart)
RPART1 <- train(type ~ ., data=train_dat1, 
                method="rpart")
RPART1
```
```{r}
rpartConf1 <- confusionMatrix(test_dat1$type, predict(RPART1, test_dat1))
```

```{r}
plot(RPART1$finalModel, uniform=TRUE, main="Classification Tree")
text(RPART1$finalModel, use.n=TRUE, all=TRUE)
```
```{r}

library(gbm)
set.seed(123)
fitControl = trainControl(method="cv", number=5, returnResamp = "all")

#gbm1 <- train(type ~ ., data=train_dat1, method="gbm",distribution="bernoulli")
```
```{r}
#gbmConf1 <- confusionMatrix(predict(gbm1, test_dat1), test_dat1$type)

```
```{r}
inTrain2 <- createDataPartition(y = wine$type, p = 0.90, list = FALSE)
train_dat2 <- wine[inTrain2,]
test_dat2 <- wine[-inTrain2,]
cor(train_dat2[-1])
```
```{r}
library(FNN)
knnGrid <- expand.grid(.k=c(2))
# Use k = 2, since we expect 2 classes
KNN2 <- train(x=train_dat2[,-1], method='knn',
             y=train_dat2$type, 
             preProcess=c('center', 'scale'), 
             tuneGrid = knnGrid)
KNN2
```

```{r}
knnconf2 <- confusionMatrix(predict(KNN2, test_dat2[,-1]), test_dat2$type)
```

```{r}
library(stats)
logit2 <- train(type~., data=train_dat2, 
               method='glm', family=binomial(link='logit'),
               preProcess=c('scale', 'center'))
logit2
```
```{r}
summary(logit2)
```
```{r}
glmconf2 <- confusionMatrix(predict(logit2, test_dat2), test_dat2$type)
```

```{r}
QDA2 <- train(type~., data=train_dat2,
             method='qda', 
             preProcess=c('scale', 'center'))
QDA2
```

```{r}
qdaConf2 <- confusionMatrix(test_dat2$type, predict(QDA2, test_dat2))
qdaConf2
```

```{r}
library(rpart)
RPART2 <- train(type ~ ., data=train_dat2, 
                method="rpart")
RPART2
```
```{r}
rpartConf2 <- confusionMatrix(test_dat2$type, predict(RPART2, test_dat2))
```

```{r}
plot(RPART2$finalModel, uniform=TRUE, main="Classification Tree")
text(RPART2$finalModel, use.n=TRUE, all=TRUE)
```
```{r}

library(gbm)
set.seed(123)
fitControl = trainControl(method="cv", number=5, returnResamp = "all")

#gbm2 <- train(type ~ ., data=train_dat2, method="gbm",distribution="bernoulli")
#summary(gbm2)
```
```{r}
#gbmConf2 <- confusionMatrix(predict(gbm2, test_dat2), test_dat2$type)

```

```{r}

library(caTools)

set.seed(222)
split <- sample.split(wine$type, SplitRatio = 0.70)

#get training and test data
winetrain2 <- subset(wine, split == TRUE)
winetest2 <- subset(wine, split == FALSE)
```

```{r}
knnGrid <- expand.grid(.k=c(2))
# Use k = 2, since we expect 2 classes
KNN2 <- train(x=winetrain2[,-1], method='knn',
             y=winetrain2$type, 
             preProcess=c('center', 'scale'), 
             tuneGrid = knnGrid)
KNN2
```

```{r}
confusionMatrix(predict(KNN2, winetest2[,-1]), winetest2$type)
```

```{r}
#for 90 10 split
#list(GBM2 = gbmConf2, RPART2 = rpartConf2, QDA2 = qdaConf2, GLM2 <- glmconf2, Knn2 <- knnconf2)
```

```{r}
#for 75 25 split
#list(GBM1 = gbmConf1, RPART1 = rpartConf1, QDA1 = qdaConf1, GLM1 <- glmConf1, Knn1 <- knnConf1)
```

