# installing packages
install.packages("Rtsne")
install.packages("ROSE")
install.packages("ROCR") 

# loading required libraries
library(randomForest)
library(cluster) 
library(Rtsne) 
library(ROCR) 
library(ROSE)
library(tidyverse)  
library(ggplot2)
library(gridExtra)
library(GGally) 
library(ggcorrplot)
library(caret)  
library("data.table")
library(rpart)
library(rpart.plot)
library(RColorBrewer)			  	 
library(party)					     
library(partykit)				       
library(caret)
library(e1071)

# Loading the file
Bank_df<-read.table(file="bank-additional-full.csv",sep=';',header = TRUE)

# Structure of the data 
str(Bank_df)

# Dimensionality of the data
dim(Bank_df)

# Summary of the data
summary(Bank_df)

# Duplicate rows checking
duplicated(Bank_df)

sum(duplicated(Bank_df)) ##The output of the duplicate values is 12.

# Now we will remove the duplicated rows from our data set
Bank_newdf =Bank_df %>% distinct

# Dimensions of the newdataset
dim(Bank_newdf)

# verifying the clean dataset
sum(duplicated(Bank_newdf))  #as we can see there are no duplicated rows or missing values in our dataset.

# Summary of the newdataset
summary(Bank_newdf)

# After carefully observing the summary of bank_cleaned data set as there are some unknowns in default,housing,loan
Bank_newdf<-subset(Bank_newdf,  loan!='unknown' & housing!='unknown' )

# removing the rows where both coloumns having unknowns and above that both the coloumns are binary so we remove them.
dim(Bank_newdf)

# as default is highly skewed data we will impute the value with the mode.
Bank_newdf$default=ifelse(Bank_newdf$default=='yes',1,0)

prop.table(table(Bank_newdf$y))

head(Bank_newdf)

# We can observe that predicted outcome y is mostly skewed to no by 88.73%
# some exploratory analysis on our data set

# age distribution plot
ggplot(Bank_newdf,aes(x=age)) + geom_histogram(color="red", fill="white", binwidth = 5) +
  ggtitle('Age distribution') +
  xlab('Age') +
  ylab('Frequency') +
  geom_vline(aes(xintercept = mean(age)),colour="black", linetype="dashed") +
  scale_x_continuous(breaks = seq(0,100,10))+theme(plot.title = element_text(hjust = 0.5))

# Education level vs Subscription
ggplot( Bank_newdf, aes(x=education, fill=y)) +
  geom_bar() +
  ggtitle("Term Deposit Subscription (Education level)") +
  xlab(" Level of Education ") +
  guides(fill=guide_legend(title="Term Deposit Subscription"))+theme(plot.title = element_text(hjust = 0.5))

# Job vs Subscription
ggplot(Bank_newdf, aes(x=job,fill=y))+geom_bar()+theme_minimal()+scale_fill_brewer(palette="Reds")+
  ggtitle("Job vs subscription")+labs(fill="subscription")+theme(plot.title = element_text(hjust = 0.5))


# Loan vs Subscription
ggplot(Bank_newdf, aes(x=loan,fill=housing))+geom_bar()+scale_fill_brewer(palette="Reds")+facet_grid(cols = vars(y)) +
  ggtitle("housing and personal loan vs subscription")+labs(fill="housing loan")+theme(plot.title = element_text(hjust = 0.5))

# Subscription based on Number of Contact during the Campaign
ggplot(Bank_newdf, aes(x=campaign, fill=y))+                         
  geom_histogram()+
  ggtitle("Subscription based on Number of Contact during the Campaign")+
  xlab("Number of Contact during the Campaign")+
  xlim(c(min=1,max=30)) +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))+
  theme(plot.title = element_text(hjust = 0.5))

#transforming the data
Bank_newdf$housing=ifelse(Bank_newdf$housing=='yes',1,0)
Bank_newdf$loan=ifelse(Bank_newdf$loan=='yes',1,0)
Bank_newdf$job=as.numeric(as.factor(Bank_newdf$job))
Bank_newdf$marital=as.numeric(as.factor(Bank_newdf$marital))
Bank_newdf$month=as.numeric(as.factor(Bank_newdf$month))
Bank_newdf$day_of_week=as.numeric(as.factor(Bank_newdf$day_of_week))
Bank_newdf$contact=as.numeric(as.factor(Bank_newdf$contact))
Bank_newdf$education=as.numeric(as.factor(Bank_newdf$education))
Bank_newdf$poutcome=as.numeric(as.factor(Bank_newdf$poutcome))
summary(Bank_newdf)

sapply(Bank_newdf,class)

Bank_newdf<-transform(
  Bank_newdf,
  job=as.factor(job),
  marital=as.factor(marital),
  education=as.factor(education),
  default=as.factor(default),
  housing=as.factor(housing),
  loan=as.factor(loan),
  contact=as.factor(contact),
  month=as.factor(month),
  day_of_week=as.factor(day_of_week)
)

sapply(Bank_newdf,class)
summary(Bank_newdf)

#performing correlation analysis change this method
ggcorr(Bank_newdf, method = c("everything", "pearson"))+  ggtitle("Correlation Analysis")+theme(plot.title = element_text(hjust = 0.5))

# Feature engineering
# Splitting data into train/test
Maindata    <- sample(1:nrow(Bank_newdf), size=0.6*nrow(Bank_newdf)) 
bank_test  <-  Bank_newdf[-Maindata ,] 
bank_train  <-  Bank_newdf[Maindata,]

#Logistic
mylogit<-glm(y~age+job+marital+education+housing+month+day_of_week+duration+campaign+pdays+previous+poutcome+cons.price.idx+cons.conf.idx+nr.employed,data=bank_test, family=binomial) 
# Summary of the regression 
summary(mylogit)
coef(mylogit)
mylogit.step = step(mylogit, direction='backward') 
summary(mylogit.step)
mylogit.probs<-predict(mylogit,bank_test,type="response")
table(bank_test$y,mylogit.probs>0.5)

mydf <-cbind(bank_test,mylogit.probs) 
mydf$response <- as.factor(ifelse(mydf$mylogit.probs>0.5, 1, 0))

logit_scores <- prediction(predictions=mydf$mylogit.probs, labels=mydf$y) 
#ROC curve 
logit_perf <- performance(logit_scores, "tpr", "fpr")  
plot(logit_perf,main="ROC Curves",xlab="1 - Specificity: False Positive Rate",ylab="Sensitivity: True Positive Rate",col="darkblue",  lwd = 3) 
abline(0,1, lty = 300, col = "green",  lwd = 3) 
grid(col="aquamarine")   
# AREA UNDER THE CURVE 
logit_auc <- performance(logit_scores, "auc")
as.numeric(logit_auc@y.values)

#lift Curve 
logit_lift1 <- performance(logit_scores, measure="lift", x.measure="rpp") 
plot(logit_lift1,main="Lift Chart",xlab="% Customer (Percentile)",ylab="Lift",col="purple", lwd = 3) 
abline(1,0,col="gray",  lwd = 3) 
grid(col="aquamarine")

#Decision tree
binary.model <- rpart(y~age+job+marital+education+housing+month+day_of_week+duration+campaign+pdays+previous+poutcome+cons.price.idx+cons.conf.idx+nr.employed, data=bank_train, cp=.02)
rpart.plot(binary.model)
summary(binary.model)
tree.pred = predict(binary.model, bank_test, type="class")
table(tree.pred,bank_test$y)

#Random forest
randomforest<-randomForest(y~age+job+marital+education+housing+month+day_of_week+duration+campaign+pdays+previous+poutcome+cons.price.idx+cons.conf.idx+nr.employed,data = bank_train,mtry=2, do.trace=10,importance = TRUE, ntree = 1000)
sur<-predict(randomforest, newdata = bank_test)  
varImpPlot(randomforest) 
exp.pred = predict(randomforest,bank_test, type = "class") 
pred_cm <- table(exp.pred,actual = bank_test$y) 
pred_cm

#SVM
svm1<-svm(y~age+job+marital+education+housing+month+day_of_week+duration+campaign+pdays+previous+poutcome+cons.price.idx+cons.conf.idx+nr.employed,data=bank_train, preProcess= c('center', 'scale'),kernel="radial", cost=5, gamma=0.0625,cross=10) 
summary(svm1) 
svm1_sol<-predict(svm1,bank_test) 
table(predict=svm1_sol, truth=bank_test$y)

#Gradient Boost
gradiant<-train(y~age+job+marital+education+housing+month+day_of_week+duration+campaign+pdays+previous+poutcome+cons.price.idx+cons.conf.idx+nr.employed, data=bank_train, method='gbm',trControl=trainControl(method="cv", number=10),verbose=FALSE)            
gradiant
summary(gradiant) 
Predic<- predict(gradiant, newdata = bank_test) 
table(Predic, bank_test$y)

#neural Networks
#bank_train$y1<-as.numeric(bank_train$y)
#bank_train$age1<-as.numeric(bank_train$age)
#bank_train$job1<-as.numeric(bank_train$job)
#bank_train$marital1<-as.numeric(bank_train$marital)
#bank_train$education1<-as.numeric(bank_train$education)
#bank_train$housing1<-as.numeric(bank_train$housing)
#bank_train$month1<-as.numeric(bank_train$month)
#bank_train$day_of_week1<-as.numeric(bank_train$day_of_week)
#bank_train$duration1<-as.numeric(bank_train$duration)
#bank_train$campaign1<-as.numeric(bank_train$campaign)
#bank_train$pdays1<-as.numeric(bank_train$pdays)
#bank_train$previos1<-as.numeric(bank_train$previous)
#bank_train$cons.conf.idx1<-as.numeric(bank_train$cons.conf.idx)


#library(datasets)
#names(infert)
#library(neuralnet)

#nn <- neuralnet(y1~age1+job1+marital1+education1+housing1+month1+day_of_week1+duration1+campaign1+pdays1+previos1+cons.conf.idx1,data=bank_train, hidden=2,linear.output=FALSE)
#nn

#names(nn)

#nn$result.matrix
#out <- cbind(nn$covariate,nn$net.result[[1]])

#dimnames(out) <- list(NULL, c("age1", "job1","marital1","education1","housing1","month1","day_of_week1","duration1","campaign1","pdays1","previos1","cons.conf.idx1","nn-output"))

#head(out)

#head(nn$generalized.weights[[1]])

#plot(nn)











