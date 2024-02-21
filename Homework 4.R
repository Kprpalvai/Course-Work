# Homework 4
library(lme4)
library(ggplot2)
d1<-read.csv("nghs.csv")
d2<-d1[,c('ID', 'SBP', 'DBP', 'AGE', 'BMI', 'RACE')]
d3<-na.omit(d2)
nghs_mixed = lmer(DBP ~ AGE +BMI + RACE + (1 | ID), data = d3)
summary(nghs_mixed)

####------------------------####
library(quantreg)
dat<-read.csv("c2_cpd.csv")
dat<-na.omit(dat)
qr1 <- rq(SBP ~ AGE+HT+TC+HDL , data=dat, tau = 0.9)
summary(qr1)
#Quantile line from fitted model
ggplot(dat, aes(AGE,SBP)) + 
  geom_point() + 
  geom_abline(intercept=coef(qr1)[1], slope=coef(qr1)[2])+
  scale_y_continuous(limits=c(70,150))

