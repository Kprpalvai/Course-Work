library(ggplot2)
# data set reading
d1 <- read.csv("HW2.csv")
d1<-na.omit(d1)
d1$new_region <- NA
d1$new_region[d1$Region == 1 | d1$Region == 2] <- 1
d1$new_region[d1$Region == 3 | d1$Region == 4] <- 2
d1$new_region[d1$Region == 5 | d1$Region == 6] <- 3
d1$new_region[d1$Region == 7] <- 4

d1$new_region<-as.factor(d1$new_region)
summary(d1)

# mean and sd for CEB1 by Region
with(d1, tapply(CEB1, new_region, function(x) {
  sprintf("M (SD) = %1.2f (%1.2f)", mean(x), sd(x))
}))
# mean and sd for CEB2 by Region
with(d1, tapply(CEB2, new_region, function(x) {
  sprintf("M (SD) = %1.2f (%1.2f)", mean(x), sd(x))
}))

# group bar chart
ggplot(d1, aes(CEB2, fill = new_region)) +
  geom_histogram(binwidth=.5, position="dodge")


# Poisson Regression Model
library(sandwich)
library(msm)
d1<-d1[d1$RH<9990,]
m1 <- glm(CEB2 ~ new_region + RH + AOR, family="poisson", data=d1)
summary(m1)
cov.m1 <- vcovHC(m1, type="HC0")
std.err <- sqrt(diag(cov.m1))
r.est <- cbind(Estimate= coef(m1), "Robust SE" = std.err,
               "Pr(>|z|)" = 2 * pnorm(abs(coef(m1)/std.err), lower.tail=FALSE),
               LL = coef(m1) - 1.96 * std.err,
               UL = coef(m1) + 1.96 * std.err)
r.est

# Overall model fit
with(m1, cbind(res.deviance = deviance, df = df.residual,p = pchisq(deviance, df.residual, lower.tail=FALSE)))
## update m1 model dropping new_region
m2 <- update(m1, . ~ . - new_region)
## test model differences with chi square test
anova(m2, m1, test="Chisq")

s <- deltamethod(list(~ exp(x1), ~ exp(x2), ~ exp(x3), ~ exp(x4), ~exp(x5),~exp(x6)), coef(m1), cov.m1)
## exponentiate old estimates dropping the p values
rexp.est <- exp(r.est[, -3])
## replace SEs with estimates for exponentiated coefficients
rexp.est[, "Robust SE"] <- s
rexp.est

(s1 <- data.frame(RH = mean(d1$RH),AOR= mean(d1$AOR),new_region = factor(1:4, levels = 1:4, labels = levels(d1$new_region))))
predict(m1, s1, type="response", se.fit=TRUE)
d1 <- d1[with(d1, order(new_region, RH,AOR)), ]
## calculate and store predicted values
d1$phat <- predict(m1, type="response")

## order by New Region and then by RH and AOR 
d1 <- d1[with(d1, order(new_region, RH,AOR)), ]
ggplot(d1, aes(x = RH, y = phat, colour = new_region)) +
  geom_point(aes(y = CEB2), alpha=.5, position=position_jitter(h=.2)) +
  geom_line(size = 1) +
  labs(x = "RH", y = "Expected CEB2 Count")

#---------------------------------------
# Negative binomial regression model
#---------------------------------------
library(foreign)
library(MASS)
d1 <- read.csv("HW2.csv")
d1<-na.omit(d1)
d1$new_region <- with(d1, ifelse(Region == 1 | Region == 2, 1,
                                 ifelse(Region == 3 | Region == 4, 2,
                                        ifelse(Region == 5 | Region == 6, 3, 4))
)
)

d1$new_region<-as.factor(d1$new_region)
summary(d1)
# ggplot
ggplot(d1, aes(CEB1, fill = new_region)) + 
  geom_histogram(binwidth = 1) + 
  facet_grid(new_region ~ ., margins = TRUE, scales = "free")
d1<-d1[d1$RH<9990,]
summary(m1 <- glm.nb(CEB1 ~ RH + AOR+ new_region, data = d1))
# model comparison
m2 <- update(m1, . ~ . - new_region)
anova(m1, m2)
# checking model assumption
m3 <- glm(CEB1 ~ RH + + AOR+new_region, family = "poisson", data = d1)
pchisq(2 * (logLik(m1) - logLik(m3)), df = 1, lower.tail = FALSE)
# estimate with confidence interval
(est <- cbind(Estimate = coef(m1), confint(m1)))

# predictive value
newdata1 <- data.frame(RH = mean(d1$RH),AOR = mean(d1$AOR), new_region = factor(1:4, levels = 1:4, 
                                                            labels = levels(d1$new_region)))
newdata1$phat <- predict(m1, newdata1, type = "response")
newdata1
# new data for figures
newdata2 <- data.frame(
  RH = rep(seq(from = min(d1$RH), to = max(d1$RH), length.out = 100), 4),AOR = mean(d1$AOR),
  new_region = factor(rep(1:4, each = 100), levels = 1:4, labels =
                  levels(d1$new_region)))

newdata2 <- cbind(newdata2, predict(m1, newdata2, type = "link", se.fit=TRUE))
newdata2 <- within(newdata2, {
  CEB1_fit<- exp(fit)
  LL <- exp(fit - 1.96 * se.fit)
  UL <- exp(fit + 1.96 * se.fit)
})
# graphs
ggplot(newdata2, aes(RH, CEB1_fit)) +
  geom_ribbon(aes(ymin = LL, ymax = UL, fill = new_region), alpha = .25) +
  geom_line(aes(colour = new_region), size = 2) +
  labs(x = "RH Value ", y = "Expected CEB1")