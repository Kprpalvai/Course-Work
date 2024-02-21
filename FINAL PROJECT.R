##################################FINAL_PROJECT#####################################
############### ARIMA, SARIMA, ARCH, GARCH, State Space Model ##############

chooseCRANmirror()
# select a mirror site; a web site close to you.


#Install packages to be used
install.packages("fma")
install.packages("rugarch")
install.packages("tseries")
install.packages("KFAS")
install.packages("datasets")

library(forecast)
library(fma)
library(rugarch)
library(tseries)
library(KFAS)
library(ggplot2)

# ARIMA
data(wheat)
tsdisplay(wheat, main="Wheat Prices in  constant 1996 pounds",xlab="Year")
(arima.mod<-auto.arima(wheat,seasonal = FALSE))
acf(arima.mod$residuals,main="Residuals ACF")
pacf(arima.mod$residuals,main="Residuals PACF")
	plot(forecast(arima.mod),ylab="Wheat Prices in  constant 1996 pounds",xlab="Year")


# SARIMA
data(milk)
tsdisplay(milk,main="Monthly milk production per cow")
(sarima.mod<-auto.arima(milk))
acf(arima.mod$residuals,main="Residuals ACF")
pacf(arima.mod$residuals,main="Residuals PACF")
plot(forecast(arima.mod),xlab="Year",ylab="pounds",
     main="Monthly milk production per cow")


# ARCH
data("sp500ret")
tsdisplay(sp500ret)
arch.mod<-garch(sp500ret,order = c(0,7))
summary(arch.mod)
acf(na.omit(arch.mod$residuals),main="Residuals ACF")
pacf(na.omit(arch.mod$residuals),main="Residuals PACF")


# GARCH
garch.mod<-garch(sp500ret,order = c(1,1))
summary(garch.mod)
acf(na.omit(garch.mod$residuals),main="Residuals ACF")
pacf(na.omit(garch.mod$residuals),main="Residuals PACF")


# State Space Model
model <- SSModel(milk ~ SSMtrend(2, Q = list(matrix(NA), matrix(NA))) + 
      SSMseasonal(period = 12, sea.type = "dummy", Q = NA), H = matrix(NA))
ssm.mod <- fitSSM(model, inits = c(0, 0, 0, 0), method = "BFGS")

forecast_values<-predict(ssm.mod$model, n.ahead = 10)
autoplot(cbind(milk,forecast_values),xlab="Year",ylab="pounds",
         main="Monthly milk production per cow")
