# Homework 4
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import matplotlib.pyplot as plt

# Read data from CSV file
d1 = pd.read_csv("nghs.csv")
d2 = d1[['ID', 'SBP', 'DBP', 'AGE', 'BMI', 'RACE']]
d3 = d2.dropna()

# Mixed-effects model (lme4 equivalent in R)
nghs_mixed = smf.mixedlm("DBP ~ AGE + BMI + RACE", d3, groups=d3["ID"]).fit()
print(nghs_mixed.summary())

# Read data from another CSV file
dat = pd.read_csv("c2_cpd.csv")
dat = dat.dropna()

# Quantile regression (quantreg equivalent in R)
qr1 = smf.quantreg("SBP ~ AGE + HT + TC + HDL", dat)
qr_fit = qr1.fit(q=0.9)
print(qr_fit.summary())

# Quantile line from fitted model
fig, ax = plt.subplots()
ax.scatter(dat['AGE'], dat['SBP'])
ax.plot(dat['AGE'], qr_fit.predict(dat), color='red')
ax.set_ylim(70, 150)
plt.show()
`