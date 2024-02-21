

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# Read data from CSV file
d1 = pd.read_csv("HW2.csv")
d1 = d1.dropna()

# Recode 'new_region' based on 'Region'
d1['new_region'] = np.select(
    [d1['Region'].isin([1, 2]), d1['Region'].isin([3, 4]), d1['Region'].isin([5, 6]), d1['Region'] == 7],
    [1, 2, 3, 4],
    default=None
)
d1['new_region'] = d1['new_region'].astype('category')

# Summary statistics
summary_d1 = d1.describe()

# Grouped mean and standard deviation
mean_std_CEB1 = d1.groupby('new_region')['CEB1'].agg(lambda x: f"M (SD) = {x.mean():.2f} ({x.std():.2f})")

# Grouped bar chart
fig, ax = plt.subplots()
d1.hist(column='CEB2', by='new_region', bins=20, alpha=0.5, ax=ax)
plt.show()

# Poisson Regression Model
d1 = d1[d1['RH'] < 9990]
m1 = sm.GLM(d1['CEB2'], sm.add_constant(d1[['new_region', 'RH', 'AOR']]), family=sm.families.Poisson()).fit()
summary_m1 = m1.summary()

# Model fit
res_deviance = m1.deviance
df_residual = m1.df_resid
p_value = 1 - stats.chi2.cdf(res_deviance, df_residual)

# Model comparison
m2 = sm.GLM(d1['CEB2'], sm.add_constant(d1[['RH', 'AOR']]), family=sm.families.Poisson()).fit()
anova_result = sm.stats.anova_lm(m2, m1, test="Chisq")

# Exponentiated estimates
coef_m1 = m1.params
cov_m1 = m1.cov_params()
std_err = np.sqrt(np.diag(cov_m1))
robust_se = sms.sandwich_covariance.cov_HC0(m1)

rexp_est = pd.DataFrame({
    'Estimate': np.exp(coef_m1),
    'Robust SE': np.exp(std_err),
    'Pr(>|z|)': 2 * (1 - stats.norm.cdf(np.abs(coef_m1 / std_err))),
    'LL': np.exp(coef_m1 - 1.96 * std_err),
    'UL': np.exp(coef_m1 + 1.96 * std_err)
})

# Predictions
s1 = pd.DataFrame({'RH': d1['RH'].mean(), 'AOR': d1['AOR'].mean(), 'new_region': range(1, 5)},
                  index=range(1, 5))
predict_m1 = m1.get_prediction(s1)
predict_m1_summary = predict_m1.summary_frame()

# Data for the scatter plot
d1 = d1.sort_values(['new_region', 'RH', 'AOR'])
d1['phat'] = predict_m1.predicted_mean

# Scatter plot
fig, ax = plt.subplots()
for region, group in d1.groupby('new_region'):
    ax.scatter(group['RH'], group['CEB2'], label=f'Region {region}', alpha=0.5)
ax.plot(d1['RH'], d1['phat'], color='black', linewidth=2)
ax.set_xlabel('RH')
ax.set_ylabel('Expected CEB2 Count')
ax.legend()
plt.show()

# Negative binomial regression model
new_region_mapping = {1: 1, 2: 2, 3: 3, 4: 4}
d1['new_region'] = d1['new_region'].map(new_region_mapping)

# Histogram
ggplot_histogram = ggplot(d1, aes('CEB1', fill='new_region')) + geom_histogram(binwidth=1) + \
                   facet_grid('new_region ~ .', margins=True, scales='free')

# Negative binomial regression model
m1 = sm.GLM(d1['CEB1'], sm.add_constant(d1[['RH', 'AOR', 'new_region']]),
            family=sm.families.NegativeBinomial()).fit()
summary_m1 = m1.summary()

# Model comparison
m2 = sm.GLM(d1['CEB1'], sm.add_constant(d1[['RH', 'AOR']]), family=sm.families.Poisson()).fit()
anova_result = sm.stats.anova_lm(m1, m2)

# Model assumption check
m3 = sm.GLM(d1['CEB1'], sm.add_constant(d1[['RH', 'AOR', 'new_region']]), family=sm.families.Poisson()).fit()
p_value = 1 - stats.chi2.cdf(2 * (m1.llf - m3.llf), 1)

# Estimates with confidence interval
est = pd.concat([pd.DataFrame({'Estimate': m1.params}), m1.conf_int().rename(columns={0: 'Lower CI', 1: 'Upper CI'})],
               axis=1)

# Predictive values
newdata1 = pd.DataFrame({'RH': d1['RH'].mean(), 'AOR': d1['AOR'].mean(),
                         'new_region': range(1, 5)}, index=range(1, 5))
newdata1['phat'] = m1.get_prediction(newdata1).predicted_mean
