import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
os.getcwd()
os.chdir('C:\\Users\\13099\\Downloads\\')

d1 = pd.read_csv('ucla_gpa.csv')
 
#Q1)

sns.boxplot(x="gre", y="rank",data= d1)
plt.title("Boxplot of rank in gre")
plt.show()
 
#Q.2)
d2=d1.copy().groupby("rank")

d3 = [rows for _, rows in d2]
len(d3) 
import statistics as ss
datalist=[]
for i in range(len(d3)):
       datalist.append(pd.DataFrame(
      {
      'gpa':[ss.mean(d3[i].gpa)],
      'Mean.gpa': [ss.mean(d3[i].gpa)],
      'Median.gpa': [ss.mean(d3[i].gpa)],
      'Min.gpa':[min(d3[i].gpa)],      # calculate minimum
      'Max.gpa':[max(d3[i].gpa)],      # calculate maximum
      'Std.gpa':[ss.stdev(d3[i].gpa)]  # calculate standard deviation
      }))
pd.concat(datalist)

#Q.3)

stats_by_rating = d1.groupby('rank').agg({'gpa': 'mean', 
                                           'gpa': 'median', 
                                           'gpa': 'min', 
                                           'gpa': 'max', 
                                           'gpa': 'std'}).reset_index()
print(stats_by_rating)
def test1(df):
    Mean=ss.mean(df.gpa)
    Minimum=min(df.gpa)
    Maximum=max(df.gpa)
    StandardDev=ss.variance(df.gpa)
    allstats=pd.DataFrame([Mean,Minimum,Maximum,StandardDev])
    return allstats
test1(d1)

#Q.4)

total = 0

N = 10

for k in range(1, N + 1):
    for j in range(1, N + 1):
        for i in range(1, N + 1):

            term = (5 * i) * ((i**20) / 3 + j + (0.9**k))
            
            total += term

print("The result of the three summations is:", total)

#Q.5)
 
x_value = [1, 2, 3, 4, 6]
y_value= [5, 7 , 8, 9, 11]
z_value = [10, 12, 13, 14, 15]

def f(x, y, z):
    return (math.sin(math.exp(x)) - math.log(z**2) ) / (5*x + y**2)

functional_values= [125]

for x in x_value:
    for y in y_value:
        for z in z_value:
            value = f(x, y, z)
            functional_values.append((x, y, z, value))
        
print(functional_values)
