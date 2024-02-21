import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

# reading data for Clustering Homework
pd.set_option('display.max_columns', 25)

d0 = pd.read_csv("C:/Users/13099/Downloads/Universities.csv")
d1 = d0.drop(columns=['State', 'Public (1)/ Private (2)'])
d2 = d1.dropna()
print(d2.shape)

#1 Compute pairwise distance matrix (by using Euclidean distance) for the d2 data.

d2.set_index('College Name', inplace=True)
d2=d2.apply(lambda x: x.astype('float64'))
# calculation
d=pairwise.pairwise_distances(d2,metric='euclidean')
pd.DataFrame(d,columns=d2.index,index=d2.index)

#2 Normalize each of the 18 variables (measurement) in the d2 data and denote it as d3.

#pandas uses sample sd
d3=(d2-d2.mean())/d2.std()

#3 Compute pairwise distance matrix (by using Euclidean distance) for the d3 data and denote it as d4.

d_norm_distance=pairwise.pairwise_distances(d3,metric='euclidean')
pd.DataFrame(d_norm_distance,columns=d3.index,index=d3.index)

#4 Construct two dendrograms for d4 data for two methods single=minimum distance and average= average distance. 
#These dendrograms pictorially show the clusters. Result could be different in some extents as two different methods are used for constructing dendrograms. 
#This clustering approach is known as hierarchical clustering.

Z = linkage(d3, method='single')
dendrogram(Z, labels=d3.index, color_threshold=2.75)

Z = linkage(d3, method='average')
dendrogram(Z, labels=d3.index, color_threshold=3.6)

#5 Write python code to generate clusters with the name of elements for each cluster.
memb = fcluster(linkage(d3, method='single'), 8,criterion='maxclust')
memb = pd.Series(memb, index=d3.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ','.join(item.index))

memb = fcluster(linkage(d3, method='average'), 12,criterion='maxclust')
memb = pd.Series(memb, index=d3.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ','.join(item.index))
    
#6 Check whether results (from graphical perspective and output from python code) match

#University Clusters in dendogram and cluster names from code output match

#7 Now take a random sample of size 30 from the 471 rows of the d2 data and denote it as n1 data. For this data,
#construct dendrogram and heatmap. Also generate clusters with name of the elements for each cluster of n1 data
n1 = d3.sample(n=30, random_state=2023)
d_norm_distance=pairwise.pairwise_distances(n1,metric='euclidean')
pd.DataFrame(d_norm_distance,columns=n1.index,index=n1.index)

Z = linkage(n1, method='average')
dendrogram(Z, labels=n1.index, color_threshold=3.6)

memb = fcluster(linkage(n1, method='average'),5,criterion='maxclust')
memb = pd.Series(memb, index=n1.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ','.join(item.index))

n1.index = ['{}: {}'.format(cluster, state)
 for cluster, state in zip(memb, n1.index)]
sns.clustermap(n1, method='average', col_cluster=False,cmap='mako_r')


#Suppose in the above method you have found n clusters. Now you have to do a K-MEANS clustering.

#K-means clustering algorithm

#1 Suppose k=n and then use d2 data to find the clustering of the data.

kmeans = KMeans(n_clusters=12,random_state=1).fit(d3)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=d3.index)
for key, item in memb.groupby(memb):
 print(key, ': ', ', '.join(item.index))    
 

#2 Find the cluster centroids and squared distances for k-MEANS with k=n

centroids = pd.DataFrame(kmeans.cluster_centers_,columns=d3.columns)
#pd.set_option('precision', 3)
round(centroids,3)

#3 Find within-cluster sum of squared distances and cluster count

distances = kmeans.transform(d3)
# find closest cluster for each data point
minSquaredDistances = distances.min(axis=1) ** 2
# combine with cluster labels into a data frame

# added needed brackets:
df = pd.DataFrame({'squaredDistance': minSquaredDistances,'cluster': kmeans.labels_}, index=d3.index)
# group by cluster and print information
for cluster, data in df.groupby('cluster'):
 count = len(data)
 withinClustSS = data.squaredDistance.sum()
 # added more brackets:
 print(f'Cluster {cluster} ({count} members): {withinClustSS:.2f} within cluster')  
pd.DataFrame(pairwise.pairwise_distances(kmeans.cluster_centers_,metric='euclidean'))

#4 Construct profile plot of cluster with different colors for each cluster

centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
plt.figure(figsize=(10,6))
parallel_coordinates(centroids, class_column='cluster',colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#5 Construct a elbow chart to determine the number of cluster.

inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(d3)
    inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(1, 11), 'inertia': inertia}) #changed here { }
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.show()
