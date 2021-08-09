# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:07:05 2019

@author: Salisu-Garba
"""
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()


# set width of bar
bar_width = 0.35

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [187, 140, 166, 185, 343, 168, 261, 168]

x2 = [1, 2, 3, 4, 5, 6, 7, 8]
y2 = [174, 124, 136, 156, 331, 160, 223, 161]


plt.bar(x, y, label='Actual categories', color='Blue')
plt.bar(x2, y2, label='Obtained categories', color='skyblue')
plt.grid(color='gray', linestyle='-', linewidth=0.2)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([50, 100, 150, 200, 250, 300, 350])
#plt.title('Clusters of clients')
plt.xlabel('No. of Categories')
plt.ylabel('No. of Services')
plt.legend()
plt.show()





import pandas as pd
from matplotlib import pyplot as plt


#RECALL
data = [['1','K-means++', 0.470], ['2', 'K-means & Agnes', 0.868], ['3', 'K-means & CSO', 0.871], ['4', 'PK-means', 0.757], ['5','M-NSA', 0.979]] 
data = pd.DataFrame(data, columns = ['Object', 'Type', 'Value']) 


colors = {'K-means++':'#376D71', 'K-means & Agnes':'#83AFC0', 'K-means & CSO':'#31AFEA', 'PK-means':'#B1E7DE', 'M-NSA':'#8CFBF5'}
c = data['Type'].apply(lambda x: colors[x])


ax = plt.subplot(111) #specify a subplot

bars = ax.bar(data['Object'], data['Value'], color=c) #Plot data on subplot axis

for i, j in colors.items(): #Loop over color dictionary
    ax.bar(data['Object'], data['Value'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified

plt.xlabel('Approach')
plt.ylabel('Recall')

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.legend(loc='upper center', ncol=3)
plt.show()



#PRECISION
data = [['1','K-means++', 0.470], ['2', 'K-means & Agnes', 0.731], ['3', 'K-means & CSO', 0.801], ['4', 'PK-means', 0.743], ['5','M-NSA', 0.933]] 
data = pd.DataFrame(data, columns = ['Object', 'Type', 'Value']) 


colors = {'K-means++':'#376D71', 'K-means & Agnes':'#83AFC0', 'K-means & CSO':'#31AFEA', 'PK-means':'#B1E7DE', 'M-NSA':'#8CFBF5'}
c = data['Type'].apply(lambda x: colors[x])


ax = plt.subplot(111) #specify a subplot

bars = ax.bar(data['Object'], data['Value'], color=c) #Plot data on subplot axis

for i, j in colors.items(): #Loop over color dictionary
    ax.bar(data['Object'], data['Value'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified

plt.xlabel('Approach')
plt.ylabel('Precision')

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.legend(loc='upper center', ncol=3)
plt.show()



#ACCURACY
data = [['1','K-means++', 0.529], ['2', 'K-means & Agnes', 0.706], ['3', 'K-means & CSO', 0.742], ['4', 'PK-means', 0.857], ['5','M-NSA', 0.915]] 
data = pd.DataFrame(data, columns = ['Object', 'Type', 'Value']) 


colors = {'K-means++':'#376D71', 'K-means & Agnes':'#83AFC0', 'K-means & CSO':'#31AFEA', 'PK-means':'#B1E7DE', 'M-NSA':'#8CFBF5'}
c = data['Type'].apply(lambda x: colors[x])


ax = plt.subplot(111) #specify a subplot

bars = ax.bar(data['Object'], data['Value'], color=c) #Plot data on subplot axis

for i, j in colors.items(): #Loop over color dictionary
    ax.bar(data['Object'], data['Value'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified

plt.xlabel('Approach')
plt.ylabel('Accuracy')

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.legend(loc='upper center', ncol=3)
plt.show()







#F-MEASURE
data = [['1','K-means++', 0.470], ['2', 'K-means & Agnes', 0.794], ['3', 'K-means & CSO', 0.835], ['4', 'PK-means', 0.750], ['5','M-NSA', 0.955]] 
data = pd.DataFrame(data, columns = ['Object', 'Type', 'Value']) 


colors = {'K-means++':'#376D71', 'K-means & Agnes':'#83AFC0', 'K-means & CSO':'#31AFEA', 'PK-means':'#B1E7DE', 'M-NSA':'#8CFBF5'}
c = data['Type'].apply(lambda x: colors[x])


ax = plt.subplot(111) #specify a subplot

bars = ax.bar(data['Object'], data['Value'], color=c) #Plot data on subplot axis

for i, j in colors.items(): #Loop over color dictionary
    ax.bar(data['Object'], data['Value'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified

plt.xlabel('Approach')
plt.ylabel('F-measure')

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.legend(loc='upper center', ncol=3)
plt.show()



#NEW DATA
# set width of bar
bar_width = 0.35

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [187, 140, 166, 185, 343, 168, 261, 168]

x2 = [1, 2, 3, 4, 5, 6, 7, 8]
y2 = [174, 124, 145, 158, 331, 160, 234, 161]


plt.bar(x, y, label='Actual categories', color='Red')
plt.bar(x2, y2, label='Obtained categories', color='skyblue')
plt.grid(color='gray', linestyle='-', linewidth=0.2)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([50, 100, 150, 200, 250, 300, 350])
#plt.title('Clusters of clients')
plt.xlabel('No. of Categories')
plt.ylabel('No. of Services')
plt.legend()
plt.show()








N = 3

ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots()

accu = [0.72, 0.99, 0.83]
rects1 = ax.bar(ind, accu, width, color='b')
prec = [0.99, 0.99, 0.81]
rects2 = ax.bar(ind+width, prec, width, color='y')
rec = [0.61, 0.99, 0.99]
rects3 = ax.bar(ind + 2 * width, rec, width, color='r')

ax.set_ylabel('Scores')
ax.set_title('Scores for each algorithm')
ax.set_xticks(ind + 1.5*width)
ax.set_xticklabels(('SVM', 'DecisionTree', 'NaiveBayes'))
ax.legend((rects1[0], rects2[0], rects3[0]), ('Accuracy', 'Precision', 'Recall'))

plt.show()





objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(objects))
performance = [10,8,6,4,2,1]

width = 0.15 

plt.bar(y_pos, performance, align='center', alpha=0.5,  color=['black', 'red', 'green', 'blue', 'cyan', 'pink'])
plt.xticks(y_pos, objects)
plt.xlabel('Precision')
plt.ylabel('Approach')
plt.title('Programming language usage')


plt.legend()
plt.show()





# generate random data for plotting
x = np.linspace(0.0,100,20)

# now there's 3 sets of points
y1 = np.random.normal(scale=0.2,size=20)
y2 = np.random.normal(scale=0.5,size=20)
y3 = np.random.normal(scale=0.8,size=20)

# plot the 3 sets
plt.plot(x,y1,label='plot 1')
plt.plot(x,y2, label='plot 2')
plt.plot(x,y3, label='plot 3')

# call with no parameters
plt.legend()

plt.show()
