
import pandas as pd
import numpy as np
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
data = load_iris()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
X = data.data
Y = data.target

from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
# x = StandardScaler().fit_transform(x)



from sklearn.decomposition import PCA
pca = PCA(2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf)



print(pca.components_)

print(pca.explained_variance_)

plt.figure(figsize=(16,10))
plt.title('Projected datapoint')
sns.scatterplot(
    x='principal component 1', y='principal component 2',
    hue="target",
    palette=sns.color_palette("hls", 3),
    data=finalDf,
    legend="full"
)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

sl=[]
mean = df['sepal length'].mean()
for x in df['sepal length']:
  if x>=mean:
    sl.append('HIGH')
  else:
    sl.append('LOW')

finalDf = pd.concat([finalDf, pd.Series(sl, name="sepal length")], axis=1)

sw=[]
mean = df['sepal width'].mean()
for x in df['sepal width']:
  if x>=mean:
    sw.append('HIGH')
  else:
    sw.append('LOW')

finalDf = pd.concat([finalDf, pd.Series(sw, name="sepal width")], axis=1)

pl=[]
mean = df['petal length'].mean()
for x in df['petal length']:
  if x>=mean:
    pl.append('HIGH')
  else:
    pl.append('LOW')

finalDf = pd.concat([finalDf, pd.Series(pl, name="petal length")], axis=1)

pw=[]
mean = df['petal width'].mean()
for x in df['petal width']:
  if x>=mean:
    pw.append('HIGH')
  else:
    pw.append('LOW')

finalDf = pd.concat([finalDf, pd.Series(pw, name="petal width")], axis=1)

plt.figure(figsize=(16,10))
plt.title('Variation of sepal width with PC1 and PC 2')
sns.scatterplot(
    x='principal component 1', y='principal component 2',
    hue="sepal width",
    palette=sns.color_palette("hls", 2),
    data=finalDf,
    legend="full"
)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

plt.figure(figsize=(16,10))
plt.title('Variation of sepal length with PC1 and PC 2')
sns.scatterplot(
    x='principal component 1', y='principal component 2',
    hue="sepal length",
    palette=sns.color_palette("hls", 2),
    data=finalDf,
    legend="full"
)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

plt.figure(figsize=(16,10))
plt.title('Variation of petal length with PC1 and PC 2')
sns.scatterplot(
    x='principal component 1', y='principal component 2',
    hue="petal length",
    palette=sns.color_palette("hls", 2),
    data=finalDf,
    legend="full"
)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

plt.figure(figsize=(16,10))
plt.title('Variation of petal width with PC1 and PC 2')
sns.scatterplot(
    x='principal component 1', y='principal component 2',
    hue="petal width",
    palette=sns.color_palette("hls", 2),
    data=finalDf,
    legend="full"
)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder



x1 = X[0:100]
y1 = Y[0:100]
label_dict = {0: 'Setosa', 1: 'Versicolor'}
lda = LDA(n_components=1)
lda1 = lda.fit_transform(x1, y1)

def plot_scikit_lda(X, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(2),('^', 's'),('blue', 'red')):

        plt.plot(X[:,0][y1 == label],
                   
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('Data points')
    plt.ylabel('LDA 1')

plot_scikit_lda(lda1, title='1D LDA transformation for Setosa and Versicolor')

x2 = X[50:150]
y2 = Y[50:150]
label_dict = {0: 'Versicolor' , 1: 'Verginica'}
lda2 = lda.fit_transform(x2, y2)


plot_scikit_lda(lda2, title='1D LDA transformation for Verginica and Versicolor')

a=X[:50]
b=X[100:]
x3=np.concatenate((a,b))
a=Y[:50]
b=Y[100:]
y3=np.concatenate((a,b)) 
label_dict = {0: 'Setosa' , 1: 'Verginica'}
lda = LDA(n_components=1)
lda3 = lda.fit_transform(x3, y3)

plot_scikit_lda(lda3, title='1D LDA transformation for Verginica and Setosa')

lda2d = LDA(n_components=2)
lda4 = pd.concat([pd.DataFrame(lda2d.fit_transform(X,Y), columns=['LDA 1', 'LDA 2']), pd.Series(Y, name = "target")], axis =1)
plt.figure(figsize=(16,10))
plt.title('transformation to 2D')
sns.scatterplot(
    x='LDA 1', y='LDA 2',
    hue="target",
    palette=sns.color_palette("hls", 3),
    data=lda4,
    legend="full"
)

plt.xlabel("lda 1")
plt.ylabel("lda 2")
plt.show()

from sklearn.manifold import TSNE
import seaborn as sb
model = TSNE(n_components =2,random_state =4,perplexity=5)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with perplexity=5')
plt.show()
model = TSNE(n_components =2,random_state =4,perplexity=15)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with perplexity=15 ')
plt.show()
model = TSNE(n_components =2,random_state =4,perplexity=25)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with perplexity=25')
plt.show()
model = TSNE(n_components =2,random_state =4,perplexity=30)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with perplexity=30 ')
plt.show()
model = TSNE(n_components =2,random_state =4,perplexity=40)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with perplexity=40 ')
plt.show()

model = TSNE(n_components =2,random_state =0, learning_rate=20)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with learning_rate=20')
plt.show()
model = TSNE(n_components =2,random_state =0,learning_rate=200)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with learning_rate=200')
plt.show()
model = TSNE(n_components =2,random_state =0,learning_rate=400)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with learning_rate=400 ')
plt.show()
model = TSNE(n_components =2,random_state =0,learning_rate=600)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with learning_rate=600')
plt.show()
model = TSNE(n_components =2,random_state =0,learning_rate=800)
tsne_data = model.fit_transform(X)
tsne_data = np.vstack((tsne_data.T,Y)).T
tsne_df = pd.DataFrame(data = tsne_data,columns =("dim_1","dim_2","label"))
sb.FacetGrid(tsne_df,hue = "label",height =4).map(plt.scatter,'dim_1','dim_2')
plt.title('2D tSNE with learning_rate=800')
plt.show()

model2 = TSNE(n_components =3,random_state =0)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with perplexity = 30')
plt.show()

model2 = TSNE(n_components =3,random_state =0,perplexity= 20)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with perplexity = 20')
plt.show()

model2 = TSNE(n_components =3,random_state =0,perplexity = 10)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with perplexity = 10')
plt.show()

model2 = TSNE(n_components =3,random_state =0,perplexity = 5)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with perplexity = 5')
plt.show()

model2 = TSNE(n_components =3,random_state =0,perplexity = 50)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with perplexity = 50')
plt.show()

model2 = TSNE(n_components =3,random_state =0,learning_rate=20)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with learning_rate=20')
plt.show()

model2 = TSNE(n_components =3,random_state =0,learning_rate=200)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with learning_rate=200')
plt.show()

model2 = TSNE(n_components =3,random_state =0,learning_rate=400)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with learning_rate=400')
plt.show()

model2 = TSNE(n_components =3,random_state =0,learning_rate=600)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with learning_rate=600')
plt.show()

model2 = TSNE(n_components =3,random_state =0,learning_rate=800)
tsne_data2 = model2.fit_transform(X)
tsne_data2 = np.vstack((tsne_data2.T,Y)).T
tsne_df2 = pd.DataFrame(data = tsne_data2,columns =("dim_1","dim_2","dim_3","label"))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
x = tsne_df2['dim_1']
y = tsne_df2['dim_2']
z = tsne_df2['dim_3']
r = tsne_df2['label']
ax.scatter(x, y, z, c=r, marker='o',s = 20)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D tSNE with learning_rate=800')
plt.show()









