
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier



lfw_dataset = datasets.fetch_lfw_people(min_faces_per_person=100)
target_names = lfw_dataset.target_names
lfw_data = lfw_dataset.data
lfw_images = lfw_dataset.images
lfw_target = lfw_dataset.target

def heat_map(con_matrix, model, target_names=target_names):
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    sns.heatmap(pd.DataFrame(con_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix for '+ model)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.style.use('ggplot')
    plt.show()

lfw_data = pd.DataFrame(lfw_data)
lfw_target_s = pd.Series(lfw_target, name="target")

x_train, x_test, y_train, y_test = model_selection.train_test_split(lfw_data, lfw_target, test_size = 0.3)

n_component = 100
pca = PCA(n_components=n_component).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

var = pca.explained_variance_
print("Variance explained by each Dimension is")
print(var)

lfw_dataFrame = pd.concat([lfw_data, lfw_target_s], axis=1)
lfw_dataFrame = lfw_dataFrame[[any([a,b,c]) for a,b,c in zip((lfw_dataFrame.target==0), (lfw_dataFrame.target==1), (lfw_dataFrame.target==2))]].reset_index(drop=True)
tsne_target = lfw_dataFrame.target
tsne_dataFrame = lfw_dataFrame.drop(columns="target")

tsne = TSNE(n_components =2, random_state =0, perplexity=50, n_iter=3000)
tsne_results = tsne.fit_transform(tsne_dataFrame)
tsne_df = pd.concat([pd.DataFrame(tsne_results).reset_index(drop=True),pd.Series(tsne_target, name="target").reset_index(drop=True)], axis=1)
tsne_df.target.loc[(tsne_df['target'] == 0)] = "Colin Powell"
tsne_df.target.loc[(tsne_df['target'] == 1)] = "Donald Rumsfeld"
tsne_df.target.loc[(tsne_df['target'] == 2)] = "George W Bush"





plt.figure(figsize=(16,10))
plt.title('tsne plot for 3 personalities')
sns.scatterplot(
    x=0, y=1,
    hue="target",
    palette=sns.color_palette("hls", 3),
    data=tsne_df,
    legend="full"
)

plt.xlabel("dimension 1")
plt.ylabel("dimension 1")
plt.show()

neigh = KNeighborsClassifier(n_neighbors=5)
predict_neigh = neigh.fit(x_train_pca, y_train).predict(x_test_pca)
con_matrix_KNN = metrics.confusion_matrix(y_test, predict_neigh)
report_KNN = metrics.classification_report(y_test, predict_neigh, target_names = target_names)
heat_map(con_matrix_KNN, "Nearest Neighbour for 100 PC")
print(report_KNN)

n_samples, h, w = lfw_images.shape
eigenfaces = pca.components_.reshape((n_component, h ,w))

def plot_gallery(images, titles, h, w, n_row=4, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

pca.components_.shape

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

var = pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

n_component = 30
pca = PCA(n_components=n_component).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

neigh = KNeighborsClassifier(n_neighbors=5)
predict_neigh = neigh.fit(x_train_pca, y_train).predict(x_test_pca)
con_matrix_KNN = metrics.confusion_matrix(y_test, predict_neigh)
report_KNN = metrics.classification_report(y_test, predict_neigh, target_names = target_names)
heat_map(con_matrix_KNN, "Nearest Neighbor for 80% variance" )
print(report_KNN,)





