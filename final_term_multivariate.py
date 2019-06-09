'''
Final project Multivariate statistic Analyasis
By:Victor Andrean (D10702808)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from matplotlib.colors import ListedColormap
#from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix
visual=0
n_components=5
preprocess='LDA' #PCA, KPCA, LDA
modelList=['Logistic Regression','KNN','SVM','Kernel SVM','Naive Bayes','Decision Tree','Random Forest']
dataSetList=['Training set','Test set']
##############################################################################
# Importing the dataset
data = pd.read_csv('glass.csv')
data.head()
data['Type']=data['Type'].replace([5, 6, 7], [4, 5, 6])

#'''Correlation Matrix'''
#correlation = data.corr()
#plt.figure(figsize=(14, 12))
#heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
#plt.savefig('correlation_matrix.png', format='png')
#plt.show()
#
#'''scatter plot'''
#color = ListedColormap(('red','blue','green','yellow','orchid','pink'))
#scatter = scatter_matrix(data.drop(['Type'],axis=1), c = data['Type'], marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(18,18), cmap = color)
#plt.suptitle('Scatter-matrix for each input variable')
#plt.savefig('scatter_matrix')
#plt.show()

'''Split the data into train and test set'''
y=data['Type'].values
X=data.drop(['Type'],axis=1).values

# Splitting the dataset into the Training set and Test set
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.20,random_state =0)
# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
preprcsName='None'
if preprocess=='PCA':
    # Applying PCA
    preprcsName='PCA'
    pca = PCA(n_components = n_components) 
    x_test = pca.fit_transform(x_test)
    x_train = pca.transform(x_train)
    explained_variance = pca.explained_variance_
    explained_variance_ratio=pca.explained_variance_ratio_
    explained_variance_ratio_total=pca.explained_variance_ratio_.cumsum()
elif preprocess=='KPCA':
    # Applying Kernel PCA
    preprcsName='KPCA'
    kpca = KernelPCA(n_components = n_components, kernel = 'rbf')
    x_train = kpca.fit_transform(x_train)
    x_test = kpca.transform(x_test)
    explained_variance = np.var(x_train, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    explained_variance_ratio_total=np.cumsum(explained_variance_ratio)#cumulative sum
elif preprocess=='LDA':
    #Applying LDA (supervised dimensionality reduction model)
    preprcsName='LDA'
    lda = LDA(n_components = n_components)
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)
    explained_variance = np.var(x_train, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    explained_variance_ratio_total=np.cumsum(explained_variance_ratio)#cumulative sum
if preprcsName!='None' and visual!=1:  
    plt.figure(figsize = (10,5)) 
    plt.plot(range(len(explained_variance)), explained_variance, 'ro-', linewidth=2)
    plt.title('Scree Plot ({})'.format(preprcsName))
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.savefig('Scree Plot ({}).png'.format(preprcsName))
    
Model=2
if Model == 1:
    classifier = LogisticRegression(random_state = 0)
elif Model==2: #K-NN
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
elif Model == 3: #SVM
    classifier = SVC(kernel = 'linear', random_state = 0)
elif  Model== 4: # Kernel SVM
    classifier = SVC(kernel = 'rbf', random_state = 0)
elif Model== 5: # Naive Bayes
    classifier = GaussianNB()
elif Model== 6: # Decision Tree
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
elif Model== 7: #  Random Forest
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cmat = confusion_matrix(y_test, y_pred)
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
F_measure_avg=0
sigma_row=cmat.sum(axis=1)
for i in range(0,len(cmat)):
    F_measure_avg=F_measure_avg+fscore[i]*sigma_row[i]
F_measure_avg=F_measure_avg/sigma_row.sum()
##############################################################################
if n_components==2 and visual==1:
    color = ListedColormap(('red','blue','green','cyan','orchid','pink'))
    # Visualising the Training set results 
    X_set, y_set = x_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap=color)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = color(i), label = j)
    if preprcsName=='None':
        plt.xlabel('Ca')
        plt.ylabel('Estimated Salary')
    else:
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    plt.legend()   
    
    fig=plt.title('{}+{} ({})'.format(preprcsName,modelList[Model-1],dataSetList[0]))
    plt.savefig('{}+{} ({}).png'.format(preprcsName,modelList[Model-1],dataSetList[0]))
    plt.show()    
        
    ## Visualising the Test set results
    X_set, y_set = x_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap=color)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = color(i), label = j)
    if preprcsName=='None':
        plt.xlabel('Ca')
        plt.ylabel('Estimated Salary')
    else:
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    plt.legend()
    fig=plt.title('{}+{} ({})'.format(preprcsName,modelList[Model-1],dataSetList[1]))
    plt.savefig('{}+{} ({}).png'.format(preprcsName,modelList[Model-1],dataSetList[1]))
    plt.show()


