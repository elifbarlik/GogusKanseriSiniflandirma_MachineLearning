import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../DATAS/data.csv')
data.drop(['id','Unnamed: 32'], inplace=True, axis=1)

data = data.rename(columns={'diagnosis':'target'})


data['target']= [1 if i.strip()=='M' else 0 for i in data.target]

print(data.head())

print(data.shape)
data.info()
describe = data.describe()
print(data.target.value_counts())

"""
standardization
missing value: none
"""

#%% EDA

#CORRELATION
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot=True, fmt=".2f")
plt.title('Correlation Between Features')
plt.show()

thershold = 0.5
filtre = np.abs(corr_matrix['target'])> thershold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot=True, fmt=".2f", annot_kws={'size':15})
plt.title('Correlation Between Features w Corr Threshold 0.75')

#BOX PLOT
data_melted= pd.melt(data, id_vars='target', var_name='features', value_name='value') #veri çerçevesini daha uzun bir forma dönüştürür.
#id_vars='target': Bu, "melt" işleminden sonra sabit kalacak sütunu belirtir
#var_name='features': Yeni oluşturulacak sütunda, eriyen sütunların isimlerinin ne olarak adlandırılacağını belirler.
#value_name='value': Eriyen sütunların değerleri bu sütunda toplanır ve adı "value" olur.
plt.figure()
sns.boxplot(x='features',y='value',hue='target', data=data_melted)
#Kutu grafikleri, verinin dağılımını ve istatistiksel özetini gösterir
#hue='target': Veriyi target değerine göre renklendirir,
plt.xticks(rotation=90)
plt.show()

"""
standardization-normalization
"""

#PAIR PLOT
sns.pairplot(data[corr_features],diag_kind='kde', markers='+', hue='target')
# KDE (Kernel Density Estimation): sürekli verinin dağılımını daha düzgün ve yumuşak bir şekilde görselleştirmeye olanak tanır.
#Diagonal (köşegen) boyunca yer alan grafikler tek bir değişkenin dağılımını gösterir.
#hue parametresi kullanılarak veri seti farklı gruplara ayrılabilir.
plt.show()


#%% OUTLIER DETECTION

y=data.target
x=data.drop(['target'], axis=1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score['score'] = x_score

threshold = -2.5
filtre = outlier_score['score'] < threshold
outlier_index = outlier_score[filtre].index.tolist()

plt.figure(figsize=(15,8))
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color='blue', s=50,label='Outlier Index')
plt.scatter(x.iloc[:,0], x.iloc[:,1], color='k',s=3, label='Data Points')
#s = 3, Noktaların boyutunu 3 birim olarak ayarlar.
radius = (x_score.max() - x_score)/(x_score.max() - x_score.min())
outlier_score['radius'] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s=1000*radius , edgecolors='r',facecolors = 'none', label='Outlier Scores')
plt.legend()
plt.show()

#DROP OUTLIERS
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

#%% TRAIN-TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=42)

#%% STANDARDIZATION
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_df = pd.DataFrame(x_train, columns=columns)
x_train_df_describe = x_train_df.describe()
x_train_df['target']=y_train

data_melted = pd.melt(x_train_df, id_vars='target',
                        var_name='features',
                        value_name='value')
plt.figure()
sns.boxplot(x='features',y='value',hue='target',data=data_melted)
plt.xticks(rotation=90)
plt.show()

sns.pairplot(x_train_df[corr_features],diag_kind='kde',markers='+',hue='target')
plt.show()

#%% Basic KNN Method

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
#accuracy_score, tahmin edilen y_pred ile gerçek y_test etiketlerini karşılaştırarak doğruluk oranını hesaplar.
score = knn.score(x_test, y_test)
#knn.score, verilen x_test verisi üzerinde knn modelini kullanarak tahminler yapar (knn.predict(x_test)) ve bu tahmin sonuçlarını y_test ile karşılaştırır
#knn.score fonksiyonu, knn.predict(x_test) fonksiyonunu çağırıp bu tahminleri kullanarak accuracy_score hesaplar ve doğrudan bu doğruluk değerini döndürür.
print(score)
print(cm)
print(acc)

#%%% Choose Best Params
def KNN_Best_Params(x_train, x_test, y_train, y_test):
    k_range = list(range(1,31))
    weight_options = ['uniform','distance']
    p_opp=1,2,float('inf')
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options,p=p_opp)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print('Test score: {}, Train Score:{}'.format(acc_test, acc_train))
    print()
    print('CM Test: ',cm_test)
    print('CM Train: ',cm_train)
    
    return grid
grid = KNN_Best_Params(x_train, x_test, y_train, y_test)
    
#%%% PCA ANLATIM
# x=[2.4, 0.6, 2.1, 2, 3,2.5, 1.9, 1.1, 1.5, 1.2]
# y=[2.5, 0.7, 2.9, 2.2, 3.0, 2.3, 2.0 ,1.1, 1.6, 0.8]

# x = np.array(x)
# y = np.array(y)

# plt.scatter(x,y)
# plt.clf()
# x_m = np.mean(x)
# y_m = np.mean(y)

# x = x - x_m
# y = y - y_m

# plt.scatter(x,y)

# c=np.cov(x,y)

# from numpy import linalg as LA

# w, v = LA.eig(c)

# # v[:,i]  w[i] ile iliskilidir
# #w= eigenvalue, v=eigenvector

# p1 = v[:,1]
# p2 = v[:,0]


# plt.plot([-2*p1[0], 2*p1[0]],[-2*p1[1],2*p1[1]]) #main component = eigenvalue p2den daha buyuk
# plt.plot([-2*p2[0],2*p2[0]],[-2*p2[1],2*p2[1]])
    

#%% PCA ANALIZI
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_reduced_pca=pca.transform(x_scaled)
pca_data = pd.DataFrame(x_reduced_pca, columns=['p1','p2'])
pca_data['target']=y
plt.figure(dpi=150)
sns.scatterplot(x='p1', y='p2', hue='target', data=pca_data, s=10)
plt.title('PCA: p1 vs p2')


x_train_pca,x_test_pca, y_train_pca,y_test_pca = train_test_split(x_reduced_pca,y,random_state=42, test_size=0.3) 

grid_pca = KNN_Best_Params(x_train_pca, x_test_pca, y_train_pca, y_test_pca)

cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])
    
h=.05
x=x_reduced_pca
x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
y_min, y_max = x[:,1].min()-1, x[:,1].max()+1    
xx, yy = np.meshgrid(np.arange(x_min, y_max,h),
                     np.arange(y_min, y_max,h))

z = grid_pca.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure(dpi=150)
plt.pcolormesh(xx,yy, z, cmap=cmap_light)
    
plt.scatter(x[:,0], x[:,1], c=y, cmap=cmap_bold, edgecolors='k', s=5)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classificatiuon (k=%i, weights = '%s')" %(len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_))
    
#%% NCA ANALIZI

nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
nca.fit(x_scaled, y)
x_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(x_reduced_nca, columns=['p1','p2'])
nca_data['target']=y
sns.scatterplot(x='p1', y='p2', data=nca_data, hue='target', s=10)
plt.title('NCA: p1 vs p2')
    
   
x_train_nca,x_test_nca, y_train_nca,y_test_nca = train_test_split(x_reduced_nca,y,random_state=42, test_size=0.3) 

grid_nca = KNN_Best_Params(x_train_nca, x_test_nca, y_train_nca, y_test_nca)

cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])
    
h=.2
x=x_reduced_nca
x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
y_min, y_max = x[:,1].min()-1, x[:,1].max()+1    
xx, yy = np.meshgrid(np.arange(x_min, y_max,h),
                     np.arange(y_min, y_max,h))

z = grid_nca.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure(dpi=150)
plt.pcolormesh(xx,yy, z, cmap=cmap_light)
    
plt.scatter(x[:,0], x[:,1], c=y, cmap=cmap_bold, edgecolors='k', s=5)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classificatiuon (k=%i, weights = '%s')" %(len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_))
     
#%%   
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(x_train_nca, y_train_nca)
y_pred_nca = knn.predict(x_test_nca)
acc_test_nca = accuracy_score(y_pred_nca, y_test_nca)
knn.score(x_test_nca, y_test_nca)
    
test_data = pd.DataFrame()
test_data['x_test_nca_p1'] = x_test_nca[:,0]
test_data['x_test_nca_p2'] = x_test_nca[:,1]
test_data['y_test_nca'] = y_pred_nca
test_data['y_test_nca'] = y_test_nca

plt.figure()
sns.scatterplot(x='x_test_nca_p1', y='x_test_nca_p2', hue='y_test_nca', data=test_data)

diff = np.where(y_pred_nca!=y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1], label='Wrong Classified',alpha=0.2, color='red', s=1000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    