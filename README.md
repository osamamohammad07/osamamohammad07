### Hello World 👋
I am a fullstack software engineer from bangladesh
- 🔭 Working on multiple Data Analytics and Machine Learning Engineering projects using Python and R and forging various ML models into a pipeline.
- 🌱 Learning multiple analytics and visualisation patterns in order to extract maximum insights from the raw data
<br>
## Connect with me
<br>
from google.colab import drive
drive.mount('/content/drive')
root_path = 'gdrive/My Drive/Colab Notebooks'
from sklearn import neighbors
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
<br>
df_botnet = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/new.xlsx')
df_botnet
df_botnet.isna().sum
df_new = df_botnet[['MI_dir_L5_mean','MI_dir_L3_mean','MI_dir_L1_mean','MI_dir_L0.1_mean','MI_dir_L0.01_mean','H_L5_mean','H_L3_mean','H_L1_mean','H_L0.1_mean','H_L0.01_mean','HH_L5_mean','HH_L3_mean','HH_L1_mean','HH_L0.1_mean','HH_L0.01_mean','HH_jit_L5_mean','HH_jit_L3_mean','HH_jit_L1_mean','HH_jit_L0.1_mean','HH_jit_L0.01_mean','HpHp_L5_mean','HpHp_L3_mean','HpHp_L1_mean','HpHp_L0.1_mean','HpHp_L0.01_mean','Attack','Botnet','MI_dir_L5_variance','MI_dir_L3_variance','MI_dir_L1_variance','MI_dir_L0.1_variance','MI_dir_L0.01_variance','H_L5_variance','H_L3_variance','H_L1_variance','H_L0.1_variance','H_L0.01_variance','HH_jit_L0.01_variance','HpHp_L0.1_covariance','HpHp_L1_covariance','HpHp_L3_covariance','HpHp_L5_covariance','HpHp_L0.01_covariance', 'MI_dir_L5_weight', 'MI_dir_L3_weight', 'MI_dir_L1_weight', 'MI_dir_L0.1_weight','MI_dir_L0.01_weight', 'H_L5_weight', 'H_L3_weight', 'H_L1_weight', 'H_L0.1_weight', 'H_L0.01_weight', 'HH_L5_weight', 'HH_L3_weight','HH_L0.01_weight','HH_jit_L5_weight', 'HH_jit_L3_weight','HH_jit_L1_weight', 'HH_jit_L0.1_weight', 'HH_jit_L0.01_weight', 'HpHp_L5_weight', 'HpHp_L3_weight', 'HpHp_L1_weight', 'HpHp_L0.01_weight','HpHp_L0.1_weight','HH_L5_pcc','HH_L5_std','HH_L5_radius','HH_L5_pcc','HH_L3_std','HpHp_L5_std','HpHp_L0.01_pcc','HpHp_L0.01_radius']]
<br>
df_new['s_mac_ip_mean'] = df_new[['MI_dir_L5_mean', 'MI_dir_L3_mean','MI_dir_L1_mean', 'MI_dir_L0.1_mean','MI_dir_L0.01_mean']].mean(axis=1)
df_new['s_mac_ip_weight'] = df_new[['MI_dir_L5_weight', 'MI_dir_L3_weight','MI_dir_L1_weight', 'MI_dir_L0.1_weight','MI_dir_L0.01_weight']].mean(axis=1)
df_new['s_ip_mean'] = df_new[['H_L5_mean', 'H_L3_mean','H_L1_mean', 'H_L0.1_mean','H_L0.01_mean']].mean(axis=1)
df_new['s_ip_weight'] = df_new[['H_L5_weight', 'H_L3_weight','H_L1_weight', 'H_L0.1_weight','H_L0.01_weight']].mean(axis=1)
df_new['channel_mean'] = df_new[['HH_L5_mean', 'HH_L3_mean','HH_L1_mean', 'HH_L0.1_mean','HH_L0.01_mean']].mean(axis=1)
df_new['channel_weight'] = df_new[['HH_L5_weight', 'HH_L3_weight', 'HH_L0.01_weight']].mean(axis=1)
df_new['channel_jitter_mean'] = df_new[['HH_jit_L5_mean', 'HH_jit_L3_mean','HH_jit_L1_mean', 'HH_jit_L0.1_mean','HH_jit_L0.01_mean']].mean(axis=1)
df_new['channel_jitter_weight'] = df_new[['HH_jit_L5_weight', 'HH_jit_L3_weight','HH_jit_L1_weight', 'HH_jit_L0.1_weight','HH_jit_L0.01_weight']].mean(axis=1)
df_new['socket_mean'] = df_new[['HpHp_L5_mean', 'HpHp_L3_mean','HpHp_L1_mean', 'HpHp_L0.1_mean','HpHp_L0.01_mean']].mean(axis=1)
df_new['socket_weight'] = df_new[['HpHp_L5_weight', 'HpHp_L3_weight', 'HpHp_L1_weight', 'HpHp_L0.01_weight', 'HpHp_L0.1_weight']].mean(axis=1)
df_new['s_mac_ip_var'] = df_new[['MI_dir_L5_variance', 'MI_dir_L3_variance','MI_dir_L1_variance', 'MI_dir_L0.1_variance','MI_dir_L0.01_variance']].mean(axis=1)
df_new['s_ip_var'] = df_new[['H_L5_variance', 'H_L3_variance','H_L1_variance', 'H_L0.1_variance','H_L0.01_variance']].mean(axis=1)
df_new['channel_jitter_var'] = df_new[['HH_jit_L0.01_variance']].mean(axis=1)
df_new['socket_covar'] = df_new[['HpHp_L5_covariance', 'HpHp_L3_covariance','HpHp_L1_covariance', 'HpHp_L0.1_covariance','HpHp_L0.01_covariance']].mean(axis=1)
df_b_traffic = df_new[['s_mac_ip_weight','s_ip_weight','channel_weight','channel_jitter_weight','socket_weight','s_mac_ip_mean','s_ip_mean','channel_mean','channel_jitter_mean','socket_mean','s_mac_ip_var','s_ip_var','channel_jitter_var','socket_covar','HH_L5_pcc','HH_L5_std','HH_L5_radius','HH_L5_pcc','HH_L3_std','HpHp_L5_std','HpHp_L0.01_pcc','HpHp_L0.01_radius','Botnet']]
df_b_traffic
<br>
print("Mean value:",df_new['s_mac_ip_mean'].mean()) 
print("Median value:",df_new['s_mac_ip_mean'].median()) 
print("Mode:",df_new['s_mac_ip_mean'].mode()) 
print("Mean value:",df_new['s_ip_mean'].mean()) 
print("Median value:",df_new['s_ip_mean'].median()) 
print("Mode:",df_new['s_ip_mean'].mode()) 
print("Mean value:",df_new['s_mac_ip_var'].mean()) 
print("Median value:",df_new['s_mac_ip_var'].median()) 
print("Mode:",df_new['s_mac_ip_var'].mode()) 
print("Mean value:",df_new['s_ip_var'].mean()) 
print("Median value:",df_new['s_ip_var'].median()) 
print("Mode:",df_new['s_ip_var'].mode()) 
print("Mean value:",df_new['channel_mean'].mean()) 
print("Median value:",df_new['channel_mean'].median()) 
print("Mode:",df_new['channel_mean'].mode()) 
print("Mean value:",df_new['channel_jitter_mean'].mean()) 
print("Median value:",df_new['channel_jitter_mean'].median()) 
print("Mode:",df_new['channel_jitter_mean'].mode()) 
print("Mean value:",df_new['socket_mean'].mean()) 
print("Median value:",df_new['socket_mean'].median()) 
print("Mode:",df_new['socket_mean'].mode()) 
print("Mode:",df_new['Botnet'].mode()) 
<br>
plt.subplots(figsize=(20,10))
sns.heatmap(df_b_traffic.corr(), annot=True, cmap="Reds")
df_attack = df_b_traffic.drop(['s_mac_ip_var','s_ip_var','s_ip_mean', 'socket_mean', 's_ip_weight', 'channel_jitter_weight', 'channel_mean','HH_L5_pcc','HH_L5_std','HH_L5_radius','HH_L5_pcc','HH_L3_std','HpHp_L5_std','HpHp_L0.01_pcc','HpHp_L0.01_radius'],axis=1)
df_attack
<br>
plt.subplots(figsize=(20,10))
sns.heatmap(df_attack.corr(), annot=True, cmap="Reds")
plt.subplots(figsize=(20,10))
sns.histplot(df_attack, x='s_mac_ip_weight', hue = 'Botnet', bins= 100)
plt.subplots(figsize=(20,10))
sns.boxplot(x = 'Botnet', y='s_mac_ip_weight', data=df_attack , width = 0.25, linewidth = 1)
plt.subplots(figsize=(20,10))
sns.scatterplot(data = df_new, x="s_mac_ip_weight", y="channel_weight", hue = 'Botnet')
mymap = {'benign': 0, 'Bashlite': 1, 'Mirai': 2}
df_ab = df_attack.applymap(lambda s: mymap.get(s) if s in mymap else s)
df_ab
<br>
g = sns.countplot(df_ab['Botnet'])
g.set_xticklabels(['Bashlite','Mirai','Benign'])
plt.show()
class_count_0, class_count_1, class_count_2 = df_ab['Botnet'].value_counts()
class_0 = df_ab[df_ab['Botnet'] == 0]
class_1 = df_ab[df_ab['Botnet'] == 1]
class_2 = df_ab[df_ab['Botnet'] == 2]
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)
print('class 2:', class_2.shape)
shuffled_df = df_ab.sample(frac=1,random_state=4)
<br>
bashlite_df = shuffled_df.loc[shuffled_df['Botnet'] == 0]

mirai_df = shuffled_df.loc[shuffled_df['Botnet'] == 1].sample(n=39100,random_state=42)
benign_df = shuffled_df.loc[shuffled_df['Botnet'] == 2].sample(n=39100,random_state=42)

df_ab = pd.concat([bashlite_df, mirai_df, benign_df])
plt.figure(figsize=(8, 8))
sns.countplot('Botnet', data=df_ab)
plt.title('Balanced Classes')
plt.show()
<br>
from sklearn.model_selection import train_test_split
from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(
    df_ab.iloc[:,0:7],df_ab.iloc[:,7], test_size=0.3)
    from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
rmse =0
mae =0
Accuracy_score=0
df_read = pd.DataFrame(columns = ['rmse','mae'])
df_read_as = pd.DataFrame(columns = ['Accuracy Score'])
#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

y_train = df_ab['Botnet']
X_train = df_ab.drop(['Botnet'], axis=1, inplace=False)
<br>
#Train the classifier.
bbc.fit(X_train, y_train)
preds = bbc.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
mae = mean_absolute_error(y_test, preds)
Accuracy_score= accuracy_score(y_test, preds)
df_read = [rmse, mae]
df_read_as = [Accuracy_score]
print(df_read)
print(" ")
print(df_read_as)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print('R^2 Value: ', regr.score(X_train,y_train))
print('Coefficients: ',regr.coef_)
print('Intercept:', regr.intercept_)
from sklearn.metrics import *
pred = regr.predict(X_test)
print('MAE: ', mean_absolute_error(y_test,pred))
print('RMSE: ', mean_squared_error(y_test,pred, squared = False))
print(pred)
# multiple models with holdout validation
from sklearn import neighbors
from sklearn.metrics import *
mm = pd.DataFrame(columns=['t_model', 'rmse', 'mae'])
index = 0
model = [linear_model.LinearRegression(), 
              linear_model.Lasso(alpha=1),
              linear_model.Ridge(alpha=1),
              neighbors.KNeighborsClassifier(n_neighbors=5)]
for t_o_m in model:
    t_o_m.fit(X_train, y_train)  
    pred = t_o_m.predict(X_test) 
    rmse = mean_squared_error(y_test, pred, squared=False)
    mae = mean_absolute_error(y_test, pred)
    mm.loc[index] = [t_o_m, rmse, mae]
    index+=1
mm
#multiple models with cross validation
from sklearn.model_selection import cross_val_score
mm_cv = pd.DataFrame(columns=['t_model', 'rmse'])
index = 0
model_cv = [linear_model.LinearRegression(), 
              linear_model.Lasso(alpha=1),
              linear_model.Ridge(alpha=1),
              neighbors.KNeighborsClassifier(n_neighbors=5)]
for tom_cv in model_cv:
    scores = cross_val_score(tom_cv, df_ab.iloc[:,0:7], df_ab.iloc[:,7], cv = 5, scoring= 'neg_root_mean_squared_error')
    mm_cv.loc[index] = [tom_cv, np.mean(scores)]
    index+=1
mm_cv
logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                                       class_weight=None, random_state=None, solver='saga', max_iter=200, multi_class= 'multinomial', 
                                                       verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
for i in range(7, 0, -1):
  X_train = df_ab.iloc[:,0:i]
  y_train = df_ab.iloc[:,7]
  X_test = df_ab.iloc[:,0:i]
  y_test = df_ab.iloc[:,7]
  logreg.fit(X_train,y_train)
  y_pred=logreg.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(accuracy)
  #KNN Classifier Backward predictor
Accuracy_score = 0
logreg = LogisticRegression()
for i in range(7, 0, -1):
  X_train = df_ab.iloc[:,0:i]
  y_train = df_ab.iloc[:,7]
  X_test = df_ab.iloc[:,0:i]
  y_test = df_ab.iloc[:,7]
  kNN_classifier = neighbors.KNeighborsClassifier(n_neighbors = 6)
  kNN_classifier.fit(X_train, y_train)
  pred = kNN_classifier.predict(X_test)
  accuracy = accuracy_score(y_test, pred)
  print(accuracy)
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3']))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
#import matplotlib.pyplot as plt  
#from sklearn.datasets import make_classification
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#X, y = make_classification(random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(
 #   df_ab.iloc[:,0:11],df_ab.iloc[:,11], test_size=0.3)
#clf = SVC(random_state=0)
#clf.fit(X_train, y_train)
#SVC(random_state=0)
#plot_confusion_matrix(clf, X_test, y_test)  
#plt.show()  
# multi-class classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# fit model
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
auc ={}

n_class = 3

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
    #auc[i] = metrics.roc_auc_score(y_test, pred_prob[:,i])
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Benign vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Scan vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Mirai vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);   
pipe_kNN = Pipeline (steps = [ ('model',neighbors.KNeighborsClassifier(n_neighbors=5))])
pipe_kNN_scaled = Pipeline (steps = [('model',neighbors.KNeighborsClassifier(n_neighbors=16))])
pipe_lr = Pipeline (steps = [ ('model',linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                                       class_weight=None, random_state=None, solver='saga', max_iter=100, multi_class= 'multinomial', 
                                                       verbose=0, warm_start=False, n_jobs=None, l1_ratio=None))])
pipe_NN = Pipeline (steps = [ ('model',MLPClassifier(learning_rate_init= 0.003, activation= 'relu'))])
pipe_DT = Pipeline (steps = [ ('model',DecisionTreeClassifier(max_depth=3))])
pipe_XG = Pipeline (steps = [('model', XGBClassifier(max_depth=2))])

rmse = 0
mae = 0
Accuracy_score = 0
df_read = pd.DataFrame(columns = ['rmse','mae'])
df_read_as = pd.DataFrame(columns = ['Accuracy Score'])
pipe_list = [pipe_kNN,pipe_kNN_scaled,pipe_lr,pipe_NN,pipe_DT, pipe_XG]
for pipe in pipe_list:
  model = pipe.fit(X_train, y_train)
  pred = model.predict(X_test)
  rmse = mean_squared_error(y_test, pred, squared=False)
  mae = mean_absolute_error(y_test, pred)
  Accuracy_score= accuracy_score(y_test, pred)
  df_read.loc[pipe] = [rmse, mae]
  df_read_as.loc[pipe] = [Accuracy_score]
print(df_read)
print(" ")
print(df_read_as)
<br>
<br>
