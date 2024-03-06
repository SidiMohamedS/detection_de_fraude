
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.metrics import pair_confusion_matrix

# from sklearn.metrics import auc, roc_curve,confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("fraude_data.csv")



data.shape


data.info()

data.isnull().sum()

data.describe()

data['nameDest'].nunique()


data['nameOrig'].nunique()

data= data.drop(labels = ['index','nameOrig', 'nameDest'], axis=1)

data.sample(5)


data['isFraud'].value_counts()


plt.figure(figsize = (12, 8))

sns.countplot(x='isFraud', data = data);


data['isFlaggedFraud'].value_counts()

plt.figure(figsize = (10, 6))

sns.catplot(x = 'type', y = 'amount', estimator = sum,
            hue = 'isFraud' , col = 'isFlaggedFraud',
            data = data);



plt.figure(figsize =(10,6))
plt.ylim([0, 8000])
sns.histplot(data['step'], kde=True)



data['step'] =data['step']  %24
data.sample(5)

plt.figure(figsize = (12, 6))

sns.lineplot(x = 'step', y = 'amount', hue = 'type', ci = None,
             estimator = 'mean', data = data);



sns.displot(data = data, x = 'step', col = 'isFraud');


plt.figure(figsize = (12, 6))

sns.countplot(x = 'type', hue = 'isFraud', data = data)



plt.figure(figsize = (12, 6))

sns.barplot(x = 'type', y = 'amount', estimator = sum, hue = 'isFraud', data = data);



data =data.loc[(data.type == 'TRANSFER') |   (data.type == 'CASH_OUT')]
data.shape


data = pd.concat([data,
                              pd.get_dummies(data['type'],
                                             prefix = 'type', drop_first = True)],
                              axis = 1)

data.head()


data = data.drop(labels = ['type','isFlaggedFraud'], axis = 1)
data.head()

data['isFraud'].value_counts()



data['origBalanceDiscrepancy'] = \
   data.newbalanceOrig + data.amount -data.oldbalanceOrg

data['destBalanceDiscrepancy'] = \
    data.oldbalanceDest + data.amount - data.newbalanceDest

sns.catplot(x = 'isFraud', y = 'origBalanceDiscrepancy', estimator = sum,
            hue = 'type_TRANSFER', data = data, aspect = 2)


sns.catplot(x = 'isFraud', y = 'destBalanceDiscrepancy', estimator = sum,
            hue = 'type_TRANSFER' , data =data, aspect = 2)


data.to_csv('preprocessed_transaction_data.csv', index=False)

data_finance= pd.read_csv('preprocessed_transaction_data.csv')
data_finance.head()



X = data_finance.drop(['isFraud'], axis = 1)

y = data_finance['isFraud']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

X_train.shape, X_test.shape




logistic_clf = LogisticRegression()

logistic_clf.fit(X_train, y_train)


y_pred_logistic = logistic_clf.predict(X_test)

y_pred_logistic[:5]





print('Logistic Regression\n')

print('Accuracy: ', accuracy_score(y_test, y_pred_logistic))
print('Precision: ', precision_score(y_test, y_pred_logistic))
print('Recall: ', recall_score(y_test, y_pred_logistic))



gnb_clf = GaussianNB()

gnb_clf.fit(X_train, y_train)

y_pred_gnb = gnb_clf.predict(X_test)

y_pred_gnb[:5]

print('Naive Bayes\n')

print('Accuracy: ', accuracy_score(y_test, y_pred_gnb))
print('Precision: ', precision_score(y_test, y_pred_gnb))
print('Recall: ', recall_score(y_test, y_pred_gnb))



svc_clf = SVC()

svc_clf.fit(X_train, y_train)

y_pred_svc = svc_clf.predict(X_test)

y_pred_svc[:5]

print('Support Vector Classifier\n')

print('Accuracy: ', accuracy_score(y_test, y_pred_svc))
print('Precision: ', precision_score(y_test, y_pred_svc))
print('Recall: ', recall_score(y_test, y_pred_svc))



rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

y_pred_rf[:5]

print('Random Forest\n')

print('Accuracy: ', accuracy_score(y_test, y_pred_rf))
print('Precision: ', precision_score(y_test, y_pred_rf))
print('Recall: ', recall_score(y_test, y_pred_rf))




fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_logistic)

AUC_logistic = auc(fpr_logistic, tpr_logistic)

print('AUC for Logistic Regression :', AUC_logistic)

fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_gnb)

AUC_gnb = auc(fpr_gnb, tpr_gnb)

print('AUC for Naive Bayes :', AUC_gnb)

""" On peut voir que le l'AUC pour le modèle Naive Bayes est plus bas seulement environ 0.71. Les metriques d'exactitude, de précison et de rappel ont également indiqué que le modèle Naive Bayes n'était pas aussi bon que le modèle de régression logistique"""

fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred_svc)

AUC_svc = auc(fpr_svc, tpr_svc)

print('AUC for Support Vector Classifier :', AUC_svc)

""" Le modèle Support Vector Classifier a un AUC de 0.79. Mieux que Naive Bayes, mais pas aussi bon que la Régression logistique."""

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

AUC_rf = auc(fpr_rf, tpr_rf)

print('AUC for Random Forest :', AUC_rf)



plt.figure(figsize=(12, 8))

plt.plot(fpr_logistic, tpr_logistic, color = 'purple',
         label = 'Logistic Regression (area = %0.2f)' % AUC_logistic)

plt.plot(fpr_gnb, tpr_gnb, color = 'blue',
         label = 'Naive Bayes (area = %0.2f)' % AUC_gnb)

plt.plot(fpr_svc, tpr_svc, color = 'orange',
         label = 'Support Vector Classifier (area = %0.2f)' % AUC_svc)

plt.plot(fpr_rf, tpr_rf, color = 'green',
         label = 'Random Forest (area = %0.2f)' % AUC_rf)

plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')

plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.0])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curves for different ML models')
plt.legend(loc = 'lower right')

# # Matrice de confusion
# conf_matrix = confusion_matrix(y_test, y_pred_rf)
# print(conf_matrix )
# # Affichage de la matrice de confusion
# plot_confusion_matrix(rf_clf, X_test, y_test, cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()

