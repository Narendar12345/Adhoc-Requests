import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

count_vect = CountVectorizer()




# import data 

data = pd.read_csv('JO_data.csv')

print(data.head())

counter = Counter(data['Job_Description'].tolist())

# print(counter)

description_list = data['Job_Description'].tolist()

# print(description_list)

count_vect1 = count_vect.fit(description_list)

# print(count_vect1.get_feature_names())

feature_dict = count_vect1.vocabulary_

X_train_counts = count_vect1.transform(description_list)

print('')

print('The shape of the dataset after count vectorisation :',X_train_counts.shape)

## Since PCA throws a sparse matrix error , moving on with Truncated SVD:

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5)

fit_svd = svd.fit(X_train_counts)

transform_svd = svd.transform(X_train_counts)

print('')
print('********************')
print('The shape of the dataset after SVD :',transform_svd.shape)




train_x, test_x, train_y, test_y = train_test_split(transform_svd, data['Class_Customercare'], test_size=0.3)


print(train_x)

print(svd.explained_variance_)

# Building a RF Classifier instead of a Naive Bayes:

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier


# random forest model creation
rfc = RandomForestClassifier()
rfc_fit = rfc.fit(train_x,train_y)


# predictions
rfc_predict = rfc_fit.predict(test_x)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

rfc_cv_score = cross_val_score(rfc,transform_svd,data['Class_Customercare'], cv=5, scoring= 'roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(test_y, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(test_y, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())







## Saving the Count vectorizer model




Pkl_Filename = "Text_classifier_Count_Vec_1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(count_vect1, file)
    print("The Count vectorizer File has been saved in your desktop")

## Saving the SVD model

Pkl_Filename = "Text_classifier_SVD_1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(fit_svd, file)
    print("The SVD pkl File has been saved in your desktop")


## Saving the RF model

Pkl_Filename = "Text_classifier_Random_Forest_1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rfc_fit, file)
    print("The Random Forest Classifier Model has been saved in your desktop")






