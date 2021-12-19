#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn import under_sampling, over_sampling
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dataset

 

# In[ ]:


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.14, random_state=0)

#n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
#X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.14, shuffle=False)


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 10)

# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [5,10]}]

sm = SMOTE(random_state=None)
X_res, y_res = sm.fit_resample(X_train, y_train.ravel())

# specify model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_res, y_res)


# In[ ]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,8))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


# model with optimal hyperparameters

# model
#model = SVC(C=5, gamma=0.001, kernel="rbf")

#model.fit(X_res, y_res)
#y_pred = model.predict(X_test)

estimator = SVC(C=5, gamma=0.001, kernel="rbf", class_weight="balanced")
classifier = OneVsRestClassifier(estimator)
classifier.fit(X_res, y_res)

y_pred = classifier.predict(X_test)


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix

class_names = [0,1,2,3,4,5,6,7,8,9]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()

#data generation for imbalanced, balanced, symmetric, asymmetric and noisy data

#imbalanced dataset
dataset = DatasetGenerator(batchSize=args.batch_size,
                               dataPath=args.data_path,
                               numOfWorkers=args.data_nums_workers,
                               noise_rate=args.nr,
                               asym=args.asym,
                               imb_type = exp,
                               imb_factor = 0.1,
                               seed=args.seed,)
dataLoader = dataset.getDataLoader()
for images, labels in tqdm(data_loader["train_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)

for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images_test, labels_test = images.to(device), labels.to(device)

estimator = SVC(C=5, gamma=0.001, kernel="rbf", class_weight="balanced")
classifier = OneVsRestClassifier(estimator)
classifier.fit(images, labels)

y_pred = classifier.predict(images_test)

# metrics
print("accuracy", metrics.accuracy_score(labels_test, y_pred), "\n")
print(metrics.confusion_matrix(labels_test, y_pred), "\n")
cnf_matrix = metrics.confusion_matrix(labels_test, y_pred)

#imbalanced asymmetric noise dataset
dataset = DatasetGenerator(batchSize=args.batch_size,
                               dataPath=args.data_path,
                               numOfWorkers=args.data_nums_workers,
                               noise_rate=0.4,
                               asym=true,
                               imb_type = exp,
                               imb_factor = 0.1,
                               seed=args.seed,)
dataLoader = dataset.getDataLoader()
for images, labels in tqdm(data_loader["train_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)

for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images_test, labels_test = images.to(device), labels.to(device)

estimator = SVC(C=5, gamma=0.001, kernel="rbf", class_weight="balanced")
classifier = OneVsRestClassifier(estimator)
classifier.fit(images, labels)

y_pred = classifier.predict(images_test)

# metrics
print("accuracy", metrics.accuracy_score(labels_test, y_pred), "\n")
print(metrics.confusion_matrix(labels_test, y_pred), "\n")
cnf_matrix = metrics.confusion_matrix(labels_test, y_pred)

#imbalanced symmetric noise dataset
dataset = DatasetGenerator(batchSize=args.batch_size,
                               dataPath=args.data_path,
                               numOfWorkers=args.data_nums_workers,
                               noise_rate=0.4,
                               asym=false,
                               imb_type = exp,
                               imb_factor = 0.1,
                               seed=args.seed,)
dataLoader = dataset.getDataLoader()
for images, labels in tqdm(data_loader["train_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)

for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images_test, labels_test = images.to(device), labels.to(device)

estimator = SVC(C=5, gamma=0.001, kernel="rbf", class_weight="balanced")
classifier = OneVsRestClassifier(estimator)
classifier.fit(images, labels)

y_pred = classifier.predict(images_test)

# metrics
print("accuracy", metrics.accuracy_score(labels_test, y_pred), "\n")
print(metrics.confusion_matrix(labels_test, y_pred), "\n")
cnf_matrix = metrics.confusion_matrix(labels_test, y_pred)

#balanced symmetric noise dataset
dataset = DatasetGenerator(batchSize=args.batch_size,
                               dataPath=args.data_path,
                               numOfWorkers=args.data_nums_workers,
                               noise_rate=0.4,
                               asym=false,
                               imb_type = none,
                               imb_factor = 0,
                               seed=args.seed,)
dataLoader = dataset.getDataLoader()
for images, labels in tqdm(data_loader["train_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)

for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images_test, labels_test = images.to(device), labels.to(device)

estimator = SVC(C=5, gamma=0.001, kernel="rbf", class_weight="balanced")
classifier = OneVsRestClassifier(estimator)
classifier.fit(images, labels)

y_pred = classifier.predict(images_test)

# metrics
print("accuracy", metrics.accuracy_score(labels_test, y_pred), "\n")
print(metrics.confusion_matrix(labels_test, y_pred), "\n")
cnf_matrix = metrics.confusion_matrix(labels_test, y_pred)

#balanced asymmetric noise dataset
dataset = DatasetGenerator(batchSize=args.batch_size,
                               dataPath=args.data_path,
                               numOfWorkers=args.data_nums_workers,
                               noise_rate=0.4,
                               asym=true,
                               imb_type = none,
                               imb_factor = 0,
                               seed=args.seed,)
dataLoader = dataset.getDataLoader()
for images, labels in tqdm(data_loader["train_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)

for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images_test, labels_test = images.to(device), labels.to(device)

estimator = SVC(C=5, gamma=0.001, kernel="rbf", class_weight="balanced")
classifier = OneVsRestClassifier(estimator)
classifier.fit(images, labels)

y_pred = classifier.predict(images_test)

# metrics
print("accuracy", metrics.accuracy_score(labels_test, y_pred), "\n")
print(metrics.confusion_matrix(labels_test, y_pred), "\n")
cnf_matrix = metrics.confusion_matrix(labels_test, y_pred)
