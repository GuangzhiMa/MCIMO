import torch
import time
from torch import nn, optim
from torch.nn import init

import scipy.stats as ss
import sys
sys.path.append("..") 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.nn as nn
import numpy as np
from sklearn import svm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import DF_interval, train_ch2, train_ch02, Data_split, Data_split_torch

import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inputlayer, hidder),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidder, hidder),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidder, outlayer),
        )

    def forward(self, inf):
        output = self.fc(inf)
        #return softmax(output)
        return output

loss = nn.CrossEntropyLoss()

np.random.seed(12345)
random.seed(12345)

# Mushroom
df_Mushrooms = pd.read_csv('California Mushrooms.csv',usecols=[1,4,5,6,7,8,9,10,11,12,13])
Mushrooms_labels0 = np.array([df_Mushrooms['Species class'][k] for k in range(245)])
Mushrooms_labels = torch.from_numpy(Mushrooms_labels0).type(torch.LongTensor)
df_fe = pd.read_csv('California Mushrooms.csv',usecols=[4,5,6,7,8,9,10,11,12,13])

feature_number = 5

Mushrooms_features = np.array(df_fe)

# oversampler
# KMeansSMOTE
from imblearn.over_sampling import KMeansSMOTE
kms = KMeansSMOTE(sampling_strategy={0: 30, 1:30, 2:30, 3: 30, 4:30, 5:30, 6: 30,
                                     7:30, 8:30, 9: 30, 10:30, 11:30, 12: 30, 13:30,
                                     14:30, 15: 30, 16:30}, random_state=120)

XMushrooms_resampled, yMushrooms_resampled = kms.fit_resample(Mushrooms_features, Mushrooms_labels0)

Mushrooms_features1 = np.zeros((XMushrooms_resampled.shape[0], 5))

for i in range(5):
    Mushrooms_features1[:, i] = (XMushrooms_resampled[:, 2*i] + XMushrooms_resampled[:, 2*i+1])/2

batch_size = 50

# Meanlogistic
Tmax = 1
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scorelog = np.zeros(Tmax)
Scorelog_auc = []
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_features1, yMushrooms_resampled, 0.2, 0.25, i)
    vali_best = 0
    vali_best0 = 0
    for j in range(len(C)):
        clf = LogisticRegression(penalty='l1', C=C[j], solver='liblinear', random_state=0, multi_class='ovr').fit(Mushrooms_train, y_train)
        clf.predict(Mushrooms_vali)
        score1 = np.zeros(T1)
        for k in range(T1):
            score1[k] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
        auc_score1 = []
        for k in range(T1):
            prob = clf.predict_proba(Mushrooms_vali)
            if len(np.unique(y_vali)) == prob.shape[1]:
                auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                auc_score1.append(auc)
        auc_score1 = np.array(auc_score1)
        if vali_best < np.mean(score1):
            vali_best = np.mean(score1)
            c_bestlog = C[j]
        if vali_best0 < np.mean(auc_score1):
            vali_best0 = np.mean(auc_score1)
            c_bestlog1 = C[j]
    clfbest = LogisticRegression(penalty='l1', C=c_bestlog, solver='liblinear', random_state=0, multi_class='ovr').fit(Mushrooms_train, y_train)
    clfbest0 = LogisticRegression(penalty='l1', C=c_bestlog1, solver='liblinear', random_state=0, multi_class='ovr').fit(Mushrooms_train, y_train)
    Scorelog[i] = balanced_accuracy_score(clfbest.predict(Mushrooms_test), y_test)
    prob = clfbest0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        Scorelog_auc.append(auc)
Scorelog_auc = np.array(Scorelog_auc)

# MeanSVM

C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
ScoreMSVM = np.zeros(Tmax)
ScoreMSVM_auc = []
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_features1, yMushrooms_resampled, 0.2, 0.25, i)
    vali_best = 0
    vali_best0 = 0
    for j in range(len(C)):
        clf = svm.SVC(kernel='linear', C=C[j], probability= True)
        score_lin = np.zeros(T1)
        for jj in range(T1):
            clf.fit(Mushrooms_train, y_train)
            score_lin[jj] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
        auc_lin = []
        for k in range(T1):
            prob = clf.predict_proba(Mushrooms_vali)
            if len(np.unique(y_vali)) == prob.shape[1]:
                auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                auc_lin.append(auc)
        auc_lin = np.array(auc_lin)
        clf = svm.SVC(kernel='rbf', C=C[j], probability= True)
        score_rbf = np.zeros(T1)
        for jj in range(T1):
            clf.fit(Mushrooms_train, y_train)
            score_rbf[jj] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
        auc_rbf = []
        for k in range(T1):
            prob = clf.predict_proba(Mushrooms_vali)
            if len(np.unique(y_vali)) == prob.shape[1]:
                auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                auc_rbf.append(auc)
        auc_rbf = np.array(auc_rbf)
        clf = svm.SVC(kernel='poly', C=C[j], probability= True)
        score_poly = np.zeros(T1)
        for jj in range(T1):
            clf.fit(Mushrooms_train, y_train)
            score_poly[jj] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
        auc_poly = []
        for k in range(T1):
            prob = clf.predict_proba(Mushrooms_vali)
            if len(np.unique(y_vali)) == prob.shape[1]:
                auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                auc_poly.append(auc)
        auc_poly = np.array(auc_poly)
        score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
        if vali_best < score:
            vali_best = score
            c_best = C[j]
        score1 = np.max((np.mean(auc_lin), np.mean(auc_rbf), np.mean(auc_poly)))
        if vali_best0 < score1:
            vali_best0 = score1
            c_best_auc = C[j]
    AUC = []
    clf = svm.SVC(kernel='linear', C=c_best, probability= True)
    clf.fit(Mushrooms_train, y_train)
    s0 = balanced_accuracy_score(clf.predict(Mushrooms_test), y_test)
    clf0 = svm.SVC(kernel='linear', C=c_best_auc, probability= True)
    clf0.fit(Mushrooms_train, y_train)
    prob = clf0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        AUC.append(auc)
    clf = svm.SVC(kernel='rbf', C=c_best, probability= True)
    clf.fit(Mushrooms_train, y_train)
    s1 = balanced_accuracy_score(clf.predict(Mushrooms_test), y_test)
    clf0 = svm.SVC(kernel='rbf', C=c_best_auc, probability= True)
    clf0.fit(Mushrooms_train, y_train)
    prob = clf0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        AUC.append(auc)
    clf = svm.SVC(kernel='poly', C=c_best, probability= True)
    clf.fit(Mushrooms_train, y_train)
    s2 = balanced_accuracy_score(clf.predict(Mushrooms_test), y_test)
    clf0 = svm.SVC(kernel='poly', C=c_best_auc, probability= True)
    clf0.fit(Mushrooms_train, y_train)
    prob = clf0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        AUC.append(auc)
    ScoreMSVM[i] = np.max((s0, s1, s2))
    AUC = np.array(AUC)
    if np.max(AUC) > 0:
        ScoreMSVM_auc.append(np.max(AUC))
ScoreMSVM_auc = np.array(ScoreMSVM_auc)

# MeanCART

C =np.arange(1,11,1)
Scoretree = np.zeros(Tmax)
Scoretree_auc = []
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_features1, yMushrooms_resampled, 0.2, 0.25, i)
    vali_best = 0
    vali_best0 = 0
    for j in range(len(C)):
        clf = DecisionTreeClassifier(min_samples_leaf=C[j], max_depth=10, random_state=0).fit(Mushrooms_train, y_train)
        clf.predict(Mushrooms_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
        auc_score = []
        for k in range(T1):
            prob = clf.predict_proba(Mushrooms_vali)
            if len(np.unique(y_vali)) == prob.shape[1]:
                auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                auc_score.append(auc)
        auc_score = np.array(auc_score)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_besttree = C[j]
        if vali_best0 < np.mean(auc_score):
            vali_best = np.mean(auc_score)
            c_besttree0 = C[j]
    clfbest = DecisionTreeClassifier(min_samples_leaf=c_besttree.astype(np.int64), max_depth=10, random_state=0).fit(Mushrooms_train, y_train)
    Scoretree[i] = balanced_accuracy_score(clfbest.predict(Mushrooms_test), y_test)
    clfbest = DecisionTreeClassifier(min_samples_leaf=c_besttree0.astype(np.int64), max_depth=10, random_state=0).fit(Mushrooms_train, y_train)
    prob = clfbest.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        Scoretree_auc.append(auc)
Scoretree_auc = np.array(Scoretree_auc)

# MeanRanF

C =np.arange(1,11,1)
Tr = np.arange(10,210,10)
Scoreran = np.zeros(Tmax)
Scoreran_auc = []
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_features1, yMushrooms_resampled, 0.2, 0.25, i)
    vali_best = 0
    vali_best0 = 0
    for j in range(len(C)):
        for k in range(len(Tr)):
            clf = RandomForestClassifier(n_estimators=Tr[k], min_samples_leaf=C[j], random_state=0).fit(Mushrooms_train, y_train)
            clf.predict(Mushrooms_vali)
            score = np.zeros(T1)
            for ii in range(T1):
                score[ii] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
            auc_score = []
            for k in range(T1):
                prob = clf.predict_proba(Mushrooms_vali)
                if len(np.unique(y_vali)) == prob.shape[1]:
                    auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                    auc_score.append(auc)
            auc_score = np.array(auc_score)
            if vali_best < np.mean(score):
                vali_best = np.mean(score)
                c_bestran = C[j]
                tr_bestran = Tr[k]
            if vali_best0 < np.mean(auc_score):
                vali_best0 = np.mean(auc_score)
                c_bestran0 = C[j]
                tr_bestran0 = Tr[k]
    clfbest = RandomForestClassifier(n_estimators=tr_bestran.astype(np.int64), min_samples_leaf=c_bestran.astype(np.int64), random_state=0).fit(Mushrooms_train, y_train)
    Scoreran[i] = balanced_accuracy_score(clfbest.predict(Mushrooms_test), y_test)
    clfbest = RandomForestClassifier(n_estimators=tr_bestran0.astype(np.int64), min_samples_leaf=c_bestran0.astype(np.int64), random_state=0).fit(Mushrooms_train, y_train)
    prob = clfbest.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        Scoreran_auc.append(auc)
Scoreran_auc = np.array(Scoreran_auc)

# DFMLP

hidder = 150
inputlayer = 5
outlayer = 17

net = Net()

T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
B = np.arange(0,1.04,0.05)
Scoredfmlp = np.zeros(Tmax)
Scoredfmlp_auc = np.zeros(Tmax)
for i in range(Tmax):
    vali_best = 0
    vali_best0 = 0
    for m in range(len(B)):
        beta = B[m]
        Mushrooms_df = DF_interval(XMushrooms_resampled, beta, feature_number)
        Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
        train_iter, vali_iter, test_iter = Data_split_torch(Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test, batch_size)
        for j in range(LR.shape[0]):
            for k in range(n_epoch.shape[0]):
                vali_Acc = np.zeros(T1)
                vali_Auc = np.zeros(T1)
                for jj in range(T1):
                    for params in net.parameters():
                        init.normal_(params, mean=0, std=0.01)
                    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR[j], betas=(0.9, 0.999), eps=1e-08)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                    train_Acc, test_Acc, train_Auc, test_Auc = train_ch2(net, train_iter, vali_iter, loss, batch_size, optimizer, scheduler, device, n_epoch[k])
                    vali_Acc[jj] = test_Acc[-1]
                    vali_Auc[jj] = test_Auc[-1]
                if vali_best < np.mean(vali_Acc):
                    vali_best = np.mean(vali_Acc)
                    lr_best = LR[j]
                    epoch_best = n_epoch[k]
                    beta_best = B[m]
                if vali_best0 < np.mean(vali_Auc):
                    vali_best0 = np.mean(vali_Auc)
                    lr_best0 = LR[j]
                    epoch_best0 = n_epoch[k]
                    beta_best0 = B[m]
    Mushrooms_df = DF_interval(XMushrooms_resampled, beta_best, feature_number)
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test, batch_size)
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_Acc0, test_Acc0, train_Auc0, test_Auc0 = train_ch2(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
    Scoredfmlp[i] = test_Acc0[-1]
    Mushrooms_df = DF_interval(XMushrooms_resampled, beta_best0, feature_number)
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test, batch_size)
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr_best0, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_Acc0, test_Acc0, train_Auc0, test_Auc0 = train_ch2(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best0)
    Scoredfmlp_auc[i] = test_Auc0[-1]

beta = beta_best
Mushrooms_df = DF_interval(XMushrooms_resampled, beta, feature_number)

train_Acc0 = []
test_Acc0 = []
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test, batch_size)
    for params in net.parameters():
               init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_ch02(train_Acc0, test_Acc0, train_Auc0, test_Auc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)

train_Acc00=np.array(train_Acc0)
test_Acc00=np.array(test_Acc0)
train_Acc00.resize((Tmax,epoch_best))
test_Acc00.resize((Tmax,epoch_best))
train_Accmean00 = np.mean(train_Acc00,0)
train_Accstd00 = np.std(train_Acc00,0)
test_Accmean00 = np.mean(test_Acc00,0)
test_Accstd00 = np.std(test_Acc00,0)
np.max(test_Accmean00)
test_Accstd00[np.where(test_Accmean00==np.max(test_Accmean00))]


plt.plot(range(1, epoch_best + 1),train_Accmean00,"r-",linewidth=2)
plt.plot(range(1, epoch_best + 1),test_Accmean00,"b--",linewidth=2)
plt.fill_between(range(1, epoch_best + 1), train_Accmean00 - train_Accstd00, train_Accmean00 + train_Accstd00 , facecolor='red', alpha=0.1)
plt.fill_between(range(1, epoch_best + 1), test_Accmean00 - test_Accstd00, test_Accmean00 + test_Accstd00, facecolor='blue', alpha=0.1)
plt.xlabel("Epochs")
plt.ylabel("Accuracy(%)")
plt.legend(['DFMLP train', 'DFMLP test'], loc='lower right')
plt.ylim((0.02, 1.05))
my_y_ticks = np.arange(0.02, 1.05, 0.08)
plt.yticks(my_y_ticks)
plt.savefig('DFMLP_mush.pdf', bbox_inches='tight')
plt.show()

beta = beta_best0
Mushrooms_df = DF_interval(XMushrooms_resampled, beta, feature_number)

train_Auc0 = []
test_Auc0 = []
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test, batch_size)
    for params in net.parameters():
               init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best0, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_ch02(train_Acc0, test_Acc0, train_Auc0, test_Auc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best0)

train_Auc00=np.array(train_Auc0)
test_Auc00=np.array(test_Auc0)
train_Auc00.resize((Tmax,epoch_best0))
test_Auc00.resize((Tmax,epoch_best0))
train_Aucmean00 = np.mean(train_Auc00,0)
train_Aucstd00 = np.std(train_Auc00,0)
test_Aucmean00 = np.mean(test_Auc00,0)
test_Aucstd00 = np.std(test_Auc00,0)
np.max(test_Aucmean00)
test_Aucstd00[np.where(test_Aucmean00==np.max(test_Aucmean00))]

plt.plot(range(1, epoch_best0 + 1),train_Aucmean00,"r-",linewidth=2)
plt.plot(range(1, epoch_best0 + 1),test_Aucmean00,"b--",linewidth=2)
plt.fill_between(range(1, epoch_best0 + 1), train_Aucmean00 - train_Aucstd00, train_Aucmean00 + train_Aucstd00 , facecolor='red', alpha=0.1)
plt.fill_between(range(1, epoch_best0 + 1), test_Aucmean00 - test_Aucstd00, test_Aucmean00 + test_Aucstd00, facecolor='blue', alpha=0.1)
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.legend(['DFMLP train', 'DFMLP test'], loc='lower right')
plt.ylim((0.02, 1.05))
my_y_ticks = np.arange(0.02, 1.05, 0.08)
plt.yticks(my_y_ticks)
plt.savefig('DFMLP_mush_auc.pdf', bbox_inches='tight')
plt.show()

# DFSVM
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
B = np.arange(0,1.04,0.05)
Scoredfsvm = np.zeros(Tmax)
Scoredfsvm_auc = []
for i in range(Tmax):
    beta_best_auc = 0.5
    c_best_auc = 1
    vali_best = 0
    vali_best0 = 0
    for m in range(len(B)):
        Mushrooms_df = DF_interval(XMushrooms_resampled, B[m], feature_number)
        Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
        for j in range(len(C)):
            clf = svm.SVC(kernel='linear', C=C[j], probability=True)
            score_lin = np.zeros(T1)
            for jj in range(T1):
                clf.fit(Mushrooms_train, y_train)
                score_lin[jj] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
            auc_lin = []
            for k in range(T1):
                prob = clf.predict_proba(Mushrooms_vali)
                if len(np.unique(y_vali)) == prob.shape[1]:
                    auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                    auc_lin.append(auc)
            auc_lin = np.array(auc_lin)
            clf = svm.SVC(kernel='rbf', C=C[j], probability=True)
            score_rbf = np.zeros(T1)
            for jj in range(T1):
                clf.fit(Mushrooms_train, y_train)
                score_rbf[jj] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
            auc_rbf = []
            for k in range(T1):
                prob = clf.predict_proba(Mushrooms_vali)
                if len(np.unique(y_vali)) == prob.shape[1]:
                    auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                    auc_rbf.append(auc)
            auc_rbf = np.array(auc_rbf)
            clf = svm.SVC(kernel='poly', C=C[j], probability=True)
            score_poly = np.zeros(T1)
            for jj in range(T1):
                clf.fit(Mushrooms_train, y_train)
                score_poly[jj] = balanced_accuracy_score(clf.predict(Mushrooms_vali), y_vali)
            auc_poly = []
            for k in range(T1):
                prob = clf.predict_proba(Mushrooms_vali)
                if len(np.unique(y_vali)) == prob.shape[1]:
                    auc = roc_auc_score(y_vali, prob, multi_class='ovr')
                    auc_poly.append(auc)
            auc_poly = np.array(auc_poly)
            score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
            if vali_best < score:
                vali_best = score
                c_best = C[j]
                beta_best = B[m]
            score1 = np.max((np.mean(auc_lin), np.mean(auc_rbf), np.mean(auc_poly)))
            if vali_best0 < score1:
                vali_best0 = score1
                c_best_auc = C[j]
                beta_best_auc = B[m]
    Mushrooms_df = DF_interval(XMushrooms_resampled, beta_best, feature_number)
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
    clf = svm.SVC(kernel='linear', C=c_best, probability=True)
    clf.fit(Mushrooms_train, y_train)
    s0 = balanced_accuracy_score(clf.predict(Mushrooms_test), y_test)
    clf = svm.SVC(kernel='rbf', C=c_best, probability=True)
    clf.fit(Mushrooms_train, y_train)
    s1 = balanced_accuracy_score(clf.predict(Mushrooms_test), y_test)
    clf = svm.SVC(kernel='poly', C=c_best, probability=True)
    clf.fit(Mushrooms_train, y_train)
    s2 = balanced_accuracy_score(clf.predict(Mushrooms_test), y_test)
    Scoredfsvm[i] = np.max((s0, s1, s2))
    Mushrooms_df = DF_interval(XMushrooms_resampled, beta_best_auc, feature_number)
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_df, yMushrooms_resampled, 0.2, 0.25, i)
    AUC = []
    clf0 = svm.SVC(kernel='linear', C=c_best_auc, probability=True)
    clf0.fit(Mushrooms_train, y_train)
    prob = clf0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        AUC.append(auc)
    clf0 = svm.SVC(kernel='rbf', C=c_best_auc, probability=True)
    clf0.fit(Mushrooms_train, y_train)
    prob = clf0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        AUC.append(auc)
    clf0 = svm.SVC(kernel='poly', C=c_best_auc, probability=True)
    clf0.fit(Mushrooms_train, y_train)
    prob = clf0.predict_proba(Mushrooms_test)
    if len(np.unique(y_test)) == prob.shape[1]:
        auc = roc_auc_score(y_test, prob, multi_class='ovr')
        AUC.append(auc)
    AUC = np.array(AUC)
    if np.max(AUC) > 0:
        Scoredfsvm_auc.append(np.max(AUC))
Scoredfsvm_auc = np.array(Scoredfsvm_auc)

# MeanMLP
T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
Scoremmlp = np.zeros(Tmax)
Scoremmlp_auc = np.zeros(Tmax)
for i in range(Tmax):
    Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test = Data_split(Mushrooms_features1, yMushrooms_resampled, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(Mushrooms_train, Mushrooms_vali, Mushrooms_test, y_train, y_vali, y_test, batch_size)
    vali_best = 0
    vali_best0 = 0
    for j in range(LR.shape[0]):
        for k in range(n_epoch.shape[0]):
            vali_Acc = np.zeros(T1)
            vali_Auc = np.zeros(T1)
            for jj in range(T1):
                for params in net.parameters():
                    init.normal_(params, mean=0, std=0.01)
                optimizer = torch.optim.Adam(params=net.parameters(), lr=LR[j], betas=(0.9, 0.999), eps=1e-08)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                train_Acc, test_Acc, train_Auc, test_Auc = train_ch2(net, train_iter, vali_iter, loss, batch_size, optimizer, scheduler, device, n_epoch[k])
                vali_Acc[jj] = test_Acc[-1]
                vali_Auc[jj] = test_Auc[-1]
            if vali_best < np.mean(vali_Acc):
                vali_best = np.mean(vali_Acc)
                lr_best = LR[j]
                epoch_best = n_epoch[k]
            if vali_best0 < np.mean(vali_Auc):
                vali_best0 = np.mean(vali_Auc)
                lr_best0 = LR[j]
                epoch_best0 = n_epoch[k]
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_Acc0, test_Acc0, train_Auc0, test_Auc0 = train_ch2(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
    Scoremmlp[i] = test_Acc0[-1]
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr_best0, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_Acc0, test_Acc0, train_Auc0, test_Auc0 = train_ch2(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best0)
    Scoremmlp_auc[i] = test_Auc0[-1]


print('Test Accuracy of Meanlogistic : %.4f ± %.4f.' %(np.mean(Scorelog), np.std(Scorelog)))
print('Test Accuracy of MeanSVM : %.4f ± %.4f.' %(np.mean(ScoreMSVM), np.std(ScoreMSVM)))
print('Test Accuracy of MeanDecisiontree : %.4f ± %.4f.' %(np.mean(Scoretree), np.std(Scoretree)))
print('Test Accuracy of MeanRandomForest : %.4f ± %.4f.' %(np.mean(Scoreran), np.std(Scoreran)))
print('Test Accuracy of MeanMLP : %.4f ± %.4f.' %(np.mean(Scoremmlp), np.std(Scoremmlp)))
print('Test Accuracy of DF-SVM : %.4f ± %.4f.' %(np.mean(Scoredfsvm), np.std(Scoredfsvm)))
print('Test Accuracy of DF-MLP : %.4f ± %.4f.' %(np.mean(Scoredfmlp), np.std(Scoredfmlp)))


print('AUC of Meanlogistic : %.4f ± %.4f.' %(np.mean(Scorelog_auc), np.std(Scorelog_auc)))
print('AUC of MeanSVM : %.4f ± %.4f.' %(np.mean(ScoreMSVM_auc), np.std(ScoreMSVM_auc)))
print('AUC of MeanDecisiontree : %.4f ± %.4f.' %(np.mean(Scoretree_auc), np.std(Scoretree_auc)))
print('AUC of MeanRandomForest : %.4f ± %.4f.' %(np.mean(Scoreran_auc), np.std(Scoreran_auc)))
print('AUC of MeanMLP : %.4f ± %.4f.' %(np.mean(Scoremmlp_auc), np.std(Scoremmlp_auc)))
print('AUC of DF-SVM : %.4f ± %.4f.' %(np.mean(Scoredfsvm_auc), np.std(Scoredfsvm_auc)))
print('AUC of DF-MLP : %.4f ± %.4f.' %(np.mean(Scoredfmlp_auc), np.std(Scoredfmlp_auc)))