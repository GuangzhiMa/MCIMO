# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 08:58:00 2022

@author: 14025959_admin
"""
import torch
import time
from torch import nn, optim
from torch.nn import init
from sklearn.datasets import make_blobs

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import DF_fuzzy, train_ch, train_ch0, Data_split, Data_split_torch

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

np.random.seed(123)
random.seed(123)

#sys data

#data generate
num_inputs = 20
num_labels = 5
sys_examples = 2000
batch_size = 1000
sys_features, sys_labels = make_blobs(n_samples=sys_examples, centers=5, n_features=20, cluster_std=[0.5, 1.2, 0.8, 1, 2], center_box=(6.5,10.0))
bais1 = np.random.uniform(-0.05, 0.05, (sys_examples, num_inputs))
bais2 = np.random.uniform(0.5, 1, (sys_examples, num_inputs))
bais3 = np.random.uniform(6, 8, (sys_examples, num_inputs))
sys_features0 = sys_features+bais1
sys_features01 = sys_features-bais2
sys_features02 = sys_features+bais3

sys_featuresdf = DF_fuzzy(sys_features0, sys_features01, sys_features02)
sys_featuresM = sys_features0/2 + (sys_features01 + sys_features02)/4

#Meanlogistic
Start = time.time()
Tmax = 20
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scorelog0 = np.zeros(Tmax)
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresM, sys_labels, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = LogisticRegression(penalty='l2', C=C[j], random_state=0, multi_class='multinomial').fit(sys_train, y_train)
        clf.predict(sys_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = clf.score(sys_vali, y_vali)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_bestlog = C[j]
    clfbest = LogisticRegression(penalty='l2', C=c_bestlog, random_state=0, multi_class='multinomial').fit(sys_train, y_train)
    Scorelog0[i] = clfbest.score(sys_test, y_test)

timeMeanlogistic = time.time() - Start

# MeanSVM

Start = time.time()

C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
ScoreMSVM = np.zeros(Tmax)
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresM, sys_labels, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = svm.SVC(kernel='linear', C=C[j])
        score_lin = np.zeros(T1)
        for jj in range(T1):
            clf.fit(sys_train, y_train)
            score_lin[jj] = clf.score(sys_vali, y_vali)
        clf = svm.SVC(kernel='rbf', C=C[j])
        score_rbf = np.zeros(T1)
        for jj in range(T1):
            clf.fit(sys_train, y_train)
            score_rbf[jj] = clf.score(sys_vali, y_vali)
        clf = svm.SVC(kernel='poly', C=C[j])
        score_poly = np.zeros(T1)
        for jj in range(T1):
            clf.fit(sys_train, y_train)
            score_poly[jj] = clf.score(sys_vali, y_vali)
        score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
        if vali_best < score:
            vali_best = score
            c_best = C[j]
    clf = svm.SVC(kernel='linear', C=c_best)
    clf.fit(sys_train, y_train)
    s0 = clf.score(sys_test, y_test)
    clf = svm.SVC(kernel='rbf', C=c_best)
    clf.fit(sys_train, y_train)
    s1 = clf.score(sys_test, y_test)
    clf = svm.SVC(kernel='poly', C=c_best)
    clf.fit(sys_train, y_train)
    s2 = clf.score(sys_test, y_test)
    ScoreMSVM[i] = np.max((s0,s1,s2))

timeMeanSVM = time.time() - Start

#MeanCART
Start = time.time()

C =np.arange(1,11,1)
Scoretree0 = np.zeros(Tmax)
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresM, sys_labels, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = DecisionTreeClassifier(min_samples_leaf=C[j], max_depth=10, random_state=0).fit(sys_train, y_train)
        clf.predict(sys_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = clf.score(sys_vali, y_vali)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_besttree = C[j]
    clfbest = DecisionTreeClassifier(min_samples_leaf=c_besttree.astype(np.int64), max_depth=10, random_state=0).fit(sys_train, y_train)
    Scoretree0[i] = clfbest.score(sys_test, y_test)

timeMeanCART = time.time() - Start

#MeanRanF
Start = time.time()

C =np.arange(1,11,1)
Tr = np.arange(10,210,10)
Scoreran0 = np.zeros(Tmax)
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresM, sys_labels, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        for k in range(len(Tr)):
            clf = RandomForestClassifier(n_estimators=Tr[k], min_samples_leaf=C[j], random_state=0).fit(sys_train, y_train)
            clf.predict(sys_vali)
            score = np.zeros(T1)
            for ii in range(T1):
                score[ii] = clf.score(sys_vali, y_vali)
            if vali_best < np.mean(score):
                vali_best = np.mean(score)
                c_bestran = C[j]
                tr_bestran = Tr[k]
    clfbest = RandomForestClassifier(n_estimators=tr_bestran.astype(np.int64), min_samples_leaf=c_bestran.astype(np.int64), random_state=0).fit(sys_train, y_train)
    Scoreran0[i] = clfbest.score(sys_test, y_test)

timeMeanRanF = time.time() - Start

#DFMLP
Start = time.time()

hidder = 150
inputlayer = 20
outlayer = 5

net = Net()
T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
Scoredfmlp = np.zeros(Tmax)
train_Acc0 = []
test_Acc0 = []
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresdf, sys_labels, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(sys_train, sys_vali, sys_test, y_train, y_vali, y_test, batch_size)
    vali_best = 0
    for j in range(LR.shape[0]):
        for k in range(n_epoch.shape[0]):
            vali_Acc = np.zeros(T1)
            for jj in range(T1):
                for params in net.parameters():
                    init.normal_(params, mean=0, std=0.01)
                optimizer = torch.optim.Adam(params=net.parameters(), lr=LR[j], betas=(0.9, 0.999), eps=1e-08)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                train_Acc, test_Acc = train_ch(net, train_iter, vali_iter, loss, batch_size, optimizer, scheduler, device, n_epoch[k])
                # vali_Acc[jj] = np.max(test_Acc)
                vali_Acc[jj] = test_Acc[-1]
            if vali_best < np.mean(vali_Acc):
                vali_best = np.mean(vali_Acc)
                lr_best = LR[j]
                epoch_best = n_epoch[k]
    for params in net.parameters():
               init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    _ , test_acc = train_ch(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
    # Scoredfmlp[i] = np.max(test_acc)
    Scoredfmlp[i] = test_acc[-1]

timeDFMLP = time.time() - Start

#Fig.3
# for i in range(Tmax):
#     sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresdf, sys_labels, 0.2, 0.25, i)
#     train_iter, vali_iter, test_iter = Data_split_torch(sys_train, sys_vali, sys_test, y_train, y_vali, y_test, batch_size)
#     for params in net.parameters():
#                init.normal_(params, mean=0, std=0.01)
#     optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
#     train_ch0(train_Acc0, test_Acc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
#
# train_Acc00=np.array(train_Acc0)
# test_Acc00=np.array(test_Acc0)
# train_Acc00.resize((Tmax,epoch_best))
# test_Acc00.resize((Tmax,epoch_best))
# train_Accmean00 = np.mean(train_Acc00,0)
# train_Accstd00 = np.std(train_Acc00,0)
# test_Accmean00 = np.mean(test_Acc00,0)
# test_Accstd00 = np.std(test_Acc00,0)
# np.max(test_Accmean00)
# test_Accstd00[np.where(test_Accmean00==np.max(test_Accmean00))]
#
# plt.plot(range(1, epoch_best + 1),train_Accmean00,"r-",linewidth=2)
# plt.plot(range(1, epoch_best + 1),test_Accmean00,"b--",linewidth=2)
# plt.fill_between(range(1, epoch_best + 1), train_Accmean00 - train_Accstd00, train_Accmean00 + train_Accstd00 , facecolor='red', alpha=0.1)
# plt.fill_between(range(1, epoch_best + 1), test_Accmean00 - test_Accstd00, test_Accmean00 + test_Accstd00, facecolor='blue', alpha=0.1)
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy(%)")
# plt.legend(['DFMLP train', 'DFMLP test'], loc='lower right')
# plt.ylim((0.02, 1.05))
# my_y_ticks = np.arange(0.02, 1.05, 0.08)
# plt.yticks(my_y_ticks)
# plt.savefig('DFMLP_sys.pdf', bbox_inches='tight')
# plt.show()
#
#DFSVM

Start = time.time()
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scoredfsvm = np.zeros(Tmax)
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresdf, sys_labels, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = svm.SVC(kernel='linear', C=C[j])
        score_lin = np.zeros(T1)
        for jj in range(T1):
            clf.fit(sys_train, y_train)
            score_lin[jj] = clf.score(sys_vali, y_vali)
        clf = svm.SVC(kernel='rbf', C=C[j])
        score_rbf = np.zeros(T1)
        for jj in range(T1):
            clf.fit(sys_train, y_train)
            score_rbf[jj] = clf.score(sys_vali, y_vali)
        clf = svm.SVC(kernel='poly', C=C[j])
        score_poly = np.zeros(T1)
        for jj in range(T1):
            clf.fit(sys_train, y_train)
            score_poly[jj] = clf.score(sys_vali, y_vali)
        score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
        if vali_best < score:
            vali_best = score
            c_best = C[j]
    clf = svm.SVC(kernel='linear', C=c_best)
    clf.fit(sys_train, y_train)
    s0 = clf.score(sys_test, y_test)
    clf = svm.SVC(kernel='rbf', C=c_best)
    clf.fit(sys_train, y_train)
    s1 = clf.score(sys_test, y_test)
    clf = svm.SVC(kernel='poly', C=c_best)
    clf.fit(sys_train, y_train)
    s2 = clf.score(sys_test, y_test)
    Scoredfsvm[i] = np.max((s0, s1, s2))

timeDFSVM = time.time() - Start

#MeanMLP
Start = time.time()
T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
lr_best = 0
Scoremmlp = np.zeros(Tmax)
for i in range(Tmax):
    sys_train, sys_vali, sys_test, y_train, y_vali, y_test = Data_split(sys_featuresM, sys_labels, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(sys_train, sys_vali, sys_test, y_train, y_vali, y_test, batch_size)
    vali_best = 0
    for j in range(LR.shape[0]):
        for k in range(n_epoch.shape[0]):
            vali_Acc = np.zeros(T1)
            for jj in range(T1):
                for params in net.parameters():
                    init.normal_(params, mean=0, std=0.01)
                optimizer = torch.optim.Adam(params=net.parameters(), lr=LR[j], betas=(0.9, 0.999), eps=1e-08)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                train_Acc, test_Acc = train_ch(net, train_iter, vali_iter, loss, batch_size, optimizer, scheduler, device, n_epoch[k])
                # vali_Acc[jj] = np.max(test_Acc)
                vali_Acc[jj] = test_Acc[-1]
            if vali_best < np.mean(vali_Acc):
                vali_best = np.mean(vali_Acc)
                lr_best = LR[j]
                epoch_best = n_epoch[k]
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_Acc0, test_Acc0 = train_ch(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
    # Scoremmlp[i] = np.max(test_Acc0)
    Scoremmlp[i] = test_Acc0[-1]

timeMMLP= time.time() - Start


print('Test Accuracy of Meanlogistic : %.4f ± %.4f. time: %.4f.' %(np.mean(Scorelog0), np.std(Scorelog0), timeMeanlogistic))
print('Test Accuracy of MeanSVM : %.4f ± %.4f. time: %.4f.' %(np.mean(ScoreMSVM), np.std(ScoreMSVM), timeMeanSVM))
print('Test Accuracy of MeanDecisiontree : %.4f ± %.4f. time: %.4f.' %(np.mean(Scoretree0), np.std(Scoretree0), timeMeanCART))
print('Test Accuracy of MeanRandomForest : %.4f ± %.4f. time: %.4f.' %(np.mean(Scoreran0), np.std(Scoreran0), timeMeanRanF))
print('Test Accuracy of MeanMLP : %.4f ± %.4f. time: %.4f.' %(np.mean(Scoremmlp), np.std(Scoremmlp), timeMMLP))
print('Test Accuracy of DF-SVM : %.4f ± %.4f. time: %.4f.' %(np.mean(Scoredfsvm), np.std(Scoredfsvm), timeDFSVM))
print('Test Accuracy of DF-MLP : %.4f ± %.4f. time: %.4f.' %(np.mean(Scoredfmlp), np.std(Scoredfmlp), timeDFMLP))