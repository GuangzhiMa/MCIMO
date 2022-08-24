# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 08:58:00 2022

@author: guangzhi
"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import train_ch, train_ch0, Data_split, Data_split_torch

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

np.random.seed(1234)
random.seed(1234)

# Perceptions experiment dataset
df_description = pd.read_excel('DESCRIPTION OF THE DATA BASE.xlsx',usecols=[9,10,11,12,13])
description_labels0 = np.array([df_description['LingF'][k] for k in range(551)])
description_labels = torch.from_numpy(description_labels0).type(torch.LongTensor)

description_features1 = np.array([(df_description['Inf1'][k] + df_description['Sup1'][k])/4 +
                                 (df_description['Inf0'][k] + df_description['Sup0'][k])/4 for k in range(551)], dtype='float32')
description_features1 = description_features1.reshape(551,1)
description_features01 = torch.from_numpy(description_features1)


description_features0 = np.array([(df_description['Inf1'][k] + df_description['Sup1'][k])/3 +
                                 (df_description['Inf0'][k] + df_description['Sup0'][k])/6 for k in range(551)], dtype='float32')
description_features0 = description_features1.reshape(551,1)

# Meanlogistic
Tmax = 20
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scorelog0 = np.zeros(Tmax)
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features1, description_labels0, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = LogisticRegression(penalty='l2', C=C[j], random_state=0, multi_class='multinomial').fit(description_train, y_train)
        clf.predict(description_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = clf.score(description_vali, y_vali)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_bestlog = C[j]
    clfbest = LogisticRegression(penalty='l2', C=c_bestlog, random_state=0, multi_class='multinomial').fit(description_train, y_train)
    Scorelog0[i] = clfbest.score(description_test, y_test)

# MeanSVM
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
ScoreMSVM = np.zeros(Tmax)
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features1, description_labels0, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = svm.SVC(kernel='linear', C=C[j])
        score_lin = np.zeros(T1)
        for jj in range(T1):
            clf.fit(description_train, y_train)
            score_lin[jj] = clf.score(description_vali, y_vali)
        clf = svm.SVC(kernel='rbf', C=C[j])
        score_rbf = np.zeros(T1)
        for jj in range(T1):
            clf.fit(description_train, y_train)
            score_rbf[jj] = clf.score(description_vali, y_vali)
        clf = svm.SVC(kernel='poly', C=C[j])
        score_poly = np.zeros(T1)
        for jj in range(T1):
            clf.fit(description_train, y_train)
            score_poly[jj] = clf.score(description_vali, y_vali)
        score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
        if vali_best < score:
            vali_best = score
            c_best = C[j]
    clf = svm.SVC(kernel='linear', C=c_best)
    clf.fit(description_train, y_train)
    s0 = clf.score(description_test, y_test)
    clf = svm.SVC(kernel='rbf', C=c_best)
    clf.fit(description_train, y_train)
    s1 = clf.score(description_test, y_test)
    clf = svm.SVC(kernel='poly', C=c_best)
    clf.fit(description_train, y_train)
    s2 = clf.score(description_test, y_test)
    ScoreMSVM[i] = np.max((s0, s1, s2))

# MeanCART
C =np.arange(1,11,1)
Scoretree0 = np.zeros(Tmax)
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features1, description_labels0, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = DecisionTreeClassifier(min_samples_leaf=C[j], max_depth=10, random_state=0).fit(description_train, y_train)
        clf.predict(description_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = clf.score(description_vali, y_vali)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_besttree = C[j]
    clfbest = DecisionTreeClassifier(min_samples_leaf=c_besttree.astype(np.int64), max_depth=10, random_state=0).fit(description_train, y_train)
    Scoretree0[i] = clfbest.score(description_test, y_test)

# MeanRanF
C =np.arange(1,11,1)
Tr = np.arange(10,210,10)
Scoreran0 = np.zeros(Tmax)
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features1, description_labels0, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        for k in range(len(Tr)):
            clf = RandomForestClassifier(n_estimators=Tr[k], min_samples_leaf=C[j], random_state=0).fit(description_train, y_train)
            clf.predict(description_vali)
            score = np.zeros(T1)
            for ii in range(T1):
                score[ii] = clf.score(description_vali, y_vali)
            if vali_best < np.mean(score):
                vali_best = np.mean(score)
                c_bestran = C[j]
                tr_bestran = Tr[k]
    clfbest = RandomForestClassifier(n_estimators=tr_bestran.astype(np.int64), min_samples_leaf=c_bestran.astype(np.int64), random_state=0).fit(description_train, y_train)
    Scoreran0[i] = clfbest.score(description_test, y_test)

# DFMLP
hidder = 100
inputlayer = 1
outlayer = 5

net = Net()
batch_size = 100
T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
Scoredfmlp = np.zeros(Tmax)
train_Acc0 = []
test_Acc0 = []
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features0, description_labels0, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(description_train, description_vali, description_test, y_train, y_vali, y_test, batch_size)
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
                vali_Acc[jj] = test_Acc[-1]
            if vali_best < np.mean(vali_Acc):
                vali_best = np.mean(vali_Acc)
                lr_best = LR[j]
                epoch_best = n_epoch[k]
    for params in net.parameters():
               init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    _ , test_Acc0 = train_ch(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
    Scoredfmlp[i] = test_Acc0[-1]

# Fig.5(a)
# for i in range(Tmax):
#     description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features0, description_labels0, 0.2, 0.25, i)
#     train_iter, vali_iter, test_iter = Data_split_torch(description_train, description_vali, description_test, y_train, y_vali, y_test, batch_size)
#     for params in net.parameters():
#                init.normal_(params, mean=0, std=0.01)
#     optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
#     train_ch0(train_Acc0, test_Acc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)

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
# plt.savefig('DFMLP_per.pdf', bbox_inches='tight')
# plt.show()

# DFSVM
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scoredfsvm = np.zeros(Tmax)
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features0, description_labels0, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = svm.SVC(kernel='linear', C=C[j])
        score_lin = np.zeros(T1)
        for jj in range(T1):
            clf.fit(description_train, y_train)
            score_lin[jj] = clf.score(description_vali, y_vali)
        clf = svm.SVC(kernel='rbf', C=C[j])
        score_rbf = np.zeros(T1)
        for jj in range(T1):
            clf.fit(description_train, y_train)
            score_rbf[jj] = clf.score(description_vali, y_vali)
        clf = svm.SVC(kernel='poly', C=C[j])
        score_poly = np.zeros(T1)
        for jj in range(T1):
            clf.fit(description_train, y_train)
            score_poly[jj] = clf.score(description_vali, y_vali)
        score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
        if vali_best < score:
            vali_best = score
            c_best = C[j]
    clf = svm.SVC(kernel='linear', C=c_best)
    clf.fit(description_train, y_train)
    s0 = clf.score(description_test, y_test)
    clf = svm.SVC(kernel='rbf', C=c_best)
    clf.fit(description_train, y_train)
    s1 = clf.score(description_test, y_test)
    clf = svm.SVC(kernel='poly', C=c_best)
    clf.fit(description_train, y_train)
    s2 = clf.score(description_test, y_test)
    Scoredfsvm[i] = np.max((s0, s1, s2))

# MeanMLP
T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
lr_best = 0
Scoremmlp = np.zeros(Tmax)
for i in range(Tmax):
    description_train, description_vali, description_test, y_train, y_vali, y_test = Data_split(description_features1, description_labels0, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(description_train, description_vali, description_test, y_train, y_vali, y_test, batch_size)
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
    Scoremmlp[i] = test_Acc0[-1]


print('Test Accuracy of Meanlogistic : %.4f ± %.4f.' %(np.mean(Scorelog0), np.std(Scorelog0)))
print('Test Accuracy of MeanSVM : %.4f ± %.4f.' %(np.mean(ScoreMSVM), np.std(ScoreMSVM)))
print('Test Accuracy of MeanDecisiontree : %.4f ± %.4f.' %(np.mean(Scoretree0), np.std(Scoretree0)))
print('Test Accuracy of MeanRandomForest : %.4f ± %.4f.' %(np.mean(Scoreran0), np.std(Scoreran0)))
print('Test Accuracy of MeanMLP : %.4f ± %.4f.' %(np.mean(Scoremmlp), np.std(Scoremmlp)))
print('Test Accuracy of DF-SVM : %.4f ± %.4f.' %(np.mean(Scoredfsvm), np.std(Scoredfsvm)))
print('Test Accuracy of DF-MLP : %.4f ± %.4f. time: %.4f' %(np.mean(Scoredfmlp), np.std(Scoredfmlp)))
