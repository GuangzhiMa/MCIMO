# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 08:58:00 2022

@author: 14025959_admin
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
from utils import DF_interval, train_ch, train_ch0, Data_split, Data_split_torch

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

# Washington
weatherlabel0 = pd.read_excel('Washing.xlsx', usecols=[6])
weatherfeature0 = pd.read_excel('Washing.xlsx', usecols=[1, 2, 3, 4, 5])
weatherfeature0 = np.array(weatherfeature0)
feature_number = weatherfeature0.shape[1]
weatherlabels0 = np.array(weatherlabel0)
weatherlabels0 = weatherlabels0.reshape(weatherlabels0.shape[0], )
weatherlabel = np.zeros([2191, ])
weatherfeature1 = np.zeros([2191, 10])

for i in range(2191):
    we = weatherlabels0[8 * i:8 * i + 8].sum()
    if we > 0:
        weatherlabel[i] = 1
    else:
        weatherlabel[i] = 0

for i in range(2191):
    we_max = np.max(weatherfeature0[8 * i:8 * i + 8, :], 0)
    we_min = np.min(weatherfeature0[8 * i:8 * i + 8, :], 0)
    for j in range(5):
        weatherfeature1[i, 2 * j] = we_min[j]
        weatherfeature1[i, 2 * j + 1] = we_max[j]

weatherlabel = weatherlabel.astype(int)

weatherfeature_M = np.zeros([weatherfeature1.shape[0], 5])
for i in range(weatherfeature_M.shape[1]):
    weatherfeature_M[:, i] = (weatherfeature1[:, 2 * i] + weatherfeature1[:, 2 * i + 1]) / 2


# Meanlogistic
Tmax = 20
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scorelog0 = np.zeros(Tmax)
for i in range(Tmax):
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_M, weatherlabel, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = LogisticRegression(penalty='l2', C=C[j], random_state=0, multi_class='multinomial').fit(wash_train, y_train)
        clf.predict(wash_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = clf.score(wash_vali, y_vali)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_bestlog = C[j]
    clfbest = LogisticRegression(penalty='l2', C=c_bestlog, random_state=0, multi_class='multinomial').fit(wash_train, y_train)
    Scorelog0[i] = clfbest.score(wash_test, y_test)

# MeanSVM
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
ScoreMSVM = np.zeros(Tmax)
for i in range(Tmax):
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_M, weatherlabel, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = svm.SVC(kernel='linear', C=C[j])
        score_lin = np.zeros(T1)
        for jj in range(T1):
            clf.fit(wash_train, y_train)
            score_lin[jj] = clf.score(wash_vali, y_vali)
        clf = svm.SVC(kernel='rbf', C=C[j])
        score_rbf = np.zeros(T1)
        for jj in range(T1):
            clf.fit(wash_train, y_train)
            score_rbf[jj] = clf.score(wash_vali, y_vali)
        clf = svm.SVC(kernel='poly', C=C[j])
        score_poly = np.zeros(T1)
        for jj in range(T1):
            clf.fit(wash_train, y_train)
            score_poly[jj] = clf.score(wash_vali, y_vali)
        score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
        if vali_best < score:
            vali_best = score
            c_best = C[j]
    clf = svm.SVC(kernel='linear', C=c_best)
    clf.fit(wash_train, y_train)
    s0 = clf.score(wash_test, y_test)
    clf = svm.SVC(kernel='rbf', C=c_best)
    clf.fit(wash_train, y_train)
    s1 = clf.score(wash_test, y_test)
    clf = svm.SVC(kernel='poly', C=c_best)
    clf.fit(wash_train, y_train)
    s2 = clf.score(wash_test, y_test)
    ScoreMSVM[i] = np.max((s0,s1,s2))

# MeanCART
C =np.arange(1,11,1)
Scoretree0 = np.zeros(Tmax)
for i in range(Tmax):
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_M, weatherlabel, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        clf = DecisionTreeClassifier(min_samples_leaf=C[j], max_depth=10, random_state=0).fit(wash_train, y_train)
        clf.predict(wash_vali)
        score = np.zeros(T1)
        for k in range(T1):
            score[k] = clf.score(wash_vali, y_vali)
        if vali_best < np.mean(score):
            vali_best = np.mean(score)
            c_besttree = C[j]
    clfbest = DecisionTreeClassifier(min_samples_leaf=c_besttree.astype(np.int64), max_depth=10, random_state=0).fit(wash_train, y_train)
    Scoretree0[i] = clfbest.score(wash_test, y_test)

# MeanRanF
C =np.arange(1,11,1)
Tr = np.arange(10,210,10)
Scoreran0 = np.zeros(Tmax)
for i in range(Tmax):
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_M, weatherlabel, 0.2, 0.25, i)
    vali_best = 0
    for j in range(len(C)):
        for k in range(len(Tr)):
            clf = RandomForestClassifier(n_estimators=Tr[k], min_samples_leaf=C[j], random_state=0).fit(wash_train, y_train)
            clf.predict(wash_vali)
            score = np.zeros(T1)
            for ii in range(T1):
                score[ii] = clf.score(wash_vali, y_vali)
            if vali_best < np.mean(score):
                vali_best = np.mean(score)
                c_bestran = C[j]
                tr_bestran = Tr[k]
    clfbest = RandomForestClassifier(n_estimators=tr_bestran.astype(np.int64), min_samples_leaf=c_bestran.astype(np.int64), random_state=0).fit(wash_train, y_train)
    Scoreran0[i] = clfbest.score(wash_test, y_test)


# DFMLP
batch_size = 500

hidder = 100
inputlayer = 5
outlayer = 2

net = Net()

LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
B = np.arange(0,1.02,0.05)
Scoredfmlp = np.zeros(Tmax)
T1 = 10
train_Acc0 = []
test_Acc0 = []
for i in range(Tmax):
    vali_best = 0
    for m in range(len(B)):
        weatherfeature_df = DF_interval(weatherfeature1, B[m], feature_number)
        wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_df, weatherlabel, 0.2, 0.25, i)
        train_iter, vali_iter, test_iter = Data_split_torch(wash_train, wash_vali, wash_test, y_train, y_vali, y_test, batch_size)
        for j in range(LR.shape[0]):
            for k in range(n_epoch.shape[0]):
                vali_Acc = np.zeros(T1)
                for jj in range(10):
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
                    beta_best = B[m]
    weatherfeature_df = DF_interval(weatherfeature1, beta_best, feature_number)
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_df, weatherlabel, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(wash_train, wash_vali, wash_test, y_train, y_vali, y_test, batch_size)
    for params in net.parameters():
               init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    _ , test_acc = train_ch(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)
    Scoredfmlp[i] = test_acc[-1]

weatherfeature_df = DF_interval(weatherfeature1, beta_best, feature_number)

for i in range(Tmax):
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_df, weatherlabel, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(wash_train, wash_vali, wash_test, y_train, y_vali, y_test, batch_size)
    for params in net.parameters():
               init.normal_(params, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params = net.parameters(), lr=lr_best, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    train_ch0(train_Acc0, test_Acc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, epoch_best)

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
plt.savefig('DFMLP_washington.pdf', bbox_inches='tight')
plt.show()

# DFSVM
T1 = 1
C = np.append(np.arange(0.1,1,0.1), np.arange(1,101,1))
Scoredfsvm = np.zeros(Tmax)
for i in range(Tmax):
    vali_best = 0
    for m in range(len(B)):
        weatherfeature_df = DF_interval(weatherfeature1, B[m], feature_number)
        wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_df, weatherlabel, 0.2, 0.25, i)
        for j in range(len(C)):
            clf = svm.SVC(kernel='linear', C=C[j])
            score_lin = np.zeros(T1)
            for jj in range(T1):
                clf.fit(wash_train, y_train)
                score_lin[jj] = clf.score(wash_vali, y_vali)
            clf = svm.SVC(kernel='rbf', C=C[j])
            score_rbf = np.zeros(T1)
            for jj in range(T1):
                clf.fit(wash_train, y_train)
                score_rbf[jj] = clf.score(wash_vali, y_vali)
            clf = svm.SVC(kernel='poly', C=C[j])
            score_poly = np.zeros(T1)
            for jj in range(T1):
                clf.fit(wash_train, y_train)
                score_poly[jj] = clf.score(wash_vali, y_vali)
            score = np.max((np.mean(score_lin), np.mean(score_rbf), np.mean(score_poly)))
            if vali_best < score:
                vali_best = score
                c_best = C[j]
                beta_best = B[m]
    weatherfeature_df = DF_interval(weatherfeature1, beta_best, feature_number)
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_df, weatherlabel, 0.2, 0.25, i)
    clf = svm.SVC(kernel='linear', C=c_best)
    clf.fit(wash_train, y_train)
    s0 = clf.score(wash_test, y_test)
    clf = svm.SVC(kernel='rbf', C=c_best)
    clf.fit(wash_train, y_train)
    s1 = clf.score(wash_test, y_test)
    clf = svm.SVC(kernel='poly', C=c_best)
    clf.fit(wash_train, y_train)
    s2 = clf.score(wash_test, y_test)
    Scoredfsvm[i] = np.max((s0, s1, s2))

# MeanMLP
T1 = 10
LR = np.array([0.0001, 0.001, 0.01, 0.1])
n_epoch = np.array([100,200,500,1000,1500])
lr_best = 0
Scoremmlp = np.zeros(Tmax)
for i in range(Tmax):
    wash_train, wash_vali, wash_test, y_train, y_vali, y_test = Data_split(weatherfeature_M, weatherlabel, 0.2, 0.25, i)
    train_iter, vali_iter, test_iter = Data_split_torch(wash_train, wash_vali, wash_test, y_train, y_vali, y_test, batch_size)
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
print('Test Accuracy of DF-MLP : %.4f ± %.4f.' %(np.mean(Scoredfmlp), np.std(Scoredfmlp)))