import torch
import time

import scipy.stats as ss
import sys
sys.path.append("..") 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import random
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..") 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


#blanced accuracy
def balanced_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    y_pre = np.array([])
    y_tru = np.array([])
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() 
                y_pre = np.concatenate((y_pre, (net(X.to(device)).argmax(dim=1)).cpu().float().numpy()))
                y_tru = np.concatenate((y_tru, (y.to(device)).cpu().float().numpy()))
                net.train() 
            else: 
                if('is_training' in net.__code__.co_varnames): 
                    y_pre = np.concatenate((y_pre, (net(X, is_training=False).argmax(dim=1)).cpu().float().numpy()))
                    y_tru = np.concatenate((y_tru, y.cpu().numpy()))
                else:
                    y_pre = np.concatenate((y_pre, (net(X).argmax(dim=1)).cpu().float().numpy()))
                    y_tru = np.concatenate((y_tru, y.cpu().numpy()))
    return balanced_accuracy_score(y_tru, y_pre)

def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    
    assert x.shape == orig_shape
    return x

def AUC(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    y_pro = np.array([])
    y_tru = np.array([])
    n = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() 
                l1 = net(X.to(device)).shape[0]
                l2 = net(X.to(device)).shape[1]
                y_pro = np.concatenate((y_pro, (net(X.to(device))).cpu().float().numpy().reshape(l1*l2,)))
                y_tru = np.concatenate((y_tru, (y.to(device)).cpu().float().numpy()))
                net.train() 
            else: 
                if('is_training' in net.__code__.co_varnames): 
                    l1 = net(X, is_training=False).shape[0]
                    l2 = net(X, is_training=False).shape[1]
                    y_pro = np.concatenate((y_pro, (net(X, is_training=False)).cpu().float().numpy().reshape(l1*l2,)))
                    y_tru = np.concatenate((y_tru, y.cpu().numpy()))
                else:
                    l1 = net(X).shape[0]
                    l2 = net(X).shape[1]
                    y_pro = np.concatenate((y_pro, net(X).cpu().float().numpy().reshape(l1*l2,)))
                    y_tru = np.concatenate((y_tru, y.cpu().numpy()))
            n += y.shape[0]
    prob = softmax(y_pro.reshape(n, 17))
    if len(np.unique(y_tru)) == prob.shape[1]:
        auc = roc_auc_score(y_tru, prob, multi_class='ovr')
    else:
        auc = 0
    return auc


def train_ch2(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    train_Acc = []
    test_Acc = []
    train_Auc = []
    test_Auc = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        y_pre0 = np.array([])
        y_pro0 = np.array([])
        y_tru0 = np.array([])
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l1 = y_hat.shape[0]
            l2 = y_hat.shape[1]
            y_pro0 = np.concatenate((y_pro0, y_hat.cpu().float().detach().numpy().reshape(l1*l2,)))
            l = loss(y_hat, y).sum()
            #l.requires_grad_(True)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            y_pre0 = np.concatenate((y_pre0, (y_hat.argmax(dim=1)).cpu().float().numpy()))
            y_tru0 = np.concatenate((y_tru0, y.cpu().numpy()))
            n += y.shape[0]
            batch_count += 1
        test_acc = balanced_accuracy(test_iter, net)
        test_AUC = AUC(test_iter, net)
        train_Acc.append(balanced_accuracy_score(y_tru0, y_pre0))
        prob0 = softmax(y_pro0.reshape(n, 17))
        if len(np.unique(y_tru0)) == prob0.shape[1]:
            train_Auc.append(roc_auc_score(y_tru0, prob0, multi_class='ovr'))
        test_Acc.append(test_acc)
        test_Auc.append(test_AUC)
        if epoch % 10 == 9:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, balanced_accuracy_score(y_tru0, y_pre0), test_acc, time.time() - start))
    scheduler.step()
    return train_Acc, test_Acc, train_Auc, test_Auc

def train_ch02(train_Acc0, test_Acc0, train_Auc0, test_Auc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        y_pre0 = np.array([])
        y_pro0 = np.array([])
        y_tru0 = np.array([])
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l1 = y_hat.shape[0]
            l2 = y_hat.shape[1]
            y_pro0 = np.concatenate((y_pro0, y_hat.cpu().float().detach().numpy().reshape(l1*l2,)))
            l = loss(y_hat, y).sum()
            #l.requires_grad_(True)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            y_pre0 = np.concatenate((y_pre0, (y_hat.argmax(dim=1)).cpu().float().numpy()))
            y_tru0 = np.concatenate((y_tru0, y.cpu().numpy()))
            n += y.shape[0]
            batch_count += 1
        test_acc = balanced_accuracy(test_iter, net)
        test_AUC = AUC(test_iter, net)
        train_Acc0.append(balanced_accuracy_score(y_tru0, y_pre0))
        prob0 = softmax(y_pro0.reshape(n, 17))
        if len(np.unique(y_tru0)) == prob0.shape[1]:
            train_Auc0.append(roc_auc_score(y_tru0, prob0, multi_class='ovr'))
        test_Acc0.append(test_acc)
        if test_AUC > 0:
            test_Auc0.append(test_AUC)
        if epoch % 10 == 9:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, balanced_accuracy_score(y_tru0, y_pre0), test_acc, time.time() - start))
    scheduler.step()

# def softmax(X):
#     X_exp = X.exp()
#     partition = X_exp.sum(dim=1, keepdim=True)
#     return X_exp / partition

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_ch(net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    train_Acc = []
    test_Acc = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            #l.requires_grad_(True)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        train_Acc.append(train_acc_sum / n)
        test_Acc.append(test_acc)
        if epoch % 10 == 9:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    scheduler.step()
    return train_Acc, test_Acc

def train_ch0(train_Acc0, test_Acc0, net, train_iter, test_iter, loss, batch_size, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            #l.requires_grad_(True)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        train_Acc0.append(train_acc_sum / n)
        test_Acc0.append(test_acc)
        if epoch % 10 == 9:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    scheduler.step()
    
    
def DF_fuzzy(data0, data1, data2):

    data = 2 * data0 / 3 + (data1 + data2) / 6

    return data


def DF_interval(data, beta, feature_number):

    data_df = np.zeros([data.shape[0], feature_number])
    for i in range(feature_number):
        data_df[:, i] = (2 * beta / 3 + 1 / 6) * data[:, 2 * i] + (-2 * beta / 3 + 5 / 6) * data[:, 2 * i + 1]

    return data_df
    
def Data_split(feature, label, vail, test, random_state):

    f_train, f_vali, y_train, y_vali = train_test_split(feature, label, test_size=vail, random_state=random_state)
    f_train, f_test, y_train, y_test = train_test_split(f_train, y_train, test_size=test, random_state=random_state)

    return f_train, f_vali, f_test, y_train, y_vali, y_test


def Data_split_torch(f_train, f_vali, f_test, y_train, y_vali, y_test, batch_size):

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(f_train, dtype=torch.float32), torch.tensor(y_train).long())
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    vali_dataset = torch.utils.data.TensorDataset(torch.tensor(f_vali, dtype=torch.float32), torch.tensor(y_vali).long())
    vali_iter = torch.utils.data.DataLoader(vali_dataset, batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(f_test, dtype=torch.float32), torch.tensor(y_test).long())
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

    return train_iter, vali_iter, test_iter