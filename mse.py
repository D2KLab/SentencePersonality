import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tkinter
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from micro_influencer_utilities import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
import joblib

import torch
import torchtext
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset

import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

def fit(num_epochs, model, loss_fn, opt, trains_dl):
    for epoch in range(num_epochs):
        for xb,yb in trains_dl:
            pred = model(xb.float())
            loss = loss_fn(pred,yb.float())
            loss.backward()
            opt.step()
            opt.zero_grad()
        

dataset = pd.read_csv("cls_table.csv", header=None)
X = dataset.iloc[:,0:768] #tweet embeddings obtained with BERT as a service 
Y = dataset.iloc[:,768:] #Big5 scores related to X 

l = 1 

#f_svm = open("svm_statistics.txt","w")
f_nn = open("predictions.txt","w")
out_mse = open("mse.txt","w")

for label in [Y.iloc[:,0], Y.iloc[:,1],Y.iloc[:,2], Y.iloc[:,3], Y.iloc[:,4]]:
    y_to_plot = []
    #SVM_y_to_plot = []
    if l==1:
        trait = "O"
    elif l==2:
        trait = "C"
    elif l==3:
        trait = "E"
    elif l==4:
        trait = "A"
    elif l==5:
        trait = "N"
    l += 1

    LnReLn_mse = []
    #SVM_model = joblib.load("Models/SVM_BERT_"+trait+".pkl") 
    
    kf = KFold(n_splits=10)
    #print("Scoring parameters with fold:" , kf )
    y = label
    
    #y = y.as_matrix()
    X = np.array(X)
    #y = y.to_numpy()
    #y = np.reshape(y, (9913, 1)) 
    y = np.array(y)
    #r2.append(r2_score(y_test, y_pred))
    num_epochs = 50
    batch_size = 100
    
    LnReLn_model = nn.Sequential(nn.Linear(768,300),nn.ReLU() ,nn.Linear(300,1))
    opt = torch.optim.Adam(LnReLn_model.parameters(), lr=1e-5)
    loss_fn = F.mse_loss
    i = 1 # iteration counter
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #y_pred_SVM = SVM_model.fit(X_train, y_train).predict(X_test)
        #SVM_y_to_plot.append(y_pred_SVM)
        
        y_train = np.reshape(y_train, (y_train.size,1))
        inputs = torch.from_numpy(np.array(X_train))
        inputs_test = torch.from_numpy(np.array(X_test))
        targets = torch.from_numpy(y_train)
        target_test = torch.from_numpy(y_test)
        trains_ds = TensorDataset(inputs.float(),targets.float())
        trains_dl = DataLoader(trains_ds,batch_size= batch_size, shuffle=True)
        fit(num_epochs, LnReLn_model, loss_fn, opt,trains_dl)
        y_pred = LnReLn_model(inputs_test.float())
        LnReLn_mse.append(mean_squared_error(target_test.float().numpy(),y_pred.float().detach().numpy()))
        print("LnReLn_mse",LnReLn_mse)
        print("LlReLN", i, "/10")
        y_to_append = y_pred.float().detach().numpy()
        y_to_plot.append(y_to_append)    
        
        print("Progress", i, "/10 ended")
        i = i+1
    print("LnReLn mean mse for trait " + trait + ": ",np.mean(LnReLn_mse))
    out_mse.write("LnReLn mean mse for trait " + trait + ": " + str(np.mean(LnReLn_mse)))

    y_asarray = np.asarray(y_to_plot).reshape(-1)
    y_asarray = np.concatenate(y_asarray)
    #print(y_asarray, y_asarray.size) #stampa su file per non rifare ogni volta
    
    for item in y_asarray[:-1]:
        str_item = str(item[0])
        f_nn.write("%s," % str_item)
    last_elem = y_asarray[-1]

    f_nn.write("%s\n" % str(last_elem[0]))

    '''
    y_asarray = np.asarray(SVM_y_to_plot).reshape(-1)
    y_asarray = np.concatenate(y_asarray)
    for item in y_asarray[:-1]:
        str_item = str(item)
        f_svm.write("%s," % str_item)
    f_svm.write("%s\n" % str(y_asarray[-1]))
    '''
    print(trait,"ended")


f_nn.close()
out_mse.close()
#f_svm.close()
