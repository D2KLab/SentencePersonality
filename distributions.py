import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tkinter
from scipy import stats
from scipy.stats import norm
import scipy

bins = 32
frequency = 3000
trait_name =["Openness","Conscentiousness","Extraversion","Agreableness","Neuroticism"]

def kl_divergence(p,q):
    return (p*np.log(p/q)).sum()

def dist_info(trait_arr, trait_name):
    print(trait_name)
    print("mean, std, var",np.mean(np.asarray(trait_arr)), np.std(np.asarray(trait_arr)), np.var(np.asarray(trait_arr)))


def read_and_display_distribution(input_csv, title):
    X = pd.read_csv(input_csv, header=None)
    o = X.iloc[0,:]
    c = X.iloc[1,:]
    e = X.iloc[2,:]
    a = X.iloc[3,:]
    n = X.iloc[4,:]    
    pos = 0
    ocean = [o,c,e,a,n]
    for trait in ocean:
        dist_info(trait, trait_name[pos])
        plt.title(title+" "+str(trait_name[pos]))
        plt.hist(trait, bins=bins, range=(1,5)) 
        plt.ylabel("frequency")
        plt.xlabel("score")
        plt.axis([1,5,0,frequency])
        plt.grid()
        plt.draw()
       
        plt.savefig("./img/"+title+"_"+str(trait_name[pos])+'.png', dpi=100)
        #plt.show()
        plt.close()
        pos = pos + 1
    
    return ocean




bert_nn = []
bert_nn = read_and_display_distribution("predictions.txt", "cls+nn+multilanguage")
bert_nn_eng = []
bert_nn_eng = read_and_display_distribution("predictions_nn.txt", "cls+nn")



dataset = pd.read_csv("train_whole_lines.csv", header=None)
Y = dataset.iloc[:,768:] #Big5 scores related to X 
o = Y.iloc[:,0] 
c = Y.iloc[:,1] 
e = Y.iloc[:,2] 
a = Y.iloc[:,3] 
n = Y.iloc[:,4] 
real = [o,c,e,a,n]
pos = 0
for trait in real:
        dist_info(trait, trait_name[pos])
        plt.title("real"+" "+str(trait_name[pos]))
        plt.hist(trait, bins=bins, range=(1,5)) 
        plt.ylabel("frequency")
        plt.xlabel("score")
        plt.axis([1,5,0,1500])
        plt.grid()
        plt.draw()
       
        plt.savefig("./img/"+"real"+"_"+str(trait_name[pos])+'.png', dpi=100)
        #plt.show()
        plt.close()
        pos = pos + 1

lines_skipped = open("lines_skipped.csv","r").read().splitlines() 
print(lines_skipped)
for trait in range(5):
    real_clean = []
    bert_nn_eng_clean = []
    pos = 0
    for elem in real[trait]:
        pos = pos + 1
        if str(pos) not in lines_skipped:
            real_clean.append(elem)

    pos = 0
    for elem2 in bert_nn_eng[trait]:
        pos = pos + 1
        if str(pos) not in lines_skipped:
            bert_nn_eng_clean.append(elem2)

    print(len(real_clean))
    kl = kl_divergence(bert_nn[trait],bert_nn_eng_clean)
    print("bert_nn vs bert_eng", kl)
    print(len(bert_nn[trait]), len(bert_nn_eng_clean))


