#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import numpy 
import matplotlib.pyplot as plt
from scipy import linalg, special, stats
from numpy import genfromtxt
import ML_support as ml
from scipy.optimize import fmin_l_bfgs_b


# In[2]:


Data, labels = ml.loadFile('./Train.txt')


# In[3]:


data_g = ml.gaussianize(Data)
data_z = ml.z_normalization(Data)


# In[4]:


def full_GMM(DTR, LTR, DTE, n_repetions):
    N = DTR.shape[1]
    t = 1e-6
    w = 1.0
    alpha = 0.1
    final_scores = {}
    gmm = {}
    
    mu, C = ml.MU_Cov_calculator(DTR, LTR)
    
    for i in numpy.unique(LTR):
        gmm[i] = [w, mu[i], C[i]]
        GMM_final = {0 : gmm[i]}
        
        for j in range (n_repetions):
            GMM_init = GMM_final.copy()
            
            for k, g in enumerate(GMM_init.values()):
                w, mu, Cov = g
                newW = w/2
                U, s, Vh = numpy.linalg.svd(Cov)
                d = U[:, 0:1]*s[0]**0.5*alpha
                newMu1 = mu+d
                newMu2 = mu-d
                GMM_final[k*2] = [newW, newMu1, Cov]
                GMM_final[k*2+1] = [newW, newMu2, Cov]
                
            GMM_final, opt_ll = ml.optimize_GMM(DTE, GMM_final)
        final_scores[i] = opt_ll
    
    r = final_scores[1]-final_scores[0]
    
    return list(r[0])


# In[5]:


def kfold(D, L, fold, params, app):
    error = 0
    N = int(D.shape[1]/fold) #numero di elementi per ogni fold
    numpy.random.seed(0) #imposto il seed del generatore di numeri casuali -> in tal modo genererò la stessa sequenza di numeri casuali aventi seed uguale
    indexes = numpy.random.permutation(D.shape[1]) #genero una sequenza di numeri casuali che vanno da 0 al num_di_campioni
    
    LTE_final = []
    llr_final = []
    for j in range(fold):
        test_indexes = indexes[(j*N):((j+1)*N)] #selezioni gli indici che identificano i campioni (casuali) del test set
        if(j > 0): #se il test set non è preso dalla prima fold (--> il test set è una fold intermedia o l'ultima fold)
            left_indexes = indexes[0:(j*N)] #allora prendo tutti gli indici che stanno a sinistra di tale fold
        else: #se il test set è preso dalla prima fold
            right_indexes = indexes[((j+1)*N):] #prendo tutti gli indici a destra della prima fold

        if(j == 0): #se il test set è preso dalla prima fold
            train_indexes = right_indexes #assegno agli indici di training quelli che stanno a destra della prima fold
        elif(j == fold-1): #se il test set è preso dall'ultima fold
            train_indexes = left_indexes #assegno agli indici di training quelli che stanno a sinistra dell'ultima fold
        else: #in questo caso il test set è preso da una fold intermedia
            train_indexes = numpy.hstack((left_indexes, right_indexes)) #pertanto assegno agli indici di training quelli appartenenti alle fold di sinistra e di destra

        DTR = D[:, train_indexes]  #definisco insieme di training e di testing
        LTR = L[train_indexes]
        DTE = D[:, test_indexes]
        LTE = L[test_indexes]
        LTE_final.extend(LTE)
        llr_final.extend(full_GMM(DTR, LTR, DTE, params))
    
    
    CM = ml.compute_optimal_B_decision(app, llr_final, LTE_final)

    app_bayes_risk=ml.compute_Bayes_risk(CM, app)
    DCF = ml.compute_norm_Bayes(app_bayes_risk, app)
    
    minDCF, _= ml.compute_min_DCF(llr_final, app, LTE_final)
    
    error = 1-(CM[0, 0]+CM[1,1])/(len(LTE_final))

    print("\-/ \-/ \-/ \-/ \-/ ")
    print(f"GMM error: {round(error, 3)}")
    print(f'{app} DCF:{round(DCF, 3)} minDCF: {round(minDCF,3)}')
    print("/-\ /-\ /-\ /-\ /-\ ")


# In[5]:


kfold(Data, labels, 5, 1, [0.5, 1, 1])
kfold(Data, labels, 5, 2, [0.5, 1, 1])
kfold(Data, labels, 5, 3, [0.5, 1, 1])
kfold(Data, labels, 5, 4, [0.5, 1, 1])


# In[20]:


kfold(data_g, labels, 5, 1, [0.5, 1, 1])
kfold(data_g, labels, 5, 2, [0.5, 1, 1])
kfold(data_g, labels, 5, 3, [0.5, 1, 1])


# In[9]:


kfold(data_z, labels, 5, 1, [0.5, 1, 1])
kfold(data_z, labels, 5, 2, [0.5, 1, 1])
kfold(data_z, labels, 5, 3, [0.5, 1, 1])


# In[ ]:


kfold(Data, labels, 5, 5, [0.5, 1, 1])
kfold(data_z, labels, 5, 4, [0.5, 1, 1])
kfold(data_z, labels, 5, 4, [0.5, 1, 1])

