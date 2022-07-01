#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import numpy 
import matplotlib.pyplot as plt
from scipy import linalg, special, stats
from numpy import genfromtxt
import ML_support as ml
from scipy.optimize import fmin_l_bfgs_b


# In[2]:


Data, labels = ml.loadFile('./Train.txt')


# In[4]:


def L_dual_wrapper(H_hat):
    def L_dual(alpha):
        one_vect = numpy.ones((len(alpha)), dtype='int32')
        L_d = 1/2 * numpy.dot( alpha.T, numpy.dot( H_hat, alpha ) ) - numpy.dot( alpha.T, one_vect )
        grad_L_d = numpy.dot( H_hat, alpha ) - one_vect

        v = numpy.array((L_d, grad_L_d), dtype=object)

        return v
    return L_dual

def radialSVM(DTR, LTR, DTE, params):
    K, C, gamma = params[0], params[1], params[2]
    
    x0 = numpy.zeros((DTR.shape[1]), dtype='int32')
    H_hat = ml.compute_H_hat2(DTR, LTR, K**2, None, None, gamma)

    boundaries = []
    for i in range(DTR.shape[1]):
        boundaries.append((0, C))

    alpha, f, dictionary = fmin_l_bfgs_b(L_dual_wrapper(H_hat), x0, bounds=boundaries, factr=1.0)

    S = ml.compute_score(alpha, DTR, LTR, DTE, K**2, None, None, gamma)
    
   
    return list(S)


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
        llr_final.extend(radialSVM(DTR, LTR, DTE, params))
        
    CM = ml.compute_optimal_B_decision(app, llr_final, LTE_final)

    app_bayes_risk=ml.compute_Bayes_risk(CM, app)
    DCF = ml.compute_norm_Bayes(app_bayes_risk, app)
    
    minDCF, _= ml.compute_min_DCF(llr_final, app, LTE_final)
    
    error = 1-(CM[0, 0]+CM[1,1])/(len(LTE_final))

    print("\-/ \-/ \-/ \-/ \-/ ")
    print("Radial Basis SVM error:", error)
    print(f'{app} DCF:{round(DCF, 3)} minDCF: {round(minDCF,3)}')
    print("/-\ /-\ /-\ /-\ /-\ ")


# In[6]:


kfold(Data, labels, 5, [0, 1, 1], [0.5, 1, 1])
kfold(Data, labels, 5, [0, 1, 10], [0.5, 1, 1])


# In[7]:


kfold(Data, labels, 5, [1, 1, 1], [0.5, 1, 1])
kfold(Data, labels, 5, [1, 1, 10], [0.5, 1, 1])


# In[8]:


data_g = ml.gaussianize(Data)
data_z = ml.z_normalization(Data)


# In[9]:


kfold(data_g, labels, 5, [0, 1, 1], [0.5, 1, 1])
kfold(data_g, labels, 5, [0, 1, 10], [0.5, 1, 1]) 
kfold(data_g, labels, 5, [1, 1, 1], [0.5, 1, 1])
kfold(data_g, labels, 5, [1, 1, 10], [0.5, 1, 1])


# In[10]:


kfold(data_z, labels, 5, [0, 1, 1], [0.5, 1, 1])
kfold(data_z, labels, 5, [0, 1, 10], [0.5, 1, 1])
kfold(data_z, labels, 5, [1, 1, 1], [0.5, 1, 1])
kfold(data_z, labels, 5, [1, 1, 10], [0.5, 1, 1])

