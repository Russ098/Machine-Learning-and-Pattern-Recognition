#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import scipy
import numpy 
import matplotlib.pyplot as plt
from scipy import linalg, special, stats
from numpy import genfromtxt
import ML_support as ml
from scipy.optimize import fmin_l_bfgs_b


# In[14]:


def logreg_obj_wrap(DTR, LTR, l):

    def logreg_obj(v):
        n = DTR.shape[1]
        w, b = v[0:-1], v[-1]
        w = w.reshape((len(w), 1))
        J = 0
        x = DTR
        for idx in range(n):
            if LTR[idx] == 0:
                c = 0
            else:
                c = 1

            J += (c * numpy.log1p(numpy.exp(numpy.dot(-w.T, x[:, idx]) - b)) + (1 - c) * numpy.log1p(numpy.exp(numpy.dot(w.T, x[:, idx]) + b)))

        return l/2 * ((numpy.linalg.norm(w))**2) + 1/n * J

    return logreg_obj


# In[15]:


def logreg(DTR, LTR, DTE, LTE, l):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    v = numpy.zeros((DTR.shape[0]+1), dtype='int32')
    x, f, d = fmin_l_bfgs_b(logreg_obj, v, approx_grad=True)
    w, b = x[0:-1], x[-1]
    
    w = ml.mcol(w)
    S = numpy.zeros((DTE.shape[1]))
    for i in range(len(S)):
        S[i] = numpy.dot(w.T, DTE[:, i]) + b

    PL = numpy.zeros((len(LTE)))

    for i in range(len(S)):
        if(S[i] > 0):
            PL[i] = 1

    PL = PL == LTE
    correctPred = sum(PL)

    e = (len(LTE)-correctPred)/len(LTE)
    print(f'| lambda = {l} |{logreg_obj(x)}|   {round(e*100, 1)}%     |\n')
        


# In[16]:


Data, label = ml.loadFile('./Train.txt')
(DTR, LTR), (DTE,LTE) = ml.split_db_2to1(Data, label, seed=42)
DTR_g, DTE_g = ml.gaussianize(DTR), ml.gaussianize(DTE)
DTR_z, DTE_z = ml.z_normalization(DTR), ml.z_normalization(DTE)


# In[5]:


logreg(DTR_g, LTR, DTE_g, LTE, 0)


# In[6]:


logreg(DTR_g, LTR, DTE_g, LTE, 0.1)


# In[7]:


logreg(DTR_g, LTR, DTE_g, LTE, 0.01)


# In[8]:


logreg(DTR_g, LTR, DTE_g, LTE, 0.001)


# In[9]:


logreg(DTR_g, LTR, DTE_g, LTE, 0.00001)


# In[11]:


logreg(DTR_g, LTR, DTE_g, LTE, 0.00000001)


# In[14]:


logreg(DTR_z, LTR, DTE_z, LTE, 0)


# In[15]:


logreg(DTR_z, LTR, DTE_z, LTE, 0.1)
logreg(DTR_z, LTR, DTE_z, LTE, 0.01)
logreg(DTR_z, LTR, DTE_z, LTE, 0.001)
logreg(DTR_z, LTR, DTE_z, LTE, 0.00001)


# In[16]:


logreg(DTR_z, LTR, DTE_z, LTE, 0.0000001)


# In[18]:


def logreg_mod2(DTR, LTR, DTE, LTE, l):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    v = numpy.zeros((DTR.shape[0]+1), dtype='int32')
    x, f, d = fmin_l_bfgs_b(logreg_obj, v, approx_grad=True)
    w, b = x[0:-1], x[-1]
    
    w = ml.mcol(w)
    S = numpy.zeros((DTE.shape[1]))
    for i in range(len(S)):
        S[i] = numpy.dot(w.T, DTE[:, i]) + b

    return list(S)


# In[19]:


def kfold(D, L, fold, l, app):
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
        llr_final.extend(logreg_mod2(DTR, LTR, DTE, LTE, l))
        
    CM = ml.compute_optimal_B_decision(app, llr_final, LTE_final)
        
    app_bayes_risk=ml.compute_Bayes_risk(CM, app)
    DCF = ml.compute_norm_Bayes(app_bayes_risk, app)

    minDCF= ml.compute_min_DCF(llr_final, app, LTE_final)
    error = 1-(CM[0, 0]+CM[1,1])/(len(LTE_final))

    print("\-/ \-/ \-/ \-/ \-/ ")
    print("Logistic Regression error:", error)
    print(f'{app} DCF:{round(DCF, 3)} minDCF: {round(minDCF,3)}')
    print("/-\ /-\ /-\ /-\ /-\ ")


# In[20]:


d_g = ml.gaussianize(Data)
d_z = ml.z_normalization(Data)


# In[8]:


kfold(d_g, label, 5, 0, [0.5, 1, 1])


# In[9]:


kfold(d_z, label, 5, 0, [0.5, 1, 1])


# In[10]:


kfold(d_z, label, 5, 0, [0.9, 1, 1])


# In[11]:


kfold(d_g, label, 5, 0, [0.9, 1, 1])


# In[11]:


kfold(d_g, label, 5, 0, [0.1, 1, 1])


# In[12]:


kfold(d_z, label, 5, 0, [0.1, 1, 1])


# In[7]:


U = ml.PCAplot(Data)


# In[8]:


Data_pca = ml.PCA(Data, label, U, 5)


# In[10]:


kfold(Data_pca, label, 5, 0, [0.1, 1, 1])


# In[21]:


kfold(d_z, label, 5, 0.1, [0.5, 1, 1])
kfold(d_g, label, 5, 0.1, [0.5, 1, 1])


# In[22]:


kfold(d_z, label, 5, 0.01, [0.5, 1, 1])
kfold(d_g, label, 5, 0.01, [0.5, 1, 1])


# In[23]:


kfold(d_z, label, 5, 0.0001, [0.5, 1, 1])
kfold(d_g, label, 5, 0.0001, [0.5, 1, 1])


# In[24]:


kfold(d_g, label, 5, 0.1, [0.1, 1, 1])
kfold(d_g, label, 5, 0.01, [0.1, 1, 1])
kfold(d_g, label, 5, 0.0001, [0.1, 1, 1])


# In[25]:


kfold(d_z, label, 5, 0.1, [0.1, 1, 1])
kfold(d_z, label, 5, 0.01, [0.1, 1, 1])
kfold(d_z, label, 5, 0.0001, [0.1, 1, 1])


# In[26]:


kfold(d_g, label, 5, 0.1, [0.9, 1, 1])
kfold(d_g, label, 5, 0.01, [0.9, 1, 1])
kfold(d_g, label, 5, 0.0001, [0.9, 1, 1])


# In[27]:


kfold(d_z, label, 5, 0.1, [0.9, 1, 1])
kfold(d_z, label, 5, 0.01, [0.9, 1, 1])
kfold(d_z, label, 5, 0.0001, [0.9, 1, 1])


# In[ ]:




