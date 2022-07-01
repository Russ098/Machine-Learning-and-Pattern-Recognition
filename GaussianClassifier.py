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


# In[2]:


class GaussianClassifier:
    def __init__(self):
        self.C = {}
        self.mu = {}

    def train(self, DTR, LTR):
        self.mu, self.C = ml.MU_Cov_calculator(DTR, LTR)

    def test(self, DTE, LTE):
        S = numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        predicted = []

        for i in numpy.unique(LTE):
            S[i, :] = ml.GAU_logpdf_ND(DTE, self.mu[i], self.C[i]) + numpy.log(1 / 2)

        Sp = scipy.special.logsumexp(S, axis=0)

        for x, p in zip(S.T, Sp): 
            tmp = x - p
            predicted.append(numpy.argmax(tmp))

        predicted = numpy.array(predicted)

        True_prediction = numpy.array([predicted == LTE])

        error = 1 - (numpy.count_nonzero(True_prediction) / True_prediction.size)

        print("Gaussian Classifier error:", error)
        return error


# In[2]:


Data, labels = ml.loadFile('./Train.txt')


# In[4]:


(DTR, LTR), (DTE,LTE) = ml.split_db_2to1(Data, labels, seed=42)


# In[5]:


G=GaussianClassifier()
G.train(DTR,LTR)
G.test(DTE,LTE)


# In[6]:


Gn=GaussianClassifier()
Gn.train(ml.z_normalization(DTR), LTR)
Gn.test(DTE, LTE)
Gn.test(ml.z_normalization(DTE), LTE)


# In[7]:


GG=GaussianClassifier()
GG.train(ml.gaussianize(DTR), LTR)
GG.test(DTE, LTE)
GG.test(ml.gaussianize(DTE), LTE)


# In[3]:


class GaussianClassifier_mod2:
    def __init__(self):
        self.C = {}
        self.mu = {}

    def train(self, DTR, LTR):
        self.mu, self.C = ml.MU_Cov_calculator(DTR, LTR)

    def test(self, DTE, LTE):
        S = numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        ll = numpy.zeros((numpy.unique(LTE).size, DTE.shape[1]))
        predicted = []

        for i in numpy.unique(LTE):
            ll[i, :] = ml.GAU_logpdf_ND(DTE, self.mu[i], self.C[i])
        
        return list(ll[1, :]-ll[0, :])
        
        
       


# In[4]:


def kfold(classifier, D, L, fold, app):
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
        classifier.train(DTR, LTR)
        llr_final.extend(classifier.test(DTE, LTE))
        
    CM = ml.compute_optimal_B_decision(app, llr_final, LTE_final)
        
    app_bayes_risk=ml.compute_Bayes_risk(CM, app)
    DCF = ml.compute_norm_Bayes(app_bayes_risk, app)

    minDCF= ml.compute_min_DCF(llr_final, app, LTE_final)
    error = 1-(CM[0, 0]+CM[1,1])/(len(LTE_final))

    print("\-/ \-/ \-/ \-/ \-/ ")
    print("Gaussian Classifier error:", error)
    print(f'{app} DCF:{round(DCF, 3)} minDCF: {round(minDCF,3)}')
    print("/-\ /-\ /-\ /-\ /-\ ")


# In[5]:


kg=GaussianClassifier_mod2()


# In[8]:


Data_g=ml.gaussianize(Data)
Data_z=ml.z_normalization(Data)


# In[35]:



kfold(kg, Data, labels, 5, [0.5, 1, 1])


# In[36]:


kfold(kg, Data, labels, 5, [0.9, 1, 1])


# In[37]:


kfold(kg, Data, labels, 5, [0.1, 1, 1])


# In[9]:


kfold(kg, Data_g, labels, 5, [0.5, 1, 1])


# In[10]:


kfold(kg, Data_g, labels, 5, [0.9, 1, 1])


# In[11]:


kfold(kg, Data_g, labels, 5, [0.1, 1, 1])


# In[12]:


kfold(kg, Data_z, labels, 5, [0.5, 1, 1])


# In[13]:


kfold(kg, Data_z, labels, 5, [0.9, 1, 1])


# In[14]:


kfold(kg, Data_z, labels, 5, [0.1, 1, 1])


# In[6]:


U = ml.PCAplot(Data)


# In[7]:


Data_pca = ml.PCA(Data, labels, U, 5)


# In[9]:


kfold(kg, Data_pca, labels, 5,  [0.5, 1, 1])


# In[10]:


kfold(kg, Data_pca, labels, 5,  [0.1, 1, 1])


# In[11]:


kfold(kg, Data_pca, labels, 5,  [0.9, 1, 1])


# In[ ]:




