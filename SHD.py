# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
# Libraries added without justification are a minus factor.
from sklearn.mixture import GaussianMixture
    


# Seed setting
seed_num = 2022
#np.random.seed(seed_num)
iteration = 100     # Number of times to repeat steps E and M.


class EM:
    """ expectation-maximization algorithm, EM algorithm
    The EM class is a class that implements an EM algorithm using GMM and kmeans.
    
    Within the fit function, the remaining functions should be used.
    Other functions can be added, but all tasks must be implemented with Python built-in functions and Numpy functions.
    You should annotate each function with a description of the function and parameters(Leave a comment).
    """
    def __init__(self, n_clusters, iteration):
        """
        Parameters
        ----------
        n_clusters (int): Num of clusters (num of GMM)
        iteration (int): Num of iteration 
            Termination conditions if the model does not converge
        mean (ndarray): Num of clusters x Num of features
            The mean vector that each cluster has.
        sigma (ndarray): Num of clusters x Num of features x Num of features     
            The covariance matrix that each cluster has.
        pi (ndarray): Num of labels (num of clusters)
            z(x), Prior probability that each cluster has.
        return None.
        -------
        None.
        """
        self.n_clusters = n_clusters   
        self.iteration = iteration  
        self.mean = np.zeros((3,4)) 
        self.sigma = np.zeros((3,4,4)) 
        self.pi = np.zeros((3))
        
    def initialization(self, clusters): # your parameter here): 
        """ 1.initialization, 10 points
        Initial values for mean, sigma, and pi should be assigned.
        It have a significant impact on performance.
        
        your comment here
        """
        # your code here
        
        
        r, _ = np.shape(data)
        
        # 각 cluster별 mean, covariance 계산
        for i in range(self.n_clusters - 1):
            clusters.append([i])
            self.mean[i] = data[i]
            self.sigma[i] = np.cov(data[i])
            self.pi[i] = 1/r
        clusters.append([i for i in range(2, r)])
        self.mean[self.n_clusters - 1] = np.mean(data[self.n_clusters:], 0)
        self.sigma[self.n_clusters - 1] = np.cov(np.transpose(data[self.n_clusters:]))
        self.pi[self.n_clusters - 1] = (r-self.n_clusters+1) / r
        #print(data - self.mean[0])
        #print(self.mean)
        #print(self.sigma)
        #print(r-self.n_clusters+1)
        

        return # something or nothing
            
    def multivariate_gaussian_distribution(self, cluster): # your parameter here):
        """ 2.multivariate_gaussian_distribution, 10 points
        Use the linear algebraic functions of Numpy. π of this function is not self.pi
        
        your comment here
        """
        # your code here
        # 전체 dataset에 대한 cluster별 mgd 계산
        exp = np.exp((-0.5) * (data[i]-self.mean[cluster]).dot(np.invert(self.sigma[cluster])).dot(np.transpose(data[cluster]-self.mean[cluster])))
        denom = ((2 * np.pi) ** (1.0 / len(cluster))) * np.sqrt(np.linalg.det(self.sigma[cluster]))
        g =  exp / denom

        return g# something or nothing
    
    def expectation(self, clusters): # your parameter here):
        """ 3.expectation step, 20 points
        The multivariate_gaussian_distribution(MVN) function must be used.
        
        your comment here
        """
        # your code here
        # data별로 각 cluster에 대한 expectation 계산
        y = np.zeros((len(data), self.n_clusters)) # N x K
        for i in range(self.n_clusters):
            y[:,i] = self.pi[i] * self.multivariate_gaussian_distribution(clusters[i]) # / marginal


        return # something or nothing

    def maximization(self, clusters): # your parameter here): 
        """ 4.maximization step, 20 points
        Hint. np.outer
        
        your comment here
        """
        # your code here
        # cluster별로 Max한
        r, _ = np.shape(data)
        for k in range(self.n_clusters):
            self.mean[k] = np.mean(data[clusters[k]], 0)
            self.sigma[k] = np.cov(np.transpose(data[self.n_clusters:]))
            self.pi[self.n_clusters - 1] = len(clusters[k])/r

        return # something or nothing
        
    def fit(self): # your parameter here):
        """ 5.fit clustering, 20 points
        Functions initialization, expectation, and maximization should be used by default.
        Termination Condition. Iteration is finished or posterior is the same as before. (Beware of shallow copy)
        Prediction for return should be formatted. Refer to iris['target'] format.
        
        your comment here
        """
        # your code here
        clusters = []
        self.initialization(clusters)
        
        for _ in range(self.iteration):
            self.expectation()

            self.maximization()
        
        prediction = 1# np array (150) as assigned by labels 0, 1, 2
        return prediction 

def plotting(data:pd.DataFrame):
    """ 6.plotting, 20 points with report
    Default = seaborn pairplot
    
    Parameters
    ----------
    data (DataFrame): dataset to plotting
    return None.
    -------
    None.
    """
    # your code here
    # 산점도 행렬을 각 변수별 커널밀도추정곡선을 볼 수 있도록,
    # labels별로 색을 다르게해서, 출력색을 bright로 출력한다.
    sns.pairplot(data, diag_kind='kde', hue="labels", palette='bright')
    plt.show(block=True)
    return # something or nothing
    
    
if __name__ == '__main__':
    # Loading and labeling data
    iris = datasets.load_iris()
    original_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['labels'])
    original_data['labels'] = original_data['labels'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
    plotting(original_data)
    
    # Only data is used W/O labels beacause EM and Kmeans are unsupervised learning
    data = iris['data']
    
    # Unsupervised learning(clustering) using EM algorithm
    EM_model = EM(n_clusters=3, iteration=iteration)
    # EM_pred = EM_model.fit()# your parameter here)
    # EM_pd = pd.DataFrame(data= np.c_[data, EM_pred], columns= iris['feature_names'] + ['labels'])
    # plotting()# your parameter here)
    
    # # Why are these two elements almost the same? Write down the reason in your report. Additional 10 points
    # print(f'pi :            {EM_model.pi}')
    # print(f'count / total : {np.bincount(EM_pred) / 150}')
    
    # # Unsupervised learning(clustering) using KMeans algorithm
    # KM_model = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data)
    # KM_pred = KM_model.predict(data)
    # KM_pd = pd.DataFrame(data= np.c_[data, KM_pred], columns= iris['feature_names'] + ['labels'])
    # plotting()# your parameter here)
    
    # # No need to explain.
    # for idx in range(2):
    #     EM_point = np.argmax(np.bincount(EM_pred[idx*50:(idx+1)*50]))
    #     KM_point = np.argmax(np.bincount(KM_pred[idx*50:(idx+1)*50]))
    #     EM_pred = np.where(EM_pred == idx, 3, EM_pred)
    #     EM_pred = np.where(EM_pred == EM_point, idx, EM_pred)
    #     EM_pred = np.where(EM_pred == 3, EM_point, EM_pred)
    #     KM_pred = np.where(KM_pred == idx, 3, KM_pred)
    #     KM_pred = np.where(KM_pred == KM_point, idx, KM_pred)
    #     KM_pred = np.where(KM_pred == 3, KM_point, KM_pred)
    
    # EM_hit = np.sum(iris['target']==EM_pred)
    # KM_hit = np.sum(iris['target']==KM_pred)
    # print(f'EM Accuracy: {round(EM_hit / 150,2)}    Hit: {EM_hit} / 150')
    # print(f'KM Accuracy: {round(KM_hit / 150,2)}    Hit: {KM_hit} / 150')
    