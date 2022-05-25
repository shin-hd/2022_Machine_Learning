# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
# Libraries added without justification are a minus factor.


# Seed setting
seed_num = 2022
np.random.seed(seed_num)
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
        
    def initialization(self, data, z):
        """ 1.initialization, 10 points
        Initial values for mean, sigma, and pi should be assigned.
        It have a significant impact on performance.
        
        데이터를 임의의 cluster에 배정하고,
        각 cluster별 mean, sigma, pi를 계산.

        Parameters
        ----------
        data (ndarray): Input dataset.
        z (ndarray): Data of each cluster.
        """
        r, _ = np.shape(data) # data 수
        
        # z[n]을 n_clusters에 균등하게 배정
        for n in range(r):
            z[n, n%self.n_clusters] = 1

        # shuffle해서 초기 cluster 랜덤
        np.random.shuffle(z)
        
        # cluster별 mean, sigma, pi 계산
        for k in range(self.n_clusters):
            self.mean[k] = np.mean(data[np.where(z[:,k] == 1)], 0)
            
            mean_centered = data[np.where(z[:,k] == 1)] - self.mean[k]
            self.sigma[k] = mean_centered.T.dot(mean_centered) / (np.sum(z[:,k])-1)
            #self.sigma[k] = (np.cov(data[np.where(z[:,k] == 1)].T))
            
            self.pi[k] = np.sum(z[:,k])/r

        #print(self.mean)
        #print(self.sigma)
        #print(mean_centered)
        #print(np.cov(np.transpose(mean_centered)))
        #print(self.pi)        

        return z
            
    def multivariate_gaussian_distribution(self, data, k, z):
        """ 2.multivariate_gaussian_distribution, 10 points
        Use the linear algebraic functions of Numpy. π of this function is not self.pi
        
        입력 data에 대한 multivariate normal distribution 계산.

        Parameters
        ----------
        data (ndarray): Input data.
        k (int): Cluster of input data.
        z (ndarray): Data of each cluster.

        Returns
        -------
        g (scalar): mvn result.
        """
        
        # 분자부분
        d_m = data-self.mean[k] # 1x4, 데이터값-평균값
        inv = np.linalg.inv(self.sigma[k]) # 4x4, inverse of covariance matrix
        exp = np.exp((-0.5) * (d_m.dot(inv).dot(d_m.T))) # 1x1 = 1x4 . 4x4 . 4x1
        # 분모부분
        det = np.linalg.det(self.sigma[k])
        denom = np.power((2 * np.pi), np.sum(z[:,k])/2) * np.sqrt(det)
        
        # mvn 계산
        g =  exp / denom
        
        return g
    
    def expectation(self, data, z):
        """ 3.expectation step, 20 points
        The multivariate_gaussian_distribution(MVN) function must be used.
        
        입력 data의 cluster별 posterior 계산. 

        Parameters
        ----------
        data (ndarray): Input data.
        z (ndarray): Data of each cluster.

        Returns
        -------
        y (ndarray): data의 cluster에 대한 posterior(NxK).
        """
        r, _ = np.shape(data) # N
        y = np.zeros((r, self.n_clusters)) # N x K
        
        # data의 cluster별 posterior 계산
        for n in range(r):
            for k in range(self.n_clusters):
                g = self.multivariate_gaussian_distribution(data[n], k, z)
                y[n,k] = self.pi[k] * g
                
                # z 초기화
                z[n,k] = 0

            y[n] = y[n]/sum(y[n])

            # posterior가 최대가 되는 cluster에 재배정
            z[n, np.argmax(y[n])] = 1
        
        return y

    def maximization(self, data, y, z):
        """ 4.maximization step, 20 points
        Hint. np.outer
        
        변경된 cluster에 따른 mu, sigma, pi 재계산
        
        Parameters
        ----------
        data (ndarray): Input data.
        z (ndarray): Data of each cluster.
        """
        
        r, _ = np.shape(data) # N
        
        # covariance matrix 초기화
        self.sigma = np.zeros((3,4,4))

        # 각 cluster별로 mu, sigma, pi 재계산
        for k in range(self.n_clusters):
            # 분모
            denom = np.sum(y[np.where(z[:,k] == 1)][:,k])

            # 분자
            num = np.sum([y[n,k]*data[n] for n in np.where(z[:,k] == 1)[0]], 0)
            
            # mean 계산
            self.mean[k] = num/denom

            # sigma 계산
            for n in np.where(z[:,k] == 1)[0]:
                # transpose 함수 사용 가능하도록 array to matrix
                d = np.array(data[n]-self.mean[k])[np.newaxis]
                self.sigma[k] += y[n,k] * np.transpose(d).dot(d)
            self.sigma[k] /= denom
            
            # 기댓값으로 pi 계산
            self.pi[k] = np.sum(y[np.where(z[:,k] == 1)][:,k])/r

        #print(self.mean)
        #print(self.sigma)
        #print(np.cov(np.transpose(data[np.where(z[:,2] == 1)] - self.mean[2])))
        #print(self.pi)

        return # nothing
        
    def fit(self, data):
        """ 5.fit clustering, 20 points
        Functions initialization, expectation, and maximization should be used by default.
        Termination Condition. Iteration is finished or posterior is the same as before. (Beware of shallow copy)
        Prediction for return should be formatted. Refer to iris['target'] format.
        
        GMM으로 학습해서 cluster 분류.

        Parameters
        ----------
        data (ndarray): Input dataset.
        
        Returns
        -------
        prediction (ndarray): 예측한 cluster.
        """
        r,_ = np.shape(data) # N

        # NxK matrix
        # data의 cluster면 1, 아니면 0
        z = np.zeros((r, self.n_clusters)).astype(int) # 0으로 초기화

        # EM 초기화
        self.initialization(data, z)
        
        # z 이전값
        z_prev = copy.copy(z)

        # 최대 iteration만큼 반복
        for _ in range(self.iteration):
            # e step
            y = self.expectation(data, z)
            
            # z가 이전과 같으면 반복 탈출 
            if (z==z_prev).all():
                break
            else: # z_prev 갱신
                z_prev = copy.copy(z)

            # m step
            self.maximization(data, y, z)
        
        # data의 cluster로 구성된 np array(150)
        prediction = np.array([np.argmax(z[i]) for i in range(r)])
        
        return prediction 

def plotting(data:pd.DataFrame):
    """ 6.plotting, 20 points with report
    Default = seaborn pairplot
    
    Parameters
    ----------
    data (DataFrame): dataset to plot
    return None.
    -------
    None.
    """
    # 산점도 행렬을 각 변수별 커널밀도추정곡선을 볼 수 있도록,
    # labels별로 색을 다르게해서, 출력색을 bright로 출력한다.
    sns.pairplot(data, diag_kind='kde', hue="labels", palette='bright')
    plt.show()#block=True)
    return
    
    
if __name__ == '__main__':
    # Loading and labeling data
    iris = datasets.load_iris()
    original_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['labels'])
    original_data['labels'] = original_data['labels'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
    #plotting(original_data)
    
    # Only data is used W/O labels beacause EM and Kmeans are unsupervised learning
    data = iris['data']

    # Unsupervised learning(clustering) using EM algorithm
    EM_model = EM(n_clusters=3, iteration=iteration)
    EM_pred = EM_model.fit(data) # EM으로 학습
    EM_pd = pd.DataFrame(data= np.c_[data, EM_pred], columns= iris['feature_names'] + ['labels'])
    plotting(EM_pd)
    
    # Why are these two elements almost the same? Write down the reason in your report. Additional 10 points
    print(f'pi :            {EM_model.pi}')
    print(f'count / total : {np.bincount(EM_pred) / 150}')
    
    # Unsupervised learning(clustering) using KMeans algorithm
    KM_model = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data)
    KM_pred = KM_model.predict(data)
    KM_pd = pd.DataFrame(data= np.c_[data, KM_pred], columns= iris['feature_names'] + ['labels'])
    plotting(KM_pd)
    
    # No need to explain.
    for idx in range(2):
        EM_point = np.argmax(np.bincount(EM_pred[idx*50:(idx+1)*50]))
        KM_point = np.argmax(np.bincount(KM_pred[idx*50:(idx+1)*50]))
        EM_pred = np.where(EM_pred == idx, 3, EM_pred)
        EM_pred = np.where(EM_pred == EM_point, idx, EM_pred)
        EM_pred = np.where(EM_pred == 3, EM_point, EM_pred)
        KM_pred = np.where(KM_pred == idx, 3, KM_pred)
        KM_pred = np.where(KM_pred == KM_point, idx, KM_pred)
        KM_pred = np.where(KM_pred == 3, KM_point, KM_pred)
    
    EM_hit = np.sum(iris['target']==EM_pred)
    KM_hit = np.sum(iris['target']==KM_pred)
    print(f'EM Accuracy: {round(EM_hit / 150,2)}    Hit: {EM_hit} / 150')
    print(f'KM Accuracy: {round(KM_hit / 150,2)}    Hit: {KM_hit} / 150')