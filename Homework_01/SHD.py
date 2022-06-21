# -*- coding: utf-8 -*-

import numpy as np
      
def feature_normalization(data): # 10 points
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]

    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num])
    mu = np.zeros([feature_num])
    std = np.zeros([feature_num])
    
    # your code here
    
    # 각 feature의 mean, std 얻어서
    mu = np.mean(data, 0)
    std = np.std(data, 0)
    # data가 표준정규분포 속성 가지도록 data-mu / std
    normal_feature = ((data + np.negative(mu))/std)
    
    # end

    return normal_feature
        
def split_data(data, label, split_factor):
    return  data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:]

def get_normal_parameter(data, label, label_num): # 20 points
    # parameter
    feature_num = data.shape[1]
    
    # you should get this parameter correctly    
    mu = np.zeros([label_num,feature_num])
    sigma = np.zeros([label_num,feature_num])

    # your code here

    # 각 label에 대해서
    for i in range(label_num):
        # 특정 label의 data index
        index = np.where(label == i)
        # index와 data를 가지고 mean, std 계산
        mu[i, :] = np.mean(data[index], 0)
        sigma[i, :] = np.std(data[index], 0)

    # end
    
    return mu, sigma

def get_prior_probability(label, label_num): # 10 points
    # parameter
    data_point = label.shape[0]
    
    # you should get this parameter correctly
    prior = np.zeros([label_num])
    
    # your code here

    # 각 label에 대해서
    for i in range(label_num):
        # label의 확률 계산
        prior[i] = np.size(np.where(label == i)) / data_point

    # end

    return prior

def Gaussian_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    pdf = 0
    
    # your code here

    # 정규분포 pdf식에 mu, sigma, x를 대입해서 pdf값 계산
    pdf = np.exp(- (x-mu)**2 / (2*(sigma**2)) ) / (np.sqrt(2*np.pi)*sigma)
    
    # end
    
    return pdf

def Gaussian_Log_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    log_pdf = 0
    
    # your code here

    # log를 취한 정규분포 pdf식에 mu, sigma, x를 대입해서 log_pdf값 계산
    log_pdf = -np.log(np.sqrt(2*np.pi)*sigma) - (x-mu)**2 / (2*(sigma**2))
    
    # end
    
    return log_pdf

def Gaussian_NB(mu, sigma, prior, data): # 40 points
    # parameter
    data_point = data.shape[0]
    label_num = mu.shape[0]
    
    # you should get this parameter correctly   
    likelihood = np.zeros([data_point, label_num])
    posterior = np.zeros([data_point, label_num])
    ## evidence can be ommitted because it is a constant
    
    # your code here
        ## Function Gaussian_PDF or Gaussian_Log_PDF should be used in this section

    # 각 data에 대해
    for i in range(data_point): #100
        # 각 label에 대해
        for j in range(label_num): #3
            # 각 feature와 특정 label의 Gaussian_Log_PDF 합을 통해 likelihood 계산
            # Log_PDF이므로 sum으로 계산가능
            likelihood[i, j] = np.sum(Gaussian_Log_PDF(data[i,:], mu[j,:], sigma[j,:]))
            # likelihood에 log를 취한 prior를 더해서 posterior 계산
            # p(x)는 uniform이므로 argmax 계산에 영향을 끼치지 않음
            posterior[i, j] = likelihood[i, j] + np.log(prior[j])

    # end
    
    return posterior

def classifier(posterior):
    data_point = posterior.shape[0]
    prediction = np.zeros([data_point])
    
    # posterior를 최대로 하는 label 선택
    prediction = np.argmax(posterior, axis=1)
    
    return prediction
        
def accuracy(pred, gnd):
    data_point = len(gnd)
    hit_num = np.sum(pred == gnd)

    # 예측값과 실제값을 비교해서 정확도 측정
    return (hit_num / data_point) * 100, hit_num

    ## total 100 point you can get 