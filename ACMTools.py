#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:54:11 2024

@author: samuelslocum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import date
import itertools
import sys


#Main class, holds the procedures to estimate the ACM model as well as other procedures for post-estimation functionality. 
class AMCModel:
    #Initialization with baseline data
    #Input: Excess return data (RX), yields (Y), risk-free rate (RF) and the number of factors (K)
    def __init__(self,RX,Y,RF,K):
        self.RX = RX
        self.Y = Y
        self.RF = RF
        self.T = RF.shape[1]
        self.K = K
        
    #This is the main function to fit the model.
    def fit(self):
        

        print("Estimating a K="+str(self.K)+" factor model")
        self.X,self.delta_0,self.delta_1,self.mu,self.lambda_0,self.Sigma,self.little_Sigma,self.lambda_1,self.phi,self.R = estimate_acm(self.RX,self.Y,self.RF,self.K)
        
        #Compute the model coefficients given the estimates
        self.A,self.B = compute_coeffs(self.delta_0,self.delta_1,self.mu,self.lambda_0,self.Sigma,self.little_Sigma,self.lambda_1,self.phi)
        
        #Compute without prices of risk -- this corresponds to interest rate expectations.
        self.A_nolambda,self.B_nolambda = compute_coeffs(self.delta_0,self.delta_1,self.mu,np.zeros((self.K,1)),self.Sigma,self.little_Sigma,np.zeros((self.K,self.K)),self.phi)
        
        #Use these to compute the prices of various maturities
        prices_fit = (self.B.transpose().dot(self.X)+self.A.transpose())
        prices_fit_nolambda = (self.B_nolambda.transpose().dot(self.X)+self.A_nolambda.transpose())
        
        #convert from prices to continuously-compounded yields
        self.maturities = np.repeat(np.array([range(1,121)]).transpose(),(self.T+1),1)
        self.yields_fit = -12*prices_fit/self.maturities
        self.yields_fit_nolambda = -12*prices_fit_nolambda/self.maturities

    #This saves the parameters to a file.
    def save_params(self, csv_filename=None):
        
        #Stack the parameters for output in a csv file
        #We have to create labels for the parameters
        label_delta0 = ["delta0"]
        label_delta1 = ["delta1_"+str(i) for i in range(len(self.delta_1))]
        label_mu = ["mu_"+str(i) for i in range(len(self.mu))]
        label_phi = list(itertools.chain.from_iterable([["phi_"+str(i)+"_"+str(j) for i in range(self.phi.shape[0])] for j in range(self.phi.shape[1])]))
        label_lambda0 = ["lambda0_"+str(i) for i in range(len(self.lambda_0))]
        label_lambda1 = list(itertools.chain.from_iterable([["lambda1_"+str(i)+"_"+str(j) for i in range(self.lambda_1.shape[0])] for j in range(self.lambda_1.shape[1])]))
        label_Sigma = list(itertools.chain.from_iterable([["Sigma_"+str(i)+"_"+str(j) for i in range(self.Sigma.shape[0])] for j in range(self.Sigma.shape[1])]))
        label_little_sigma = ["little_Sigma"]

        labels = list(itertools.chain.from_iterable([label_delta0,label_delta1,label_mu,label_phi,label_lambda0,label_lambda1,label_Sigma,label_little_sigma]))
        values = np.concatenate([self.delta_0,self.delta_1,self.mu,self.phi.reshape(-1,1),self.lambda_0,self.lambda_1.reshape(-1,1),self.Sigma.reshape(-1,1),np.array([[self.little_Sigma]])],0)
        
        if csv_filename is None:
            csv_filename = "ACMmodel_params_"+date.today().strftime("%Y%m%d")+".csv"
            print("Dumping estimated model parameters in "+csv_filename)
       
        pd.DataFrame(data=values,index=labels,columns=["Value"]).to_csv(csv_filename,index_label="Parameter")
        
    #This takes a set of forecasts for the short rate (target) over various horizons (proj_horizon)
    #It walks the yield curve forward conditional on the short rate passing through these values.
    def walk_forward(self, proj_horizon, target):
        
        
        X0 = self.X[:,[-1]]
        P0 = np.eye(self.K)*0.0
        V = self.delta_0.transpose()
        H = self.delta_1.transpose()
        alpha=0.75
        
        curve_projection = {}
        
        for k in range(len(proj_horizon)):
            
        
            nan_list = [np.nan for x in range(proj_horizon[k])]
    
            for i in range(k+1):
                nan_list[proj_horizon[i]-1] = target[i]/12
                
            Z = np.array([nan_list])
            X_filt,P0 = ForwardInference(self.phi,self.Sigma,self.mu,X0,P0,V,H,0*self.R,Z)
    
            #Now generate some conifdence intervals
            rand_data = np.random.normal(loc=0, scale=1, size=[self.K,10000])
            rnums = np.linalg.cholesky(P0).dot(rand_data)+X_filt[:,[-1]]
            
            curve_projection[proj_horizon[k]] = {}
            prices_fit_fwd1y = (self.B.transpose().dot(X_filt[:,[proj_horizon[k]-1]])+self.A.transpose())
            yields_fit_forward = (-12*prices_fit_fwd1y/self.maturities[:,[-1]])
            curve_projection[proj_horizon[k]]["expected"] = yields_fit_forward
            
            yff = []
            for i in range(10000):
                prices_fit_fwd1y_sample = (self.B.transpose().dot(rnums[:,[i]])+self.A.transpose())
                yields_fit_forward_sample = (-12*prices_fit_fwd1y_sample/self.maturities[:,[-1]])
                yff.append(yields_fit_forward_sample)
            ysample = np.concatenate(yff,1)
            
            y_low = np.quantile(ysample,alpha,1)
            y_high = np.quantile(ysample,1-alpha,1)
    
            curve_projection[proj_horizon[k]]["high"] = y_high
            curve_projection[proj_horizon[k]]["low"] = y_low
            
        return curve_projection


#FUNCTION: fit_yields_returns
#INPUT: The yield tenors we want to produce (MONTHS). The Svennson yield curve parameter dataframe (monthly_data), the length of the final panel (T).
#OUTPUT: The fitted excess returns (RX), yields (Y), and risk-free rate (RF) panels based off of the Svensson model.
#Notes: This takes the raw monthly yield curve data and generates the inputs for the ACM model.
def fit_yields_returns(MONTHS, monthly_data, T):
    #Construct the yields using those parameters
    yield_df = Svensson(MONTHS,monthly_data)
    
    MONTHS = yield_df.columns
    
    log_prices_df = -(1/12)*np.repeat([MONTHS],len(yield_df),0)*yield_df
    yc_today = yield_df.iloc[-1]
    date_final=yield_df.iloc[-1].name
    print("Today's Date: "+date.today().strftime("%B %d, %Y"))
    print("Latest Fed data: "+date_final.strftime("%B %d, %Y"))

    #Also construct the prices of those assets in the next period
    MONTHS_forward = [x-1 for x in MONTHS]
    yield_df_forward = Svensson(MONTHS_forward,monthly_data)
    yield_df_forward.columns=MONTHS
    log_prices_df_forward = (-(1/12)*np.repeat([MONTHS_forward],len(yield_df_forward),0)*yield_df_forward).shift(-1)

    #Construct the risk-free rate as well. This should be in monthly terms
    rf_df = Svensson([1],monthly_data)/12
    rf_panel = pd.concat([rf_df] * len(MONTHS), axis=1, ignore_index=True)
    rf_panel.columns = MONTHS

    #This is a panel of the excess-returns of zero-coupon bonds with maturities defined by MONTHS
    rx_df = log_prices_df_forward-log_prices_df-rf_panel
    
    # The model transitions to use numpy matrices instead of pandas dataframes, so we do the conversion here.
    Y = yield_df.values[-(T+1):,:].transpose()
    RX = rx_df.values[-(T+1):-1,:].transpose()
    RF = rf_df.values[-(T+1):-1,:].transpose()
    
    return RX,Y,RF


#FUNCTION: pull_yc_data
#INPUT: link(or path) yo the Fed's yield curve data
#OUTPUT: Monthly yield curve data
#Notes: Pulls the yield curve data from the specified link, returns monthly yield curve parameters
def pull_yc_data(link):
    #Pull the data at the link, format dates
    storage_options = {'User-Agent': 'Mozilla/5.0'}
    raw_data = pd.read_csv(link,header=7,storage_options= storage_options)
    raw_data.Date = pd.to_datetime(raw_data.Date)
    raw_data = raw_data.set_index('Date')    
    #Drop all days where one of the critical parameters are nan
    raw_data = raw_data.dropna(subset=['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2'])
    #Use the last yield curve in each month for estimation
    monthly_data = raw_data.groupby(100*raw_data.index.year+raw_data.index.month).tail(1)
    return monthly_data


#FUNCTION: Svensson
#INPUT: Desired maturities (in months), A panel of yield curve parameters from the Fed
#OUTPUT: Fitted yields for the desired maturities
#Notes: Uses the Svensson model defined in the source: https://www.federalreserve.gov/pubs/feds/2006/200628/200628abs.html
def Svensson(maturities_months,monthly_data):
    #Pull the parameters out of the panel
    beta0 = monthly_data['BETA0']
    beta1 = monthly_data['BETA1']
    beta2 = monthly_data['BETA2']
    beta3 = monthly_data['BETA3']
    tau1 = monthly_data['TAU1']
    tau2 = monthly_data['TAU2']
    
    v1 = pd.concat([beta0,beta1,beta2,beta3],axis=1)
    Total_Obs = len(v1)
    dates = v1.index
    one_vector = pd.DataFrame(np.ones((Total_Obs,1)),index=dates)
    
    Fitted_yield_panels = []
    for month in maturities_months:
        #Compute the Svensson fitted yields according to parameters
        year = month/12
        v2 = pd.concat([one_vector,(1-np.exp(-year/tau1))/(year/tau1),(1-np.exp(-year/tau1))/(year/tau1)-np.exp(-year/tau1),(1-np.exp(-year/tau2))/(year/tau2)-np.exp(-year/tau2)],axis=1)
        Fitted_yield_panels.append(pd.DataFrame(np.sum(v2.values*v1.values,1),index=dates,columns=[month]))
    return pd.concat(Fitted_yield_panels,axis=1)/100


#FUNCTION: regress
#INPUT: X and Y data, intercept flag
#OUTPUT: coefficients + intercept + fitted residuals
#Notes: Standard linear regression
def regress(X,Y,const=False):
    if const:
        X_ = np.concatenate([np.ones((1,X.shape[1])),X],axis=0)
        COEF = Y.dot(X_.transpose().dot(np.linalg.inv(X_.dot(X_.transpose()))))
        intercept = COEF[:,[0]]
        beta = COEF[:,1:]
        residuals = Y-COEF.dot(X_)
        return intercept,beta,residuals
    else:
        beta = Y.dot(X.transpose()).dot(np.linalg.inv(X.dot(X.transpose())))
        residuals = Y-beta.dot(X)
        return beta,residuals
    

#FUNCTION: compute_coeffs
#INPUT: The fitted parameters required to compute the ACM model
#OUTPUT: The coefficients that can be used to compute an ACM-fitted yield curve.
#Notes: Computes Coefficients for the ACM model
def compute_coeffs(delta_0,delta_1,mu,lambda_0,Sigma_hat,little_sigma_hat,lambda_1,phi):

    A = -delta_0
    B = -delta_1
    
    Alist = [A]
    Blist = [B]
    
    for n in range(119):
        A = A+B.transpose().dot(mu-lambda_0)+.5*(B.transpose().dot(Sigma_hat.dot(B))+little_sigma_hat)-delta_0
        B = -delta_1+(phi-lambda_1).transpose().dot(B)
        Alist.append(A)
        Blist.append(B)
    
    
    Avec = np.concatenate(Alist,1)
    Bvec = np.concatenate(Blist,1)
    
    return Avec, Bvec

#FUNCTION: estimate_acm
#INPUT: Excess returns for a panel of bonds, yields for those same bonds, a short rate, and a number of factors K.
#OUTPUT: The estimate ACM model parameters
#Notes: Estimates the acm model using the 3-step procedure described in Adrian, Crump, Moench (2013)
def estimate_acm(RX,Y,RF,K):

    #The factors in the yield curve model will be the principal-components
    pca = PCA(n_components=K, svd_solver='full')
    X = pca.fit_transform(Y.transpose()).transpose()
    X_lag = X[:,:-1]
    T = X_lag.shape[1]
    N = Y.shape[0]

    #STEP 1: regress X on its' lag
    mu,phi,V = regress(X[:,:-1],X[:,1:],const=True)
    Sigma = V.dot(V.transpose())/T
    
    #STEP 2: time series regression of RX on a constant, X, and V
    Z = np.concatenate([V.transpose(),X_lag.transpose()],1).transpose()
    a,beta_c,E = regress(Z,RX,const=True)
    beta = beta_c[:,:K].transpose()
    c = beta_c[:,(K):]
    little_Sigma = np.trace(E.dot(E.transpose()))/(N*T)
    
    #STEP 3: cross sectional regression
    B_star = np.concatenate([beta[:,[i]].dot(beta[:,[i]].transpose()).reshape(K**2,1) for i in range(N)],axis=1).transpose()
    M1 = np.linalg.inv(beta.dot(beta.transpose()))
    M2 = a + .5*(B_star.dot(Sigma.reshape(K**2,1))+little_Sigma*np.ones((N,1)))
    lambda_0 = M1.dot(beta.dot(M2))
    lambda_1 = np.linalg.inv(beta.dot(beta.transpose())).dot(beta).dot(c)
    
    #Finally, estimate delta_0, delta_1 by regression.
    delta_0_trans,delta_1_trans,E = regress(X_lag,RF,const=True)
    delta_0 = delta_0_trans.transpose()
    delta_1 = delta_1_trans.transpose()
    R = E.dot(E.transpose())/T

    
    return X,delta_0,delta_1,mu,lambda_0,Sigma,little_Sigma,lambda_1,phi,R

#FUNCTION: ForwardInference
#INPUT: See wikipedia page for Kalman filtering
#phi: The state transition matrix.
#mu: Constant in state transition.
#Sigma: Variance on state transition.
#X0: Initial value
#P0: Initial variance (could be small)
#V: Constant in observation equation
#H: Coefficient in observation equation.
#R: Covariance in observation equation.
#Z: Observations
#OUTPUT: The forward-filtered factors (Xfilt) as well as the point estimate covariances (P0)
#Notes: This just uses Kalman filtering based on the observations found in Z using the dynamic factor system laid out by the parameters.
def ForwardInference(phi,Sigma,mu,X0,P0,V,H,R,Z):
    #State Transition
    F = phi
    Q = Sigma
    #H = ACM.B[:,[0,119]].transpose()
    #V = ACM.A[:,[0,119]].transpose()
    #R = np.zeros([2,2])
    B = mu
    #X0 = self.X[:,[-1]]
    #P0 = self.P_term
    
    nobs = Z.shape[1]
    X_filt = np.zeros([X0.shape[0],nobs])
    X_filt_Km1 = np.zeros([X0.shape[0],nobs])

    sz = X0.shape[0]
    t=0
    
    P_list = []
    C_list = []
    
    while t<=nobs-1:
        
        #This is x_K+1|K
        X1 = F.dot(X0)+B
        P1 = F.dot(P0).dot(F.transpose())+Q
        C_list.append(P0.dot(F.transpose().dot(np.linalg.inv(P1))))
        
        X_filt_Km1[:,[t]] = X1
        
        Z_candidate = Z[:,[t]]
        selection = list(~np.isnan(Z_candidate).transpose()[0])
        
        Z_tilde = Z_candidate[selection,:]
        
        if len(Z_tilde)>0:
            
            H_tilde = H[selection,:]
            V_tilde = V[selection,:]
        
            y_tilde=Z_tilde-H_tilde.dot(X1)-V_tilde
            
            
            S = H_tilde.dot(P1).dot(H_tilde.transpose())+R[selection,selection]
            K = P1.dot(H_tilde.transpose()).dot(np.linalg.inv(S))
            X0 = X1 + K.dot(y_tilde)
            
            X_filt[:,[t]] = X0
            P0 = (np.eye(sz)-K.dot(H_tilde)).dot(P1)
            
        else:
            P0 = P1
            X0 = X1
            X_filt[:,[t]] = X0
            
        t=t+1
        P_list.append(P0)
        
    #Can we now apply an RTF smoother?
    times = list(range(X_filt.shape[1]))
    times.reverse()
    Xinit = X_filt[:,[-1]]
    for k in times[1:-1]:
        Xinit = X_filt[:,[k-1]]+C_list[k].dot(Xinit-X_filt_Km1[:,[k]])
        X_filt[:,[k]] = Xinit
        
    return X_filt,P0


#Just iterates forward to compute the implied path of interest rates
def FPath(X0,months,phi,mu,Sigma):
    
    S = X0.shape[0]
    P0 = np.zeros([X0.shape[0],X0.shape[0]])
    
    I = np.zeros([S,months])

    for t in range(months):
        X0 = phi.dot(X0)+mu
        P0 = phi.dot(P0).dot(phi.transpose())+Sigma
        I[:,[t]] = X0[:,[-1]]
    
    return I





    