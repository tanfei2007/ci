import pandas as pd
import numpy as np
import glob, os, re
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from tabulate import tabulate 
from regressors import stats
import sympy
import copy
from scipy.stats import norm
from DeepEstimator import *
from summary import *


class doubly_robust:
    def __init__(self, df_Xy_Z, y_idx, XZ_start_idx):
        self.df_Xy_Z = df_Xy_Z
        self.y_idx = y_idx
        self.XZ_start_idx = XZ_start_idx
        
    def est_ate_continuous(self, t_name, cols, t_level):
        
        t = self.df_Xy_Z[t_name].values # retrieve treatments
        XZ = self.df_Xy_Z.iloc[:, self.XZ_start_idx:][cols].values # retrieve confounders and other treatments
        y = self.df_Xy_Z.iloc[:, self.y_idx].values
        
        
        
        # summary statistics
        s = Summary(XZ, y, t, 'c', t_name, cols)
        s.basic_statistics()
        
        
        
        est_con = est_continuous(XZ, y, t, t_name[0])
        est_con.ps_est_fit()
        est_con.outcome_est_fit()
        
        print(est_con.sigma)

        res = [est_con.outcome_est_pred(i) for i in t_level]
        
        #return est_con.t_mu
        
        return res
    
    
    def est_ate_discrete(self, t_name, cols):
    #estimate average treatment effects
        #t_idx = [self.df_Xy_Z.columns.get_loc(t) for t in t_name]
        M = len(t_name)
        if M == 1: # binary 
            t = self.df_Xy_Z[t_name].values # retrieve treatments
            t_label = t.ravel()
            XZ = self.df_Xy_Z.iloc[:, self.XZ_start_idx:][cols].values # retrieve confounders and other treatments
            y = self.df_Xy_Z.iloc[:, self.y_idx].values
            
            est_bin = est_discrete(XZ, y, t, 'b', t_name[0]) # initialization
            
            
            # summary statistics
            s = Summary(XZ, y, t, 'b', t_name[0], cols)
            s.basic_statistics()
            
            
            # propensity score
            est_bin.ps_est_fit()
            ps = est_bin.ps_est_pred()
            ps = ps.ravel() # flattent to one-dimension array
            
            
            # predict outcomes
            model = est_bin.outcome_est_fit()
            y_1 = est_bin.outcome_est_pred(model, np.ones_like(t))
            y_0 = est_bin.outcome_est_pred(model, np.zeros_like(t))
            y_1 = y_1.ravel()
            y_0 = y_0.ravel()
            
            
            # doubly robust
            idx_1 = np.where(t_label==1)
            idx_0 = np.where(t_label==0)
            dr1 = np.zeros(len(y))
            dr0 = np.zeros(len(y))
           
            dr1[idx_1] = y[idx_1]/ps[idx_1] - y_1[idx_1]*(1-ps[idx_1])/ps[idx_1]
            dr1[idx_0] = y_1[idx_0]
            dr0[idx_1] = y_0[idx_1]
            dr0[idx_0] = y[idx_0]/(1-ps[idx_0]) - y_0[idx_0]*ps[idx_0]/(1-ps[idx_0])  
            
            ate = np.nanmean(dr1) - np.nanmean(dr0)
            
            return ate
            
        elif M > 1: # multi-valued treatments 
            t = self.df_Xy_Z[t_name].values # retrieve treatments
            t_label = np.argmax(t, axis=1)
            XZ = self.df_Xy_Z.iloc[:, self.XZ_start_idx:][cols].values # retrieve confounders and other treatments
            y = self.df_Xy_Z.iloc[:, self.y_idx].values
            
            est_mult = est_discrete(XZ, y, t, 'm', t_name[0]) # initialization
            
            
            # summary statistics
            s = Summary(XZ, y, t, 'm', t_name, cols)
            s.basic_statistics()
            
            
            # propensity score
            est_mult.ps_est_fit()
            ps = est_mult.ps_est_pred()
 
            
            
            # predict outcomes and doubly robust     
            
            dr = np.zeros((len(y), M))
            
            for i in range(M):
                
                model = est_mult.outcome_est_fit(t_level=i)
                
                t_hat = np.zeros((len(y), M))
                t_hat[:,i] = 1
                y_hat = est_mult.outcome_est_pred(model, t_hat)
                y_hat = y_hat.ravel()
                
                idx_1 = np.where(t[:,i]==1)
                idx_0 = np.where(t[:,i]==0)
                
                dr[idx_1,i] = y[idx_1]/ps[idx_1,i] - (1-ps[idx_1,i])*y_hat[idx_1]/ps[idx_1,i]
                #dr[idx_1,i] = y[idx_1]
                dr[idx_0,i] = y_hat[idx_0]
            
            
            # average treatment effects    
            ate = np.zeros(M) 
            for i in range(M):
                ate[i] = np.nanmean(dr[:,i])
                    
            return ate
        
        
        