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


class doubly_robust:
    def __init__(self, df_Xy_Z, y_idx, XZ_start_idx):
        self.df_Xy_Z = df_Xy_Z
        self.y_idx = y_idx
        self.XZ_start_idx = XZ_start_idx
    
    def est_ate_continuous_fit(self, t_name, ps_cols):
        t = self.df_Xy_Z[t_name].values # retrieve treatments
        t_level = t.ravel()
        self.XZ_continuous = self.df_Xy_Z.iloc[:, self.XZ_start_idx:][ps_cols].values # retrieve confounders and other treatments
        y = self.df_Xy_Z.iloc[:, self.y_idx].values
        # fit 
        self.prop_score_continuous_fit(t_level)
        gps = self.prop_score_continuous_predict(t_level)
        self.est_idv_outcome_continuous_fit(t_level, gps, y)
        return gps
        
    def est_ate_continuous_predict(self, t_name, t_level=None):
        #if not t_level.all: # study the mean of t level
        #    t = self.df_Xy_Z[t_name].values # retrieve treatments
        #    t_level = t.ravel().mean()
        # predict the mean effects at t_level
        gps = [self.prop_score_continuous_predict(i) for i in t_level]
        res = [self.est_idv_outcome_continuous_predict(i, j) for i, j in zip(t_level, gps)]
        
        return res
    
    
    def est_ate_discrete(self, t_name, outcome_cols, ps_cols):
    #estimate average treatment effects
        #t_idx = [self.df_Xy_Z.columns.get_loc(t) for t in t_name]
        M = len(t_name)
        if M == 1: # binary 
            t = self.df_Xy_Z[t_name].values # retrieve treatments
            t_label = t.ravel()
            XZ_outcome = self.df_Xy_Z.iloc[:, self.XZ_start_idx:][outcome_cols].values # retrieve confounders and other treatments
            
            XZ_ps = self.df_Xy_Z.iloc[:, self.XZ_start_idx:][ps_cols].values #
            
            ps = self.est_prop_score_discrete(XZ_ps, t_label, mode='b')
            ps = ps[:,1]
            
            
            # predict outcomes
            tXZ = np.concatenate((XZ_outcome, t), axis=1)
            y = self.df_Xy_Z.iloc[:, self.y_idx].values
            reg = self.est_idv_outcome_discrete(tXZ, y)
            
            del tXZ
            
            y_1 = reg.predict(np.concatenate((XZ_outcome, np.ones_like(t)), axis=1))
            y_0 = reg.predict(np.concatenate((XZ_outcome, np.zeros_like(t)), axis=1))
            
            #print('t:', t.shape)
            #print('y:', y.shape)
            #print('ps:', ps.shape)
            #print('y_0:', y_0.shape)
            
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
            XZ = self.df_Xy_Z.iloc[:, self.XZ_start_idx:].drop(t_name, axis=1).values # retrieve confounders and other treatments
            #ps = self.est_prop_score_discrete(XZ, t_label, mode='m')
            
            # predict outcomes
            tXZ = np.concatenate((XZ, t), axis=1)
            y = self.df_Xy_Z.iloc[:, self.y_idx].values
            reg = self.est_idv_outcome_discrete(tXZ, y)
            
            del tXZ
            
            dr = np.zeros((len(y), M))
            
            for i in range(M):
                t_hat = np.zeros((len(y), M))
                t_hat[:,i] = 1
                y_hat = reg.predict(np.concatenate((XZ, t_hat), axis=1))
                
                idx_1 = np.where(t[:,i]==1)
                idx_0 = np.where(t[:,i]==0)
                
                #dr[idx_1,i] = y[idx_1]/ps[idx_1,i] - (1-ps[idx_1,i])*y_hat[idx_1]/ps[idx_1,i]
                dr[idx_1,i] = y[idx_1]
                dr[idx_0,i] = y_hat[idx_0]
                
            ate = np.zeros((M, M)) # ate[i,j] = ate_i - ate_j
            for i in range(M):
                for j in range(M):
                    ate[i,j] = np.nanmean(dr[:,i]) - np.nanmean(dr[:,j])
                    
            return ate
            
            
    def est_prop_score_discrete(self, XZ, t, mode='b'):
    # propensity score estimation for binary treatments
    # propensity score: p(t|Z) where Z includes both confounders 
    # and other treatments and t is a discrete treatment
        if mode == 'b':
            clf = LogisticRegression(random_state=0, solver='liblinear').fit(XZ, t)
        elif mode == 'm':
            clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', max_iter=500).fit(XZ, t)
        return clf.predict_proba(XZ)
    
    def est_idv_outcome_discrete(self, tXZ, y):
        ols = LinearRegression()
        reg = ols.fit(tXZ, y)
        return reg
    
    
    def prop_score_continuous_fit(self, t):
        ols = LinearRegression()
        self.reg_ps = ols.fit(self.XZ_continuous, t)
        t_pred = self.reg_ps.predict(self.XZ_continuous)
        self.sigma = (((t - t_pred)**2).sum()/(len(t)-1))**(0.5)
    
    
    def prop_score_continuous_predict(self, t):
        t_pred = self.reg_ps.predict(self.XZ_continuous)
        t = np.array(t)
        gps = norm.pdf(t, t_pred, self.sigma)
        return gps
    
    
    def est_idv_outcome_continuous_fit(self, t, gps, y):
        ols = LinearRegression()
        t = t.reshape(-1,1)
        gps = gps.reshape(-1,1)
        t_gps = np.concatenate((t, t**2, gps, gps**2, t*gps), axis=1)
        #print(t_gps.shape, y.shape)
        self.reg_outcome = ols.fit(t_gps, y)
    
    
    def est_idv_outcome_continuous_predict(self, t, gps):
        
        N = self.XZ_continuous.shape[0]
        t = np.ones((N,1))*t
        gps = gps.reshape(-1,1)
        t_gps = np.concatenate((t, t**2, gps, gps**2, t*gps), axis=1)
        res = self.reg_outcome.predict(t_gps)
        return res.mean()