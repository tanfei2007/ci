#***********************
# Fei Tan
# tanfei2007@gmail.com
# 5/01/2019
#**********************

import tensorflow as tf
import numpy as np
from tabulate import tabulate
import pandas as pd


class Summary(object):
    """
    summarize covariates/confounders XZ, treatments, outcomes
    """
    
    def __init__(self, XZ, y, t, mode, t_name, col_names, K=3):
        self.XZ = XZ    # features/covariates (X), confounders (Z)
        self.y = y      # responses (y)
        self.t = t      # treatment (t)
        self.mode = mode # mode of treatment (binary ('b') / multivalued ('m') / continuous ('c'))
        self.t_name = t_name
        self.col_names = col_names
        self.K = K # number of bins for continuous treatments
        
    
    def basic_statistics(self):
        """
        basic statistics
        """
        
        if self.mode == 'b': 
            idx_t = np.where(self.t == 1)
            idx_c = np.where(self.t == 0)
            
            # number
            t_N = len(idx_t[0])
            c_N = len(idx_c[0])
            
            # mean
            t_mean = self.XZ[idx_t[0],].mean(axis=0)
            c_mean = self.XZ[idx_c[0],].mean(axis=0)
            y_t_mean = self.y[idx_t[0]].mean()
            y_c_mean = self.y[idx_c[0]].mean()
            
            # standard deviation
            t_sd = np.sqrt(self.XZ[idx_t[0],].var(axis=0, ddof=1))
            c_sd = np.sqrt(self.XZ[idx_c[0],].var(axis=0, ddof=1))
            y_t_sd = np.sqrt(self.y[idx_t[0]].var(ddof=1))
            y_c_sd = np.sqrt(self.y[idx_c[0]].var(ddof=1))
            
            
            # difference
            y_diff = y_t_mean - y_c_mean
            XZ_ndiff = calc_ndiff(t_mean, c_mean, t_sd, c_sd)
            
            
            
            df_y_stat = pd.DataFrame({'t_N': t_N, 'c_N': c_N, 'y_t_mean': y_t_mean, 'y_c_mean':y_c_mean, 'y_t_sd':y_t_sd, 'y_c_sd':y_c_sd, 'y_diff': y_diff}, index=['y'])
            df_XZ_stat = pd.DataFrame({'t_mean':t_mean, 'c_mean': c_mean, 't_sd': t_sd, 'c_sd': c_sd, 'XZ_ndiff': XZ_ndiff}, index=self.col_names)
            
            
            print('----------' + self.t_name + '------------')
            print('y statistics:')
            print(tabulate(df_y_stat, headers=['t_N', 'c_N', 'y_t_mean', 'y_c_mean', 'y_t_sd', 'y_c_sd', 'y_diff'], tablefmt='orgtbl'))
            print('XZ statistics:')
            print(tabulate(df_XZ_stat, headers=['t_mean', 'c_mean', 't_sd', 'c_sd', 'XZ_ndiff'], tablefmt='orgtbl'))
            print('----------' + 'end' + '------------')
            
            
        elif self.mode == 'm':
            
            M = len(self.t_name)
            
            for i in range(M):
                
                idx_t = np.where(self.t[:,i]==1)
                idx_c = np.where(self.t[:,i]==0)

                # number
                t_N = len(idx_t[0])
                c_N = len(idx_c[0])

                
                # mean
                t_mean = self.XZ[idx_t[0],].mean(axis=0)
                c_mean = self.XZ[idx_c[0],].mean(axis=0)
                y_t_mean = self.y[idx_t[0]].mean()
                y_c_mean = self.y[idx_c[0]].mean()

                
                # standard deviation
                t_sd = np.sqrt(self.XZ[idx_t[0],].var(axis=0, ddof=1))
                c_sd = np.sqrt(self.XZ[idx_c[0],].var(axis=0, ddof=1))
                y_t_sd = np.sqrt(self.y[idx_t[0]].var(ddof=1))
                y_c_sd = np.sqrt(self.y[idx_c[0]].var(ddof=1))


                # difference
                y_diff = y_t_mean - y_c_mean
                XZ_ndiff = calc_ndiff(t_mean, c_mean, t_sd, c_sd)
            
                df_y_stat = pd.DataFrame({'t_N': t_N, 'c_N': c_N, 'y_t_mean': y_t_mean, 'y_c_mean':y_c_mean, 'y_t_sd':y_t_sd, 'y_c_sd':y_c_sd, 'y_diff': y_diff}, index=['y'])
                df_XZ_stat = pd.DataFrame({'t_mean':t_mean, 'c_mean': c_mean, 't_sd': t_sd, 'c_sd': c_sd, 'XZ_ndiff': XZ_ndiff}, index=self.col_names)


                print('----------' + self.t_name[i] + '------------')
                print('y statistics:')
                print(tabulate(df_y_stat, headers=['t_N', 'c_N', 'y_t_mean', 'y_c_mean', 'y_t_sd', 'y_c_sd', 'y_diff'], tablefmt='orgtbl'))
                print('XZ statistics:')
                print(tabulate(df_XZ_stat, headers=['t_mean', 'c_mean', 't_sd', 'c_sd', 'XZ_ndiff'], tablefmt='orgtbl'))
                print('----------' + 'end' + '------------')
            
        else:    # continuous values and bin continuous values and calculating the overlapping
            
            intervals = pd.cut(self.t.ravel(), self.K, retbins=True)[1]
            
            
            for i in range(self.K):
                
                idx = (self.t > intervals[i]) & (self.t <= intervals[i+1])
                idx_t = np.where(idx)
                idx_c = np.where(~idx)

                # number
                t_N = len(idx_t[0])
                c_N = len(idx_c[0])

                
                # mean
                t_mean = self.XZ[idx_t[0],].mean(axis=0)
                c_mean = self.XZ[idx_c[0],].mean(axis=0)
                y_t_mean = self.y[idx_t[0]].mean()
                y_c_mean = self.y[idx_c[0]].mean()

                
                # standard deviation
                t_sd = np.sqrt(self.XZ[idx_t[0],].var(axis=0, ddof=1))
                c_sd = np.sqrt(self.XZ[idx_c[0],].var(axis=0, ddof=1))
                y_t_sd = np.sqrt(self.y[idx_t[0]].var(ddof=1))
                y_c_sd = np.sqrt(self.y[idx_c[0]].var(ddof=1))


                # difference
                y_diff = y_t_mean - y_c_mean
                XZ_ndiff = calc_ndiff(t_mean, c_mean, t_sd, c_sd)
            
                df_y_stat = pd.DataFrame({'t_N': t_N, 'c_N': c_N, 'y_t_mean': y_t_mean, 'y_c_mean':y_c_mean, 'y_t_sd':y_t_sd, 'y_c_sd':y_c_sd, 'y_diff': y_diff}, index=['y'])
                df_XZ_stat = pd.DataFrame({'t_mean':t_mean, 'c_mean': c_mean, 't_sd': t_sd, 'c_sd': c_sd, 'XZ_ndiff': XZ_ndiff}, index=self.col_names)


                print('----------' + str(intervals[i]) + 'to' + str(intervals[i+1]) + '------------')
                print('y statistics:')
                print(tabulate(df_y_stat, headers=['t_N', 'c_N', 'y_t_mean', 'y_c_mean', 'y_t_sd', 'y_c_sd', 'y_diff'], tablefmt='orgtbl'))
                print('XZ statistics:')
                print(tabulate(df_XZ_stat, headers=['t_mean', 'c_mean', 't_sd', 'c_sd', 'XZ_ndiff'], tablefmt='orgtbl'))
                print('----------' + 'end' + '------------')
                
                
                
                
def calc_ndiff(t_mean, c_mean, t_sd, c_sd):
    """
    calculate the normalized difference between two different groups
    """
    return (t_mean-c_mean) / np.sqrt((t_sd**2+c_sd**2)/2)



