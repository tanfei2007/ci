#***********************
# Fei Tan
# tanfei2007@gmail.com
# 4/23/2019
#**********************

import tensorflow as tf
import numpy as np

import h5py
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Dense
from scipy.stats import norm


class est_discrete(object):
    """
    estimate propensity score and outcome for discrete treatments
    """
    
    def __init__(self, XZ, y, t, mode, t_name):
        self.XZ = XZ    # features/covariates (X), confounders (Z)
        self.y = y      # responses (y)
        self.t = t      # treatment (t)
        self.mode = mode # mode of treatment (binary/discrete)
        self.tXZ = np.concatenate((self.t, self.XZ), axis=1)
        self.t_name = t_name
        
    def ps_est_fit(self):
        """
        propensity score fit
        """
        
        main_input = Input(shape=(self.XZ.shape[1],))
        
        main_path = Dense(units=10, activation='relu', kernel_initializer='he_normal')(main_input)
        main_path = Dense(units=10, activation='relu', kernel_initializer='he_normal')(main_path)
        
        if self.mode == 'b':
            main_path = Dense(units=1, activation='sigmoid')(main_path)  # one output
        elif self.mode == 'm':
            main_path = Dense(units=self.t.shape[1], activation='softmax')(main_path) # multiple outputs
        else:
            raise ValueError('mode has to be binary (b) or multiple (m)')
            
        model = Model(inputs=main_input, outputs=main_path)
    
        adam = Adam(lr=0.001, decay=1e-03, clipnorm=1.0)
        
        if self.mode == 'b':
            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
            ps_target = self.t.ravel()
        elif self.mode == 'm':
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            ps_target = self.t
        
        checkpointer = ModelCheckpoint(filepath=self.t_name + '_ps_est.hdf5', verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        
        print(model.summary())
        
        model.fit(self.XZ, ps_target, batch_size=128, epochs=100, shuffle=True, \
                  validation_split=0.2, verbose=2, callbacks=[checkpointer,earlystopper])
        
        self.ps_model = model
    
    
    def ps_est_pred(self):
        """
        propensity score prediction 
        """
        
        prob = self.ps_model.predict(self.XZ, verbose=1, batch_size=1000)
        self.ps = prob
        
        return prob
    
    
    def outcome_est_fit(self, t_level=None):
        """
        outcome fit
        """
        
        main_input = Input(shape=(self.tXZ.shape[1],))
        
        main_path = Dense(units=10, activation='relu', kernel_initializer='he_normal')(main_input)
        main_path = Dense(units=10, activation='relu')(main_path)
        
        main_path = Dense(units=1, activation='linear')(main_path)
        
        model = Model(inputs=main_input, outputs=main_path)
    
        adam = Adam(lr=0.001, decay=1e-03, clipnorm=1.0)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
        checkpointer = ModelCheckpoint(filepath=self.t_name + '_outcome_est.hdf5', verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        
        print(model.summary())
        
        if self.mode == 'b':
            model.fit(self.tXZ, self.y, batch_size=128, epochs=100, shuffle=True, \
                      validation_split=0.2, verbose=2, callbacks=[checkpointer,earlystopper])
        elif self.mode == 'm':
            idx = np.where(np.argmax(self.t, axis=1) == t_level)
            model.fit(self.tXZ[idx], self.y[idx], batch_size=128, epochs=100, shuffle=True, \
                      validation_split=0.2, verbose=2, callbacks=[checkpointer,earlystopper])
        
        return model
     
    
    def outcome_est_pred(self, model, t_add):
        """
        outcome prediction
        """
        
        t_addXZ = np.concatenate((t_add, self.XZ), axis=1)
        
        response = model.predict(t_addXZ, verbose=1, batch_size=1000)
        
        return response 
    
    
    def trim(self, cutoff):
        """
        trim based on propensity score
        This method should only be executed after the propensity score has been estimated.
        """
        self.cutoff = cutoff # propensity score cutoff
        
        if 0 <= self.cutoff <= 0.5:
            pscore = self.ps
            
            if self.mode == 'b':
                keep = np.where((pscore.ravel() >= self.cutoff) & (pscore.ravel() <= 1-self.cutoff))
                
            elif self.mode == 'm':
                keep = np.where((np.min(pscore, axis=1) >= self.cutoff) & (np.max(pscore, axis=1) <= 1-self.cutoff))
            
            else:
                raise ValueError('Invalid treatment modes.')
            
            
            # updates
            self.XZ = self.XZ[keep]
            self.tXZ = self.tXZ[keep]
            self.y = self.y[keep]
            self.t = self.t[keep]
            
        else:
            raise ValueError('Invalid Cutoff.')
            
        return keep
    
            
            
    def trim_s(self):
        """
        Trims data based on propensity score using the cutoff selection algorithm suggested by [1]_.
        This method should only be executed after the propensity score has been estimated.

        References
        ----------
         .. [1] Crump, R., Hotz, V., Imbens, G., & Mitnik, O. (2009).
         Dealing with Limited Overlap in Estimation of
         Average Treatment Effects. Biometrika, 96, 187-199.
            """
            
        pscore = self.ps
        
        if self.mode == 'b':
            pscore = pscore.ravel()
            g = 1.0/(pscore*(1-pscore))  # 1 over Bernoulli variance
            self.cutoff = select_cutoff(g)
            
        elif self.mode == 'm':
            cutoff = 0
            for i in range(pscore.shape[1]):
                g = 1.0/(pscore[:,i]*(1-pscore[:,i]))
                cutoff = max([select_cutoff(g), cutoff]) # max of all levels
                
            self.cutoff = cutoff
            
        else:
                raise ValueError('Invalid treatment modes.')
                
        keep = self.trim(self.cutoff)
                
        return keep
        
        
class est_continuous(object):
    """
    estimate propensity score and outcome for continuous treatments
    """
    
    def __init__(self, XZ, y, t, t_name):
        self.XZ = XZ    # features/covariates (X), confounders (Z)
        self.y = y      # responses (y)
        self.t = t      # treatment (t)
        self.tXZ = np.concatenate((self.t, self.XZ), axis=1)
        self.t_name = t_name
    
    
    def ps_est_fit(self):
        """
        propensity score fit
        """

        main_input = Input(shape=(self.XZ.shape[1],))

        main_path = Dense(units=500, activation='relu')(main_input)
        main_path = Dense(units=500, activation='relu')(main_path)
        main_path = Dense(units=1, activation='linear')(main_path)  # one output

        model = Model(inputs=main_input, outputs=main_path)

        adam = Adam(lr=0.001, decay=1e-03)

        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
        ps_target = self.t.ravel()

        checkpointer = ModelCheckpoint(filepath=self.t_name + '_ps_est.hdf5', verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        print(model.summary())

        model.fit(self.XZ, ps_target, batch_size=128, epochs=100, shuffle=True, \
                  validation_split=0.2, verbose=2, callbacks=[checkpointer,earlystopper])

        self.ps_model = model
        t_mu = self.ps_model.predict(self.XZ, verbose=1, batch_size=10000)
        self.t_mu = t_mu.ravel()
        self.sigma = (((ps_target - self.t_mu)**2).sum()/(len(self.t)-1))**(0.5) # unadjusted sample variance


    def _ps_est_pred(self, t_add):
        """
        propensity score prediction 
        """

        gps = norm.pdf(t_add, self.t_mu, self.sigma)

        return gps


    def outcome_est_fit(self):
        
        gps = self._ps_est_pred(self.t.ravel())
        gps = gps.reshape((-1, 1))
        #gps_t = np.concatenate((gps, self.t, gps**2, self.t**2, gps*self.t), axis=1)
        gps_t = np.concatenate((gps, self.t), axis=1)
        
        main_input = Input(shape=(gps_t.shape[1],))

        main_path = Dense(units=500, activation='relu')(main_input)
        main_path = Dense(units=500, activation='relu')(main_path)
        main_path = Dense(units=1, activation='linear')(main_path)  # one output

        model = Model(inputs=main_input, outputs=main_path)

        adam = Adam(lr=0.001, decay=1e-03)

        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
        ps_target = self.t.ravel()

        checkpointer = ModelCheckpoint(filepath=self.t_name + '_outcome_est.hdf5', verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        print(model.summary())

        model.fit(gps_t, self.y, batch_size=128, epochs=100, shuffle=True, \
                  validation_split=0.2, verbose=2, callbacks=[checkpointer,earlystopper])

        self.outcome_model = model


    def outcome_est_pred(self, t_add):
        
        gps = self._ps_est_pred(t_add.ravel())
        gps = gps.reshape((-1, 1))
        t_add = np.ones((len(gps),1))*t_add
        #gps_t = np.concatenate((gps, t_add, gps**2, t_add**2, gps*t_add), axis=1)
        gps_t = np.concatenate((gps, t_add), axis=1)

        res = self.outcome_model.predict(gps_t, verbose=1, batch_size=10000)

        return res.mean()
    

def select_cutoff(g):

    if g.max() <= 2*g.mean():
        cutoff = 0
    else:
        sorted_g = np.sort(g)
        cumsum_1 = range(1, len(g)+1)
        LHS = g * sumlessthan(g, sorted_g, cumsum_1)
        cumsum_g = np.cumsum(sorted_g)
        RHS = 2 * sumlessthan(g, sorted_g, cumsum_g)
        gamma = np.max(g[LHS <= RHS])
        cutoff = 0.5 - np.sqrt(0.25 - 1./gamma)
     
    return cutoff


def sumlessthan(g, sorted_g, cumsum):
    deduped_values = dict(zip(sorted_g, cumsum))

    return np.array([deduped_values[x] for x in g])
