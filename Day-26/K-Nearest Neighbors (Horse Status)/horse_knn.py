import numpy as np
import pandas as pd
import streamlit as st

class KNN:
    def __init__(self, k, p, weights):
        self.P = p
        self.K = k
        self.Weights = weights
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def distance(self, x1, x2):
        diff = np.abs(x1 - x2)
        axis_val = 1 if x1.ndim > 1 else 0
        
        if self.P == 1:
            return np.sum(diff, axis=axis_val)
        elif self.P == 2:
            return np.sqrt(np.sum(diff ** 2, axis=axis_val))
        elif self.P == float('inf') or self.P == 'chebyshev':
            return np.max(diff, axis=axis_val)
        else:
            return np.power(np.sum(diff ** self.P, axis=axis_val), 1/self.P)
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        dists = self.distance(self.X_train, x)
        best_k = np.argsort(dists)[:self.K]
        label_k = self.y_train[best_k]
        
        if self.Weights == 'uniform':
            values, counts = np.unique(label_k, return_counts=True)
            return values[np.argmax(counts)]
        
        elif self.Weights == 'distance':
            weights = 1 / (dists[best_k] + 1e-6)
            vote_scores = {}
            
            for label, weight in zip(label_k, weights):
                vote_scores[label] = vote_scores.get(label, 0) + weight
            return max(vote_scores, key=vote_scores.get)
        
@st.cache_resource
def load_model():
    df = pd.read_csv('train.csv')
    
    ## Kolom Numeric Diisi Rata-Rata (Mean)
    for cols in df.select_dtypes(include=['number']).columns:
        df[cols] = df[cols].fillna(df[cols].mean())
        
    ## Kolom Object Diisi Kata Yang Sering Muncul (Modus/Mode)
    for cols in df.select_dtypes(include=['object']).columns:
        df[cols] = df[cols].fillna(df[cols].mode()[0])