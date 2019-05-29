#!/usr/bin/env python
"""
lambdata - A useful collection of Data Science helper functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_val_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    X_train = pd.concat(X_train,X_train2)
    y_train = pd.concat(y_train,y_train2)
    return X_train, y_train, X_val, y_val, X_test, y_test
