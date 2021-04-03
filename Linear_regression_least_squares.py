#!/usr/bin/env python
# coding: utf-8

# In[119]:


'''
Linear Regression model

Model built using least-squares method

y = intercept + slope * x

m = slope
m = Σ(x - x̅) * (y - y̅) / Σ(x -x̅)²

b = intercept
b = y̅ - m * x̅

Object requires x and y

x = explanatory variable
y = dependent variable

'''
class Linearregression:
    import numpy as np
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.least_square(x,y)
              
    
    def least_square(self, x, y):
        
        m = sum((self.x - np.mean(self.x)) * (self.y - np.mean(self.y))) / sum(((self.x - np.mean(self.x))**2)) # slope
        
        b = np.mean(self.y) - m * np.mean(self.x) # intercept
        
        sum_residuals = sum((self.y - np.mean(self.y)**2))
        
        self.y_intercept = b + m * self.x
        
        self.r2 = sum((self.y_intercept - np.mean(self.y))**2) / ((self.y - np.mean(self.y))**2)
        
        self.linear = ([self.x.min(),self.x.max()], [self.y_intercept.min(), self.y_intercept.max()])
        
        self.b = b
        self.m = m     
                       
        return self.y_intercept, self.r2, self.linear, self.b, self.m
          
        
    def predict(self,feature_x,true_y, mse = False):
        self.feature_x = feature_x
        self.true_y = true_y
        
        predicted_values = []
        
        for i in range(len(self.feature_x)):
            pred_y = self.b + self.m * self.feature_x[i]
            predicted_values.append(pred_y)
        
        self.pred_y =  pred_y
        self.predicted_values = predicted_values
        
        def mean_square_error(self,true_y, pred_y):
            
            pred_values = np.array(pred_y)
            true_values = np.array(true_y)
            
            mse = np.mean(np.square(true_values - pred_values))        
        
            self.mse = mse
            return self.mse
        
        mean_square_error(self,true_y, pred_y)
        
        
        if mse == True:
            return self.predicted_values, self.pred_y, self.mse
        else:
            return self.predicted_values, self.pred_y

