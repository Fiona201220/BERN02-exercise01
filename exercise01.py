## Xiaofei Wang 2024-09-03 ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loess(x, y, x_pred, k):
   
    y_pred = np.zeros(len(x_pred))
    y_pred_se = np.zeros(len(x_pred))

    # for each x_pred_i
    for j, x0 in enumerate(x_pred):
        distances = np.abs(x - x0)
        nearest_points= np.argsort(distances)[:k]
        x_near = x[nearest_points]
        #print(x_near)
        y_near = y[nearest_points]

        # calculate weight
        max_distance = distances[nearest_points[-1]]
        weights = 1 - distances[nearest_points] / max_distance
        weights /= np.sum(weights) 
        mean_x = np.sum(weights * x_near) 
        mean_y = np.sum(weights * y_near) 
        
        # 
        beta_1 = np.sum(weights * (x_near - mean_x) * (y_near - mean_y)) / np.sum(weights * (x_near - mean_x)**2)
        beta_0 = mean_y - beta_1 * mean_x
        
        # predict y at x0
        y_pred[j] = beta_0 + beta_1 * x0
        # since we don't know the true value of y_pred, but we konw the points near x0
        r2 = (y_near - (beta_0 + beta_1 * x_near))**2
        rss = np.sum(weights * r2)
        # standard error
        y_pred_se[j] = np.sqrt(rss / (k - 2))
    
    return y_pred, y_pred_se
    
   


df = pd.read_csv('/Users/wangxiaofei/Desktop/pollution_cleaneddata.csv')

x = df['POOR'].values
y = df['MORT'].values
k = 5
# prediction
x_pred = [10,18,25]
y_pred = loess(x, y, x_pred, k)
y_pred_se = loess(x, y, x_pred, k)
# predictions with the expected value and the standard error.
print( y_pred)