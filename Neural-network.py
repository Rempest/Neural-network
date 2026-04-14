import numpy as np
class Neural_network:
  def __init__(self, lr = 0.01, epoch = 1000.0):
    self.lr = lr
    self.epoch = epoch
    self.w = None
    self.b = None
  def fit(self, x, y):
    n_samples = x.shape[0]
    self.w = 0.0
    self.b = 0.0
    for i in range(self.epoch):
       y_pred = self.w * X + self.b
       dw = (2/n_samples) * np.sum((y_pred - y) * x)
       db = (2/n_samples) * np.sum(y_pred - y)
       self.w -= self.lr * dw
       self.b -= self.lr * db
      #i will be come back:)
    
    
