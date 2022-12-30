# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 19:01:24 2022

@author: Kert PC
"""

import matplotlib.pyplot as plt


file1 = open('test.txt', 'r')
l2_norms = []
mse_loss = []
indices = range(1,161)

for line in file1:
    l2_norms.append(float(line.split(':')[2]))
    mse_loss.append((float(line.split(':')[2]) / (44100 * 16))**2)
    
# Closing files
file1.close()


plt.figure(figsize=(12, 4))

plt.plot(indices, l2_norms)
plt.suptitle('L2 norm on validation set')
plt.show()


plt.figure(figsize=(12, 4))

plt.plot(indices, mse_loss)
plt.suptitle('MSE loss on validation set')
plt.show()