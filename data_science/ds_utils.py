import numpy as np
import matplotlib.pyplot as plt

###

def sigmoid(x):
   return 1/(1+np.exp(-x))

def relu(x):
   return np.maximum(0,x)

def calc_contingency_table(df, a1, a2):
    group1 = np.unique(df[a1]);
    group2 = np.unique(df[a2])
    con_table = np.zeros((len(group1),len(group2)))
    for i in range(len(group1)):
        for j in range(len(group2)):
            con_table[i][j] = sum((df[a1] == group1[i]) & (df[a2] == group2[j]))
    return con_table, group1, group2

def plot_decision_boundary(model, X, y):
    
    fig, ax = plt.subplots(1,2,figsize=(10,4));

    ax[0].scatter(X[:, 0], X[:, 1], marker='o', c=y,
                s=40, edgecolor='black');
