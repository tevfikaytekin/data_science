import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
   return 1/(1+np.exp(-x))

def relu(x):
   return np.maximum(0,x)

def calc_contingency_table(df, a1, a2):
    group1 = np.unique(df[a1])
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

    n_classes = 2
    plot_colors = "rgb"

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plot_step_x = (x_max-x_min) / 100
    plot_step_y = (y_max-y_min) / 100


    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step_x),
                             np.arange(y_min, y_max, plot_step_y))
    #plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].set_xlim(ax[0].get_xlim())
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax[1].scatter(X[idx, 0], X[idx, 1], c=color, 
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=40)