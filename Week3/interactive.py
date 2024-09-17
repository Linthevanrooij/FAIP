
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import sklearn.metrics

import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from ipywidgets import interact_manual

X_b = None
X = None
y = None
X_new_b = None
X_new = None
pa = None
pb = None
zz_P = None
par_range = None
MSE_best = None
theta_list = None
MSE_list = None

def init():

    global X_b
    global X
    global y
    global X_new_b
    global X_new
    global pa
    global pb
    global zz_P
    global par_range
    global MSE_best
    global theta_list
    global MSE_list

    X = 2 * np.random.rand(100, 1) - 1
    y = 5 + 3 * X + np.random.randn(100, 1)

    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    theta_list = np.array([[0],[0]]).T

    theta = np.array([[0],[0]])
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)

    MSE_list = [MSE]

    X_new = np.array([[-2], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    par_range = [-10,10,-15,15] # b, a

    pb, pa = np.meshgrid(np.linspace(par_range[0], par_range[1], 200).reshape(-1, 1),
        np.linspace(par_range[2], par_range[3], 200).reshape(-1, 1))
    P = np.c_[pb.ravel(), pa.ravel()]

    Y_P = np.matmul(P,X_b.T)
    bla = Y_P-y.T
    SE_P = bla**2
    MSE_P = np.mean(SE_P,axis=1)

    zz_P = MSE_P.reshape(pb.shape)

    theta_best
    y_predict_best = X_b.dot(theta_best)
    MSE_best = sklearn.metrics.mean_squared_error(y, y_predict_best)
    
    wa = widgets.FloatSlider(min=-10, max=10, step=0.1, continuous_update=False)
    wb = widgets.FloatSlider(min=-15, max=15, step=0.1, continuous_update=False)

    interactive_plot = interactive(f, a=wa, b=wb)
    return interactive_plot

def f(a, b):
    global theta_list  
    global MSE_list
    theta = np.array([[b],[a]])
    
    theta_list = np.concatenate((theta_list,theta.T),axis=0)
    
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)
    MSE_list.append(MSE)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    y_predict = X_new_b.dot(theta)
    
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    grad_norm = np.sqrt(np.sum(gradients**2))
    gradients = 1/grad_norm*gradients

    ax1.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    ax1.plot(X, y, "b.")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$", rotation=0)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.axis([-1, 1, -2, 15])
    ax1.set_title('regression problem')

    CS = ax2.contour(pb, pa, zz_P, levels=10)
    ax2.clabel(CS, inline=True, fontsize=10)
    ax2.scatter(theta_list[:,0],theta_list[:,1])
    ax2.scatter(theta_list[-1,0],theta_list[-1,1],c='r')
    ax2.arrow(theta_list[-1,0],theta_list[-1,1], -gradients[0,0], -gradients[1,0],width = 0.25,color='r')
    ax2.axis(par_range)
    ax2.set_xlabel("$b$")
    ax2.set_ylabel("$a$", rotation=0)
    ax2.set_title('parameter space')
    
    ax3.set_title('Training Curve, current MSE: %.1f' % MSE)
    ax3.plot(MSE_list)
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax3.plot(x_best_plot,y_best_plot)
    ax3.set_xlabel('tries')
    ax3.set_ylabel('MSE')

    
def init2():

    global X_b
    global X
    global y
    global X_new_b
    global X_new
    global pa
    global pb
    global zz_P
    global par_range
    global MSE_best
    global theta_list
    global MSE_list
    global wa_label
    global wb_label

    X = 2 * np.random.rand(100, 1) - 1
    y = 5 + 3 * X + np.random.randn(100, 1)

    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    theta_init = np.array([[-10],[-5]])


    theta_list = theta_init.copy().T

    theta = theta_init.copy()
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)

    MSE_list = [MSE]

    X_new = np.array([[-2], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    par_range = [-10,10,-15,15] # b, a

    pb, pa = np.meshgrid(np.linspace(par_range[0], par_range[1], 200).reshape(-1, 1),
        np.linspace(par_range[2], par_range[3], 200).reshape(-1, 1))
    P = np.c_[pb.ravel(), pa.ravel()]

    Y_P = np.matmul(P,X_b.T)
    bla = Y_P-y.T
    SE_P = bla**2
    MSE_P = np.mean(SE_P,axis=1)

    zz_P = MSE_P.reshape(pb.shape)

    theta_best
    y_predict_best = X_b.dot(theta_best)
    MSE_best = sklearn.metrics.mean_squared_error(y, y_predict_best)

    wa = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta[0])
    wb = widgets.FloatSlider(min=par_range[0], max=par_range[1], step=0.1, continuous_update=False, value=theta[1])
    wa_label = widgets.Label(value="+10")
    wb_label = widgets.Label(value="+5")

    bar_a = widgets.HBox([wa,wa_label])
    bar_b = widgets.HBox([wb,wb_label])
    
    ui = widgets.VBox([bar_a, bar_b])

    out = widgets.interactive_output(g, {'a':wa, 'b':wb})
    return display(ui, out)
    
    
def g(a, b):
    global theta_list  
    global MSE_list
    theta = np.array([[b],[a]])
    
    theta_list = np.concatenate((theta_list,theta.T),axis=0)
    
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)
    MSE_list.append(MSE)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    y_predict = X_new_b.dot(theta)
    
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    grad_norm = np.sqrt(np.sum(gradients**2))
    gradients_unit = 1/grad_norm*gradients.copy()
    
    global wa_label
    global wb_label
    
    wa_label.value = ('-dJ/da = %+f' % -gradients[1,0])
    wb_label.value = ('-dJ/db = %+f' % -gradients[0,0])

    ax1.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    ax1.plot(X, y, "b.")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$", rotation=0)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.axis([-1, 1, -2, 15])
    ax1.set_title('regression problem')

    CS = ax2.contour(pb, pa, zz_P, levels=10)
    ax2.clabel(CS, inline=True, fontsize=10)
    ax2.scatter(theta_list[:,0],theta_list[:,1])
    ax2.scatter(theta_list[-1,0],theta_list[-1,1],c='r')
    ax2.arrow(theta_list[-1,0],theta_list[-1,1], -gradients_unit[0,0], -gradients_unit[1,0],width = 0.25,color='r')
    ax2.axis(par_range)
    ax2.set_xlabel("$b$")
    ax2.set_ylabel("$a$", rotation=0)
    ax2.set_title('parameter space')
    
    ax3.set_title('Training Curve, current MSE: %.1f' % MSE)
    ax3.plot(MSE_list)
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax3.plot(x_best_plot,y_best_plot)
    ax3.set_xlabel('tries')
    ax3.set_ylabel('MSE')

def init3(show_grad = False):

    global X_b
    global X
    global y
    global X_new_b
    global X_new
    global pa
    global pb
    global zz_P
    global par_range
    global MSE_best
    global theta_list
    global MSE_list
    global w1_label
    global w2_label
    global w3_label
    global w4_label
    global w5_label
    
    global wb_label

    X = 2 * np.random.rand(100, 1) - 1

    feat_extract = sklearn.preprocessing.PolynomialFeatures(degree=5,include_bias=False)
    X_2 = feat_extract.fit_transform(X)
    X_b = np.c_[np.ones((100, 1)), X_2]  # add x0 = 1 to each instance

    theta_opt = np.array([10,1,5,8,6,8])
    y = np.matmul(X_b,theta_opt) + 0.25*np.random.randn(100, 1).T
    y = y.T

    theta_best = theta_opt
    #theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    theta_init = theta_best.copy()*0 
    #np.array([[-10],[-5]])

    theta_list = theta_init.copy()

    theta = theta_init.copy()
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)

    MSE_list = [MSE]

    y_predict_best = X_b.dot(theta_best)
    MSE_best = sklearn.metrics.mean_squared_error(y, y_predict_best)

    w1 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[0])
    w2 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[1])
    w3 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[2])
    w4 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[3])
    w5 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[4])

    wb = widgets.FloatSlider(min=par_range[0], max=par_range[1], step=0.1, continuous_update=False, value=theta_init[5])
    w1_label = widgets.Label(value="w1")
    w2_label = widgets.Label(value="w2")
    w3_label = widgets.Label(value="w3")
    w4_label = widgets.Label(value="w4")
    w5_label = widgets.Label(value="w5")

    wb_label = widgets.Label(value="w0")

    bar_1 = widgets.HBox([w1,w1_label])
    bar_2 = widgets.HBox([w2,w2_label])
    bar_3 = widgets.HBox([w3,w3_label])
    bar_4 = widgets.HBox([w4,w4_label])
    bar_5 = widgets.HBox([w5,w5_label])

    bar_b = widgets.HBox([wb,wb_label])

    X_new = np.arange(-1,1,0.01).reshape(-1,1)
    X_new_2 = feat_extract.transform(X_new)
    X_new_b = np.c_[np.ones((len(X_new_2), 1)), X_new_2]  # add x0 = 1 to each instance
    
    ui = widgets.VBox([bar_b, bar_1, bar_2, bar_3, bar_4, bar_5])

    if show_grad:
        out = widgets.interactive_output(h, {'w1':w1, 'w2':w2, 'w3': w3, 'w4': w4, 'w5': w5, 'wb': wb})
    else:
        out = widgets.interactive_output(h_nograd, {'w1':w1, 'w2':w2, 'w3': w3, 'w4': w4, 'w5': w5, 'wb': wb})
    return display(ui, out)

def h(w1, w2, w3, w4, w5, wb):

    global theta_list  
    global MSE_list
    theta = np.array([[wb],[w1],[w2],[w3],[w4],[w5]])
    
    #theta_list = np.concatenate((theta_list,theta.T),axis=0)
    
    y_trn = np.matmul(X_b,theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)
    MSE_list.append(MSE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    
    y_predict = np.matmul(X_new_b,theta)
    
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    grad_norm = np.sqrt(np.sum(gradients**2))
    gradients_unit = 1/grad_norm*gradients.copy()
    
    global w1_label
    global w2_label
    global w3_label
    global w4_label
    global w5_label
    
    global wb_label
    
    wb_label.value = ('-dJ/dw0 = %+f' % -gradients[0,0])
    w1_label.value = ('-dJ/dw1 = %+f' % -gradients[1,0])
    w2_label.value = ('-dJ/dw2 = %+f' % -gradients[2,0])
    w3_label.value = ('-dJ/dw3 = %+f' % -gradients[3,0])
    w4_label.value = ('-dJ/dw4 = %+f' % -gradients[4,0])
    w5_label.value = ('-dJ/dw5 = %+f' % -gradients[5,0])
    

    ax1.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    ax1.plot(X, y, "b.")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$", rotation=0)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.axis([-1, 1, 0, 40])
    ax1.set_title('regression problem')
    
    ax2.set_title('Training Curve, current MSE: %.1f' % MSE)
    ax2.plot(MSE_list)
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax2.plot(x_best_plot,y_best_plot)
    ax2.set_xlabel('tries')
    ax2.set_ylabel('MSE')
    
def h_nograd(w1, w2, w3, w4, w5, wb):

    global theta_list  
    global MSE_list
    theta = np.array([[wb],[w1],[w2],[w3],[w4],[w5]])
    
    #theta_list = np.concatenate((theta_list,theta.T),axis=0)
    
    y_trn = np.matmul(X_b,theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)
    MSE_list.append(MSE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    
    y_predict = np.matmul(X_new_b,theta)
    
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    grad_norm = np.sqrt(np.sum(gradients**2))
    gradients_unit = 1/grad_norm*gradients.copy()
    

    ax1.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    ax1.plot(X, y, "b.")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$", rotation=0)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.axis([-1, 1, 0, 40])
    ax1.set_title('regression problem')
    
    ax2.set_title('Training Curve, current MSE: %.1f' % MSE)
    ax2.plot(MSE_list)
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax2.plot(x_best_plot,y_best_plot)
    ax2.set_xlabel('tries')
    ax2.set_ylabel('MSE')
    
    
    
def init4():
    
    global X_b
    global X
    global y
    global X_new_b
    global X_new
    global pa
    global pb
    global zz_P
    global par_range
    global MSE_best
    global theta_list
    global MSE_list

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_path_sgd = []
    m = len(X_b)
    np.random.seed(42)
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    X_new = np.array([[-2], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    par_range = [-10,10,-15,15] # b, a

    pb, pa = np.meshgrid(np.linspace(par_range[0], par_range[1], 200).reshape(-1, 1),
        np.linspace(par_range[2], par_range[3], 200).reshape(-1, 1))
    P = np.c_[pb.ravel(), pa.ravel()]

    Y_P = np.matmul(P,X_b.T)
    bla = Y_P-y.T
    SE_P = bla**2
    MSE_P = np.mean(SE_P,axis=1)

    zz_P = MSE_P.reshape(pb.shape)

    theta_best
    y_predict_best = X_b.dot(theta_best)
    MSE_best = sklearn.metrics.mean_squared_error(y, y_predict_best)
    
    cont_update = False
    wa = widgets.fixed(-10)
    wb = widgets.fixed(-5)
    #wa = widgets.FloatSlider(value=-10,min=par_range[2], max=par_range[3], step=0.1, continuous_update=cont_update, description='Init a')
    wb = widgets.FloatSlider(value=-5,min=par_range[0], max=par_range[1], step=0.1, continuous_update=cont_update, description='Init b')
    weta = widgets.FloatLogSlider(value=0.01, base=10, min=-5, max=-0.1, step=0.2, description='Eta', continuous_update=cont_update)
    we = widgets.IntSlider(value=5, min=1, max=50, continuous_update=cont_update, description='Epochs')

    #init_box = widgets.HBox([wa, wb])
    ui = widgets.HBox([weta, we])
    #ui = widgets.VBox([init_box, hyp_box])

    out = widgets.interactive_output(sgd, {'a':wa, 'b':wb, 'eta': weta, 'epochs': we})

    return display(ui, out)

def sgd(a, b, eta, epochs):
    
    global theta_list
    global MSE_list
    theta_list = np.array([[b],[a]]).T

    theta = np.array([[b],[a]])
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)

    MSE_list = [MSE]
    
    n_epochs = epochs
    m = len(X_b)
    theta_path_sgd = []
    
    theta = np.array([[b],[a]])
    
    for epoch in range(n_epochs):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

        y_predict = X_b.dot(theta) 
        MSE = sklearn.metrics.mean_squared_error(y, y_predict)
        MSE_list.append(MSE)

        theta_list = np.concatenate((theta_list,theta.T),axis=0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    y_predict = X_new_b.dot(theta)           
    style = "r--"        
    ax1.plot(X_new, y_predict, style)   
    
    ax1.plot(X, y, "b.")                                 
    ax1.set_xlabel("$x$", fontsize=18)                     
    ax1.set_ylabel("$y$", rotation=0, fontsize=18)           
    ax1.axis([0, 2, 0, 15])    
    
    CS = ax2.contour(pb, pa, zz_P, levels=10)
    ax2.clabel(CS, inline=True, fontsize=10)
    ax2.scatter(theta_list[:,0],theta_list[:,1])
    ax2.scatter(theta_list[-1,0],theta_list[-1,1],c='r')
    #ax2.arrow(theta_list[-1,0],theta_list[-1,1], -gradients[0,0], -gradients[1,0],width = 0.25,color='r')
    ax2.axis(par_range)
    ax2.set_xlabel("$b$")
    ax2.set_ylabel("$a$", rotation=0)
    ax2.set_title('parameter space')
    
    ax3.set_title('Training Curve, MSE after training: %.1f' % MSE)
    ax3.plot(MSE_list,'o-')
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax3.plot(x_best_plot,y_best_plot)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('MSE')
                                   
