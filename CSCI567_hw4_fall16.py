# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 18:17:40 2016

@author: mjhundekar
"""
from hw_utils import *
import os

os.system("taskset -p 0xffff %d" % os.getpid())

x_train, y_train, x_test, y_test = loaddata('MiniBooNE_PID.txt');
norm_x_train, norm_x_test = normalize(x_train, x_test);

achitecture_1 = [[50, 2], [50, 50, 2], [50, 50, 50, 2], [50, 50, 50, 50, 2]]

achitecture_2 = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]

# d.Linear activations:------------------------------------------------------------------------------------------
print('\nLinear activations with architecture: ' + str(achitecture_1))
architecture, regulariztn, decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_1, actfn='linear', last_act='softmax',
               reg_coeffs=[0.0], num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
               sgd_Nesterov=False, EStop=False, verbose=0)

print('\nLinear activations with architecture: ' + str(achitecture_2))
architecture, regulariztn, decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_2, actfn='linear', last_act='softmax',
               reg_coeffs=[0.0], num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
               sgd_Nesterov=False, EStop=False, verbose=0)

# (e) Sigmoid activation:-----------------------------------------------------------------------------------------
print('\nSigmoid activations with architecture: ' + str(achitecture_2))
architecture, regulariztn, decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_1, actfn='sigmoid', last_act='softmax',
               reg_coeffs=[0.0], num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
               sgd_Nesterov=False, EStop=False, verbose=0)

# (f) ReLu activation:-------------------------------------------------------------------------------------------
print('\nReLu activations with architecture: ' + str(achitecture_1))
architecture, regulariztn, decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_1, actfn='relu', last_act='softmax',
               reg_coeffs=[0.0], num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
               sgd_Nesterov=False, EStop=False, verbose=0)

# (g) L2-Regularization:-------------------------------------------------------------------------
reg_arch = [[50, 800, 500, 300, 2]]

regularization = [pow(10, -7), 5 * pow(10, -7), pow(10, -6), 5 * pow(10, -6), pow(10, -5)]
print('\nL2-Regularization with architecture: ' + str(reg_arch) + ' ' + str(regularization))
architecture, regulariztn, decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, reg_arch, actfn='relu', last_act='softmax',
               reg_coeffs=regularization, num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0],
               sgd_moms=[0.0],
               sgd_Nesterov=False, EStop=False, verbose=0)

print 'Best value of regularization parameter: ', regulariztn, ' with accuracy: ', accuracy

# (h) Early Stopping and L2-regularization:--------------------------------------------------------
print('\nEarly Stopping with architecture: ' + str(reg_arch) + ' ' + str(regularization))
architecture, op_regulariztn, decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, reg_arch, actfn='relu', last_act='softmax',
               reg_coeffs=regularization, num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0],
               sgd_moms=[0.0],
               sgd_Nesterov=False, EStop=True, verbose=0)

print 'Best value of regularization parameter: ', op_regulariztn, ' with accuracy: ', accuracy

# (i) SGD with weight decay:
weight_decay = [pow(10, -5), 5 * pow(10, -5), pow(10, -4), 3 * pow(10, -4), 7 * pow(10, -4), pow(10, -3)]
i_arch = [[50, 800, 500, 300, 2]]
architecture, regulariztn, op_decay, momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, i_arch, actfn='relu', last_act='softmax',
               reg_coeffs=[5 * pow(10, -7)], num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=weight_decay,
               sgd_moms=[0.0], sgd_Nesterov=False, EStop=False, verbose=0)

print 'Best value of decay parameter: ', op_decay, ' with accuracy: ', accuracy

# (j) Momentum:-----------------------------------------------------------------

architecture, regulariztn, decay, op_momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, i_arch, actfn='relu', last_act='softmax',
               reg_coeffs=[0.0], num_epoch=50, batch_size=1000, sgd_lr=0.00001, sgd_decays=[op_decay],
               sgd_moms=[0.99, 0.98, 0.95, 0.9, 0.85], sgd_Nesterov=True, EStop=False, verbose=0)

print ('Best value of momentum coefficient: ', op_momentum, ' with accuracy: ', accuracy)

# ---------------Part k-----------------

architecture, regulariztn, optimal_decay, optimal_momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, i_arch, actfn='relu', last_act='softmax',
               reg_coeffs=[op_regulariztn], num_epoch=100, batch_size=1000, sgd_lr=0.00001,
               sgd_decays=[op_decay], sgd_moms=[op_momentum], sgd_Nesterov=True, EStop=True, verbose=0)

# -------------Part l-------------------

decay_list = [pow(10, -5), 5 * pow(10, -5), pow(10, -4)]

architecture, op_regulariztn, op_decay, op_momentum, accuracy = \
    testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_2, actfn='relu', last_act='softmax',
               reg_coeffs=regularization, num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=decay_list,
               sgd_moms=[0.99], sgd_Nesterov=True, EStop=True, verbose=0)

print ('Best test accuracy:', accuracy)
print('Best architecture: ', architecture, ' Best regularization parameter: ', op_regulariztn,
      ' Best decay: ', optimal_decay)
