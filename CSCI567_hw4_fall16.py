# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 18:17:40 2016

@author: mjhundekar
"""
from hw_utils import *
x_train, y_train, x_test, y_test = loaddata('MiniBooNE_PID.txt');
norm_x_train, norm_x_test = normalize(x_train, x_test);

linear_activations = [[50,2],[50,50,2],[50,50,50,2],[50,50,50,50,2]]



testmodels(norm_x_train, y_train, norm_x_test, y_test, linear_activations, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0);