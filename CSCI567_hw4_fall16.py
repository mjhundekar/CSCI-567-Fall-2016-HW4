# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 18:17:40 2016

@author: mjhundekar
"""
from hw_utils import *
x_train, y_train, x_test, y_test = loaddata('MiniBooNE_PID.txt');
norm_x_train, norm_x_test = normalize(x_train, x_test);

achitecture_1 = [[50,2],[50,50,2],[50,50,50,2],[50,50,50,50,2]]

achitecture_2 = [[50,50,2],[50,500,2],[50,500,300,2],[50,800,500,300,2]. [50, 800, 800, 500, 300, 2]]

testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_1, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
           
testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_2, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)

testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_1, actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)


testmodels(norm_x_train, y_train, norm_x_test, y_test, achitecture_1, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)

reg_arch =  [[50, 800, 500, 300, 2]]

regularization = [pow(10,-7), 5*pow(10,-7), pow(10, -6), 5*pow(10, -6), pow(10, -5)]

testmodels(norm_x_train, y_train, norm_x_test, y_test, reg_arch, actfn='relu', last_act='softmax', reg_coeffs=regularization,
				num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)
