#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:04:28 2021

@author: dawei
"""
"""
Input arguments for KD
"""

import subprocess
            
par = 1
alpha = 0.1
Temp  = 1
cmd = 'python3 joint_trainfixlr_loso_individual.py --par %d --alpha %f --Temp %d --batch_size 128 --num_epochs 1 --num-classes 23 --learning_rate 0.001' %(par, alpha, Temp)
subprocess.run(cmd.split(' '))
