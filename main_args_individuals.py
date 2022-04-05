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
    
#cmd = 'python3 /home/dawei/research-socialbit/clf.py'
#subprocess.run(cmd.split(' '))


for par in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
 for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for Temp in [1, 2, 3, 4, 5, 6, 7, 8]:
            # bs = batch_size/2 per gpu
            cmd = 'python joint_trainfixlr_loso_individual.py --par %d --alpha %f --Temp %d --batch_size 256 --num_epochs 100 --num-classes 23 --learning_rate 0.001' %(par, alpha, Temp)
            subprocess.run(cmd.split(' '))
            
#for par in [3]:
 #   for alpha in [0.1, 0.2]:
 #       for Temp in [1, 2]:
 #           cmd = 'python3 joint_trainfixlr_loso_individual.py --par %d --alpha %f --Temp %d --batch_size 128 --num_epochs 1 --num-classes 23 --learning_rate 0.001' %(par, alpha, Temp)
 #           subprocess.run(cmd.split(' '))
