# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 03:49:43 2018

@author: Tom
"""
import pickle
import numpy as np
import GenerationColouring2 as gc

if __name__ == '__main__':
    
    with open('Test2_Output18','rb') as fp:
        gens = pickle.load(fp)