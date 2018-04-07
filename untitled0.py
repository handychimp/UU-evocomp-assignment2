# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:16:01 2018

@author: tomor
"""
import pickle
import numpy as np
import GenerationColouring2 as gc
<<<<<<< HEAD
import copy

if __name__=='__main__':
    with open('Test2_Output18','rb') as fp:
        gens = pickle.load(fp)
        
        
    selected1,selected2,child1,child2 = gens[0].gpx_crossover(gens[0].population[0],gens[0].population[1])
    
    test = gens[0].population[0].colouring_from_chromosome(child1)
    
    
    np.sort(test.colouring, axis=0)
    test.colouring
    
    
    sort_test = np.argsort(test_c[:,0])
    
    test_c2=copy.deepcopy(test_c)
    pointer = 0
    for x in sort_test:
        test_c2[pointer] = test_c[x]
        pointer += 1
    
    test_c[0]
    
    np.random.randint(100)
=======

if __name__=='__main__':
    with open('Output18','rb') as fp:
        gens = pickle.load(fp)
        
        
>>>>>>> 5494ebd6f1f8fff21f149b3d2433bccfa441e8da
