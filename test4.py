# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:47:15 2018

@author: Tom
"""

import GenerationColouring2 as gc
import numpy as np
import pickle
import copy

if __name__ == '__main__':
   """
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    graph = np.genfromtxt('le450_15c_edges.csv',delimiter=',')
    #bug where first read character is shown as NaN instead of its value. Manually override this.
    graph[0,0] = 1 
    
    c=18
    generations = []
    fitnesses =[]
    best_individual=[]
    non_improvement=0
    no_solution = True
    best_fitness = None
    
    current_gen = gc.Generation(graph=graph,colours=c,pop_size=100)
    with open('diversity_check','wb') as fp:
        pickle.dump(current_gen,fp)
   """
    
   with open('diversity_check','rb') as fp:
        gen0 = pickle.load(fp)
    
   individuals = gen0.population
    
   ls_2 = copy.deepcopy(gen0)
   ls_1 = copy.deepcopy(gen0)
   for i in range(0,10):
       ls_2.population[i].chromosome=ls_2.population[i].local_search2() 
        

   for i in range(0,10):
       ls_1.population[i].local_search2() 
        
   population = []
   population.append(gen0.population)
   population.append(ls_1.population)
   population.append(ls_2.population)
    
   with open('localsearch_check','wb') as fp:
       pickle.dump(population,fp)

"""
    test = [1,2,2,3,1,5]
    test = list(enumerate(test))
    test[:][:][1]
    
    adjacent_vertex = [i for i,x in enumerate(individuals[0].graph[1,:]) if x]
    
    col = [[i,1] for i in range(0,451)]
    col = np.asarray(col)
    col2= [list(range(0,451))]
    
    test_c = gc.Colouring(graph=individuals[0].graph,colours=1,chromosome=col2)
    test_c.calc_fitness2(test_c.chromosome)
"""

if np.random.randint(0,2):
    t = 1
else:
    t=0
    print('smelly')