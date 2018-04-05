# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:43:06 2018

@author: Tom
"""

import GenerationColouring2 as gc
import numpy as np
import pickle

if __name__ == '__main__':
#    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#    graph = np.genfromtxt('le450_15c_edges.csv',delimiter=',')
#    #bug where first read character is shown as NaN instead of its value. Manually override this.
#    graph[0,0] = 1 
#    
#    gen1 = gc.Generation(graph=graph,colours=2,pop_size=2)
#        
#    gen2 = gen1.create_next_gen()
#    
#    gen1_p = gen1.population
#    gen2_p = gen2.population
#    
#    generations = []
#    generations.append(gen1_p)
#    generations.append(gen2_p)
#    
#    with open('gen_mptest','wb') as fp:
#        pickle.dump(generations,fp) 
        
    with open('gen_mptest','rb') as fp:
        generations = pickle.load(fp)
        
        
#    gen1.gpx_crossover(gen1.population[0],gen1.population[1])
#    children = gen1.children
#    
#    gen2 = gen1.create_next_gen()
#    gen1_p = gen1.population
#    gen1_ng = gen1.next_gen
#    gen2_p = gen2.population
#    gen2.calc_avg_fitness()
#    gen1.calc_avg_fitness()
#    gen2.avg_fitness
#    gen1.avg_fitness
    
#        p=Pool()
#        Trial_results=p.map(map_func,range(0,25))
#        p.close()
#        p.join()
    