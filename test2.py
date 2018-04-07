# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import GenerationColouring2 as gc
import numpy as np
import pickle

if __name__ == '__main__':
    graph = np.genfromtxt('le450_15c_edges.csv',delimiter =',')
    graph[0,0] = 1
    fitnesses=[]
    generations=[]
    best_individual=[]
    non_improvement = 0
    best_fitness = None
    c=18

    current_gen = gc.Generation(graph=graph,colours=c,pop_size=4)
    generations.append(current_gen)
    current_gen.calc_avg_fitness()
    fitnesses.append(current_gen.avg_fitness)
    current_gen.calc_best_fitness()
    best_fitness = current_gen.best_fitness
    best_individual.append(current_gen.best_fitness)  
    
    while non_improvement < 5 and best_fitness != 0 and current_gen.gen_number < 5:      
        if current_gen.gen_number !=0:
             np.random.shuffle(current_gen.population)
                
        current_gen = current_gen.create_next_gen()
        generations.append(current_gen)
        current_gen.calc_avg_fitness()
        fitnesses.append(current_gen.avg_fitness)
        current_gen.calc_best_fitness()
        best_individual.append(current_gen.best_fitness)            
            
        if best_individual[current_gen.gen_number-1] <= best_individual[current_gen.gen_number]:
            non_improvement+=1
        else:
            non_improvement = 0
            best_fitness = current_gen.best_fitness
        
            
        print('Generation ' +str(current_gen.gen_number) + ' ... Gens Best fitness: ' + str(current_gen.best_fitness))
        print('Best Fitness: ' + str(best_fitness))
        print('Generations without improvement: ' + str(non_improvement))
    
    filename = 'Test2_Output' + str(c)
    with open(filename,'wb') as fp:
        pickle.dump(generations,fp) 