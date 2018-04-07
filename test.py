# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import GenerationColouring2 as gc
import numpy as np
import pickle

if __name__ == '__main__':
    graph = np.genfromtxt('test_graph.csv',delimiter =',')
    graph[0,0] = 1
    
    c=6
    no_solution = True
    
    while no_solution and c > 0:
        
        generations = []
        fitnesses =[]
        best_individual = []
        non_improvement=0
        no_solution = True
        best_fitness = None
            
        no_solution = True
        current_gen = gc.Generation(graph=graph,colours=c,pop_size=100)
        generations.append(current_gen)
        current_gen.calc_avg_fitness()
        fitnesses.append(current_gen.avg_fitness)
        current_gen.calc_best_fitness()
        best_individual.append(current_gen.best_fitness)
        print('Generation 0 ... Avg_Fitness: ' + str(current_gen.avg_fitness))
        
        while non_improvement < 2:
            
            current_gen = current_gen.create_next_gen()
            generations.append(current_gen)
            current_gen.calc_avg_fitness()
            fitnesses.append(current_gen.avg_fitness)
            current_gen.calc_best_fitness()
            
            
            if fitnesses[current_gen.gen_number-1] <= fitnesses[current_gen.gen_number]:
                non_improvement+=1
            else:
                non_improvement = 0
                best_fitness = current_gen.avg_fitness
            
            print('Colours ' + str(c) + 'Generation ' +str(current_gen.gen_number) + ' ... Avg_Fitness: ' + str(current_gen.avg_fitness))
            print('Best Fitness: ' + str(best_fitness))
            print('Generations without improvement: ' + str(non_improvement))
            
        filename = 'Test_Output' + str(c)
        with open(filename,'wb') as fp:
            pickle.dump(generations,fp) 
        if best_fitness != 0:
            no_solution = False
        else:
            c=c-1