
"""
Created on Fri Mar 16 16:58:30 2018

@author: Tom
"""
import GenerationColouring2 as gc
import numpy as np
import pickle

if __name__ == '__main__':
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
    
    while no_solution and c>0:
        current_gen = gc.Generation(graph=graph,colours=c,pop_size=100)
        generations.append(current_gen)
        current_gen.calc_avg_fitness()
        fitnesses.append(current_gen.avg_fitness)
        current_gen.calc_best_fitness()
        best_individual.append(current_gen.best_fitness) 
        best_fitness = current_gen.best_fitness

        print('Generation 0 ... Avg_Fitness: ' + str(current_gen.avg_fitness))
        
        while non_improvement < 5 and best_fitness != 0:
            
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
            
        filename = 'Output' + str(c)
        with open(filename,'wb') as fp:
            pickle.dump(generations,fp) 
        if current_gen.avg_fitness != 0:
            no_solution = False
        else:
            c=c-1