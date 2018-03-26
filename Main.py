
"""
Created on Fri Mar 16 16:58:30 2018

@author: Tom
"""
import GenerationColouring2 as gc
import numpy as np
import pickle
import multiprocessing as mp

def getKey(item):
    return item[0]

if __name__ == '__main__':
   # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    graph = np.genfromtxt('le450_15c_edges.csv',delimiter=',')
    #bug where first read character is shown as NaN instead of its value. Manually override this.
    graph[0,0] = 1 
   
    colouring = np.asarray([range(0,451),[1]*451])
    colouring = colouring.transpose()  
    
    
    
    col_1 = gc.Colouring(graph,colouring)
    
    col_1.calc_fitness()
    col_1.fitness
    
    col_1.colours = 2
    col_1.local_search()
    #test=col_1.colouring
    #col_1.calc_fitness(col_1.colouring)
    #col_1.colouring 
  
    #test = col_1.colouring
    #test = test[test[:,0].argsort()]
   #col_1.colouring
  
  #    colouring.view().sort(order=['f0'], axis = 0)
    

#graph[(graph[:,0]==450) | (graph[:,1]==450)]
#    col_1.fitness
#    col_1.vertex_fitness
#    col_1.colouring
    
   # test = np.asarray([[1,2],[2,80],[3,4]])

#        adjacent_edges = col_1.graph[(col_1.graph[:,0]==1) | (col_1.graph[:,1]==1)]
#        print(adjacent_edges)
#        for vertex1, vertex2 in adjacent_edges:
#            vertex1_col = col_1.colouring[(col_1.colouring[:,0]==vertex1)]
#            vertex2_col = col_1.colouring[(col_1.colouring[:,0]==vertex2)]
#            
#            vertex1_col[0][1] == vertex2_col[0][1]



      #   test = col_1.vertex_fitness[(col_1.vertex_fitness[:][0]==1)]
               #test = col_1.vertex_fitness[(col_1.vertex_fitness[:][0]==1)]
               #test[1]
  #  col_1.calc_fitness()
 
    #col_1.vertex_fitness[,][1]
    
    #for vertex in col_1.colouring[:,0]:
     #   this_vertex = vertex
 
    """
    generations = []
    this_gen = gc.Generation(0,graph)
    this_gen.generate_pop()
    
    #add number of colours parameter and reduce this until solution found
    #pickle results each experiment into a filename related to number of colours
    best_solution = this_gen.best_solution()
    L=0
    while L<=100:
        children = this_gen.pop_crossover()
        next_gen = gc.selection(this_gen,children)
        
        if next_gen.best_solution() > best_solution:
            L=0
            best_solution = next_gen.best_solution
        else:
            L+=1
        
    pickle.dump()


    Example colouring:
        
    colouring = np.asarray([range(1,451),[1]*450])
    colouring = colouring.transpose()
    
    col_1 = gc.Colouring(raw_graph,colouring)
    col_1.fitness
    col_1.calc_fitness()

    
    """
        