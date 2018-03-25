
"""
Created on Fri Mar 16 16:13:31 2018

@author: Tom
"""
import random
import numpy as np
import copy
import multiprocessing as mp

#class Graph:
#    
#    def __init__(self,graph,partition_count = 1):
#        self.graph = graph
#        self.partition_count = partition_count
#        if self.partition_count > 1:
#            self.calc_partition()
#    
#    def graph_partition(self,cut_size):
#        #Fiduccia Matheyses Algorith
   #     pass


######## PARTITIONING REPRESENTATION?
   ##### All vertices of one colour represent a set or partition.
   ##### What is the gain or loss for swapping a vertex?
   ##### Need to do this as multiple passes in random order, storing the one
   ##### with most gain
class Colouring:
    
    def __init__(self,graph,colouring,colours=1,gen=None):
        self.graph=graph
        self.colouring = colouring
        self.colours = colours
        self.generation_created = gen
        self.vertex_fitness = [[]]
        self.children=0
        self.fitness=None
        self.seed=None
    
    def set_graph(self,graph):
        self.graph=graph
        
    def calc_fitness(self,colouring=[]):
        #check number of incorrect edges
        if len(colouring) == 0:
            colouring = self.colouring
            commit_result = True
        else:
            commit_result = False
        
        running_fitness = 0
        v_fitness = []
        for vertex in colouring[:,0]:
            inv_edges = 0
            #Only have to consider this side of graph as edges errors are symmetric
            adjacent_edges = self.graph[(self.graph[:,0]==vertex) | (self.graph[:,1]==vertex)]
            for vertex1, vertex2 in adjacent_edges:
                vertex1_col = colouring[(colouring[:,0]==vertex1)]
                vertex2_col = colouring[(colouring[:,0]==vertex2)]
               
                if vertex1_col[0][1] == vertex2_col[0][1]:
                    inv_edges += 1
            v_fitness.append([vertex,inv_edges])
            running_fitness += inv_edges
                   
        if commit_result:
            self.fitness = int(running_fitness/2)
            self.vertex_fitness=np.asarray(v_fitness)
        
        return [int(running_fitness/2),np.asarray(v_fitness)]
    
    def calc_vertex_fitness(self,vertex,colouring=[]):
        
        if len(colouring) == 0:
            colouring = self.colouring
        
        inv_edges = 0
        #Only have to consider this side of graph as edges errors are symmetric
        adjacent_edges = self.graph[(self.graph[:,0]==vertex) | (self.graph[:,1]==vertex)]

        for vertex1, vertex2 in adjacent_edges:
            vertex1_col = colouring[(colouring[:,0]==vertex1)]
            vertex2_col = colouring[(colouring[:,0]==vertex2)]

            if vertex1_col[0][1] == vertex2_col[0][1]:
                inv_edges += 1
                
        return inv_edges
        
    
    ###multi thread this
    def local_search(self):
        #local search for optimum using vertex descent
        
        if self.vertex_fitness == []:
            self.calc_fitness()
        
        ###set for repeated run
        best_colouring = copy.copy(self.colouring)
        best_total_fitness = self.fitness
        non_improvement = 0
        
        ###while not same for 100 iterations
        while non_improvement < 100:
            challenger_colouring = copy.copy(self.colouring)
            np.random.shuffle(challenger_colouring)

            cpu_count = mp.cpu_count()
            step = int(len(challenger_colouring)/cpu_count)
            print(step)
            splits= []
            
            for i in range(0,cpu_count):
                if i == cpu_count:
                    splits.append(challenger_colouring[(i*step):(len(challenger_colouring)-1)])    
                else:
                    splits.append(challenger_colouring[(i*step):(step*(i+1))])
            print(splits)
            p=mp.Pool()
            return_colouring = p.map(self.vertex_descent,splits)
            p.close()
            p.join()
            print('RETURN COLOURING:')
            print(return_colouring)
            challenger_colouring = np.asarray([])
            for i in range(0,cpu_count):
                challenger_colouring += return_colouring[i]
            
            print(challenger_colouring)
            results = self.calc_fitness(challenger_colouring)        
            
            print(results[0])
            if results[0] < best_total_fitness:
                best_total_fitness=results[0]
                best_vertex_fitness = results[1]
                best_colouring=copy.copy(challenger_colouring)
                non_improvement = 0
            else:
                non_improvement +=1
            
            print (non_improvement)
                
        self.colouring = best_colouring
        self.fitness = best_total_fitness
        self.vertex_fitness = best_vertex_fitness 
        return self.fitness
    
    def vertex_descent(self,colouring):
        print('Starting job')
        for vertex in colouring[:,0]:
            print('vertex ' + str(vertex) + '...calculating')
            best_colour = colouring[(colouring[:,0]==vertex)]
            best_colour = best_colour[0][1]
                
            ###recalculate
            best_fitness = self.calc_vertex_fitness(vertex,colouring)
            colour_palette = [c for c in range (1,self.colours + 1) if c != best_colour]         
            inv_edges = 0        
            adjacent_edges = self.graph[(self.graph[:,0]==vertex) | (self.graph[:,1]==vertex)]
            for c in colour_palette:
                for vertex1, vertex2 in adjacent_edges:
                    vertex2_col = colouring[(colouring[:,0]==vertex2)] if vertex2 != vertex else colouring[(colouring[:,0]==vertex1)] 
                
                    if c == vertex2_col[0][1]:
                        inv_edges +=1
                
                if inv_edges < best_fitness:
                    best_colour = c
                    best_fitness = inv_edges
                elif inv_edges == best_fitness:
                    if random.randint(0,1) == 0:
                        best_colour = c
                        best_fitness = inv_edges
                        
            print('vertex ' + str(vertex) + ' ...colour: ' + str(best_colour) + ' ...fitness: ' + str (best_fitness))
            colouring[(colouring[:,0]==vertex)] = [vertex,best_colour]        
        return colouring
        
        pass
    
    def crossover (self,other):
        #crossover between two parents - return a generation class object
        pass

    ##method overiding the == , < , > ,etc logical operators    
        
class Generation:  
    
    def __init__(self,k,graph,pop,c):
        self.generation=k
        self.population=pop
        self.mean_fitness=None
        self.sdev_fitness=None
        self.graph=graph
        self.colours=c
    
    def generate_pop(self,n,c):
        #generate a population of size n colourings
        
        pass
        
    def pop_fitness(self):
        #compute fitness of all in gen without fitness
        pass
    
    def pop_crossover(self):
        #perform crossover
        pass
    
    def best_solution(self):
        #return a colouring that is the best so far
        pass
    
def selection(x,y):
    #return a generation class object
    pass
