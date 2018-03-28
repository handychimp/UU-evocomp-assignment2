# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:19:50 2018

@author: Tom
"""

import numpy as np
import copy
import random
import operator

class Colouring:
    
    def __init__(self,graph,colouring=[[]],colours=1,m_id=None):
        
        if graph.shape[0] != graph.shape[1]:
            self.initialise_graph(graph)
        else:
            self.graph = graph
        
        self.colouring = colouring
        self.colours = colours
        self.fitness=None
        self.seed=None
        self.m_id=m_id

    def initialise_graph(self,graph):
        
        max1=max(graph[:,0])
        max2=max(graph[:,1])
        
        vertex = int(max(max1,max2))
    
        graph_base = graph.astype(int)
        
        max1=max(graph[:,0])
        max2=max(graph[:,1])
        
        vertex = int(max(max1,max2))
        
        graph_adj = [False] * (vertex+1)
        graph_adj =  [graph_adj] * (vertex+1)
        graph_adj = np.asarray(graph_adj)
       
        running_edges = 0
        for i in range(0,vertex+1):         
            edges = graph_base[(graph_base[:,0]==i)] 
            if np.any(edges):
                for vertex1, vertex2 in edges:
                    graph_adj[i,vertex2] = True
                    graph_adj[vertex2,i] = True
                    
                print('vertex = ' + str(i+1))
                print('Edges added: ' + str(len(edges)))
                running_edges += len(edges)
                print('Total edges: ' + str(running_edges))
                
        self.graph = graph_adj
        
    def random_colouring(self):
        size = len(self.graph)
        colours = self.colours
        colouring=[range(0,size),np.random.randint(1,colours+1,size=size)]
        colouring = np.asarray(colouring)
        colouring = colouring.transpose()
        
        return colouring
    
    def calc_vertex_fitness(self,vertex,colouring=[[]]):
        if colouring==[[]]:
            colouring=self.colouring


        self_colour = colouring[vertex]
        adjacent_colours = colouring[self.graph[vertex,:]]
        invalid_colours = adjacent_colours[(adjacent_colours[:,1]==self_colour[1])]
        
        return len(invalid_colours)
            
    def calc_fitness(self,colouring=[[]]):
        if colouring ==[[]]:                  
            colouring = self.colouring
            commit=True
        else:
            commit=False
        
        v_fitness = []
        total_fitness = 0
        for i in range(0,len(colouring)):
           fitness_result = int(self.calc_vertex_fitness(i,colouring))
           total_fitness += fitness_result
           v_fitness.append([i,fitness_result])
        
        v_fitness = np.asarray(v_fitness)
        total_fitness=total_fitness/2
        if commit:
            self.vertex_fitness = v_fitness
            self.fitness = total_fitness
        
        return [int(total_fitness),v_fitness]
    
    def local_search(self):
        best_colouring = copy.copy(self.colouring)
        best_vertex_fitness = self.vertex_fitness
        best_total_fitness = self.fitness
        non_improvement = 0
        
        print('Initial Fitness: ' + str(best_total_fitness))
        while non_improvement < 100:
            challenger_colouring=copy.copy(self.colouring)
            np.random.shuffle(challenger_colouring)
            
            for vertex in challenger_colouring[:,0]:
                best_colour = best_colouring[vertex]
                best_colour = best_colour[1]
                
                best_fitness = self.calc_vertex_fitness(vertex,challenger_colouring)
                colour_palette = [c for c in range (1,self.colours + 1) if c != best_colour]  
                
                for c in colour_palette:
                    challenger_colouring[vertex,1]=c
                    c_fitness = self.calc_vertex_fitness(vertex,challenger_colouring)
                    
                    if c_fitness < best_fitness:
                        best_fitness = c_fitness
                        best_colour = c
                    elif c_fitness == best_fitness:
                        if random.randint(0,1) == 0:
                            challenger_colouring[vertex,1] = best_colour
                    
                challenger_colouring[vertex,1] = best_colour
             
            challenger_colouring = challenger_colouring[challenger_colouring[:,0].argsort()]
            results = self.calc_fitness(challenger_colouring)
            
            if results[0] < best_total_fitness:
                best_total_fitness = results[0]
                best_vertex_fitness = results[1]
                best_colouring=copy.copy(challenger_colouring)
                non_improvement = 0
            else:
                non_improvement+=1
            
            print('Best Fitness: ' + str(best_total_fitness) + ' Challenger Fitness: ' + str(results[0]) + ' Non-Improvement: ' + str(non_improvement))
            
        self.colouring = best_colouring
        self.fitness = best_total_fitness
        self.vertex_fitness = best_vertex_fitness 
        return self.fitness
    
    def make_chromosome(self):
        chromosome=[]
        for i in range (0,self.colours):
            coloured_edges = self.colouring[(self.colouring[:,1]==i)]
            chromosome.append(coloured_edges[0])
        self.chromosome=chromosome
        
class Generation:
    
    def __init__(self,graph,colours,population=[],pop_size=None,gen_number=0):
        self.gen_number=gen_number
        self.initialise_graph(graph)
        self.colours = colours
        self.population = population
        if population == []:
            ###can probably parallelise this
            for i in range(0,pop_size):
                self.create_member(i)
            self.pop_size=pop_size
            self.id_counter=pop_size
        else:
            for i in range(0,pop_size):
                population[i].m_id = i
            self.pop_size=len(population)
            self.id_counter = len(self.population)
        
        self.children=[]
        self.next_gen=[]
            
    def create_member(self,m_id):
        colouring = Colouring(self.graph,colours=self.colours,m_id=m_id)
        colouring.random_colouring()
        colouring.calc_fitness()
        colouring.local_search()
        colouring.make_chromosome()
        self.population.append(colouring)        
    
    def initialise_graph(self,graph):
        
        max1=max(graph[:,0])
        max2=max(graph[:,1])
        
        vertex = int(max(max1,max2))
    
        graph_base = graph.astype(int)
        
        max1=max(graph[:,0])
        max2=max(graph[:,1])
        
        vertex = int(max(max1,max2))
        
        graph_adj = [False] * (vertex+1)
        graph_adj =  [graph_adj] * (vertex+1)
        graph_adj = np.asarray(graph_adj)
       
        running_edges = 0
        for i in range(0,vertex+1):         
            edges = graph_base[(graph_base[:,0]==i)] 
            if np.any(edges):
                for vertex1, vertex2 in edges:
                    graph_adj[i,vertex2] = True
                    graph_adj[vertex2,i] = True
                    
                print('vertex = ' + str(i+1))
                print('Edges added: ' + str(len(edges)))
                running_edges += len(edges)
                print('Total edges: ' + str(running_edges))
                
        self.graph = graph_adj
        
    def calc_avg_fitness(self):
        pop_fitness = []
        for colouring in self.population:
            pop_fitness.append(colouring.fitness)
        
        avg_fitness = avg(pop_fitness)
        
        self.avg_fitness = avg_fitness
        
    def gpx_crossover(self,x,y):
        
        x_chromosome1 = x.chromosome
        y_chromosome1 = y.chromosome
        x_chromosome2 = x.chromosome
        y_chromosome2 = y.chromosome
        
        child1=[]
        child2=[]
        
        ###for odd and even --
        first_parent = True
        for i in range (0,self.colours):
            x_index,x_max = max(enumerate(x_chromosome1), key=operator.itemgetter(1))
            y_index,y_max = max(enumerate(y_chromosome1), key=operator.itemgetter(1))
            
            if first_parent:
                child1.append(x_chromosome1[x_index])
                child2.append(y_chromosome2[y_index])
                first_parent = False
                
                for vertex in x_chromosome1[x_index]:
                    [z.remove(vertex)  for z in x_chromosome1 if vertex in z]
                    [z.remove(vertex)  for z in y_chromosome1 if vertex in z]
                
                for vertex in y_chromosome2[y_index]:
                    [z.remove(vertex)  for z in x_chromosome2 if vertex in z]
                    [z.remove(vertex)  for z in y_chromosome2 if vertex in z]
                    
            else:
                child1.append(y_chromosome1[y_index])
                child2.append(x_chromosome2[x_index])
                first_parent = True
                
                for vertex in y_chromosome1[x_index]:
                    [z.remove(vertex)  for z in x_chromosome1 if vertex in z]
                    [z.remove(vertex)  for z in y_chromosome1 if vertex in z]
                
                for vertex in x_chromosome2[y_index]:
                    [z.remove(vertex)  for z in x_chromosome2 if vertex in z]
                    [z.remove(vertex)  for z in y_chromosome2 if vertex in z]
                    
        ##RANDOMLY ADD LEFT OVER POINTS IN BEST SET
        ###shuffle
        np.random.shuffle(x_chromosome1)
        for colourset, vertex in x_chromosome1:
            adjacent_vertex = [i for i,x in enumerate(self.graph[vertex,:]) if x]
            
            invalid_edges = []
            for partition in child1:
                invalid_edges.append(len(set(adjacent_vertex) & set(partition)))
            
            p_index,p_min = max(enumerate(invalid_edges),key=operator.itemgetter(1))
            child1[p_index].append(vertex)
            
            #Don't need to remove as cycling through the list
            #[z.remove(vertex)  for z in x_chromosome1 if vertex in z]
            #Dont need as all gathered from x
            #[z.remove(vertex)  for z in y_chromosome1 if vertex in z]

        np.random.shuffle(x_chromosome2)
        for colourset, vertex in x_chromosome2:
            adjacent_vertex = [i for i,x in enumerate(self.graph[vertex,:]) if x]
            
            invalid_edges = []
            for partition in child2:
                invalid_edges.append(len(set(adjacent_vertex) & set(partition)))
            
            p_index,p_max = min(enumerate(invalid_edges),key=operator.itemgetter(1))
            child2[p_index].append(vertex)
            
            ##Don't need to remove as cycling through the list
            #[z.remove(vertex)  for z in x_chromosome2 if vertex in z]
            #Dont need as all gathered from x
            #[z.remove(vertex)  for z in y_chromosome1 if vertex in z]
        child1 = self.colouring_from_chromosome(child1)
        child2 = self.colouring_from_chromosome(child2)
            
        selected1, selected2 = self.family_competition(x,y,child1,child2)
        self.next_gen.append(selected1)
        self.next_gen.append(selected2)
    
    def colouring_from_chromosome(self,chromosome):
        colouring = []
        colour = 1
        for colourset in chromosome:
            for vertex in colourset:
                colouring.append[vertex,colour]
            colour += 1
        
        new_colouring = Colouring(self.graph,colouring=colouring,colours=self.colours)
        new_colouring.calc_fitness()
        new_colouring.local_search()
        new_colouring.make_chromosome()
        self.children.append(new_colouring)
        
        return new_colouring        
        
        
    def family_tournament(self,p1,p2,c1,c2):
        parents=[]
        parents.append(p1)
        parents.append(p2)
        
        family = [] 
        family.append(c1)
        family.append(c2)
        family.append(p1)
        family.append(p2)
        
        family_fitness = [x.fitness for x in family]
        f_index,f_min = min(enumerate(family_fitness),key=operator.itemgetter(1))
        
        selected = []
        selected.append(family[f_index])
        del family_fitness[f_index]
        del family[f_index]
        
        f_index,f_min = min(enumerate(family_fitness),key=operator.itemgetter(1))
        selected.append(family[f_index])
    
        return (selected[0],selected[1])
    
    
    def create_next_gen(self):
        half_pop = int(self.pop_size/2)
        
        for i in range (0,half_pop):
            self.gpx_crossover(self.population[2*i],self.population[2*i+1])
        
        new_gen = Generation(graph=self.graph,colours=self.colours,population=self.next_gen,gen_number=self.gen_number+1)
        
        return new_gen