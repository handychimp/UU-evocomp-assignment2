# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:19:50 2018

@author: Tom
"""

import numpy as np
import copy
import random

class Colouring:
    
    def __init__(self,graph,colouring,colours=1):
        self.initialise_graph(graph)
        self.colouring = colouring
        self.colours = colours
        self.fitness=None
        self.seed=None

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
            print('Why are you here?')
        else:
            commit=False
        
        v_fitness = []
        total_fitness = 0
        for i in range(0,len(colouring)):
           fitness_result = int(self.calc_vertex_fitness(i,colouring))
           total_fitness += fitness_result
           v_fitness.append([i,fitness_result])
        
        v_fitness = np.asarray(v_fitness)
        print(total_fitness)
        total_fitness=total_fitness/2
        print(total_fitness)        
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