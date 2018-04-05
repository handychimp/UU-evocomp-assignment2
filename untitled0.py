# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:09:54 2018

@author: tomor
"""

import GenerationColouring2 as gc
import numpy as np
import pickle
import copy

if __name__ == '__main__':
    
    with open('generation0','rb') as fp:
        generations = pickle.load(fp)
    
    generations[0].population[0].colouring = generations[0].population[0].random_colouring()
    generations[0].population[0].colouring
    generations[0].population[0].local_search()
    generations[0].population[0].make_chromosome()
    
    generations[0].population[0].make_chromosome()
    generations[0].population[1].make_chromosome()
    a1 = generations[0].population[0].chromosome
    a2 = generations[0].population[1].chromosome
    
    child1,child2,chromo1,chromo2,inv_edg1,inv_edg2=generations[0].gpx_crossover(generations[0].population[0],generations[0].population[1])
#    
    c1 = generations[0].population[0].chromosome
    c2 = generations[0].population[1].chromosome
    
    if any(chromo1):
        print('Why is it not working?')
#    col0 = generations[0].population[0].colouring
#    col1 = generations[0].population[0].colouring
#    graph=generations[0].graph
#    col1[graph[1,:]]
#
#    test = range(450)
#    testcol = gc.Colouring(graph=generations[0].graph)
#    c1 = [x for x in range(0,100)]
#    c2 = [x for x in range(100,200)]
#    c3 = [x for x in range(200,300)]
#    c4 = [x for x in range(300,400)]
#    c5 = [x for x in range(400,451)]
#    testcol.chromosome=[[],c1,c2,[],[],c3,c4,c5]
#    testcol.colouring_from_chromosome(testcol.chromosome)
#    testcol.colouring
#    testcol.chromosome
    #est = copy.copy(generations[0].population[0].chromosome)
    #test2 = [1,2,3,4,80,90,150,300,301,350]
    #for vertex in test2:
     #   [z.remove(vertex)  for z in test if vertex in z]
    
    #print(test)
#    graph = np.genfromtxt('le450_15c_edges.csv',delimiter=',')
#    graph[0,0]=1
#
#    col = gc.Colouring(graph=graph,colours=2)
#
#    col.colouring
#    col.colouring = col.random_colouring()
#
#    col.colouring[col.graph[:,2]]
#
#    graph2 = col.graph
#
#    test = [[1,2,3],[4,5,6]]
#    testlen = [len(x) for x in test]
#    len(test[:])
#    col.make_chromosome()
#    print(col.colours)
#
#    for colourset in generations[0].population[0].chromosome:
#        for x in colourset:
#            print(x)
#            int(generations[0].pop_size/2)
#    test.append([])
#    any(test)
#    with open('generation0','wb') as fp:
#            pickle.dump(generations,fp) 
#            
#
#        #chromosome=[]
#        #colours = [i for i in range(1,self.colours+1)]
#        #for c in colours:
#        #     print(c)
#        
#        #for i in range(1,colours+1):
#        
#            #coloured_edges = self.colouring[(self.colouring[:,1]==i)]
#            #print(i)
#            #print(coloured_edges)
#            #chromosome.append(coloured_edges[:,0])
#        #self.chromosome=chromosome
        #return chromosome