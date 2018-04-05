# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:46:25 2018

@author: tomor
"""
import copy
import operator
import numpy as np

def gpx_crossover(self,x,y):
    print('Crossover Started')
    ###take the chromosome from the Colouring opjects and copy to local variables for editing
		###Store variables a list of the length of each sublist, for selecting largest list
    x_chromosome1 = copy.deepcopy(x.chromosome)
    x_chromosome1_len = [len(x) for x in x_chromosome1]
    y_chromosome1 = copy.deepcopy(y.chromosome)
    y_chromosome1_len = [len(y) for y in y_chromosome1]
    x_chromosome2 = copy.deepcopy(x.chromosome)
    x_chromosome2_len = [len(x) for x in x_chromosome2]
    y_chromosome2 = copy.deepcopy(y.chromosome)
    y_chromosome2_len = [len(y) for y in y_chromosome2]
    
		##Initialise child solutions
    print('Chromosomes set - Ready to go!')
    child1=[]
    child2=[]
    
    ###alternate between which parent list to use
    first_parent = True
		###Repeat until the child has a number of lists equal to the number of colours set in this Generation object
    for i in range (0,self.colours):
        print('Colourset ' + str(i))
        if first_parent:
				###Get the index of the longest lists for the x chromosome for first child and y chromosme for second child
            x_index,x_max = max(enumerate(x_chromosome1_len), key=operator.itemgetter(1))
            y_index,y_max = max(enumerate(y_chromosome2_len), key=operator.itemgetter(1))
        
				###Add the lists from appropriate index (largest sublist) to the solution
            child1.append(x_chromosome1[x_index])
            child2.append(y_chromosome2[y_index])
            print(child1)
            print(child2)
            first_parent = False
            
            print('Group appended')
            ##For all points in the sublist selected for the x chromosome, remove it from the x and y chromosomes for that child
            for vertex in x_chromosome1[x_index]:
                new_x1=[]
                new_y1=[]
                for partition in x_chromosome1:
                    ##Lists are recreated here...printing suggests that these are working as expected
                    print('original partition:' + str(partition))
                    partition[:]=[x for x in partition if x != vertex]     
                    new_x1.append(partition)
                    print('vertex: ' + str(vertex))
                    print(partition)
                    
                for partition in y_chromosome1:
                    print('original partition:' + str(partition))
                    partition[:]=[y for y in partition if y != vertex]
                    new_y1.append(partition)
                    print('vertex: ' + str(vertex))
                    print(partition)
                x_chromosome1 = copy.copy(new_x1)
                y_chromosome1 = copy.copy(new_y1)
            ##For all points in the sublist selected for the x chromosome, remove it from the x and y chromosomes for that child
            for vertex in y_chromosome2[y_index]:
                new_x2=[]
                new_y2=[]
                for partition in x_chromosome2:
                    print('original partition:' + str(partition))
                    partition[:]=[x for x in partition if x != vertex]  
                    new_x2.append(partition)
                    print('vertex: ' + str(vertex))
                    print(partition)
            
                for partition in y_chromosome2:
                    print('original partition:' + str(partition))
                    partition[:]=[y for y in partition if y != vertex]
                    new_y2.append(partition)
                    print('vertex: ' + str(vertex))
                    print(partition)
                x_chromosome2 = copy.copy(new_x2)
                y_chromosome2 = copy.copy(new_y2)
            print('placed groups removed')
            
				##Recalculate the lengths of sublists for all chromosome
            x_chromosome1_len = [len(x) for x in x_chromosome1]
            y_chromosome1_len = [len(y) for y in y_chromosome1]
            y_chromosome2_len = [len(y) for y in y_chromosome2]
            x_chromosome2_len = [len(x) for x in x_chromosome2]
        
        ##Same process as above but starting with y for frst child and x for second
        else:
            x_index,x_max = max(enumerate(x_chromosome2_len), key=operator.itemgetter(1))
            y_index,y_max = max(enumerate(y_chromosome1_len), key=operator.itemgetter(1))

            child1.append(y_chromosome1[y_index])
            child2.append(x_chromosome2[x_index])
            print(child1)
            print(child2)
            first_parent = True
            print('Colourset appended')
            
            for vertex in y_chromosome1[y_index]:
                new_x1 = []
                new_y1 = []
                for partition in x_chromosome1:
                    partition[:]=[x for x in partition if x != vertex]     
                    new_x1.append(partition)
                    
                for partition in y_chromosome1:
                    partition[:]=[y for y in partition if y != vertex]
                    new_y1.append(partition)
                
                x_chromosome1 = copy.copy(new_x1)
                y_chromosome1 = copy.copy(new_y2)
            
            for vertex in x_chromosome2[x_index]:
                new_x2 = []
                new_y2 = []
                
                for partition in x_chromosome2:
                    partition[:]=[x for x in partition if x != vertex]     
                    new_x2.append(partition)    
                    
                for partition in y_chromosome2:
                    partition[:]=[y for y in partition if y != vertex]
                    new_y2.append(partition)
                
                x_chromosome2 = copy.copy(new_x2)
                y_chromosome2 = copy.copy(new_y2)
                
            print('Places set removed')

            x_chromosome1_len = [len(x) for x in x_chromosome1]
            y_chromosome1_len = [len(y) for y in y_chromosome1]
            y_chromosome2_len = [len(y) for y in y_chromosome2]
            x_chromosome2_len = [len(x) for x in x_chromosome2]

        
    ##RANDOMLY ADD LEFT OVER POINTS IN BEST SET
    ##shuffle
    print('Random placing the rest...')
    if any(x_chromosome1):
        loop_count = 0
        vertex_remaining1 = []
        for colourset in x_chromosome1:
            for x in colourset:
                vertex_remaining1.append(x)
                
            
        np.random.shuffle(vertex_remaining1)
        for vertex in vertex_remaining1:
            loop_count+=1
            adjacent_vertex = [i for i,x in enumerate(self.graph[vertex,:]) if x]
            invalid_edges1 = []
            for partition in child1:
                invalid_edges1.append(len(set(adjacent_vertex) & set(partition)))
        
            p_index,p_min = min(enumerate(invalid_edges1),key=operator.itemgetter(1))
            child1[p_index].append(vertex)

        print('finished placing...')
        print(str(loop_count) + ' vertex placed')
        #Don't need to remove as cycling through the list
        #[z.remove(vertex)  for z in x_chromosome1 if vertex in z]
        #Dont need as all gathered from x
        #[z.remove(vertex)  for z in y_chromosome1 if vertex in z]

    if any(x_chromosome2):
        loop_count = 0
        vertex_remaining2 = []
        for colourset in x_chromosome2:
            for x in colourset:
                vertex_remaining2.append(x)
                
            
        np.random.shuffle(vertex_remaining2)
        for vertex in vertex_remaining2:
            loop_count+=1
            adjacent_vertex = [i for i,x in enumerate(self.graph[vertex,:]) if x]
            invalid_edges2 = []
            for partition in child2:
                invalid_edges2.append(len(set(adjacent_vertex) & set(partition)))
        
            p_index,p_min = min(enumerate(invalid_edges2),key=operator.itemgetter(1))
            child2[p_index].append(vertex)
            
        print('finished placing...')
        print(str(loop_count) + ' vertex placed')
        ##Don't need to remove as cycling through the list
        #[z.remove(vertex)  for z in x_chromosome2 if vertex in z]
        #Dont need as all gathered from x
        #[z.remove(vertex)  for z in y_chromosome1 if vertex in z]
    return (child1,child2,x_chromosome1,x_chromosome2,invalid_edges1,invalid_edges2)