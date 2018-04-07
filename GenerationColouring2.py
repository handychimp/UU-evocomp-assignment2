# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:19:50 2018

@author: Tom
"""

import numpy as np
import copy
import random
import operator
from multiprocessing import Pool
import pickle

class Colouring:
    
    def __init__(self,graph,colouring=[[]],colours=1,m_id=None):
        
        if graph.shape[0] != graph.shape[1]:
            self.initialise_graph(graph)
        else:
            self.graph = graph
        
        self.colouring = colouring
        self.colours = colours
        self.fitness=None
        self.rand_state=None
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
        graph_adj = np.asarray(graph_adj,dtype=bool)
       
        running_edges = 0
        for i in range(0,vertex+1):         
            edges = graph_base[(graph_base[:,0]==i)] 
            if np.any(edges):
                for vertex1, vertex2 in edges:
                    graph_adj[i,vertex2] = True
                    graph_adj[vertex2,i] = True
                    
                #print('vertex = ' + str(i+1))
                #print('Edges added: ' + str(len(edges)))
                running_edges += len(edges)
                print('Total edges: ' + str(running_edges))
                
        self.graph = graph_adj
        
    def random_colouring(self):
        size = len(self.graph)
        colours = self.colours
        colouring=[range(0,size),self.rand_state.randint(1,colours+1,size=size)]
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
            challenger_colouring=copy.deepcopy(self.colouring)
            self.rand_state.shuffle(challenger_colouring)
            
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
                        if self.rand_state.randint(0,1) == 0:
                            challenger_colouring[vertex,1] = best_colour
                    
                challenger_colouring[vertex,1] = best_colour
             
            challenger_colouring = challenger_colouring[challenger_colouring[:,0].argsort()]
            results = self.calc_fitness(challenger_colouring)
            
            if results[0] < best_total_fitness:
                best_total_fitness = results[0]
                best_vertex_fitness = results[1]
                best_colouring=copy.deepcopy(challenger_colouring)
                non_improvement = 0
                #print('New Fitness:' + str(best_total_fitness))
            else:
                non_improvement+=1
            
            #print('Best Fitness: ' + str(best_total_fitness) + ' Challenger Fitness: ' + str(results[0]) + ' Non-Improvement: ' + str(non_improvement))
        if self.m_id != None:
            print('Final Fitness ' + str(self.m_id) +': ' + str(best_total_fitness))
        else:
            print('Final Fitness:' + str(best_total_fitness))   
            
        self.colouring = best_colouring
        self.fitness = best_total_fitness
        self.vertex_fitness = best_vertex_fitness 
        return self.fitness
    
    def make_chromosome(self):
        chromosome=[]
        
        for i in range(1,self.colours+1):
            coloured_edges = self.colouring[(self.colouring[:,1]==i)]
            chromosome.append(list(coloured_edges[:,0]))
        self.chromosome=chromosome
        print('Chromosome made.')
        return chromosome
    
    def colouring_from_chromosome(self,chromosome):
        colouring = []
        colour = 1
        for colourset in chromosome:
            #print(colourset)
            for vertex in colourset:
                #print(vertex)
                colouring.append([vertex,colour])
            colour += 1
        colouring=np.asarray(colouring)
        sort_index = np.argsort(colouring[:,0])
        colouring2=copy.deepcopy(colouring)
        pointer = 0
        for x in sort_index:
            colouring2[pointer] = colouring[x]
            pointer += 1
        
        
        new_colouring = Colouring(self.graph,colouring=colouring2,colours=self.colours)
        new_colouring.calc_fitness()
        #new_colouring.local_search()
        #new_colouring.make_chromosome()
        new_colouring.chromosome = chromosome
        
        return new_colouring        
        
class Generation:
    
    def __init__(self,graph,colours,population=[],pop_size=None,gen_number=0):
        self.gen_number=gen_number
        if graph.shape[0] != graph.shape[1]:
            self.initialise_graph(graph)
        else:
            self.graph = graph
        self.colours = colours
        self.population = copy.deepcopy(population)
        if population == []:
            ###can probably parallelise this
            p=Pool()
            results = p.map(self.create_member,range(0,pop_size))
            p.close()
            p.join()
#            for i in range(0,pop_size):
#                print('Population Member: ' + str(i))
#                self.create_member(i)
            self.population=results
            self.pop_size=pop_size
            self.id_counter=pop_size
        else:
            for i in range(0,len(population)):
                population[i].m_id = i
            self.pop_size=len(population)
            self.id_counter = len(self.population)
        
        self.children=[]
        self.next_gen=[]
            
    def create_member(self,m_id):
        print('Generation ' + str(self.gen_number) + ' Member ID: ' + str(m_id))
        colouring = Colouring(self.graph,colours=self.colours,m_id=m_id)
        colouring.rand_state=np.random.RandomState(colouring.m_id)
        colouring.colouring = colouring.random_colouring()
        colouring.calc_fitness()
        colouring.local_search()
        colouring.make_chromosome()

        self.population.append(colouring)
        
        return colouring        
    
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
        graph_adj = np.asarray(graph_adj,dtype=bool)
       
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
        
        avg_fitness = np.mean(pop_fitness)
        
        self.avg_fitness = avg_fitness
        
        return avg_fitness
    
    def calc_best_fitness(self):
        pop_fitness = []
        for colouring in self.population:
            pop_fitness.append(colouring.fitness)
        
        x_index,x_min = min(enumerate(pop_fitness), key=operator.itemgetter(1))
        
        self.best_fitness = x_min
        
        return x_min
        
    def gpx_crossover(self,x_parent,y_parent):
        #print('Crossover Started, Members: ' + str(x_parent.m_id), + ', ' + str(y_parent.m_id))
        ###take the chromosome from the Colouring opjects and copy to local variables for editing
		###Store variables a list of the length of each sublist, for selecting largest list
        print('Starting Crossover')
        x_chromosome1 = copy.deepcopy(x_parent.chromosome)
        x_chromosome1_len = [len(x) for x in x_chromosome1]
        y_chromosome1 = copy.deepcopy(y_parent.chromosome)
        y_chromosome1_len = [len(y) for y in y_chromosome1]
        x_chromosome2 = copy.deepcopy(x_parent.chromosome)
        x_chromosome2_len = [len(x) for x in x_chromosome2]
        y_chromosome2 = copy.deepcopy(y_parent.chromosome)
        y_chromosome2_len = [len(y) for y in y_chromosome2]

        rand_seed = x_parent.rand_state.randint(100) + y_parent.rand_state.randint(100)
        local_rand = numpy.random.RandomState(rand_seed)
        
		##Initialise child solutions
       # print('Chromosomes set - Ready to go!')
        child1=[]
        child2=[]
        
        ###alternate between which parent list to use
        first_parent = True
		###Repeat until the child has a number of lists equal to the number of colours set in this Generation object
        for i in range (0,self.colours):
            if first_parent:
				###Get the index of the longest lists for the x chromosome for first child and y chromosme for second child
                x_index,x_max = max(enumerate(x_chromosome1_len), key=operator.itemgetter(1))
                y_index,y_max = max(enumerate(y_chromosome2_len), key=operator.itemgetter(1))
            
				###Add the lists from appropriate index (largest sublist) to the solution
                child1.append(x_chromosome1[x_index])
                child2.append(y_chromosome2[y_index])
                first_parent = False
                
               # print('Group appended')
                ##For all points in the sublist selected for the x chromosome, remove it from the x and y chromosomes for that child
                for vertex in x_chromosome1[x_index]:
                    new_y1=[]

                        
                    for partition in y_chromosome1:
                        partition[:]=[y for y in partition if y != vertex]
                        new_y1.append(partition)

                    y_chromosome1 = copy.copy(new_y1)
                
                del x_chromosome1[x_index]
                ##For all points in the sublist selected for the x chromosome, remove it from the x and y chromosomes for that child
                for vertex in y_chromosome2[y_index]:
                    new_x2=[]
                    #new_y2=[]
                    for partition in x_chromosome2:
                        #print('original partition:' + str(partition))
                        partition[:]=[x for x in partition if x != vertex]  
                        new_x2.append(partition)
                        #print('vertex: ' + str(vertex))
                        #print(partition)
                
#                    for partition in y_chromosome2:
#                        #print('original partition:' + str(partition))
#                        partition[:]=[y for y in partition if y != vertex]
#                        new_y2.append(partition)
#                        #print('vertex: ' + str(vertex))
#                        #print(partition)
                    
                    #print('x_chromo_id:' + str(id(x_chromosome2)))
                    #print('new_x_chromo_id:' + str(id(new_x2)))
                    x_chromosome2 = copy.copy(new_x2)
                    #y_chromosome2 = copy.copy(new_y2)
                    #print('Assigned new_x_chromo_id:' + str(id(x_chromosome2)))
                del y_chromosome2[y_index]
               # print('placed groups removed')
                
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
                
                first_parent = True
                #print('Colourset appended')
                
                for vertex in y_chromosome1[y_index]:
                    new_x1 = []
                    #new_y1 = []
                    for partition in x_chromosome1:
                        partition[:]=[x for x in partition if x != vertex]     
                        new_x1.append(partition)
                        
#                    for partition in y_chromosome1:
#                        partition[:]=[y for y in partition if y != vertex]
#                        new_y1.append(partition)
                    
                    x_chromosome1 = copy.copy(new_x1)
                    #y_chromosome1 = copy.copy(new_y2)
                del y_chromosome1[y_index]

                for vertex in x_chromosome2[x_index]:
                    #new_x2 = []
                    new_y2 = []
                    
#                    for partition in x_chromosome2:
#                        partition[:]=[x for x in partition if x != vertex]     
#                        new_x2.append(partition)    
                        
                    for partition in y_chromosome2:
                        partition[:]=[y for y in partition if y != vertex]
                        new_y2.append(partition)
                    
                    x_chromosome2 = copy.copy(new_x2)
                    y_chromosome2 = copy.copy(new_y2)
                
                del x_chromosome2[x_index]
                    
               # print('Places set removed')
    
                x_chromosome1_len = [len(x) for x in x_chromosome1]
                y_chromosome1_len = [len(y) for y in y_chromosome1]
                y_chromosome2_len = [len(y) for y in y_chromosome2]
                x_chromosome2_len = [len(x) for x in x_chromosome2]

            
        ##RANDOMLY ADD LEFT OVER POINTS IN BEST SET
        ##shuffle
        #print('Random placing the rest...')
        if any(x_chromosome1):
            loop_count = 0
            vertex_remaining1 = []
            for colourset in x_chromosome1:
                for x in colourset:
                    vertex_remaining1.append(x)
                    
                
            local_rand.shuffle(vertex_remaining1)
            for vertex in vertex_remaining1:
                loop_count+=1
                adjacent_vertex = [i for i,x in enumerate(self.graph[vertex,:]) if x]
                invalid_edges1 = []
                for partition in child1:
                    invalid_edges1.append(len(set(adjacent_vertex) and set(partition)))
            
                p_index,p_min = min(enumerate(invalid_edges1),key=operator.itemgetter(1))
                child1[p_index].append(vertex)

            #print('finished placing...')
            #print(str(loop_count) + ' vertex placed')
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
                    
                
            local_rand.shuffle(vertex_remaining2)
            for vertex in vertex_remaining2:
                loop_count+=1
                adjacent_vertex = [i for i,x in enumerate(self.graph[vertex,:]) if x]
                invalid_edges2 = []
                for partition in child2:
                    invalid_edges2.append(len(set(adjacent_vertex) and set(partition)))
            
                p_index,p_min = min(enumerate(invalid_edges2),key=operator.itemgetter(1))
                child2[p_index].append(vertex)
                
            #print('finished placing...')
            #print(str(loop_count) + ' vertex placed')
            ##Don't need to remove as cycling through the list
            #[z.remove(vertex)  for z in x_chromosome2 if vertex in z]
            #Dont need as all gathered from x
            #[z.remove(vertex)  for z in y_chromosome1 if vertex in z]
        #return (child1,child2,x_chromosome1,x_chromosome2,vertex_remaining1,vertex_remaining2)
        c1_colouring = Colouring(graph=self.graph,colours=self.colours)
        c2_colouring = Colouring(graph=self.graph,colours=self.colours)
        c1_colouring = c1_colouring.colouring_from_chromosome(child1)
        c2_colouring = c2_colouring.colouring_from_chromosome(child2)
        rand_1 = x_parent.rand_state.randint(100) + y_parent.rand_state.randint(100)
        rand_2 = x_parent.rand_state.randint(100) + y_parent.rand_state.randint(100)
        c1_colouring.rand_state = np.random.rand_state(rand_1)
        c1_colouring.rand_state = np.random.rand_state(rand_2)
        
        c1_colouring.local_search()
        c2_colouring.local_search()
        
        self.children.append(c1_colouring)
        self.children.append(c2_colouring)
        selected1, selected2 = self.family_tournament(x_parent,y_parent,c1_colouring,c2_colouring)
        self.next_gen.append(selected1)
        self.next_gen.append(selected2)
        
        return (selected1,selected2)
    
    def colouring_from_chromosome(self,chromosome):
        colouring = []
        colour = 1
        for colourset in chromosome:
            for vertex in colourset:
                colouring.append([vertex,colour])
            colour += 1
        colouring=np.asarray(colouring)
        new_colouring = Colouring(self.graph,colouring=colouring,colours=self.colours)
        new_colouring.calc_fitness()
        new_colouring.local_search()
        new_colouring.make_chromosome()
        self.children.append(new_colouring)
        
        return new_colouring        
        
        
    def family_tournament(self,p1,p2,c1,c2):
        print('Starting family tournament')
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
    
        print('Selection complete')
        return (selected[0],selected[1])
    
    
    def create_next_gen(self):
        half_pop = int(self.pop_size/2)
        print('Population length:' + str(len(self.population)))
        print('Pop_size: ' + str(self.pop_size))
        print('Half_pop: ' + str(half_pop))
        pool_args = []
        for i in range (0,half_pop):
            pool_args.append((self.population[2*i],self.population[2*i+1]))    
         
        p=Pool()
        results = p.starmap(self.gpx_crossover,pool_args)
        p.close()
        p.join()

        with open('results_mptest','wb') as fp:
            pickle.dump(results,fp)  
       
        for i in results:
            for j in i:
                self.next_gen.append(j)

        #print('Length results:' + str(len(results)))
        print('Next Generation Created, Size: ' + str(len(self.next_gen)))
        #with open('gen_mptest','wb') as fp:
        #pickle.dump(generations,fp) 
        new_gen = Generation(graph=self.graph,colours=self.colours,population=self.next_gen,gen_number=self.gen_number+1)
        
        return new_gen