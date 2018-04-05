# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:43:06 2018

@author: Tom
"""

import GenerationColouring2 as gc
import numpy as np

if __name__ == '__main__':
    graph = np.genfromtxt('le450_15c_edges.csv',delimiter=',')
    #bug where first read character is shown as NaN instead of its value. Manually override this.
    graph[0,0] = 1 
    
    gen1 = gc.Generation(graph=graph,colours=2,pop_size=2)
    gen2 = gen1.create_next_gen()    
    children = gen1.children
