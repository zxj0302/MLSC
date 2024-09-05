"""
To determine whether a pattern graph F is with htw(F)<=3.
"""
import networkx as nx
import sys
import queue
from multiprocessing import Process, cpu_count, Manager
from gmatch.subcounting.utils import subgraph_count_nx
from gmatch.main import ROOT_DIR

def partition(collection):
    """
    Get all partitions of the set collection.
    """
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]): # recursion
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller


def graph_partitions(G, minimum=5):
    """
    Get all partitions of a graph G.
    Args:
        G: the pattern graph.
        minimum: The partition(or quotient) graph G/P should have at least `minimum` nodes.
    """

    nodes = list(G.nodes())
    all_partitions = partition(nodes)
    for i, P in enumerate(all_partitions):
        if len(P) < minimum:
            continue
        P = list(map(lambda x: set(x), P))
        try:
            gp = nx.quotient_graph(G, P, relabel=True) # partition graph
        except:
            import ipdb; ipdb.set_trace()
            sys.exit('Error in nx.quotient_graph')
            
        if gp.number_of_edges() < 10:
            continue

        yield gp

def worker(qg_queue, count_queue, qg_queue_finished):
    while True:
        try:
            t = qg_queue.get(timeout=0.1)
            k, fm, qg = t
            count = subgraph_count_nx(fm, qg)
            if count > 0:
                count_queue.put((False, k))
                break
        except queue.Empty:
            if qg_queue_finished.qsize() > 0:
                return
            else:
                pass

def tasker(qg_queue, quotient_graphs, fm3, qg_queue_finished):
    for qg in quotient_graphs:
        for k, fm in fm3.items():
            qg_queue.put((k, fm, qg))
    qg_queue_finished.put(True)
    

def htw3_determine(F, single_process=False):
    """
    Determine whether htw(F) <= 3 or not.
    F is a networkx object (or a file?)
    Args:
        F: a pattern graph, networkx object
    Return: 
        True or False
    """

    patterns_dir = ROOT_DIR/'data'/'forbidden_minors'
    fm3 = {
        'K5': patterns_dir/'K5.graph',
        'octahedron': patterns_dir/'octahedron.graph',
        'pentagonal': patterns_dir/'pentagonal.graph',
        'wagner': patterns_dir/'wagner.graph'
    }
    
    quotient_graphs = graph_partitions(F, minimum=5)

    if single_process:
        for qg in quotient_graphs:
            for k, fm in fm3.items():
                count = subgraph_count_nx(fm, qg)
                if count > 0:
                    return False, k
        return True, None
    

    # multiprocessing manner
    qg_queue = Manager().Queue() 
    qg_queue_finished = Manager().Queue() # flag indicating all qg_graphs have been put.
    count_queue = Manager().Queue()
    taskers = Process(target=tasker, args=(qg_queue, quotient_graphs, fm3, qg_queue_finished))
    taskers.start()

    num_workers = max(int(cpu_count()*0.4), 1)
    workers = [Process(target=worker, args=(qg_queue, count_queue, qg_queue_finished)) for i in range(num_workers) ]
    for each in workers:
        each.start()
    
    while any([each.is_alive() for each in workers]):
        if not count_queue.empty():
            for each in workers:
                if each.is_alive(): each.kill()
            if taskers.is_alive(): taskers.kill()
            res = count_queue.get()
            flag, k = res
            return flag, k
    
    for each in workers:
        if each.is_alive(): each.kill()
    if taskers.is_alive(): taskers.kill()

    return True, None