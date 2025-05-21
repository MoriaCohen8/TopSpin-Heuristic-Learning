from heuristics import BaseHeuristic
from heuristics import BellmanUpdateHeuristic,BootstrappingHeuristic
from BWAS import BWAS
from topspin import TopSpinState

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance
print("~~~~~~~~~~~~ easy instance ~~~~~~~~~~~")    

print("base_heuristic: ")    
start1 = TopSpinState(instance_1, 4)
base_heuristic = BaseHeuristic(11, 4)
path, expansions = BWAS(start1, 5, 10, base_heuristic.get_h_values, 1000000)
if path is not None:
    print("expansions: "+str(expansions))
    print("path length: "+str(len(path)))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")
print("BellmanUpdateHeuristic: ")    
start1 = TopSpinState(instance_1, 4)
BU_heuristic = BellmanUpdateHeuristic(11, 4)
path, expansions = BWAS(start1, 5, 10, BU_heuristic.get_h_values, 1000000)
if path is not None:
    print("expansions: "+str(expansions))
    print("path length: "+str(len(path)))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")

print("BootstrappingHeuristic: ")    
start1 = TopSpinState(instance_1, 4)
BO_heuristic = BootstrappingHeuristic(11, 4)
path, expansions = BWAS(start1, 5, 10, BO_heuristic.get_h_values, 1000000)
if path is not None:
    print("expansions: "+str(expansions))
    print("path length: "+str(len(path)))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")

print("~~~~~~~~~~~~ hard instance ~~~~~~~~~~~")    
print("base_heuristic: ")    
start2 = TopSpinState(instance_2, 4)
base_heuristic = BaseHeuristic(11, 4)
path, expansions = BWAS(start2, 5, 10, base_heuristic.get_h_values, 1000000)
if path is not None:
    print("expansions: "+str(expansions))
    print("path length: "+str(len(path)))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")
print("BellmanUpdateHeuristic: ")    
start2 = TopSpinState(instance_2, 4)
BU_heuristic = BellmanUpdateHeuristic(11, 4)
path, expansions = BWAS(start2, 5, 10, BU_heuristic.get_h_values, 1000000)
if path is not None:
    print("expansions: "+str(expansions))
    print("path length: "+str(len(path)))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")

print("BootstrappingHeuristic: ")    
start2 = TopSpinState(instance_2, 4)
BO_heuristic = BootstrappingHeuristic(11, 4)
path, expansions = BWAS(start2, 5, 10, BO_heuristic.get_h_values, 1000000)
if path is not None:
    print("expansions: "+str(expansions))
    print("path length: "+str(len(path)))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")