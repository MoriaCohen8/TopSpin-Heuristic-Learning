from queue import PriorityQueue 

class Node:
    def __init__(self,s,g,p,f):
        self.s=s
        self.g=g
        self.p=p
        self.f=f
    def __lt__(self, other):
        if (self.f < other.f):
            return True
        elif(self.f == other.f):
            return self.g > other.g
        return False

    def __eq__(self, other):
        return self.f == other.f and self.g == other.g and self.s==other.s


def path_to_goal(node):
    path = []
    while node:
        path.append(node.s.state_list)
        node = node.p
    if (len(path)==0):
        return None
    path.reverse()  # Reverse the path to get it from start to goal
    return path

def BWAS(start, W, B, heuristic_function, T):
    open = PriorityQueue() # priority queue of nodes based on minimal f
    closed={} # maps states to their shortest discovered path cost
    UB=float('inf')
    LB=0
    nUB=None
    expansions=0
    f_start=heuristic_function([start])[0]
    n_start=Node(start,0,None,f_start)
    open.put(n_start)
    while not open.empty() and expansions<=T:
        generated=[]
        butch_expansion=0
        while (not open.empty()) and butch_expansion<B and (expansions<=T):
            n=open.get()
            expansions=expansions+1
            butch_expansion+=1
            if len(generated)==0:
                LB=max(n.f,LB)
            if (n.s.is_goal()):
                if (UB>n.g):
                    UB=n.g
                    nUB=n
                    continue
            for successor in n.s.get_neighbors():
                g_successor=n.g+successor[1]
                if (successor[0] not in closed) or (g_successor < closed[successor[0]]):
                    closed[successor[0]]=g_successor
                    generated.append((successor[0],g_successor,n))
        if LB>=UB:
            return path_to_goal(nUB),expansions
        generated_states=[x[0] for x in generated]
        if len(generated_states)==0:
            continue
        heuristics=heuristic_function(generated_states)
        if len(heuristics)==0:
            continue
        for i in range(0,len(generated_states)):
            s,g,p=generated[i]
            h=heuristics[i]
            f_s=g+W*h
            n_s=Node(s,g,p,f_s)
            open.put(n_s)
    return path_to_goal(nUB),expansions
