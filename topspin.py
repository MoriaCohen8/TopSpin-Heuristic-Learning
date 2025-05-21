
class TopSpinState:
    def __init__(self, state, k=4):
        self.state_list=state
        self.k=k
        self.n=len(state)

    def is_goal(self):
        for i in range(0,self.n):
            if (self.state_list[i]!=i+1):
                return False
        return True

    def get_state_as_list(self):
        return self.state_list

    def get_neighbors(self):
        n1=self.clockwise_rotation()
        n2=self.counter_clockwise_rotation()
        n3=self.reverse_first_k()
        return [(n3,1),(n1,1),(n2,1)]


    def clockwise_rotation(self):
        return TopSpinState(self.state_list[-1:]+self.state_list[:-1], self.k)

    def counter_clockwise_rotation(self):
        return TopSpinState(self.state_list[1:]+self.state_list[:1], self.k)

    def reverse_first_k(self):
        return TopSpinState(self.state_list[:self.k][::-1]+self.state_list[self.k:], self.k)

    def __repr__(self):
        return str(self.state_list)
    def __hash__(self):
        return hash(tuple(self.state_list))
    def __eq__(self, other):
        return self.state_list == other.state_list and self.k==other.k


t=TopSpinState([3,2,1],2)
n1=t.clockwise_rotation()
n2=t.counter_clockwise_rotation()
n3=t.reverse_first_k()
n=t.get_neighbors
print(t)