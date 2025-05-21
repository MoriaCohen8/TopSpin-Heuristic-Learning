import random
from topspin import TopSpinState
from heuristics import BellmanUpdateHeuristic, BootstrappingHeuristic, BaseHeuristic
from BWAS import BWAS

def generate_random_states(n, k, num_states=1000):
    states = []
    goal_state = list(range(1, n + 1))
    for _ in range(num_states):
        state = goal_state.copy()
        for _ in range(random.randint(1, 150)):  # Apply a random number of random moves
            move = random.choice(['clockwise', 'counter_clockwise', 'reverse'])
            if move == 'clockwise':
                state = TopSpinState(state, k).clockwise_rotation().get_state_as_list()
            elif move == 'counter_clockwise':
                state = TopSpinState(state, k).counter_clockwise_rotation().get_state_as_list()
            else:
                state = TopSpinState(state, k).reverse_first_k().get_state_as_list()
        states.append(TopSpinState(state, k))
    return states

def bellmanUpdateTraining(bellman_update_heuristic, epochs=100, batch_size=1000):
    for ep in range(1000):    
        for epoch in range(epochs):
            print("external epoch "+str(ep+1)+ "    epoch  "+str(epoch))
            states = generate_random_states(bellman_update_heuristic._n, bellman_update_heuristic._k, num_states=batch_size)
            inputs = []
            outputs = []
            for state in states:
                neighbors = state.get_neighbors()
                best_neighbor_h_value = float('inf')
                for neighbor, cost in neighbors:
                    if neighbor.is_goal():
                        best_neighbor_h_value = 0
                        break
                    h_value = bellman_update_heuristic.get_h_values([neighbor])[0]
                    best_neighbor_h_value = min(best_neighbor_h_value, 1 + h_value)
                inputs.append(state)
                outputs.append(best_neighbor_h_value)
            bellman_update_heuristic.train_model(inputs, outputs)
        bellman_update_heuristic.save_model()

def bootstrappingTraining(bootstrapping_heuristic, epochs=5000, batch_size=100, T=10000):
    for epoch in range(epochs):
        states = generate_random_states(bootstrapping_heuristic._n, bootstrapping_heuristic._k, num_states=batch_size)
        inputs = []
        outputs = []
        all_no_solution=True
        for state in states:
            path, _ = BWAS(state, 5, 10, bootstrapping_heuristic.get_h_values, T)
            if path is None:  # Increase the number of expansions if the problem was not solved
                continue
            else:
                all_no_solution=False
            # print("external epoch "+str(e+1)+ "    epoch: "+str( e*10+epoch+1)+",       len batch= "+str(len(inputs))+"        len path= "+str(len(path) - 1)) # "external epoch"+str(e+1)+    e*10+
            solution_length = len(path) - 1
            for i, s in enumerate(path[:-1]):
                inputs.append(TopSpinState(s))
                outputs.append(solution_length - i)
        if all_no_solution:
            T=T*2
            all_no_solution=True
            for state in states:
                path, _ = BWAS(state, 5, 10, bootstrapping_heuristic.get_h_values, T)
                if path is None:  # Increase the number of expansions if the problem was not solved
                    continue
                else:
                    all_no_solution=False
                # print("epoch: "+str(epoch+1)+",       len batch= "+str(len(inputs))+"        len path= "+str(len(path) - 1))
                solution_length = len(path) - 1
                for i, s in enumerate(path[:-1]):
                    inputs.append(TopSpinState(s))
                    outputs.append(solution_length - i)   
            T=T/2   
            if  all_no_solution:
                continue
        print(len(inputs))
        bootstrapping_heuristic.train_model(inputs, outputs,5)
    bootstrapping_heuristic.save_model()


if __name__ == "__main__":
    bellman_heuristic = BellmanUpdateHeuristic(11, 4)
    bootstrapping_heuristic = BootstrappingHeuristic(11, 4)

    bellmanUpdateTraining(bellman_heuristic)
    bootstrappingTraining(bootstrapping_heuristic)