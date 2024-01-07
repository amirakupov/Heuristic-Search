import numpy as np
import heapq
import time
import random
from statistics import mean, stdev

class Puzzle8Solver:
    def __init__(self, initial_state):
        self.goal_state = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.initial_state = np.array(initial_state)
        self.size = 3

    def find_blank(self, state):
        # Finds the empty tile
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return i, j

    def move(self, state, direction):
        # Moves the tile in the specified direction
        i, j = self.find_blank(state)
        if direction == 'up' and i > 0:
            new_state = state.copy()
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
            return new_state
        elif direction == 'down' and i < self.size - 1:
            new_state = state.copy()
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
            return new_state
        elif direction == 'left' and j > 0:
            new_state = state.copy()
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
            return new_state
        elif direction == 'right' and j < self.size - 1:
            new_state = state.copy()
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
            return new_state
        return None

    def h_manhattan(self, state):
        # Manhattan heuristic distance calculation
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != 0:
                    goal_row, goal_col = divmod(state[i][j] - 1, self.size)
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance

    def is_goal(self, state):
        return np.array_equal(state, self.goal_state)

    def display_board(self, state):
        for row in state:
            print(" ".join(map(str, row)))
        print()

    # Existing code remains mostly the same

    def a_star_search(self, heuristic_func, visualize=False):
        open_list = []
        closed_list = set()

        heapq.heappush(open_list, (heuristic_func(self.initial_state), 0, self.initial_state.tobytes()))

        start_time = time.time()
        expanded_nodes = 0

        while open_list:
            _, cost, current_state = heapq.heappop(open_list)
            current_state = np.frombuffer(current_state, dtype=int).reshape((self.size, self.size))
            if self.is_goal(current_state):
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Solution found in {execution_time:.6f} seconds.")
                print("Total number of expanded nodes:", expanded_nodes)
                return cost, expanded_nodes

            current_state_str = current_state.tobytes()
            if current_state_str not in closed_list:
                closed_list.add(current_state_str)
                expanded_nodes += 1

                i, j = self.find_blank(current_state)
                directions = ['up', 'down', 'left', 'right']
                for direction in directions:
                    new_state = self.move(current_state, direction)
                    if new_state is not None:
                        new_cost = cost + 1
                        heapq.heappush(open_list, (
                            new_cost + heuristic_func(new_state), new_cost, new_state.tobytes()))

        print("No solution found.")
        return None, expanded_nodes

    def h_hamming(self, state):
        # Hamming heuristic (number of misplaced tiles)
        hamming_distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != self.goal_state[i][j] and state[i][j] != 0:
                    hamming_distance += 1
        return hamming_distance

    def a_star_search_hamming(self):
        open_list = []
        closed_list = set()

        heapq.heappush(open_list, (self.h_hamming(self.initial_state), 0, self.initial_state.tobytes(), []))

        start_time = time.time()
        expanded_nodes = 0

        while open_list:
            _, cost, current_state, path = heapq.heappop(open_list)
            current_state = np.frombuffer(current_state, dtype=int).reshape((self.size, self.size))
            if self.is_goal(current_state):
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Solution found in {execution_time:.6f} seconds.")
                print("Solution path:", path)
                print("Total number of expanded nodes:", expanded_nodes)
                return path

            current_state_str = current_state.tobytes()
            if current_state_str not in closed_list:
                closed_list.add(current_state_str)
                expanded_nodes += 1

                i, j = self.find_blank(current_state)
                directions = ['up', 'down', 'left', 'right']
                for direction in directions:
                    new_state = self.move(current_state, direction)
                    if new_state is not None:
                        new_cost = cost + 1
                        new_path = path + [direction]
                        heapq.heappush(open_list, (
                            new_cost + self.h_hamming(new_state), new_cost, new_state.tobytes(), new_path))

        print("No solution found.")
        return None
    def generate_random_state(self):
        # Generate a solvable random start state
        while True:
            temp_state = list(range(9))
            random.shuffle(temp_state)
            random_state = np.array(temp_state).reshape((self.size, self.size))
            if self.check_solvability(random_state):
                return random_state

    def check_solvability(self, state):
        # Check if a given state is solvable
        inversions = 0
        state_list = state.flatten()
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] != 0 and state_list[j] != 0 and state_list[i] > state_list[j]:
                    inversions += 1
        return inversions % 2 == 0

    def measure_performance(self, heuristic_func, iterations=100):
        memory_usage = []
        expanded_nodes_list = []
        execution_time = []

        for _ in range(iterations):
            random_state = self.generate_random_state()
            self.initial_state = random_state

            start_time = time.time()
            path, expanded_nodes = self.a_star_search(heuristic_func)
            end_time = time.time()

            if path:
                memory_usage.append(expanded_nodes)
            else:
                memory_usage.append(0)
            expanded_nodes_list.append(expanded_nodes)
            execution_time.append(end_time - start_time)

        return memory_usage, expanded_nodes_list, execution_time

initial_board = [
    [1, 2, 3],
    [0, 4, 5],
    [6, 7, 8]
]
solver = Puzzle8Solver(initial_board)

manhattan_path, manhattan_expanded = solver.a_star_search(solver.h_manhattan, visualize=True)
manhattan_memory, manhattan_expanded, manhattan_time = solver.measure_performance(solver.h_manhattan)
# Measure performance for Hamming heuristic with visualization
hamming_path, hamming_expanded = solver.a_star_search(solver.h_hamming, visualize=True)
# Measure performance for Hamming heuristic
hamming_memory, hamming_expanded, hamming_time = solver.measure_performance(solver.h_hamming)

# Calculate mean and standard deviation
manhattan_mean_memory = mean(manhattan_memory)
manhattan_std_memory = stdev(manhattan_memory)
manhattan_mean_expanded = mean(manhattan_expanded)
manhattan_std_expanded = stdev(manhattan_expanded)
manhattan_mean_time = mean(manhattan_time)
manhattan_std_time = stdev(manhattan_time)

hamming_mean_memory = mean(hamming_memory)
hamming_std_memory = stdev(hamming_memory)
hamming_mean_expanded = mean(hamming_expanded)
hamming_std_expanded = stdev(hamming_expanded)
hamming_mean_time = mean(hamming_time)
hamming_std_time = stdev(hamming_time)

print("Manhattan Heuristic:")
print(f"Mean Memory Usage: {manhattan_mean_memory}, Standard Deviation: {manhattan_std_memory}")
print(f"Mean Expanded Nodes: {manhattan_mean_expanded}, Standard Deviation: {manhattan_std_expanded}")
print(f"Mean Execution Time: {manhattan_mean_time}, Standard Deviation: {manhattan_std_time}")

print("\nHamming Heuristic:")
print(f"Mean Memory Usage: {hamming_mean_memory}, Standard Deviation: {hamming_std_memory}")
print(f"Mean Expanded Nodes: {hamming_mean_expanded}, Standard Deviation: {hamming_std_expanded}")
print(f"Mean Execution Time: {hamming_mean_time}, Standard Deviation: {hamming_std_time}")