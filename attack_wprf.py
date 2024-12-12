import numpy as np
import time
from math import log2, floor, sqrt, ceil, comb
from itertools import combinations

def initialize_parameters(L):
    n = 2 * L
    m = floor(7.06 * L)
    t = int(2 * L / log2(3))
    return n, m, t

def random_binary_vector(n):
    return np.random.randint(0, 2, size=n, dtype=np.uint8)

def random_binary_matrix(m, n):
    return np.random.randint(0, 2, size=(m, n), dtype=np.uint8)

def random_matrix_F3(t, m):
    return np.random.randint(0, 3, size=(t, m), dtype=np.int8)

def f(x, k):
    return np.multiply(k, x)

def g(x, A):
    return np.mod(np.dot(A, x), 2)

def h(x, B):
    return np.mod(np.dot(B, x), 3)

def F(x, k, A, B):
    return h(g(f(x, k), A), B)

def find_collision(F, n, k, A, B, seen_outputs, seen_inputs):
    num_checked = 0 # Counter for how many inputs have been checked
    while True:
        x = np.random.randint(0, 2, size=n, dtype=np.uint8)
        x_tuple = tuple(x)

        # Ensure we are not checking the same input twice
        if x_tuple not in seen_inputs:
            seen_inputs.add(x_tuple)
            Fx = tuple(F(x, k, A, B))
            num_checked += 1

            if Fx in seen_outputs:
                x_prev = seen_outputs[Fx]
                return x_prev, x, num_checked, seen_outputs, seen_inputs
            else:
                seen_outputs[Fx] = x # Store the output and the corresponding input for future comparison

def index_set_difference(x_a, x_b):
    return {i for i in range(len(x_a)) if x_a[i] != x_b[i]}

def update_K(K, index_set):
    K[list(index_set)] = 0
    return K

def all_match_check(F, K, k, A, B, seen_inputs_list):
    input_checks = 0
    for x in seen_inputs_list[:3]:
        input_checks += 1
        if not np.array_equal(F(np.array(x), K, A, B), F(np.array(x), k, A, B)):
            return False, input_checks
    return True, input_checks

def get_variants_by_hamming_distance(K, distance):
    indices = np.where(K == 1)[0]
    for idx_set in combinations(indices, distance):
        K_variant = K.copy()
        for idx in idx_set:
            K_variant[idx] ^= 1 # Flip the bit at this position
        yield K_variant

def generalized_find_collision_update(F, n, k, A, B, L):
    seen_outputs = {}
    seen_inputs = set()
    collision_complexity = 0
    exhaustive_search_complexity = 0
    counter = 1
    K = np.ones(n, dtype=np.uint8) # Initialize K as a vector of ones
    max_dist_sufficient = True 

    while True:
        # Find a collision (x_a, x_b) such that F(x_a) = F(x_b)
        x_a, x_b, num_checks, seen_outputs, seen_inputs = find_collision(F, n, k, A, B, seen_outputs, seen_inputs)
        collision_complexity += num_checks
        
        # Get index set where x_a and x_b differ
        index_set = index_set_difference(x_a, x_b)

        # Update the vector K using the index set
        K = update_K(K, index_set)

        # If we have collected at least 3 inputs in seen_inputs, check F(x, K, A, B)
        seen_inputs_list = list(seen_inputs)
        is_match, input_checks = all_match_check(F, K, k, A, B, seen_inputs_list)
        if is_match:
            print(f"YOU WIN in {counter} steps.")
            print("\nResults:")
            print(f"Collision complexity: {collision_complexity}")
            print(f"Exhaustive search complexity: {exhaustive_search_complexity}")
            print(f"Number of collisions: {counter}")
            print(f"Total complexity: {collision_complexity + exhaustive_search_complexity}")
            total_complexity = collision_complexity + exhaustive_search_complexity
            return collision_complexity, exhaustive_search_complexity, counter, max_dist_sufficient, total_complexity
            break

        H_1 = np.sum(K)
        LHS = 3 * sum(comb(H_1, j) for j in range(1, ceil(L / (2 ** counter)) + 1))
        RHS = (2 ** ((L + 1) / 2)) * (sqrt(counter + 1) - sqrt(counter))

        found_key = False
        max_dist = None  # Initialize max_dist
        
        if LHS < RHS:
            max_dist = ceil(L / (2 ** counter))
            for dist in range(1, max_dist + 1):
                for K_variant in get_variants_by_hamming_distance(K, dist):
                    is_match, input_checks = all_match_check(F, K_variant, k, A, B, seen_inputs_list)
                    exhaustive_search_complexity += input_checks  # Count inputs checked

                    if is_match:
                        print(f"YOU WIN with Hamming distance {dist} of K after {counter} steps.")
                        found_key = True
                        print("\nResults:")
                        print(f"Collision complexity: {collision_complexity}")
                        print(f"Exhaustive search complexity: {exhaustive_search_complexity}")
                        print(f"Number of collisions: {counter}")
                        print(f"Max_dist sufficient: {max_dist_sufficient}")
                        print(f"Total complexity: {collision_complexity + exhaustive_search_complexity}")
                        total_complexity = collision_complexity + exhaustive_search_complexity
                        return collision_complexity, exhaustive_search_complexity, counter, max_dist_sufficient, total_complexity
            if found_key:
                break

            else:
                print(f"Key not found within max_dist = {max_dist}. Additional collision required.")
                max_dist_sufficient = False

        counter += 1

def main():
    L = int(input("Enter security parameter L: "))
    num_experiments = 100
    results = {
        "collision_complexity": [],
        "exhaustive_search_complexity": [],
        "number_collisions": [],
        "total_complexity": [],
        "max_dist_sufficient": 0,  # Count of experiments where max_dist was sufficient
    }
    for _ in range(num_experiments):
        print(f"Starting experiment {_}.")
        n, m, t = initialize_parameters(L)
        k = random_binary_vector(n)
        A = random_binary_matrix(m, n)
        B = random_matrix_F3(t, m)

        result = generalized_find_collision_update(F, n, k, A, B, L)
        results["collision_complexity"].append(result[0])
        results["exhaustive_search_complexity"].append(result[1])
        results["number_collisions"].append(result[2])
        results["total_complexity"].append(result[4])
        if result[3]:  # If max_dist_sufficient is True
            results["max_dist_sufficient"] += 1

    print("\nFinal Results:")
    print(f"Average Collision Complexity: {np.mean(results['collision_complexity'])}")
    print(f"Average Exhaustive Search Complexity: {np.mean(results['exhaustive_search_complexity'])}")
    print(f"Average Number of Collisions: {np.mean(results['number_collisions'])}")
    print(f"Average Total Complexity: {np.mean(results['total_complexity'])}")
    print(f"Percentage of Max_dist Sufficient Cases: {results['max_dist_sufficient'] / 100 * 100:.2f}%")

if __name__ == "__main__":
    main()
