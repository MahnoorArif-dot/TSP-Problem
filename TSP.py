import random
import numpy as np

NUM_CITIES = 10
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.05

np.random.seed(0)
distance_matrix = np.random.randint(10, 100, size=(NUM_CITIES, NUM_CITIES))
np.fill_diagonal(distance_matrix, 0)

def fitness(route):
    total_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    total_distance += distance_matrix[route[-1], route[0]] 
    return 1 / total_distance


def create_population():
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(POPULATION_SIZE)]


def select_parent(population, fitness_scores):
    tournament = random.sample(population, 5)
    best = max(tournament, key=lambda x: fitness_scores[tuple(x)])
    return best

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [-1] * NUM_CITIES
    child[start:end] = parent1[start:end]

    for i in range(start, end):
        if parent2[i] not in child:
            pos = i
            while start <= pos < end:
                pos = parent2.index(parent1[pos])
            child[pos] = parent2[i]

    for i in range(NUM_CITIES):
        if child[i] == -1:
            child[i] = parent2[i]
    
    return child

def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm_tsp():
    population = create_population()
    best_route = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        fitness_scores = {tuple(route): fitness(route) for route in population}
        
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = select_parent(population, fitness_scores)
            parent2 = select_parent(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population

        current_best_route = max(population, key=lambda x: fitness_scores.get(tuple(x), fitness(x)))
        current_best_fitness = fitness_scores.get(tuple(current_best_route), fitness(current_best_route))
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_route = current_best_route

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_route, 1 / best_fitness  


best_route, best_distance = genetic_algorithm_tsp()
print(f"Best Route: {best_route}")
print(f"Best Distance: {best_distance}")
