import numpy as np

import numpy as np

# Define the Ackley function
def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.exp(1)

# Function to initialize the population
def initialize_population(mu, dimensions):
    return np.random.uniform(-5, 5, size=(mu, dimensions))

# Function to create offspring
def create_offspring(mu_population, lambda_, mutation_strength):
    offspring = np.zeros_like(mu_population)
    for i in range(lambda_):
        parent_index = i % mu_population.shape[0]
        offspring[i] = mu_population[parent_index] + mutation_strength * np.random.randn(mu_population.shape[1])
    return offspring

# Function to evaluate the fitness of each individual
def evaluate_population(population):
    return np.array([ackley(x, y) for x, y in population])

# Function to select the top mu individuals
def select_survivors(mu_population, offspring_population, mu):
    combined_population = np.vstack((mu_population, offspring_population))
    fitness_values = evaluate_population(combined_population)
    sorted_indices = np.argsort(fitness_values)
    return combined_population[sorted_indices][:mu]

# Evolution Strategy function
def evolution_strategy(mu, lambda_, generations, mutation_strength):
    dimensions = 2
    mu_population = initialize_population(mu, dimensions)
    for _ in range(generations):
        offspring_population = create_offspring(mu_population, lambda_, mutation_strength)
        mu_population = select_survivors(mu_population, offspring_population, mu)
    best_solution = mu_population[0]
    best_fitness = ackley(*best_solution)
    return best_solution, best_fitness

if __name__ == "__main__":
    np.random.seed(42)
    mu = 5
    lambda_ = 10
    generations = 100
    mutation_strength = 0.1

    best_solution, best_fitness = evolution_strategy(mu, lambda_, generations, mutation_strength)

    print("Best solution found at x =", best_solution[0], "y =", best_solution[1])
    print("Best fitness:", best_fitness)
