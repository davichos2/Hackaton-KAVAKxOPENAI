"""
Genetic Algorithm Heuristic for LLM Parameter Optimization

This module receives the current parameters and iteration summary,
then uses a genetic algorithm to propose better parameters.
"""

import numpy as np
import copy
from typing import Dict, Any, Tuple, List


# Genetic Algorithm Parameters
POPULATION_SIZE = 20  # Reduced from 100 for faster iterations
MUTATION_RATE = 0.2
NUM_TO_EVALUATE = 5  # How many individuals to actually test (rest use similarity)


def vector_to_params(vector: Tuple) -> Dict[str, Any]:
    """Convert vector representation to parameter dict"""
    retrieval_top_k, top_p, max_tokens, temperature, presence_penalty, frequency_penalty, num_beams = vector

    return {
        "retrieval_top_k": int(retrieval_top_k),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "presence_penalty": float(presence_penalty),
        "frequency_penalty": float(frequency_penalty),
        "num_beams": int(num_beams),
    }


def params_to_vector(params: Dict[str, Any]) -> Tuple:
    """Convert parameter dict to vector representation"""
    return (
        params.get("retrieval_top_k", 50),
        params.get("top_p", 0.9),
        params.get("max_tokens", 100),
        params.get("temperature", 0.7),
        params.get("presence_penalty", 0.0),
        params.get("frequency_penalty", 0.0),
        params.get("num_beams", 1),
    )


def mutate_vector(vector: Tuple, mutation_rate: float) -> Tuple:
    """Apply mutation to a vector with given probability"""
    prob = np.random.rand()

    if prob < mutation_rate:
        # Mutate: generate new random values within bounds
        retrieval_top_k = np.random.randint(10, 100)
        top_p = np.random.uniform(0.6, 1.0)
        max_tokens = np.random.randint(20, 200)
        temperature = np.random.uniform(0.2, 1.5)
        presence_penalty = np.random.uniform(-2.0, 2.0)
        frequency_penalty = np.random.uniform(-2.0, 2.0)
        num_beams = np.random.randint(1, 5)

        return (retrieval_top_k, top_p, max_tokens, temperature,
                presence_penalty, frequency_penalty, num_beams)
    else:
        return vector


def crossover(parent1: Tuple, parent2: Tuple) -> Tuple[Tuple, Tuple]:
    """Two-point crossover between two parents"""
    length = len(parent1)

    # Select two crossover points
    point1 = np.random.randint(0, length)
    point2 = np.random.randint(point1, length)

    # Create children
    child1 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    child2 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]

    return child1, child2


def initialize_population(current_params: Dict[str, Any], size: int) -> List[Tuple]:
    """Initialize population around current parameters"""
    population = []

    # Keep current params as first individual
    current_vector = params_to_vector(current_params)
    population.append(current_vector)

    # Generate rest with variations
    for i in range(size - 1):
        retrieval_top_k = np.random.randint(10, 100)
        top_p = np.random.uniform(0.6, 1.0)
        max_tokens = np.random.randint(20, 200)
        temperature = np.random.uniform(0.2, 1.5)
        presence_penalty = np.random.uniform(-2.0, 2.0)
        frequency_penalty = np.random.uniform(-2.0, 2.0)
        num_beams = np.random.randint(1, 5)

        population.append((retrieval_top_k, top_p, max_tokens, temperature,
                          presence_penalty, frequency_penalty, num_beams))

    return population


# Global population (persists across calls)
_population = None
_fitness_scores = None
_generation = 0


def propose(params: Dict[str, Any], iteration_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function called by evaluator.

    Args:
        params: Current parameter configuration
        iteration_summary: Results from testing current params

    Returns:
        New parameter configuration to test
    """
    global _population, _fitness_scores, _generation

    print(f"\n🧬 Genetic Algorithm - Generation {_generation}")

    # Extract fitness from summary
    current_fitness = iteration_summary.get("mean_quality", 50.0)
    print(f"   Current fitness: {current_fitness:.2f}")

    # Initialize population on first call
    if _population is None:
        print(f"   Initializing population of {POPULATION_SIZE} individuals...")
        _population = initialize_population(params, POPULATION_SIZE)
        _fitness_scores = np.zeros(POPULATION_SIZE)
        _fitness_scores[0] = current_fitness  # First individual is current params
        _generation = 1
    else:
        # Update fitness for the individual we just tested
        current_vector = params_to_vector(params)

        # Find which individual in population matches current params (approximately)
        distances = [np.linalg.norm(np.array(current_vector) - np.array(ind))
                     for ind in _population]
        closest_idx = np.argmin(distances)
        _fitness_scores[closest_idx] = current_fitness

        print(f"   Updated fitness for individual {closest_idx}")

    # Selection: Tournament selection
    def tournament_select(k=3):
        """Select best individual from k random candidates"""
        candidates = np.random.choice(POPULATION_SIZE, k, replace=False)
        best_idx = candidates[np.argmax(_fitness_scores[candidates])]
        return _population[best_idx]

    # Generate offspring
    children = []
    num_children = POPULATION_SIZE // 2

    for _ in range(num_children // 2):
        parent1 = tournament_select()
        parent2 = tournament_select()

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate_vector(child1, MUTATION_RATE)
        child2 = mutate_vector(child2, MUTATION_RATE)

        children.append(child1)
        children.append(child2)

    # Replacement: Keep best individuals from current population
    sorted_indices = np.argsort(_fitness_scores)[::-1]
    num_keep = POPULATION_SIZE - len(children)

    new_population = [_population[idx] for idx in sorted_indices[:num_keep]]
    new_population.extend(children)

    # Update global population
    _population = new_population[:POPULATION_SIZE]

    # Extend fitness scores (new children get 0 initially)
    new_fitness = np.zeros(POPULATION_SIZE)
    for i, idx in enumerate(sorted_indices[:num_keep]):
        new_fitness[i] = _fitness_scores[idx]
    _fitness_scores = new_fitness

    _generation += 1

    # Select best untested individual to evaluate next
    # Find individuals with fitness == 0 (untested)
    untested_indices = np.where(_fitness_scores == 0)[0]

    if len(untested_indices) > 0:
        # Test a random untested individual
        next_idx = np.random.choice(untested_indices)
        print(f"   Proposing untested individual {next_idx}")
    else:
        # All tested, propose best one with small mutation
        best_idx = np.argmax(_fitness_scores)
        print(f"   All tested. Mutating best individual (fitness: {_fitness_scores[best_idx]:.2f})")
        _population[best_idx] = mutate_vector(_population[best_idx], 0.3)
        next_idx = best_idx

    next_vector = _population[next_idx]
    next_params = vector_to_params(next_vector)

    print(f"   Best fitness so far: {np.max(_fitness_scores):.2f}")
    print(f"   Mean fitness: {np.mean(_fitness_scores[_fitness_scores > 0]):.2f}")

    return next_params


def reset():
    """Reset the genetic algorithm state (useful for testing)"""
    global _population, _fitness_scores, _generation
    _population = None
    _fitness_scores = None
    _generation = 0
