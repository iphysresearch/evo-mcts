import random

def parent_selection_weight(pop, weights, m):
    """
    Selects parents from the population using weighted random selection.
    
    Args:
        pop (list): List of individuals in the population.
        weights (list): List of weights for each individual.
        m (int): Number of parents to select.
    
    Returns:
        list: Selected parent individuals.
    
    Note:
        - Uses random.choices for weighted random selection with replacement
        - Normalizes weights to ensure they sum to 1
    """
    # Check if inputs are valid
    if not pop:
        print("Warning: Empty population provided")
        return []
    
    if len(pop) != len(weights):
        print(f"Warning: Population size ({len(pop)}) doesn't match weights size ({len(weights)})")
        # Adjust weights to match population size
        if len(weights) < len(pop):
            weights = weights + [0] * (len(pop) - len(weights))
        else:
            weights = weights[:len(pop)]
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        # If all weights are zero, use uniform weights
        normalized_weights = [1.0 / len(weights) for _ in weights]
    
    # Ensure m is not larger than population size
    m = min(m, len(pop))
    
    # Select m parents with replacement according to normalized probabilities
    try:
        parents = random.choices(pop, weights=normalized_weights, k=m)
        return parents
    except Exception as e:
        print(f"Error in parent selection: {e}")
        print(f"Population size: {len(pop)}, Weights size: {len(normalized_weights)}, m: {m}")
        # Fallback to uniform selection
        return random.choices(pop, k=m)

def parent_selection(pop, m, unique=False):
    """
    Selects parents from the population using rank-based selection with inverse rank probability.
    
    Args:
        pop (list): List of individuals in the population, typically sorted by objective value.
        m (int): Number of parents to select.
        unique (bool): If True, ensures no duplicate parents are selected (without replacement).
                      If False, allows duplicates (with replacement). Default is False.
    
    Returns:
        list: Selected parent individuals.
    
    Note:
        - Assigns probability inversely proportional to rank (lower ranks = better individuals get higher probability)
        - Adds len(pop) to denominator to reduce selection pressure for small populations
        - Uses random.choices for weighted random selection with replacement
        - Uses random.sample for selection without replacement when unique=True
    """
    # sort pop by objective value
    pop = sorted(pop, key=lambda x: x['objective'])
    
    # Create ranks for each individual (0 to len(pop)-1)
    ranks = [i for i in range(len(pop))]
    
    # Calculate selection probability for each individual
    # Formula: 1/(rank + 1 + len(pop)) gives higher probability to lower ranks
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    
    if unique:
        # Ensure m is not larger than population size when sampling without replacement
        m = min(m, len(pop))
        # Use weighted sampling without replacement
        parents = random.sample(population=pop, k=m, counts=[int(p * 1000) for p in probs])
    else:
        # Select m parents with replacement according to calculated probabilities
        parents = random.choices(pop, weights=probs, k=m)
    
    return parents

def parent_selection_e1(pop, m):
    """
    Selects parents from the population using uniform random selection.
    
    Args:
        pop (list): List of individuals in the population.
        m (int): Number of parents to select.
    
    Returns:
        list: Selected parent individuals.
    
    Note:
        - All individuals have equal probability of selection (uniform weights)
        - Used for exploration rather than exploitation
        - Uses random.choices for random selection with replacement
    """
    # Assign equal probability to all individuals
    probs = [1 for i in range(len(pop))]
    
    # Select m parents with replacement using uniform probability
    parents = random.choices(pop, weights=probs, k=m)
    return parents