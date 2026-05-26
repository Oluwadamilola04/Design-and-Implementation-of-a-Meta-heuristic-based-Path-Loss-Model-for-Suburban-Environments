"""
Metaheuristic Optimization Algorithms for Path Loss Model Parameters
Implements PSO and GA optimization with history tracking
"""

import numpy as np
from typing import Tuple, Callable, Dict, List
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class PSOptimizer:
    """
    Particle Swarm Optimization (PSO) with convergence history tracking
    
    Optimizes COST-231 parameters to minimize RMSE
    """
    
    def __init__(self, n_particles: int = 50, n_dimensions: int = 9,
                 n_iterations: int = 1000, c1: float = 0.5, 
                 c2: float = 0.3, w: float = 0.9,
                 bounds: Tuple[np.ndarray, np.ndarray] = None):
        """
        Initialize PSO
        
        Args:
            n_particles: Number of particles in swarm
            n_dimensions: Number of dimensions (parameters)
            n_iterations: Number of iterations
            c1: Cognitive parameter
            c2: Social parameter
            w: Inertia weight
            bounds: Tuple of (lower_bounds, upper_bounds)
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
        if bounds is None:
            self.lower_bounds = np.zeros(n_dimensions)
            self.upper_bounds = np.ones(n_dimensions) * 100
        else:
            self.lower_bounds, self.upper_bounds = bounds
        
        # Initialize particles and velocities
        self.positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds, 
            (n_particles, n_dimensions)
        )
        self.velocities = np.random.uniform(
            -1, 1, (n_particles, n_dimensions)
        )
        
        # Track best solutions
        self.best_positions = self.positions.copy()
        self.best_costs = np.full(n_particles, np.inf)
        self.global_best_position = None
        self.global_best_cost = np.inf
        
        # History
        self.cost_history = []
        self.position_history = []
    
    def optimize(self, objective_func: Callable, data: Tuple) -> Tuple[float, np.ndarray, List[float]]:
        """
        Run PSO optimization
        
        Args:
            objective_func: Function to minimize, takes (params, data) and returns RMSE
            data: Data tuple to pass to objective function
            
        Returns:
            Tuple of (best_cost, best_params, cost_history)
        """
        for iteration in range(self.n_iterations):
            # Evaluate fitness for all particles
            costs = np.array([
                objective_func(self.positions[i], data) 
                for i in range(self.n_particles)
            ])
            
            # Update best solutions
            improved = costs < self.best_costs
            self.best_costs[improved] = costs[improved]
            self.best_positions[improved] = self.positions[improved]
            
            # Update global best
            best_idx = np.argmin(self.best_costs)
            if self.best_costs[best_idx] < self.global_best_cost:
                self.global_best_cost = self.best_costs[best_idx]
                self.global_best_position = self.best_positions[best_idx].copy()
            
            # Update velocities and positions
            r1 = np.random.random((self.n_particles, self.n_dimensions))
            r2 = np.random.random((self.n_particles, self.n_dimensions))
            
            self.velocities = (self.w * self.velocities +
                              self.c1 * r1 * (self.best_positions - self.positions) +
                              self.c2 * r2 * (self.global_best_position - self.positions))
            
            self.positions = self.positions + self.velocities
            
            # Enforce bounds
            self.positions = np.clip(self.positions, self.lower_bounds, self.upper_bounds)
            
            # Record history
            self.cost_history.append(self.global_best_cost)
            self.position_history.append(self.global_best_position.copy())
            
            if (iteration + 1) % 50 == 0:
                print(f"PSO Iteration {iteration + 1}/{self.n_iterations} - "
                      f"Best RMSE: {self.global_best_cost:.4f}")
        
        return self.global_best_cost, self.global_best_position, self.cost_history


class GAOptimizer:
    """
    Genetic Algorithm Optimizer for path loss model parameters
    Uses DEAP library
    """
    
    def __init__(self, pop_size: int = 100, n_generations: int = 1000,
                 cxpb: float = 0.9, mutpb: float = 0.5,
                 bounds: Tuple[np.ndarray, np.ndarray] = None):
        """
        Initialize GA
        
        Args:
            pop_size: Population size
            n_generations: Number of generations
            cxpb: Crossover probability
            mutpb: Mutation probability
            bounds: Tuple of (lower_bounds, upper_bounds)
        """
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        
        if bounds is None:
            self.lower_bounds = np.zeros(9)
            self.upper_bounds = np.ones(9) * 100
        else:
            if len(bounds) == 2 and np.asarray(bounds[0]).ndim > 0:
                self.lower_bounds = np.asarray(bounds[0], dtype=float)
                self.upper_bounds = np.asarray(bounds[1], dtype=float)
            else:
                bounds_array = np.asarray(bounds, dtype=float)
                if bounds_array.ndim == 2 and bounds_array.shape[1] == 2:
                    self.lower_bounds = bounds_array[:, 0]
                    self.upper_bounds = bounds_array[:, 1]
                else:
                    raise ValueError(
                        "bounds must be either (lower_bounds, upper_bounds) "
                        "or a list of (lower, upper) pairs"
                    )
        
        self.cost_history = []
        self.best_individual = None
        self.best_cost = np.inf
    
    def optimize(self, objective_func: Callable, data: Tuple) -> Tuple[float, np.ndarray, List[float]]:
        """
        Run GA optimization
        
        Args:
            objective_func: Function to minimize
            data: Data tuple to pass to objective function
            
        Returns:
            Tuple of (best_cost, best_params, cost_history)
        """
        try:
            from deap import base, creator, tools, algorithms
            import random
        except ImportError:
            raise ImportError("DEAP library required. Install with: pip install deap")
        
        # Clear previous DEAP definitions if they exist
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Define fitness and individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        n_dims = len(self.lower_bounds)
        
        # Attribute generators
        for i in range(n_dims):
            toolbox.register(f"attr_float_{i}", random.uniform, 
                           self.lower_bounds[i], self.upper_bounds[i])
        
        # Individual and population
        attr_funcs = tuple(getattr(toolbox, f"attr_float_{i}") for i in range(n_dims))
        toolbox.register("individual", tools.initCycle, creator.Individual, 
                        attr_funcs, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Evaluation, crossover, mutation
        def evaluate_individual(individual):
            return (objective_func(np.array(individual), data),)

        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population
        population = toolbox.population(n=self.pop_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        hof = tools.HallOfFame(1)
        
        print(f"Starting GA optimization: {self.n_generations} generations, "
              f"population size {self.pop_size}...")
        
        # Run algorithm
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=self.n_generations, stats=stats, halloffame=hof, verbose=False
        )
        
        # Extract history
        self.cost_history = logbook.select("min")
        self.best_individual = hof[0]
        self.best_cost = hof[0].fitness.values[0]
        
        # Print progress every 100 generations
        for gen in range(0, self.n_generations, 100):
            if gen < len(self.cost_history):
                print(f"GA Generation {gen}/{self.n_generations} - "
                      f"Best RMSE: {self.cost_history[gen]:.4f}")
        
        print(f"GA Generation {self.n_generations}/{self.n_generations} - "
              f"Best RMSE: {self.best_cost:.4f}")
        
        return self.best_cost, np.array(self.best_individual), self.cost_history


def create_objective_function(model_func: Callable) -> Callable:
    """
    Create an objective function for optimization
    
    Args:
        model_func: Function that calculates COST-231 model predictions
        
    Returns:
        Objective function that calculates RMSE
    """
    def objective(params, data):
        f_mhz, h_b_m, h_m_m, d_km, measured_path_loss_db = data
        
        # Predict using model with current parameters
        predicted = model_func(f_mhz, h_b_m, h_m_m, d_km, params)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(measured_path_loss_db, predicted))
        return rmse
    
    return objective
