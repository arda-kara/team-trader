"""
Strategy optimizer for optimizing trading strategy parameters.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from ..config.settings import settings
from ..models.base import (
    Strategy, OptimizationResult, OptimizationMethod, ObjectiveFunction,
    StrategyParameter
)
from ..generators.strategy_generator import strategy_generator
from ..backtester.backtester import backtester
from ..generators.redis_client import optimization_cache

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """Optimizer for trading strategies."""
    
    def __init__(self):
        """Initialize strategy optimizer."""
        self.max_trials = settings.optimizer.max_trials
        self.timeout = settings.optimizer.timeout
        self.n_jobs = settings.optimizer.n_jobs
        self.default_objective = settings.optimizer.default_objective
        self.max_drawdown_constraint = settings.optimizer.max_drawdown_constraint
        self.min_trades_constraint = settings.optimizer.min_trades_constraint
        self.min_win_rate_constraint = settings.optimizer.min_win_rate_constraint
    
    def _generate_optimization_id(self) -> str:
        """Generate unique optimization ID.
        
        Returns:
            str: Optimization ID
        """
        return f"optimization_{uuid.uuid4().hex[:8]}"
    
    async def optimize_strategy(self, strategy_id: str, start_date: datetime, end_date: datetime,
                              initial_capital: float = 10000.0, method: Optional[str] = None,
                              objective: Optional[str] = None, parameters: Optional[Dict[str, Dict[str, Any]]] = None,
                              constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize strategy parameters.
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            method: Optimization method
            objective: Objective function
            parameters: Parameters to optimize with ranges
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult: Optimization result
        """
        # Get strategy
        strategy = strategy_generator.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        # Set defaults if not provided
        if method is None:
            method = settings.optimizer.default_method
        
        if objective is None:
            objective = settings.optimizer.default_objective
        
        if parameters is None:
            # Use strategy parameters with their min/max values
            parameters = {}
            for param in strategy.parameters:
                if param.min_value is not None and param.max_value is not None:
                    parameters[param.name] = {
                        "min": param.min_value,
                        "max": param.max_value,
                        "step": param.step
                    }
                elif param.choices is not None:
                    parameters[param.name] = {
                        "choices": param.choices
                    }
        
        if constraints is None:
            constraints = {
                "max_drawdown": self.max_drawdown_constraint,
                "min_trades": self.min_trades_constraint,
                "min_win_rate": self.min_win_rate_constraint
            }
        
        # Run optimization based on method
        if method == "grid_search":
            result = await self._optimize_grid_search(
                strategy, start_date, end_date, initial_capital, 
                objective, parameters, constraints
            )
        elif method == "random_search":
            result = await self._optimize_random_search(
                strategy, start_date, end_date, initial_capital, 
                objective, parameters, constraints
            )
        elif method == "bayesian":
            result = await self._optimize_bayesian(
                strategy, start_date, end_date, initial_capital, 
                objective, parameters, constraints
            )
        elif method == "genetic":
            result = await self._optimize_genetic(
                strategy, start_date, end_date, initial_capital, 
                objective, parameters, constraints
            )
        else:
            # Default to Bayesian optimization
            result = await self._optimize_bayesian(
                strategy, start_date, end_date, initial_capital, 
                objective, parameters, constraints
            )
        
        # Cache result
        optimization_cache.set(f"optimization:{result.id}", result.dict())
        
        return result
    
    async def _optimize_grid_search(self, strategy: Strategy, start_date: datetime, end_date: datetime,
                                  initial_capital: float, objective: str, 
                                  parameters: Dict[str, Dict[str, Any]],
                                  constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize strategy using grid search.
        
        Args:
            strategy: Strategy
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            objective: Objective function
            parameters: Parameters to optimize with ranges
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult: Optimization result
        """
        # Generate optimization ID
        optimization_id = self._generate_optimization_id()
        
        # Start time
        start_time = datetime.now()
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(parameters)
        
        # Initialize results
        all_results = []
        best_score = float('-inf')
        best_parameters = {}
        
        # Run grid search
        for param_set in param_grid:
            # Run backtest with parameters
            backtest_result = await backtester.run_backtest(
                strategy.id, start_date, end_date, initial_capital, param_set
            )
            
            # Calculate objective score
            score = self._calculate_objective_score(backtest_result, objective)
            
            # Check constraints
            constraints_met = self._check_constraints(backtest_result, constraints)
            
            # Record result
            result = {
                "parameters": param_set,
                "score": score,
                "constraints_met": constraints_met,
                "metrics": {
                    "total_return": backtest_result.total_return,
                    "sharpe_ratio": backtest_result.sharpe_ratio,
                    "max_drawdown": backtest_result.max_drawdown,
                    "win_rate": backtest_result.win_rate
                }
            }
            all_results.append(result)
            
            # Update best if constraints met and score is better
            if constraints_met and score > best_score:
                best_score = score
                best_parameters = param_set
        
        # End time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create optimization result
        result = OptimizationResult(
            id=optimization_id,
            strategy_id=strategy.id,
            method=OptimizationMethod.GRID_SEARCH,
            objective=ObjectiveFunction(objective),
            start_date=start_date,
            end_date=end_date,
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            execution_time=execution_time
        )
        
        return result
    
    def _generate_parameter_grid(self, parameters: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search.
        
        Args:
            parameters: Parameters with ranges
            
        Returns:
            List[Dict[str, Any]]: Parameter grid
        """
        # Generate grid for each parameter
        param_values = {}
        for param_name, param_range in parameters.items():
            if "choices" in param_range:
                # Discrete choices
                param_values[param_name] = param_range["choices"]
            elif "min" in param_range and "max" in param_range:
                # Numeric range
                min_val = param_range["min"]
                max_val = param_range["max"]
                step = param_range.get("step", 1)
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer range
                    param_values[param_name] = list(range(min_val, max_val + 1, step))
                else:
                    # Float range
                    values = []
                    val = min_val
                    while val <= max_val:
                        values.append(val)
                        val += step
                    param_values[param_name] = values
        
        # Generate all combinations
        import itertools
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]
        
        grid = []
        for values in itertools.product(*param_value_lists):
            param_set = {name: value for name, value in zip(param_names, values)}
            grid.append(param_set)
        
        return grid
    
    async def _optimize_random_search(self, strategy: Strategy, start_date: datetime, end_date: datetime,
                                    initial_capital: float, objective: str, 
                                    parameters: Dict[str, Dict[str, Any]],
                                    constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize strategy using random search.
        
        Args:
            strategy: Strategy
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            objective: Objective function
            parameters: Parameters to optimize with ranges
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult: Optimization result
        """
        # Generate optimization ID
        optimization_id = self._generate_optimization_id()
        
        # Start time
        start_time = datetime.now()
        
        # Initialize results
        all_results = []
        best_score = float('-inf')
        best_parameters = {}
        
        # Run random search
        num_trials = min(self.max_trials, 100)  # Limit to 100 trials for random search
        
        for _ in range(num_trials):
            # Generate random parameters
            param_set = self._generate_random_parameters(parameters)
            
            # Run backtest with parameters
            backtest_result = await backtester.run_backtest(
                strategy.id, start_date, end_date, initial_capital, param_set
            )
            
            # Calculate objective score
            score = self._calculate_objective_score(backtest_result, objective)
            
            # Check constraints
            constraints_met = self._check_constraints(backtest_result, constraints)
            
            # Record result
            result = {
                "parameters": param_set,
                "score": score,
                "constraints_met": constraints_met,
                "metrics": {
                    "total_return": backtest_result.total_return,
                    "sharpe_ratio": backtest_result.sharpe_ratio,
                    "max_drawdown": backtest_result.max_drawdown,
                    "win_rate": backtest_result.win_rate
                }
            }
            all_results.append(result)
            
            # Update best if constraints met and score is better
            if constraints_met and score > best_score:
                best_score = score
                best_parameters = param_set
        
        # End time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create optimization result
        result = OptimizationResult(
            id=optimization_id,
            strategy_id=strategy.id,
            method=OptimizationMethod.RANDOM_SEARCH,
            objective=ObjectiveFunction(objective),
            start_date=start_date,
            end_date=end_date,
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            execution_time=execution_time
        )
        
        return result
    
    def _generate_random_parameters(self, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate random parameters within ranges.
        
        Args:
            parameters: Parameters with ranges
            
        Returns:
            Dict[str, Any]: Random parameter set
        """
        param_set = {}
        
        for param_name, param_range in parameters.items():
            if "choices" in param_range:
                # Discrete choices
                param_set[param_name] = np.random.choice(param_range["choices"])
            elif "min" in param_range and "max" in param_range:
                # Numeric range
                min_val = param_range["min"]
                max_val = param_range["max"]
                step = param_range.get("step", None)
                
                if isinstance(min_val, int) and isinstance(max_val, int) and step is not None:
                    # Integer range with step
                    values = list(range(min_val, max_val + 1, step))
                    param_set[param_name] = np.random.choice(values)
                elif isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer range
                    param_set[param_name] = np.random.randint(min_val, max_val + 1)
                elif step is not None:
                    # Float range with step
                    values = []
                    val = min_val
                    while val <= max_val:
                        values.append(val)
                        val += step
                    param_set[param_name] = np.random.choice(values)
                else:
                    # Float range
                    param_set[param_name] = np.random.uniform(min_val, max_val)
        
        return param_set
    
    async def _optimize_bayesian(self, strategy: Strategy, start_date: datetime, end_date: datetime,
                               initial_capital: float, objective: str, 
                               parameters: Dict[str, Dict[str, Any]],
                               constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize strategy using Bayesian optimization.
        
        Args:
            strategy: Strategy
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            objective: Objective function
            parameters: Parameters to optimize with ranges
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult: Optimization result
        """
        # Generate optimization ID
        optimization_id = self._generate_optimization_id()
        
        # Start time
        start_time = datetime.now()
        
        # Initialize results
        all_results = []
        
        # Create Optuna study
        sampler = TPESampler()
        pruner = MedianPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective function for Optuna
        async def optuna_objective(trial):
            # Generate parameters from trial
            param_set = {}
            
            for param_name, param_range in parameters.items():
                if "choices" in param_range:
                    # Discrete choices
                    param_set[param_name] = trial.suggest_categorical(
                        param_name, param_range["choices"]
                    )
                elif "min" in param_range and "max" in param_range:
                    # Numeric range
                    min_val = param_range["min"]
                    max_val = param_range["max"]
                    step = param_range.get("step", None)
                    
                    if isinstance(min_val, int) and isinstance(max_val, int) and step is not None:
                        # Integer range with step
                        param_set[param_name] = trial.suggest_int(
                            param_name, min_val, max_val, step=step
                        )
                    elif isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer range
                        param_set[param_name] = trial.suggest_int(
                            param_name, min_val, max_val
                        )
                    elif step is not None:
                        # Float range with step
                        param_set[param_name] = trial.suggest_discrete_uniform(
                            param_name, min_val, max_val, step
                        )
                    else:
                        # Float range
                        param_set[param_name] = trial.suggest_float(
                            param_name, min_val, max_val
                        )
            
            # Run backtest with parameters
            backtest_result = await backtester.run_backtest(
                strategy.id, start_date, end_date, initial_capital, param_set
            )
            
            # Calculate objective score
            score = self._calculate_objective_score(backtest_result, objective)
            
            # Check constraints
            constraints_met = self._check_constraints(backtest_result, constraints)
            
            # Record result
            result = {
                "parameters": param_set,
                "score": score,
                "constraints_met": constraints_met,
                "metrics": {
                    "total_return": backtest_result.total_return,
                    "sharpe_ratio": backtest_result.sharpe_ratio,
                    "max_drawdown": backtest_result.max_drawdown,
                    "win_rate": backtest_result.win_rate
                }
            }
            all_results.append(result)
            
            # Apply constraints as penalties
            if not constraints_met:
                score = float('-inf')
            
            return score
        
        # Run optimization
        await self._run_optuna_optimization(study, optuna_objective)
        
        # Get best parameters and score
        best_parameters = study.best_params
        best_score = study.best_value
        
        # End time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create optimization result
        result = OptimizationResult(
            id=optimization_id,
            strategy_id=strategy.id,
            method=OptimizationMethod.BAYESIAN,
            objective=ObjectiveFunction(objective),
            start_date=start_date,
            end_date=end_date,
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            execution_time=execution_time
        )
        
        return result
    
    async def _run_optuna_optimization(self, study: optuna.Study, objective_func: Callable) -> None:
        """Run Optuna optimization.
        
        Args:
            study: Optuna study
            objective_func: Objective function
        """
        # Define synchronous wrapper for async objective
        def objective_wrapper(trial):
            import asyncio
            return asyncio.run(objective_func(trial))
        
        # Run optimization
        study.optimize(
            objective_wrapper,
            n_trials=self.max_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )
    
    async def _optimize_genetic(self, strategy: Strategy, start_date: datetime, end_date: datetime,
                              initial_capital: float, objective: str, 
                              parameters: Dict[str, Dict[str, Any]],
                              constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize strategy using genetic algorithm.
        
        Args:
            strategy: Strategy
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            objective: Objective function
            parameters: Parameters to optimize with ranges
            constraints: Optimization constraints
            
        Returns:
            OptimizationResult: Optimization result
        """
        # This is a simplified implementation of genetic algorithm optimization
        # In a real implementation, this would use a proper genetic algorithm library
        
        # For now, we'll use a simple approach similar to random search
        # but with some "evolution" between generations
        
        # Generate optimization ID
        optimization_id = self._generate_optimization_id()
        
        # Start time
        start_time = datetime.now()
        
        # Initialize results
        all_results = []
        best_score = float('-inf')
        best_parameters = {}
        
        # Parameters for genetic algorithm
        population_size = 20
        num_generations = min(5, self.max_trials // population_size)
        mutation_rate = 0.2
        
        # Generate initial population
        population = []
        for _ in range(population_size):
            param_set = self._generate_random_parameters(parameters)
            population.append(param_set)
        
        # Run genetic algorithm
        for generation in range(num_generations):
            # Evaluate population
            fitness_scores = []
            
            for param_set in population:
                # Run backtest with parameters
                backtest_result = await backtester.run_backtest(
                    strategy.id, start_date, end_date, initial_capital, param_set
                )
                
                # Calculate objective score
                score = self._calculate_objective_score(backtest_result, objective)
                
                # Check constraints
                constraints_met = self._check_constraints(backtest_result, constraints)
                
                # Apply constraints as penalties
                if not constraints_met:
                    score = float('-inf')
                
                fitness_scores.append(score)
                
                # Record result
                result = {
                    "parameters": param_set,
                    "score": score,
                    "constraints_met": constraints_met,
                    "metrics": {
                        "total_return": backtest_result.total_return,
                        "sharpe_ratio": backtest_result.sharpe_ratio,
                        "max_drawdown": backtest_result.max_drawdown,
                        "win_rate": backtest_result.win_rate
                    }
                }
                all_results.append(result)
                
                # Update best if constraints met and score is better
                if constraints_met and score > best_score:
                    best_score = score
                    best_parameters = param_set
            
            # Create next generation
            if generation < num_generations - 1:
                # Select parents based on fitness
                parents = self._select_parents(population, fitness_scores)
                
                # Create offspring through crossover and mutation
                offspring = self._create_offspring(parents, parameters, mutation_rate)
                
                # Replace population with offspring
                population = offspring
        
        # End time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create optimization result
        result = OptimizationResult(
            id=optimization_id,
            strategy_id=strategy.id,
            method=OptimizationMethod.GENETIC,
            objective=ObjectiveFunction(objective),
            start_date=start_date,
            end_date=end_date,
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            execution_time=execution_time
        )
        
        return result
    
    def _select_parents(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Select parents for genetic algorithm.
        
        Args:
            population: Population of parameter sets
            fitness_scores: Fitness scores for population
            
        Returns:
            List[Dict[str, Any]]: Selected parents
        """
        # Convert negative infinity to very negative number for selection
        adjusted_scores = [score if score != float('-inf') else -1e10 for score in fitness_scores]
        
        # Shift scores to be positive for selection
        min_score = min(adjusted_scores)
        if min_score < 0:
            adjusted_scores = [score - min_score + 1 for score in adjusted_scores]
        
        # Select parents with probability proportional to fitness
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            # If all scores are zero, select randomly
            return np.random.choice(population, size=len(population), replace=True).tolist()
        
        probabilities = [score / total_fitness for score in adjusted_scores]
        parents = np.random.choice(
            population, size=len(population), replace=True, p=probabilities
        ).tolist()
        
        return parents
    
    def _create_offspring(self, parents: List[Dict[str, Any]], parameters: Dict[str, Dict[str, Any]],
                        mutation_rate: float) -> List[Dict[str, Any]]:
        """Create offspring through crossover and mutation.
        
        Args:
            parents: Parent parameter sets
            parameters: Parameter definitions
            mutation_rate: Mutation rate
            
        Returns:
            List[Dict[str, Any]]: Offspring parameter sets
        """
        offspring = []
        
        for i in range(0, len(parents), 2):
            # Get two parents
            parent1 = parents[i]
            parent2 = parents[min(i + 1, len(parents) - 1)]
            
            # Create two children through crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Apply mutation
            child1 = self._mutate(child1, parameters, mutation_rate)
            child2 = self._mutate(child2, parameters, mutation_rate)
            
            offspring.append(child1)
            offspring.append(child2)
        
        return offspring[:len(parents)]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Two children
        """
        child1 = {}
        child2 = {}
        
        # Randomly select parameters from each parent
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutate(self, child: Dict[str, Any], parameters: Dict[str, Dict[str, Any]],
              mutation_rate: float) -> Dict[str, Any]:
        """Apply mutation to child.
        
        Args:
            child: Child parameter set
            parameters: Parameter definitions
            mutation_rate: Mutation rate
            
        Returns:
            Dict[str, Any]: Mutated child
        """
        mutated_child = child.copy()
        
        for param_name, param_value in child.items():
            # Apply mutation with probability
            if np.random.random() < mutation_rate:
                param_range = parameters[param_name]
                
                if "choices" in param_range:
                    # Discrete choices
                    choices = param_range["choices"]
                    mutated_child[param_name] = np.random.choice(choices)
                elif "min" in param_range and "max" in param_range:
                    # Numeric range
                    min_val = param_range["min"]
                    max_val = param_range["max"]
                    step = param_range.get("step", None)
                    
                    if isinstance(min_val, int) and isinstance(max_val, int) and step is not None:
                        # Integer range with step
                        values = list(range(min_val, max_val + 1, step))
                        mutated_child[param_name] = np.random.choice(values)
                    elif isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer range
                        mutated_child[param_name] = np.random.randint(min_val, max_val + 1)
                    elif step is not None:
                        # Float range with step
                        values = []
                        val = min_val
                        while val <= max_val:
                            values.append(val)
                            val += step
                        mutated_child[param_name] = np.random.choice(values)
                    else:
                        # Float range
                        mutated_child[param_name] = np.random.uniform(min_val, max_val)
        
        return mutated_child
    
    def _calculate_objective_score(self, backtest_result: Any, objective: str) -> float:
        """Calculate objective score from backtest result.
        
        Args:
            backtest_result: Backtest result
            objective: Objective function
            
        Returns:
            float: Objective score
        """
        if objective == "sharpe_ratio":
            return backtest_result.sharpe_ratio
        elif objective == "sortino_ratio":
            return backtest_result.sortino_ratio
        elif objective == "calmar_ratio":
            return backtest_result.metrics.get("calmar_ratio", 0.0)
        elif objective == "total_return":
            return backtest_result.total_return
        elif objective == "risk_adjusted_return":
            # Custom risk-adjusted return metric
            if backtest_result.max_drawdown > 0:
                return backtest_result.total_return / backtest_result.max_drawdown
            else:
                return backtest_result.total_return
        elif objective == "max_drawdown":
            # Negative because we want to minimize drawdown
            return -backtest_result.max_drawdown
        elif objective == "win_rate":
            return backtest_result.win_rate
        elif objective == "profit_factor":
            return backtest_result.profit_factor
        else:
            # Default to Sharpe ratio
            return backtest_result.sharpe_ratio
    
    def _check_constraints(self, backtest_result: Any, constraints: Dict[str, Any]) -> bool:
        """Check if backtest result meets constraints.
        
        Args:
            backtest_result: Backtest result
            constraints: Constraints
            
        Returns:
            bool: True if constraints are met
        """
        # Check max drawdown constraint
        if "max_drawdown" in constraints:
            max_drawdown_limit = constraints["max_drawdown"]
            if backtest_result.max_drawdown > max_drawdown_limit:
                return False
        
        # Check minimum number of trades
        if "min_trades" in constraints:
            min_trades = constraints["min_trades"]
            if len(backtest_result.trades) < min_trades:
                return False
        
        # Check minimum win rate
        if "min_win_rate" in constraints:
            min_win_rate = constraints["min_win_rate"]
            if backtest_result.win_rate < min_win_rate:
                return False
        
        return True

# Create optimizer instance
strategy_optimizer = StrategyOptimizer()
