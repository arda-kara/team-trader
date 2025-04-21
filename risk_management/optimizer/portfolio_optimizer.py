"""
Portfolio optimizer for optimal capital allocation.
"""

import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from scipy.optimize import minimize

from ..config.settings import settings
from ..models.base import (
    PortfolioAllocation, OptimizationMethod,
    OptimizePortfolioRequest, OptimizePortfolioResponse
)
from ..database.models import (
    PortfolioAllocationRepository, PortfolioRiskRepository
)

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Manager for portfolio optimization."""
    
    def __init__(self):
        """Initialize portfolio optimizer."""
        self.optimization_settings = settings.portfolio_optimization
        self.risk_free_rate = self.optimization_settings.risk_free_rate
        self.min_weight = self.optimization_settings.min_weight
        self.max_weight = self.optimization_settings.max_weight
        self.risk_aversion = self.optimization_settings.risk_aversion
        self.regularization_factor = self.optimization_settings.regularization_factor
    
    async def optimize_portfolio(self, request: OptimizePortfolioRequest) -> OptimizePortfolioResponse:
        """Optimize portfolio allocation.
        
        Args:
            request: Portfolio optimization request
            
        Returns:
            OptimizePortfolioResponse: Portfolio optimization response
        """
        # Get historical returns for assets
        returns, cov_matrix = await self._get_historical_data(request.assets)
        
        # Perform optimization based on method
        if request.optimization_method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._optimize_mean_variance(returns, cov_matrix, request.target_return, request.constraints)
        elif request.optimization_method == OptimizationMethod.RISK_PARITY:
            weights = self._optimize_risk_parity(cov_matrix)
        elif request.optimization_method == OptimizationMethod.MIN_VARIANCE:
            weights = self._optimize_min_variance(cov_matrix)
        elif request.optimization_method == OptimizationMethod.MAX_SHARPE:
            weights = self._optimize_max_sharpe(returns, cov_matrix)
        elif request.optimization_method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._optimize_equal_weight(len(returns))
        else:
            # Default to equal weight
            weights = self._optimize_equal_weight(len(returns))
        
        # Calculate expected return and risk
        expected_return = np.sum(returns.mean() * weights) * 252  # Annualized
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0
        
        # Create allocation dictionary
        allocations = {}
        for i, asset in enumerate(request.assets):
            allocations[asset] = float(weights[i])
        
        # Create portfolio allocation
        allocation = PortfolioAllocation(
            id=f"alloc_{uuid.uuid4().hex[:8]}",
            portfolio_id=request.portfolio_id,
            timestamp=datetime.utcnow(),
            allocations=allocations,
            expected_return=float(expected_return),
            expected_risk=float(expected_risk),
            sharpe_ratio=float(sharpe_ratio),
            optimization_method=request.optimization_method,
            constraints=request.constraints
        )
        
        # Save allocation to database
        PortfolioAllocationRepository.create(allocation.dict())
        
        # Create response
        response = OptimizePortfolioResponse(
            allocation=allocation
        )
        
        logger.info(f"Portfolio optimized for {request.portfolio_id} using {request.optimization_method.value}")
        
        return response
    
    async def _get_historical_data(self, assets: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
        """Get historical return data for assets.
        
        Args:
            assets: List of asset symbols
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: Returns and covariance matrix
        """
        # In a real system, you would fetch historical price data from a database or API
        # and calculate returns and covariance matrix
        
        # For simplicity, generate random returns and covariance matrix
        num_assets = len(assets)
        
        # Generate random returns (mean between 5% and 15% annualized)
        returns = pd.Series(np.random.uniform(0.05/252, 0.15/252, num_assets))
        
        # Generate random correlation matrix
        corr = np.random.uniform(-0.2, 0.8, (num_assets, num_assets))
        corr = (corr + corr.T) / 2  # Make symmetric
        np.fill_diagonal(corr, 1)  # Set diagonal to 1
        
        # Generate random volatilities (between 15% and 35% annualized)
        vols = np.random.uniform(0.15/np.sqrt(252), 0.35/np.sqrt(252), num_assets)
        
        # Calculate covariance matrix
        cov_matrix = np.outer(vols, vols) * corr
        
        return returns, cov_matrix
    
    def _optimize_mean_variance(self, returns: pd.Series, cov_matrix: pd.DataFrame, 
                              target_return: Optional[float] = None, 
                              constraints: Dict[str, Any] = {}) -> np.ndarray:
        """Optimize portfolio using mean-variance optimization.
        
        Args:
            returns: Asset returns
            cov_matrix: Covariance matrix
            target_return: Target return
            constraints: Optimization constraints
            
        Returns:
            np.ndarray: Optimal weights
        """
        num_assets = len(returns)
        
        # Define objective function (minimize variance)
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return portfolio_variance
        
        # Initial guess (equal weight)
        initial_weights = np.ones(num_assets) / num_assets
        
        # Constraints
        constraints_list = []
        
        # Weights sum to 1
        constraints_list.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Target return constraint
        if target_return is not None:
            constraints_list.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(returns * x) * 252 - target_return
            })
        
        # Long-only constraint
        if constraints.get('long_only', True):
            bounds = [(self.min_weight, self.max_weight) for _ in range(num_assets)]
        else:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(num_assets)]
        
        # Solve optimization problem
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Check if optimization was successful
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return initial_weights
        
        return result.x
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio using risk parity.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            np.ndarray: Optimal weights
        """
        num_assets = cov_matrix.shape[0]
        
        # Define objective function (minimize risk concentration)
        def objective(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)  # Normalize
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            asset_contrib = weights * np.dot(cov_matrix, weights) / portfolio_risk
            return np.sum((asset_contrib - portfolio_risk / num_assets) ** 2)
        
        # Initial guess (equal weight)
        initial_weights = np.ones(num_assets) / num_assets
        
        # Bounds
        bounds = [(0.001, 1.0) for _ in range(num_assets)]
        
        # Solve optimization problem
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds
        )
        
        # Normalize weights
        weights = np.abs(result.x)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _optimize_min_variance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio to minimize variance.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            np.ndarray: Optimal weights
        """
        num_assets = cov_matrix.shape[0]
        
        # Define objective function (minimize variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Initial guess (equal weight)
        initial_weights = np.ones(num_assets) / num_assets
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(num_assets)]
        
        # Solve optimization problem
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def _optimize_max_sharpe(self, returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio to maximize Sharpe ratio.
        
        Args:
            returns: Asset returns
            cov_matrix: Covariance matrix
            
        Returns:
            np.ndarray: Optimal weights
        """
        num_assets = len(returns)
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(returns * weights) * 252
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return -(portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Initial guess (equal weight)
        initial_weights = np.ones(num_assets) / num_assets
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(num_assets)]
        
        # Solve optimization problem
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def _optimize_equal_weight(self, num_assets: int) -> np.ndarray:
        """Create equal weight portfolio.
        
        Args:
            num_assets: Number of assets
            
        Returns:
            np.ndarray: Equal weights
        """
        return np.ones(num_assets) / num_assets

# Create portfolio optimizer instance
portfolio_optimizer = PortfolioOptimizer()
