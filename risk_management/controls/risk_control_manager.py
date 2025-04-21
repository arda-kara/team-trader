"""
Risk control manager for enforcing risk limits and policies.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from ..config.settings import settings
from ..models.base import (
    RiskLimit, RiskCheck, RiskLevel, RiskType, RiskAction,
    RiskCheckRequest, RiskCheckResponse
)
from ..database.models import (
    RiskLimitRepository, RiskCheckRepository, PortfolioRiskRepository
)

logger = logging.getLogger(__name__)

class RiskControlManager:
    """Manager for risk controls and limit enforcement."""
    
    def __init__(self):
        """Initialize risk control manager."""
        self.risk_limits = settings.risk_limits
        self.log_risk_checks = settings.logging.log_risk_checks
    
    async def check_order_risk(self, request: RiskCheckRequest) -> RiskCheckResponse:
        """Check if an order complies with risk limits.
        
        Args:
            request: Risk check request
            
        Returns:
            RiskCheckResponse: Risk check response
        """
        # Get relevant risk limits
        limits = self._get_relevant_limits(request)
        
        # Perform risk checks
        checks = []
        is_approved = True
        
        for limit in limits:
            check = await self._perform_risk_check(limit, request)
            checks.append(check)
            
            # If any check has a BLOCK action, the order is not approved
            if check.action == RiskAction.BLOCK:
                is_approved = False
        
        # Create response
        message = "Order approved" if is_approved else "Order rejected due to risk limits"
        response = RiskCheckResponse(
            is_approved=is_approved,
            checks=checks,
            message=message
        )
        
        # Log risk check result
        if self.log_risk_checks:
            if is_approved:
                logger.info(f"Risk check passed for order: {request.order_id}")
            else:
                logger.warning(f"Risk check failed for order: {request.order_id}")
        
        return response
    
    def _get_relevant_limits(self, request: RiskCheckRequest) -> List[RiskLimit]:
        """Get relevant risk limits for the request.
        
        Args:
            request: Risk check request
            
        Returns:
            List[RiskLimit]: List of relevant risk limits
        """
        limits = []
        
        # Get active limits from database
        db_limits = RiskLimitRepository.get_active_limits()
        
        # Convert to RiskLimit objects
        for db_limit in db_limits:
            limit = RiskLimit(
                id=db_limit.id,
                name=db_limit.name,
                description=db_limit.description,
                risk_type=db_limit.risk_type,
                threshold=db_limit.threshold,
                warning_threshold=db_limit.warning_threshold,
                action=db_limit.action,
                scope=db_limit.scope,
                scope_id=db_limit.scope_id,
                is_active=db_limit.is_active,
                created_at=db_limit.created_at,
                updated_at=db_limit.updated_at,
                metadata=db_limit.metadata
            )
            
            # Filter limits based on request
            if self._is_limit_applicable(limit, request):
                limits.append(limit)
        
        return limits
    
    def _is_limit_applicable(self, limit: RiskLimit, request: RiskCheckRequest) -> bool:
        """Check if a limit is applicable to the request.
        
        Args:
            limit: Risk limit
            request: Risk check request
            
        Returns:
            bool: Whether the limit is applicable
        """
        # Check if limit scope matches request
        if limit.scope == "portfolio" and request.portfolio_id:
            if limit.scope_id is None or limit.scope_id == request.portfolio_id:
                return True
        
        elif limit.scope == "strategy" and request.strategy_id:
            if limit.scope_id is None or limit.scope_id == request.strategy_id:
                return True
        
        elif limit.scope == "asset" and request.symbol:
            if limit.scope_id is None or limit.scope_id == request.symbol:
                return True
        
        # Check if check_types is specified and limit's risk_type is included
        if request.check_types and limit.risk_type.value not in request.check_types:
            return False
        
        return False
    
    async def _perform_risk_check(self, limit: RiskLimit, request: RiskCheckRequest) -> RiskCheck:
        """Perform a risk check for a specific limit.
        
        Args:
            limit: Risk limit
            request: Risk check request
            
        Returns:
            RiskCheck: Risk check result
        """
        # Calculate current value based on risk type
        value = await self._calculate_risk_value(limit.risk_type, request)
        
        # Check if limit is breached
        is_breached = value > limit.threshold
        
        # Determine risk level
        risk_level = self._determine_risk_level(value, limit)
        
        # Determine action
        action = limit.action if is_breached else RiskAction.ALLOW
        
        # Create context
        context = {
            "order_id": request.order_id,
            "strategy_id": request.strategy_id,
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "price": request.price,
            "portfolio_id": request.portfolio_id,
            "limit_name": limit.name,
            "limit_description": limit.description,
            "value": value,
            "threshold": limit.threshold
        }
        
        # Create risk check
        check = RiskCheck(
            id=f"check_{uuid.uuid4().hex[:8]}",
            limit_id=limit.id,
            value=value,
            threshold=limit.threshold,
            is_breached=is_breached,
            risk_level=risk_level,
            action=action,
            timestamp=datetime.utcnow(),
            context=context
        )
        
        # Save risk check to database
        RiskCheckRepository.create(check.dict())
        
        return check
    
    async def _calculate_risk_value(self, risk_type: RiskType, request: RiskCheckRequest) -> float:
        """Calculate risk value based on risk type.
        
        Args:
            risk_type: Risk type
            request: Risk check request
            
        Returns:
            float: Calculated risk value
        """
        # This is a simplified implementation
        # In a real system, you would have more sophisticated risk calculations
        
        if risk_type == RiskType.CONCENTRATION:
            # Calculate position size as percentage of portfolio
            return await self._calculate_position_size_pct(request)
        
        elif risk_type == RiskType.EXPOSURE:
            # Calculate sector or asset class exposure
            return await self._calculate_exposure(request)
        
        elif risk_type == RiskType.VOLATILITY:
            # Calculate volatility-based risk
            return await self._calculate_volatility_risk(request)
        
        elif risk_type == RiskType.DRAWDOWN:
            # Calculate drawdown risk
            return await self._calculate_drawdown_risk(request)
        
        elif risk_type == RiskType.LEVERAGE:
            # Calculate leverage risk
            return await self._calculate_leverage(request)
        
        elif risk_type == RiskType.LIQUIDITY:
            # Calculate liquidity risk
            return await self._calculate_liquidity_risk(request)
        
        elif risk_type == RiskType.MARKET:
            # Calculate market risk (VaR)
            return await self._calculate_var(request)
        
        # Default to a safe value
        return 0.0
    
    async def _calculate_position_size_pct(self, request: RiskCheckRequest) -> float:
        """Calculate position size as percentage of portfolio.
        
        Args:
            request: Risk check request
            
        Returns:
            float: Position size percentage
        """
        # Get portfolio value
        portfolio_value = await self._get_portfolio_value(request.portfolio_id)
        
        if portfolio_value <= 0:
            return 1.0  # Maximum risk if portfolio value is invalid
        
        # Calculate position value
        position_value = request.quantity * request.price if request.quantity and request.price else 0
        
        # Calculate percentage
        return position_value / portfolio_value
    
    async def _calculate_exposure(self, request: RiskCheckRequest) -> float:
        """Calculate sector or asset class exposure.
        
        Args:
            request: Risk check request
            
        Returns:
            float: Exposure value
        """
        # In a real system, you would look up the sector/asset class of the symbol
        # and calculate the total exposure including the new position
        
        # For simplicity, return a random value between 0 and 0.5
        import random
        return random.uniform(0, 0.5)
    
    async def _calculate_volatility_risk(self, request: RiskCheckRequest) -> float:
        """Calculate volatility-based risk.
        
        Args:
            request: Risk check request
            
        Returns:
            float: Volatility risk value
        """
        # In a real system, you would calculate historical volatility
        # and determine the risk based on position size and volatility
        
        # For simplicity, return a random value between 0 and 0.3
        import random
        return random.uniform(0, 0.3)
    
    async def _calculate_drawdown_risk(self, request: RiskCheckRequest) -> float:
        """Calculate drawdown risk.
        
        Args:
            request: Risk check request
            
        Returns:
            float: Drawdown risk value
        """
        # In a real system, you would calculate the current drawdown
        # and estimate the additional drawdown from the new position
        
        # Get the latest portfolio risk
        if request.portfolio_id:
            portfolio_risk = PortfolioRiskRepository.get_latest_by_portfolio_id(request.portfolio_id)
            if portfolio_risk:
                return portfolio_risk.max_drawdown
        
        # Default to a moderate value
        return 0.1
    
    async def _calculate_leverage(self, request: RiskCheckRequest) -> float:
        """Calculate leverage risk.
        
        Args:
            request: Risk check request
            
        Returns:
            float: Leverage value
        """
        # In a real system, you would calculate the current leverage
        # and estimate the new leverage after the position is added
        
        # Get the latest portfolio risk
        if request.portfolio_id:
            portfolio_risk = PortfolioRiskRepository.get_latest_by_portfolio_id(request.portfolio_id)
            if portfolio_risk:
                return portfolio_risk.leverage
        
        # Default to a moderate value
        return 1.0
    
    async def _calculate_liquidity_risk(self, request: RiskCheckRequest) -> float:
        """Calculate liquidity risk.
        
        Args:
            request: Risk check request
            
        Returns:
            float: Liquidity risk value
        """
        # In a real system, you would calculate the liquidity of the asset
        # and determine the risk based on position size and liquidity
        
        # For simplicity, return a random value between 0 and 0.2
        import random
        return random.uniform(0, 0.2)
    
    async def _calculate_var(self, request: RiskCheckRequest) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            request: Risk check request
            
        Returns:
            float: VaR as percentage of portfolio
        """
        # In a real system, you would calculate VaR using historical simulation,
        # parametric method, or Monte Carlo simulation
        
        # Get the latest portfolio risk
        if request.portfolio_id:
            portfolio_risk = PortfolioRiskRepository.get_latest_by_portfolio_id(request.portfolio_id)
            if portfolio_risk:
                return portfolio_risk.var_95 / portfolio_risk.total_value
        
        # Default to a moderate value
        return 0.02
    
    async def _get_portfolio_value(self, portfolio_id: Optional[str]) -> float:
        """Get portfolio value.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            float: Portfolio value
        """
        if not portfolio_id:
            return 100000.0  # Default value
        
        # Get the latest portfolio risk
        portfolio_risk = PortfolioRiskRepository.get_latest_by_portfolio_id(portfolio_id)
        if portfolio_risk:
            return portfolio_risk.total_value
        
        return 100000.0  # Default value
    
    def _determine_risk_level(self, value: float, limit: RiskLimit) -> RiskLevel:
        """Determine risk level based on value and limit.
        
        Args:
            value: Risk value
            limit: Risk limit
            
        Returns:
            RiskLevel: Risk level
        """
        if value > limit.threshold:
            return RiskLevel.HIGH
        
        if limit.warning_threshold and value > limit.warning_threshold:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW

# Create risk control manager instance
risk_control_manager = RiskControlManager()
