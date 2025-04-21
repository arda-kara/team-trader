"""
API endpoints for the risk management module.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional

from ..models.base import (
    RiskCheckRequest, RiskCheckResponse,
    OptimizePortfolioRequest, OptimizePortfolioResponse,
    GetPortfolioRiskRequest, GetPortfolioRiskResponse,
    GetExposureAnalysisRequest, GetExposureAnalysisResponse,
    GetDrawdownEventsRequest, GetDrawdownEventsResponse,
    CreateRiskProfileRequest, CreateRiskProfileResponse
)
from ..controls.risk_control_manager import risk_control_manager
from ..optimizer.portfolio_optimizer import portfolio_optimizer
from ..exposure.exposure_manager import exposure_manager
from ..controls.drawdown_protection import drawdown_protection_manager
from ..database.models import (
    RiskProfileRepository, PortfolioRiskRepository
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["risk"])

@router.post("/check", response_model=RiskCheckResponse)
async def check_risk(request: RiskCheckRequest):
    """Check if an order complies with risk limits."""
    try:
        response = await risk_control_manager.check_order_risk(request)
        return response
    except Exception as e:
        logger.error(f"Error checking risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize", response_model=OptimizePortfolioResponse)
async def optimize_portfolio(request: OptimizePortfolioRequest):
    """Optimize portfolio allocation."""
    try:
        response = await portfolio_optimizer.optimize_portfolio(request)
        return response
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/risk", response_model=GetPortfolioRiskResponse)
async def get_portfolio_risk(request: GetPortfolioRiskRequest):
    """Get portfolio risk metrics."""
    try:
        # Get the latest portfolio risk from database
        portfolio_risk = PortfolioRiskRepository.get_latest_by_portfolio_id(request.portfolio_id)
        
        if not portfolio_risk:
            raise HTTPException(status_code=404, detail="Portfolio risk not found")
        
        # Create response
        response = GetPortfolioRiskResponse(
            risk={
                "id": portfolio_risk.id,
                "portfolio_id": portfolio_risk.portfolio_id,
                "timestamp": portfolio_risk.timestamp,
                "total_value": portfolio_risk.total_value,
                "cash": portfolio_risk.cash,
                "invested": portfolio_risk.invested,
                "leverage": portfolio_risk.leverage,
                "var_95": portfolio_risk.var_95,
                "var_99": portfolio_risk.var_99,
                "expected_shortfall": portfolio_risk.expected_shortfall,
                "volatility": portfolio_risk.volatility,
                "beta": portfolio_risk.beta,
                "sharpe_ratio": portfolio_risk.sharpe_ratio,
                "sortino_ratio": portfolio_risk.sortino_ratio,
                "max_drawdown": portfolio_risk.max_drawdown,
                "correlation_matrix": portfolio_risk.correlation_matrix if request.include_correlations else {},
                "exposures": portfolio_risk.exposures if request.include_exposures else {},
                "stress_tests": portfolio_risk.stress_tests if request.include_stress_tests else {},
                "metadata": portfolio_risk.metadata
            }
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/exposure", response_model=GetExposureAnalysisResponse)
async def get_exposure_analysis(request: GetExposureAnalysisRequest):
    """Get exposure analysis."""
    try:
        response = await exposure_manager.get_exposure_analysis(request)
        return response
    except Exception as e:
        logger.error(f"Error getting exposure analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/drawdown", response_model=GetDrawdownEventsResponse)
async def get_drawdown_events(request: GetDrawdownEventsRequest):
    """Get drawdown events."""
    try:
        response = await drawdown_protection_manager.get_drawdown_events(request)
        return response
    except Exception as e:
        logger.error(f"Error getting drawdown events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profile", response_model=CreateRiskProfileResponse)
async def create_risk_profile(request: CreateRiskProfileRequest):
    """Create a new risk profile."""
    try:
        # Check if profile with same name already exists
        existing_profile = RiskProfileRepository.get_by_name(request.name)
        if existing_profile:
            raise HTTPException(status_code=400, detail=f"Risk profile with name '{request.name}' already exists")
        
        # Create risk profile
        profile_data = request.dict()
        
        # Handle risk limits separately
        risk_limits = profile_data.pop("risk_limits", None)
        
        # Create profile
        profile = RiskProfileRepository.create(profile_data)
        
        # Create risk limits if provided
        if risk_limits:
            for limit_data in risk_limits:
                limit_data["profile_id"] = profile.id
                # Create limit using repository
                # (This would be implemented in a real system)
        
        # Create response
        response = CreateRiskProfileResponse(
            profile={
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "max_drawdown_pct": profile.max_drawdown_pct,
                "max_leverage": profile.max_leverage,
                "max_position_size_pct": profile.max_position_size_pct,
                "max_sector_exposure_pct": profile.max_sector_exposure_pct,
                "var_limit_pct": profile.var_limit_pct,
                "target_volatility": profile.target_volatility,
                "target_beta": profile.target_beta,
                "is_active": profile.is_active,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
                "metadata": profile.metadata
            }
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating risk profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles", response_model=List[Dict[str, Any]])
async def get_risk_profiles(active_only: bool = Query(True, description="Whether to return only active profiles")):
    """Get all risk profiles."""
    try:
        if active_only:
            profiles = RiskProfileRepository.get_active_profiles()
        else:
            # This would get all profiles in a real system
            profiles = RiskProfileRepository.get_active_profiles()
        
        # Convert to response format
        result = []
        for profile in profiles:
            result.append({
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "max_drawdown_pct": profile.max_drawdown_pct,
                "max_leverage": profile.max_leverage,
                "max_position_size_pct": profile.max_position_size_pct,
                "max_sector_exposure_pct": profile.max_sector_exposure_pct,
                "var_limit_pct": profile.var_limit_pct,
                "target_volatility": profile.target_volatility,
                "target_beta": profile.target_beta,
                "is_active": profile.is_active,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting risk profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/limits/check", response_model=Dict[str, bool])
async def check_exposure_limits(portfolio_id: str = Query(..., description="Portfolio ID")):
    """Check if exposures are within limits."""
    try:
        result = await exposure_manager.check_exposure_limits(portfolio_id)
        return result
    except Exception as e:
        logger.error(f"Error checking exposure limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/drawdown/monitor", response_model=Dict[str, Any])
async def monitor_drawdown(portfolio_id: str = Query(..., description="Portfolio ID")):
    """Monitor portfolio drawdown."""
    try:
        event = await drawdown_protection_manager.monitor_drawdown(portfolio_id)
        
        if event:
            return {
                "has_drawdown": True,
                "drawdown_pct": event.drawdown_pct,
                "duration_days": event.drawdown_duration_days,
                "is_active": event.is_active,
                "actions_taken": event.actions_taken
            }
        else:
            return {
                "has_drawdown": False
            }
    except Exception as e:
        logger.error(f"Error monitoring drawdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hedge", response_model=Dict[str, Any])
async def hedge_exposures(portfolio_id: str = Query(..., description="Portfolio ID")):
    """Hedge portfolio exposures."""
    try:
        actions = await exposure_manager.hedge_exposures(portfolio_id)
        return {
            "hedging_actions": actions
        }
    except Exception as e:
        logger.error(f"Error hedging exposures: {e}")
        raise HTTPException(status_code=500, detail=str(e))
