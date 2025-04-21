"""
Drawdown protection manager for limiting portfolio drawdowns.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from ..config.settings import settings
from ..models.base import (
    DrawdownEvent, DrawdownMethod, ReductionMethod,
    GetDrawdownEventsRequest, GetDrawdownEventsResponse
)
from ..database.models import (
    DrawdownEventRepository, PortfolioRiskRepository
)

logger = logging.getLogger(__name__)

class DrawdownProtectionManager:
    """Manager for drawdown protection."""
    
    def __init__(self):
        """Initialize drawdown protection manager."""
        self.drawdown_settings = settings.drawdown_protection
        self.enabled = self.drawdown_settings.enabled
        self.max_drawdown_pct = self.drawdown_settings.max_drawdown_pct
        self.calculation_method = self.drawdown_settings.drawdown_calculation_method
        self.action_threshold_pct = self.drawdown_settings.action_threshold_pct
        self.recovery_threshold_pct = self.drawdown_settings.recovery_threshold_pct
        self.reduction_method = self.drawdown_settings.reduction_method
        self.reduction_factor = self.drawdown_settings.reduction_factor
        self.stop_trading_threshold_pct = self.drawdown_settings.stop_trading_threshold_pct
        self.restart_trading_threshold_pct = self.drawdown_settings.restart_trading_threshold_pct
        self.use_time_based_recovery = self.drawdown_settings.use_time_based_recovery
        self.recovery_time_days = self.drawdown_settings.recovery_time_days
        self.log_drawdown_events = settings.logging.log_drawdown_events
    
    async def monitor_drawdown(self, portfolio_id: str) -> Optional[DrawdownEvent]:
        """Monitor portfolio drawdown.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Optional[DrawdownEvent]: Drawdown event if active, None otherwise
        """
        if not self.enabled:
            return None
        
        # Get portfolio value history
        portfolio_values = await self._get_portfolio_value_history(portfolio_id)
        
        if not portfolio_values:
            return None
        
        # Calculate current drawdown
        current_value = portfolio_values[-1]["value"]
        current_date = portfolio_values[-1]["date"]
        
        # Get active drawdown events
        active_events = DrawdownEventRepository.get_active_by_portfolio_id(portfolio_id)
        
        if active_events:
            # Update existing drawdown event
            event = active_events[0]
            peak_value = event.peak_value
            start_date = event.start_date
            
            # Calculate drawdown
            drawdown_pct = (peak_value - current_value) / peak_value
            drawdown_duration_days = (current_date - start_date).days
            
            # Check if drawdown has recovered
            if drawdown_pct <= self.recovery_threshold_pct:
                # End drawdown event
                event_data = {
                    "current_value": current_value,
                    "drawdown_pct": drawdown_pct,
                    "drawdown_duration_days": drawdown_duration_days,
                    "is_active": False,
                    "end_date": current_date
                }
                
                DrawdownEventRepository.update(event.id, event_data)
                
                if self.log_drawdown_events:
                    logger.info(f"Drawdown recovered for {portfolio_id}: {drawdown_pct:.2%} after {drawdown_duration_days} days")
                
                return None
            else:
                # Update drawdown event
                event_data = {
                    "current_value": current_value,
                    "drawdown_pct": drawdown_pct,
                    "drawdown_duration_days": drawdown_duration_days,
                    "current_date": current_date
                }
                
                DrawdownEventRepository.update(event.id, event_data)
                
                # Check if action is needed
                if drawdown_pct >= self.action_threshold_pct:
                    action = self._determine_drawdown_action(drawdown_pct, drawdown_duration_days)
                    
                    if action:
                        # Add action to event
                        actions_taken = event.actions_taken
                        actions_taken.append({
                            "date": current_date.isoformat(),
                            "action": action["type"],
                            "reduction_pct": action.get("reduction_pct"),
                            "description": action["description"]
                        })
                        
                        event_data["actions_taken"] = actions_taken
                        DrawdownEventRepository.update(event.id, event_data)
                        
                        if self.log_drawdown_events:
                            logger.warning(f"Drawdown action for {portfolio_id}: {action['description']}")
                
                # Convert to DrawdownEvent model
                return DrawdownEvent(
                    id=event.id,
                    portfolio_id=event.portfolio_id,
                    start_date=event.start_date,
                    current_date=current_date,
                    peak_value=event.peak_value,
                    current_value=current_value,
                    drawdown_pct=drawdown_pct,
                    drawdown_duration_days=drawdown_duration_days,
                    is_active=True,
                    actions_taken=event.actions_taken
                )
        else:
            # Check if we're in a drawdown
            peak_value = max(v["value"] for v in portfolio_values)
            peak_date = next(v["date"] for v in portfolio_values if v["value"] == peak_value)
            
            # Calculate drawdown
            drawdown_pct = (peak_value - current_value) / peak_value
            
            if drawdown_pct >= self.action_threshold_pct:
                # Create new drawdown event
                drawdown_duration_days = (current_date - peak_date).days
                
                event_data = {
                    "id": f"dd_{uuid.uuid4().hex[:8]}",
                    "portfolio_id": portfolio_id,
                    "start_date": peak_date,
                    "current_date": current_date,
                    "peak_value": peak_value,
                    "current_value": current_value,
                    "drawdown_pct": drawdown_pct,
                    "drawdown_duration_days": drawdown_duration_days,
                    "is_active": True,
                    "actions_taken": []
                }
                
                # Determine action
                action = self._determine_drawdown_action(drawdown_pct, drawdown_duration_days)
                
                if action:
                    event_data["actions_taken"].append({
                        "date": current_date.isoformat(),
                        "action": action["type"],
                        "reduction_pct": action.get("reduction_pct"),
                        "description": action["description"]
                    })
                
                # Create event in database
                DrawdownEventRepository.create(event_data)
                
                if self.log_drawdown_events:
                    logger.warning(f"New drawdown detected for {portfolio_id}: {drawdown_pct:.2%}")
                    if action:
                        logger.warning(f"Drawdown action for {portfolio_id}: {action['description']}")
                
                # Convert to DrawdownEvent model
                return DrawdownEvent(**event_data)
        
        return None
    
    async def get_drawdown_events(self, request: GetDrawdownEventsRequest) -> GetDrawdownEventsResponse:
        """Get drawdown events.
        
        Args:
            request: Drawdown events request
            
        Returns:
            GetDrawdownEventsResponse: Drawdown events response
        """
        # Get drawdown events from database
        db_events = DrawdownEventRepository.get_by_portfolio_id(
            request.portfolio_id,
            request.include_active_only,
            request.start_date,
            request.end_date
        )
        
        # Convert to DrawdownEvent models
        events = []
        for db_event in db_events:
            event = DrawdownEvent(
                id=db_event.id,
                portfolio_id=db_event.portfolio_id,
                start_date=db_event.start_date,
                current_date=db_event.current_date,
                end_date=db_event.end_date,
                peak_value=db_event.peak_value,
                current_value=db_event.current_value,
                drawdown_pct=db_event.drawdown_pct,
                drawdown_duration_days=db_event.drawdown_duration_days,
                is_active=db_event.is_active,
                actions_taken=db_event.actions_taken,
                metadata=db_event.metadata
            )
            events.append(event)
        
        # Create response
        response = GetDrawdownEventsResponse(
            events=events
        )
        
        return response
    
    def _determine_drawdown_action(self, drawdown_pct: float, duration_days: int) -> Optional[Dict[str, Any]]:
        """Determine action to take based on drawdown.
        
        Args:
            drawdown_pct: Drawdown percentage
            duration_days: Drawdown duration in days
            
        Returns:
            Optional[Dict[str, Any]]: Action to take, or None if no action
        """
        # Check if we should stop trading
        if drawdown_pct >= self.stop_trading_threshold_pct:
            return {
                "type": "stop",
                "description": f"Stopped trading due to {drawdown_pct:.2%} drawdown"
            }
        
        # Check if we should reduce exposure
        if drawdown_pct >= self.action_threshold_pct:
            # Calculate reduction percentage based on method
            if self.reduction_method == ReductionMethod.PROPORTIONAL:
                reduction_pct = min(0.9, drawdown_pct / self.max_drawdown_pct * self.reduction_factor)
            elif self.reduction_method == ReductionMethod.VOLATILITY_BASED:
                # In a real system, you would calculate based on volatility
                reduction_pct = min(0.9, drawdown_pct * self.reduction_factor)
            else:  # ReductionMethod.EQUAL
                reduction_pct = self.reduction_factor
            
            return {
                "type": "reduce",
                "reduction_pct": reduction_pct,
                "description": f"Reduced exposure by {reduction_pct:.2%} due to {drawdown_pct:.2%} drawdown"
            }
        
        return None
    
    async def _get_portfolio_value_history(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get portfolio value history.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List[Dict[str, Any]]: Portfolio value history
        """
        # Get portfolio risks from database
        portfolio_risks = PortfolioRiskRepository.get_by_portfolio_id(
            portfolio_id,
            start_date=datetime.utcnow() - timedelta(days=90),
            limit=90
        )
        
        # Convert to value history
        values = []
        for risk in portfolio_risks:
            values.append({
                "date": risk.timestamp,
                "value": risk.total_value
            })
        
        # Sort by date
        values.sort(key=lambda x: x["date"])
        
        return values

# Create drawdown protection manager instance
drawdown_protection_manager = DrawdownProtectionManager()
