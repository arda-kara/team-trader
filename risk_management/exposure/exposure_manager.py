"""
Exposure manager for monitoring and managing portfolio exposures.
"""

import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from ..config.settings import settings
from ..models.base import (
    ExposureAnalysis, SectorClassification, FactorModel,
    GetExposureAnalysisRequest, GetExposureAnalysisResponse
)
from ..database.models import (
    ExposureAnalysisRepository, PortfolioRiskRepository
)

logger = logging.getLogger(__name__)

class ExposureManager:
    """Manager for monitoring and managing portfolio exposures."""
    
    def __init__(self):
        """Initialize exposure manager."""
        self.exposure_settings = settings.exposure_management
        self.sector_classification = self.exposure_settings.sector_classification
        self.factor_model = self.exposure_settings.factor_model
        self.max_net_exposure = self.exposure_settings.max_net_exposure
        self.max_gross_exposure = self.exposure_settings.max_gross_exposure
        self.target_beta = self.exposure_settings.target_beta
        self.beta_tolerance = self.exposure_settings.beta_tolerance
        self.auto_hedge = self.exposure_settings.auto_hedge
        self.exposure_limits = self.exposure_settings.exposure_limits
        self.log_exposure_changes = settings.logging.log_exposure_changes
    
    async def analyze_exposures(self, portfolio_id: str) -> Dict[str, ExposureAnalysis]:
        """Analyze all exposures for a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Dict[str, ExposureAnalysis]: Dictionary of exposure analyses by type
        """
        exposure_types = []
        
        if self.exposure_settings.monitor_sectors:
            exposure_types.append("sector")
        
        if self.exposure_settings.monitor_asset_classes:
            exposure_types.append("asset_class")
        
        if self.exposure_settings.monitor_factors:
            exposure_types.append("factor")
        
        if self.exposure_settings.monitor_geographies:
            exposure_types.append("geography")
        
        if self.exposure_settings.monitor_currencies:
            exposure_types.append("currency")
        
        # Analyze each exposure type
        results = {}
        for exposure_type in exposure_types:
            analysis = await self.analyze_exposure(portfolio_id, exposure_type)
            results[exposure_type] = analysis
        
        return results
    
    async def analyze_exposure(self, portfolio_id: str, exposure_type: str) -> ExposureAnalysis:
        """Analyze a specific exposure type for a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            exposure_type: Exposure type
            
        Returns:
            ExposureAnalysis: Exposure analysis
        """
        # Get portfolio positions
        positions = await self._get_portfolio_positions(portfolio_id)
        
        # Calculate exposures based on type
        if exposure_type == "sector":
            exposures, benchmark_exposures = await self._calculate_sector_exposures(positions)
        elif exposure_type == "asset_class":
            exposures, benchmark_exposures = await self._calculate_asset_class_exposures(positions)
        elif exposure_type == "factor":
            exposures, benchmark_exposures = await self._calculate_factor_exposures(positions)
        elif exposure_type == "geography":
            exposures, benchmark_exposures = await self._calculate_geography_exposures(positions)
        elif exposure_type == "currency":
            exposures, benchmark_exposures = await self._calculate_currency_exposures(positions)
        else:
            exposures = {}
            benchmark_exposures = {}
        
        # Calculate active exposures
        active_exposures = {}
        for key in exposures:
            if key in benchmark_exposures:
                active_exposures[key] = exposures[key] - benchmark_exposures[key]
            else:
                active_exposures[key] = exposures[key]
        
        # Calculate aggregate exposures
        long_exposure = sum(v for v in exposures.values() if v > 0)
        short_exposure = sum(abs(v) for v in exposures.values() if v < 0)
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Create exposure analysis
        analysis = ExposureAnalysis(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            portfolio_id=portfolio_id,
            timestamp=datetime.utcnow(),
            exposure_type=exposure_type,
            exposures=exposures,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            benchmark_exposures=benchmark_exposures,
            active_exposures=active_exposures
        )
        
        # Save analysis to database
        ExposureAnalysisRepository.create(analysis.dict())
        
        # Log exposure analysis
        if self.log_exposure_changes:
            logger.info(f"Exposure analysis for {portfolio_id} - {exposure_type}: net={net_exposure:.2f}, gross={gross_exposure:.2f}")
        
        return analysis
    
    async def get_exposure_analysis(self, request: GetExposureAnalysisRequest) -> GetExposureAnalysisResponse:
        """Get exposure analysis.
        
        Args:
            request: Exposure analysis request
            
        Returns:
            GetExposureAnalysisResponse: Exposure analysis response
        """
        # Get latest exposure analysis from database
        db_analysis = ExposureAnalysisRepository.get_latest_by_portfolio_and_type(
            request.portfolio_id, request.exposure_type
        )
        
        if db_analysis:
            # Convert to ExposureAnalysis object
            analysis = ExposureAnalysis(
                id=db_analysis.id,
                portfolio_id=db_analysis.portfolio_id,
                timestamp=db_analysis.timestamp,
                exposure_type=db_analysis.exposure_type,
                exposures=db_analysis.exposures,
                net_exposure=db_analysis.net_exposure,
                gross_exposure=db_analysis.gross_exposure,
                long_exposure=db_analysis.long_exposure,
                short_exposure=db_analysis.short_exposure,
                benchmark_exposures=db_analysis.benchmark_exposures if request.include_benchmark else None,
                active_exposures=db_analysis.active_exposures if request.include_benchmark else None,
                metadata=db_analysis.metadata
            )
        else:
            # Create new analysis
            analysis = await self.analyze_exposure(request.portfolio_id, request.exposure_type)
        
        # Create response
        response = GetExposureAnalysisResponse(
            exposure=analysis
        )
        
        return response
    
    async def check_exposure_limits(self, portfolio_id: str) -> Dict[str, bool]:
        """Check if exposures are within limits.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Dict[str, bool]: Dictionary of exposure types and whether they are within limits
        """
        # Analyze all exposures
        analyses = await self.analyze_exposures(portfolio_id)
        
        # Check limits for each exposure type
        results = {}
        
        for exposure_type, analysis in analyses.items():
            # Check net exposure
            if analysis.net_exposure > self.max_net_exposure:
                results[f"{exposure_type}_net"] = False
            else:
                results[f"{exposure_type}_net"] = True
            
            # Check gross exposure
            if analysis.gross_exposure > self.max_gross_exposure:
                results[f"{exposure_type}_gross"] = False
            else:
                results[f"{exposure_type}_gross"] = True
            
            # Check individual exposures
            if exposure_type in self.exposure_limits:
                for name, limit in self.exposure_limits[exposure_type].items():
                    if name in analysis.exposures and abs(analysis.exposures[name]) > limit:
                        results[f"{exposure_type}_{name}"] = False
                    else:
                        results[f"{exposure_type}_{name}"] = True
        
        return results
    
    async def hedge_exposures(self, portfolio_id: str) -> Dict[str, Any]:
        """Hedge portfolio exposures.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Dict[str, Any]: Hedging actions
        """
        # This is a simplified implementation
        # In a real system, you would calculate optimal hedges and generate orders
        
        # Analyze all exposures
        analyses = await self.analyze_exposures(portfolio_id)
        
        # Determine hedging actions
        actions = {}
        
        for exposure_type, analysis in analyses.items():
            # Check if hedging is needed
            if abs(analysis.net_exposure) > self.exposure_settings.hedge_threshold_pct:
                actions[exposure_type] = {
                    "hedge_amount": -analysis.net_exposure,
                    "hedge_instruments": self._get_hedge_instruments(exposure_type)
                }
        
        # Log hedging actions
        if self.log_exposure_changes and actions:
            logger.info(f"Hedging actions for {portfolio_id}: {actions}")
        
        return actions
    
    async def _get_portfolio_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get portfolio positions.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        # In a real system, you would fetch positions from a database or API
        
        # For simplicity, generate random positions
        import random
        
        # Generate 10-20 random positions
        num_positions = random.randint(10, 20)
        
        # Define possible sectors, asset classes, etc.
        sectors = ["technology", "financials", "healthcare", "consumer_discretionary", 
                  "consumer_staples", "industrials", "energy", "materials", 
                  "utilities", "real_estate", "communication_services"]
        
        asset_classes = ["equity", "fixed_income", "commodities", "currencies", "crypto"]
        
        factors = ["momentum", "value", "size", "quality", "volatility"]
        
        geographies = ["north_america", "europe", "asia_pacific", "emerging_markets"]
        
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY"]
        
        # Generate positions
        positions = []
        for i in range(num_positions):
            position = {
                "symbol": f"STOCK{i+1}",
                "quantity": random.uniform(100, 1000),
                "market_value": random.uniform(10000, 100000),
                "weight": random.uniform(0.01, 0.1),
                "sector": random.choice(sectors),
                "asset_class": random.choice(asset_classes),
                "geography": random.choice(geographies),
                "currency": random.choice(currencies),
                "factors": {factor: random.uniform(-1, 1) for factor in factors}
            }
            positions.append(position)
        
        # Normalize weights to sum to 1
        total_weight = sum(p["weight"] for p in positions)
        for position in positions:
            position["weight"] = position["weight"] / total_weight
        
        return positions
    
    async def _calculate_sector_exposures(self, positions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate sector exposures.
        
        Args:
            positions: List of positions
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Sector exposures and benchmark exposures
        """
        # Calculate sector exposures
        exposures = {}
        for position in positions:
            sector = position["sector"]
            weight = position["weight"]
            
            if sector in exposures:
                exposures[sector] += weight
            else:
                exposures[sector] = weight
        
        # Generate benchmark exposures
        # In a real system, you would fetch benchmark exposures from a database or API
        benchmark_exposures = {
            "technology": 0.20,
            "financials": 0.15,
            "healthcare": 0.12,
            "consumer_discretionary": 0.10,
            "consumer_staples": 0.08,
            "industrials": 0.10,
            "energy": 0.05,
            "materials": 0.05,
            "utilities": 0.03,
            "real_estate": 0.04,
            "communication_services": 0.08
        }
        
        return exposures, benchmark_exposures
    
    async def _calculate_asset_class_exposures(self, positions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate asset class exposures.
        
        Args:
            positions: List of positions
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Asset class exposures and benchmark exposures
        """
        # Calculate asset class exposures
        exposures = {}
        for position in positions:
            asset_class = position["asset_class"]
            weight = position["weight"]
            
            if asset_class in exposures:
                exposures[asset_class] += weight
            else:
                exposures[asset_class] = weight
        
        # Generate benchmark exposures
        benchmark_exposures = {
            "equity": 0.60,
            "fixed_income": 0.30,
            "commodities": 0.05,
            "currencies": 0.03,
            "crypto": 0.02
        }
        
        return exposures, benchmark_exposures
    
    async def _calculate_factor_exposures(self, positions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate factor exposures.
        
        Args:
            positions: List of positions
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Factor exposures and benchmark exposures
        """
        # Calculate factor exposures
        exposures = {}
        for position in positions:
            weight = position["weight"]
            for factor, factor_exposure in position["factors"].items():
                factor_contribution = weight * factor_exposure
                
                if factor in exposures:
                    exposures[factor] += factor_contribution
                else:
                    exposures[factor] = factor_contribution
        
        # Generate benchmark exposures
        benchmark_exposures = {
            "momentum": 0.05,
            "value": 0.02,
            "size": -0.01,
            "quality": 0.03,
            "volatility": -0.02
        }
        
        return exposures, benchmark_exposures
    
    async def _calculate_geography_exposures(self, positions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate geography exposures.
        
        Args:
            positions: List of positions
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Geography exposures and benchmark exposures
        """
        # Calculate geography exposures
        exposures = {}
        for position in positions:
            geography = position["geography"]
            weight = position["weight"]
            
            if geography in exposures:
                exposures[geography] += weight
            else:
                exposures[geography] = weight
        
        # Generate benchmark exposures
        benchmark_exposures = {
            "north_america": 0.55,
            "europe": 0.20,
            "asia_pacific": 0.15,
            "emerging_markets": 0.10
        }
        
        return exposures, benchmark_exposures
    
    async def _calculate_currency_exposures(self, positions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate currency exposures.
        
        Args:
            positions: List of positions
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Currency exposures and benchmark exposures
        """
        # Calculate currency exposures
        exposures = {}
        for position in positions:
            currency = position["currency"]
            weight = position["weight"]
            
            if currency in exposures:
                exposures[currency] += weight
            else:
                exposures[currency] = weight
        
        # Generate benchmark exposures
        benchmark_exposures = {
            "USD": 0.60,
            "EUR": 0.15,
            "GBP": 0.05,
            "JPY": 0.10,
            "CHF": 0.02,
            "CAD": 0.03,
            "AUD": 0.03,
            "CNY": 0.02
        }
        
        return exposures, benchmark_exposures
    
    def _get_hedge_instruments(self, exposure_type: str) -> List[str]:
        """Get hedge instruments for an exposure type.
        
        Args:
            exposure_type: Exposure type
            
        Returns:
            List[str]: List of hedge instruments
        """
        # In a real system, you would have a database of hedge instruments
        
        if exposure_type == "sector":
            return ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLB", "XLU", "XLRE", "XLC"]
        elif exposure_type == "asset_class":
            return ["SPY", "AGG", "GSG", "FXE", "GBTC"]
        elif exposure_type == "factor":
            return ["MTUM", "VLUE", "SIZE", "QUAL", "USMV"]
        elif exposure_type == "geography":
            return ["SPY", "VGK", "VPL", "VWO"]
        elif exposure_type == "currency":
            return ["UUP", "FXE", "FXB", "FXY", "FXF", "FXC", "FXA", "CYB"]
        else:
            return []

# Create exposure manager instance
exposure_manager = ExposureManager()
