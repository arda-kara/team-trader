"""
Agent base class and factory for the agentic oversight system.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from ..config.settings import settings
from ..core.models import (
    Agent, AgentType, AgentStatus, Task, TaskStatus,
    CreateAgentRequest, CreateAgentResponse
)
from ..memory.database import AgentModel, AgentRepository
from ..reasoning.engine import reasoning_engine

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class."""
    
    def __init__(self, agent_data: Agent):
        """Initialize base agent.
        
        Args:
            agent_data: Agent data
        """
        self.agent = agent_data
        self.id = agent_data.id
        self.name = agent_data.name
        self.type = agent_data.type
        self.status = agent_data.status
        self.capabilities = agent_data.capabilities
        self.config = agent_data.config
        self.created_at = agent_data.created_at
        self.last_active = agent_data.last_active
        self.metadata = agent_data.metadata
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task.
        
        Args:
            task: Task to process
            
        Returns:
            Dict[str, Any]: Task result
        """
        logger.info(f"Agent {self.id} processing task {task.id}")
        
        # Update agent status
        self.status = AgentStatus.BUSY
        self._update_agent_status()
        
        try:
            # Perform reasoning
            reasoning = await reasoning_engine.reason(self.agent, task)
            
            # Process task based on agent type
            result = await self._process_task_impl(task, reasoning)
            
            # Update task status
            task_update = {
                "status": TaskStatus.COMPLETED,
                "completed_at": datetime.utcnow(),
                "output_data": result
            }
            
            # Update agent status
            self.status = AgentStatus.IDLE
            self._update_agent_status()
            
            return task_update
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            
            # Update task status
            task_update = {
                "status": TaskStatus.FAILED,
                "output_data": {
                    "error": str(e)
                }
            }
            
            # Update agent status
            self.status = AgentStatus.ERROR
            self._update_agent_status()
            
            return task_update
    
    async def _process_task_impl(self, task: Task, reasoning: Any) -> Dict[str, Any]:
        """Process task implementation.
        
        Args:
            task: Task to process
            reasoning: Reasoning result
            
        Returns:
            Dict[str, Any]: Task result
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _process_task_impl")
    
    def _update_agent_status(self):
        """Update agent status in database."""
        try:
            AgentRepository.update(self.id, {
                "status": self.status,
                "last_active": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")

class MonitoringAgent(BaseAgent):
    """Monitoring agent class."""
    
    async def _process_task_impl(self, task: Task, reasoning: Any) -> Dict[str, Any]:
        """Process task implementation.
        
        Args:
            task: Task to process
            reasoning: Reasoning result
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Extract monitoring data from task
        monitoring_data = task.input_data.get("monitoring_data", {})
        
        # Analyze monitoring data
        analysis_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_id": reasoning.id,
            "conclusion": reasoning.conclusion,
            "confidence": reasoning.confidence,
            "anomalies_detected": False,
            "metrics": {},
            "alerts": []
        }
        
        # Check for anomalies based on reasoning conclusion
        if "anomaly" in reasoning.conclusion.lower() or "issue" in reasoning.conclusion.lower():
            analysis_result["anomalies_detected"] = True
            
            # Extract alert from reasoning
            alert = {
                "level": "warning",
                "title": task.title,
                "message": reasoning.conclusion,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add alert to result
            analysis_result["alerts"].append(alert)
        
        # Extract metrics from monitoring data
        if "metrics" in monitoring_data:
            analysis_result["metrics"] = monitoring_data["metrics"]
        
        return analysis_result

class DecisionAgent(BaseAgent):
    """Decision agent class."""
    
    async def _process_task_impl(self, task: Task, reasoning: Any) -> Dict[str, Any]:
        """Process task implementation.
        
        Args:
            task: Task to process
            reasoning: Reasoning result
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Create decision based on reasoning
        decision_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_id": reasoning.id,
            "decision": reasoning.conclusion,
            "confidence": reasoning.confidence,
            "alternatives": reasoning.alternatives,
            "requires_approval": False,
            "justification": ""
        }
        
        # Extract decision type from task
        decision_type = task.input_data.get("decision_type", "")
        
        # Check if decision requires approval
        requires_approval = decision_type in settings.human_interface.human_approval_required_for
        decision_result["requires_approval"] = requires_approval
        
        # Extract justification from reasoning
        for step in reasoning.steps:
            if isinstance(step, dict) and "justification" in step:
                decision_result["justification"] = step["justification"]
                break
            elif isinstance(step, dict) and "content" in step and "justification" in step["content"].lower():
                lines = step["content"].split("\n")
                for line in lines:
                    if "justification" in line.lower():
                        decision_result["justification"] = line.split(":", 1)[1].strip() if ":" in line else line
                        break
        
        # If no justification found, use the conclusion
        if not decision_result["justification"]:
            decision_result["justification"] = reasoning.conclusion
        
        return decision_result

class ExplanationAgent(BaseAgent):
    """Explanation agent class."""
    
    async def _process_task_impl(self, task: Task, reasoning: Any) -> Dict[str, Any]:
        """Process task implementation.
        
        Args:
            task: Task to process
            reasoning: Reasoning result
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Extract explanation parameters from task
        detail_level = task.input_data.get("detail_level", "high")
        audience = task.input_data.get("audience", "trader")
        decision_id = task.input_data.get("decision_id", "")
        
        # Create explanation based on reasoning
        explanation_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_id": reasoning.id,
            "explanation": reasoning.conclusion,
            "detail_level": detail_level,
            "audience": audience,
            "decision_id": decision_id,
            "factors": []
        }
        
        # Extract factors from reasoning steps
        factors = []
        for step in reasoning.steps:
            if isinstance(step, dict):
                # Check if step contains factor information
                if "factor" in step:
                    factors.append(step)
                elif "content" in step and "factor" in step["content"].lower():
                    # Extract factor from content
                    lines = step["content"].split("\n")
                    for line in lines:
                        if "factor" in line.lower():
                            factor = {
                                "name": line.split(":", 1)[0].strip() if ":" in line else "Factor",
                                "description": line.split(":", 1)[1].strip() if ":" in line else line,
                                "importance": "medium"
                            }
                            factors.append(factor)
        
        # If no factors found, try to extract from conclusion
        if not factors:
            lines = reasoning.conclusion.split("\n")
            for line in lines:
                if ":" in line and len(line) < 100:  # Simple heuristic for factor-like lines
                    factor = {
                        "name": line.split(":", 1)[0].strip(),
                        "description": line.split(":", 1)[1].strip(),
                        "importance": "medium"
                    }
                    factors.append(factor)
        
        explanation_result["factors"] = factors
        
        return explanation_result

class LearningAgent(BaseAgent):
    """Learning agent class."""
    
    async def _process_task_impl(self, task: Task, reasoning: Any) -> Dict[str, Any]:
        """Process task implementation.
        
        Args:
            task: Task to process
            reasoning: Reasoning result
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Extract learning parameters from task
        learning_type = task.input_data.get("learning_type", "performance_analysis")
        data = task.input_data.get("data", {})
        
        # Create learning result based on reasoning
        learning_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_id": reasoning.id,
            "learning_type": learning_type,
            "insights": reasoning.conclusion,
            "improvements": []
        }
        
        # Extract improvements from reasoning steps
        improvements = []
        for step in reasoning.steps:
            if isinstance(step, dict):
                # Check if step contains improvement information
                if "improvement" in step:
                    improvements.append(step["improvement"])
                elif "content" in step and "improvement" in step["content"].lower():
                    # Extract improvement from content
                    lines = step["content"].split("\n")
                    for line in lines:
                        if "improvement" in line.lower():
                            improvement = line.split(":", 1)[1].strip() if ":" in line else line
                            improvements.append(improvement)
        
        # If no improvements found, try to extract from conclusion
        if not improvements:
            if "improve" in reasoning.conclusion.lower():
                lines = reasoning.conclusion.split("\n")
                for line in lines:
                    if "improve" in line.lower():
                        improvement = line.strip()
                        improvements.append(improvement)
        
        learning_result["improvements"] = improvements
        
        return learning_result

class HumanInterfaceAgent(BaseAgent):
    """Human interface agent class."""
    
    async def _process_task_impl(self, task: Task, reasoning: Any) -> Dict[str, Any]:
        """Process task implementation.
        
        Args:
            task: Task to process
            reasoning: Reasoning result
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Extract interface parameters from task
        interface_type = task.input_data.get("interface_type", "notification")
        user_id = task.input_data.get("user_id", "")
        
        # Create interface result based on reasoning
        interface_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_id": reasoning.id,
            "interface_type": interface_type,
            "user_id": user_id,
            "content": reasoning.conclusion,
            "requires_response": False,
            "options": []
        }
        
        # Handle different interface types
        if interface_type == "approval_request":
            interface_result["requires_response"] = True
            interface_result["options"] = ["approve", "reject"]
        elif interface_type == "notification":
            interface_result["requires_response"] = False
        elif interface_type == "question":
            interface_result["requires_response"] = True
            
            # Extract options from reasoning
            options = []
            for step in reasoning.steps:
                if isinstance(step, dict) and "options" in step:
                    options = step["options"]
                    break
                elif isinstance(step, dict) and "content" in step and "option" in step["content"].lower():
                    lines = step["content"].split("\n")
                    for line in lines:
                        if "option" in line.lower() or line.strip().startswith(("-", "*", "•")):
                            option = line.split(":", 1)[1].strip() if ":" in line else line.strip().lstrip("-*• ")
                            options.append(option)
            
            interface_result["options"] = options
        
        return interface_result

class AgentFactory:
    """Factory for creating agents."""
    
    @staticmethod
    async def create_agent(request: CreateAgentRequest) -> CreateAgentResponse:
        """Create a new agent.
        
        Args:
            request: Agent creation request
            
        Returns:
            CreateAgentResponse: Agent creation response
        """
        # Generate agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Create agent data
        agent_data = {
            "id": agent_id,
            "name": request.name,
            "type": request.type,
            "status": AgentStatus.IDLE,
            "capabilities": request.capabilities,
            "config": request.config,
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow(),
            "metadata": request.metadata
        }
        
        # Create agent in database
        try:
            agent_model = AgentRepository.create(agent_data)
            
            # Convert to Agent model
            agent = Agent(
                id=agent_model.id,
                name=agent_model.name,
                type=agent_model.type,
                status=agent_model.status,
                capabilities=agent_model.capabilities,
                config=agent_model.config,
                created_at=agent_model.created_at,
                last_active=agent_model.last_active,
                metadata=agent_model.metadata
            )
            
            # Create response
            response = CreateAgentResponse(agent=agent)
            
            return response
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    @staticmethod
    def get_agent_instance(agent_data: Agent) -> BaseAgent:
        """Get agent instance based on type.
        
        Args:
            agent_data: Agent data
            
        Returns:
            BaseAgent: Agent instance
        """
        # Create agent instance based on type
        if agent_data.type == AgentType.MONITORING:
            return MonitoringAgent(agent_data)
        elif agent_data.type == AgentType.DECISION:
            return DecisionAgent(agent_data)
        elif agent_data.type == AgentType.EXPLANATION:
            return ExplanationAgent(agent_data)
        elif agent_data.type == AgentType.LEARNING:
            return LearningAgent(agent_data)
        elif agent_data.type == AgentType.HUMAN_INTERFACE:
            return HumanInterfaceAgent(agent_data)
        else:
            raise ValueError(f"Unsupported agent type: {agent_data.type}")

# Create agent factory instance
agent_factory = AgentFactory()
