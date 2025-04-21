"""
Agent coordinator for the agentic oversight system.
"""

import logging
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from ..config.settings import settings
from ..core.models import (
    Agent, Task, TaskStatus, TaskPriority, Coordination,
    CoordinationStrategy, ConflictResolutionStrategy,
    CreateTaskRequest, CreateTaskResponse
)
from ..memory.database import (
    TaskModel, TaskRepository, CoordinationModel, AgentRepository
)
from ..agents.base import agent_factory

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """Coordinator for managing multiple agents."""
    
    def __init__(self):
        """Initialize agent coordinator."""
        self.coordination_strategy = settings.coordinator.coordination_strategy
        self.max_coordination_rounds = settings.coordinator.max_coordination_rounds
        self.coordination_timeout_seconds = settings.coordinator.coordination_timeout_seconds
        self.conflict_resolution_strategy = settings.coordinator.conflict_resolution_strategy
        self.task_prioritization_method = settings.coordinator.task_prioritization_method
        self.task_assignment_method = settings.coordinator.task_assignment_method
        self.coordination_check_interval_seconds = settings.coordinator.coordination_check_interval_seconds
        self.max_tasks_per_agent = settings.coordinator.max_tasks_per_agent
        self.task_timeout_multiplier = settings.coordinator.task_timeout_multiplier
        
        # Initialize task queue
        self.task_queue = []
        
        # Initialize active tasks
        self.active_tasks = {}
        
        # Initialize coordination sessions
        self.coordination_sessions = {}
    
    async def start(self):
        """Start the coordinator."""
        logger.info("Starting agent coordinator")
        
        # Start background task processing
        asyncio.create_task(self._process_tasks())
        
        # Start coordination check
        asyncio.create_task(self._check_coordinations())
    
    async def create_task(self, request: CreateTaskRequest) -> CreateTaskResponse:
        """Create a new task.
        
        Args:
            request: Task creation request
            
        Returns:
            CreateTaskResponse: Task creation response
        """
        # Generate task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create task data
        task_data = {
            "id": task_id,
            "title": request.title,
            "description": request.description,
            "type": request.type,
            "status": TaskStatus.PENDING,
            "priority": request.priority,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "assigned_to": request.assigned_to,
            "created_by": request.created_by,
            "due_by": request.due_by,
            "input_data": request.input_data,
            "output_data": {},
            "dependencies": request.dependencies,
            "tags": request.tags,
            "metadata": request.metadata
        }
        
        # Create task in database
        try:
            task_model = TaskRepository.create(task_data)
            
            # Convert to Task model
            task = Task(
                id=task_model.id,
                title=task_model.title,
                description=task_model.description,
                type=task_model.type,
                status=task_model.status,
                priority=task_model.priority,
                created_at=task_model.created_at,
                updated_at=task_model.updated_at,
                assigned_to=task_model.assigned_to,
                created_by=task_model.created_by,
                due_by=task_model.due_by,
                input_data=task_model.input_data,
                output_data=task_model.output_data,
                dependencies=task_model.dependencies,
                tags=task_model.tags,
                metadata=task_model.metadata
            )
            
            # Add task to queue if not assigned
            if not task.assigned_to:
                self.task_queue.append(task)
                self._prioritize_tasks()
            
            # Create response
            response = CreateTaskResponse(task=task)
            
            return response
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            raise
    
    async def _process_tasks(self):
        """Process tasks in the queue."""
        while True:
            try:
                # Check for pending tasks
                if not self.task_queue:
                    # Get pending tasks from database
                    pending_tasks = TaskRepository.get_pending_tasks()
                    
                    # Add to queue
                    for task_model in pending_tasks:
                        task = Task(
                            id=task_model.id,
                            title=task_model.title,
                            description=task_model.description,
                            type=task_model.type,
                            status=task_model.status,
                            priority=task_model.priority,
                            created_at=task_model.created_at,
                            updated_at=task_model.updated_at,
                            assigned_to=task_model.assigned_to,
                            created_by=task_model.created_by,
                            due_by=task_model.due_by,
                            input_data=task_model.input_data,
                            output_data=task_model.output_data,
                            dependencies=task_model.dependencies,
                            tags=task_model.tags,
                            metadata=task_model.metadata
                        )
                        
                        # Only add if not already in queue or active
                        if task.id not in self.active_tasks and task not in self.task_queue:
                            self.task_queue.append(task)
                    
                    # Prioritize tasks
                    self._prioritize_tasks()
                
                # Process tasks
                if self.task_queue:
                    # Get next task
                    task = self.task_queue[0]
                    
                    # Check if task can be processed
                    if await self._can_process_task(task):
                        # Remove from queue
                        self.task_queue.pop(0)
                        
                        # Process task
                        asyncio.create_task(self._process_task(task))
                
                # Sleep before checking again
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error processing tasks: {e}")
                await asyncio.sleep(5)
    
    async def _can_process_task(self, task: Task) -> bool:
        """Check if a task can be processed.
        
        Args:
            task: Task to check
            
        Returns:
            bool: Whether task can be processed
        """
        # Check if task is already assigned
        if task.assigned_to:
            return True
        
        # Check if task has dependencies
        if task.dependencies:
            # Check if all dependencies are completed
            for dependency_id in task.dependencies:
                dependency = TaskRepository.get_by_id(dependency_id)
                if not dependency or dependency.status != TaskStatus.COMPLETED:
                    return False
        
        # Check if there are available agents
        available_agents = AgentRepository.get_by_status(AgentStatus.IDLE)
        if not available_agents:
            return False
        
        # Check if task requires coordination
        if self._requires_coordination(task):
            # Check if there are enough available agents
            if len(available_agents) < 2:
                return False
        
        return True
    
    def _requires_coordination(self, task: Task) -> bool:
        """Check if a task requires coordination.
        
        Args:
            task: Task to check
            
        Returns:
            bool: Whether task requires coordination
        """
        # Check task type
        coordination_task_types = [
            "strategy_change",
            "risk_limit_override",
            "portfolio_rebalance",
            "market_analysis",
            "anomaly_investigation"
        ]
        
        if task.type in coordination_task_types:
            return True
        
        # Check task priority
        if task.priority == TaskPriority.CRITICAL:
            return True
        
        # Check task metadata
        if task.metadata.get("requires_coordination", False):
            return True
        
        return False
    
    def _prioritize_tasks(self):
        """Prioritize tasks in the queue."""
        if not self.task_queue:
            return
        
        # Sort based on prioritization method
        if self.task_prioritization_method == "fifo":
            # Already in FIFO order
            pass
        elif self.task_prioritization_method == "importance":
            # Sort by priority only
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        elif self.task_prioritization_method == "urgency":
            # Sort by due date
            self.task_queue.sort(key=lambda t: t.due_by if t.due_by else datetime.max)
        elif self.task_prioritization_method == "importance_urgency":
            # Sort by priority and due date
            self.task_queue.sort(key=lambda t: (
                t.priority,
                t.due_by if t.due_by else datetime.max
            ), reverse=True)
        else:
            # Default to priority
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)
    
    async def _process_task(self, task: Task):
        """Process a task.
        
        Args:
            task: Task to process
        """
        try:
            # Add to active tasks
            self.active_tasks[task.id] = task
            
            # Check if task requires coordination
            if self._requires_coordination(task) and not task.assigned_to:
                # Start coordination
                await self._coordinate_task(task)
            else:
                # Assign task if not already assigned
                if not task.assigned_to:
                    agent_id = await self._assign_task(task)
                    if not agent_id:
                        # Put back in queue
                        self.task_queue.append(task)
                        self._prioritize_tasks()
                        return
                    
                    # Update task with assigned agent
                    task.assigned_to = agent_id
                    TaskRepository.update(task.id, {"assigned_to": agent_id, "status": TaskStatus.ASSIGNED})
                
                # Get agent
                agent_model = AgentRepository.get_by_id(task.assigned_to)
                if not agent_model:
                    logger.error(f"Agent {task.assigned_to} not found for task {task.id}")
                    # Put back in queue
                    self.task_queue.append(task)
                    self._prioritize_tasks()
                    return
                
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
                
                # Get agent instance
                agent_instance = agent_factory.get_agent_instance(agent)
                
                # Process task
                task_update = await agent_instance.process_task(task)
                
                # Update task
                TaskRepository.update(task.id, task_update)
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            
            # Update task status
            TaskRepository.update(task.id, {
                "status": TaskStatus.FAILED,
                "output_data": {
                    "error": str(e)
                }
            })
        finally:
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _assign_task(self, task: Task) -> Optional[str]:
        """Assign a task to an agent.
        
        Args:
            task: Task to assign
            
        Returns:
            Optional[str]: Agent ID or None
        """
        # Get available agents
        available_agents = AgentRepository.get_by_status(AgentStatus.IDLE)
        if not available_agents:
            return None
        
        # Assign based on assignment method
        if self.task_assignment_method == "round_robin":
            # Simple round robin
            return available_agents[0].id
        elif self.task_assignment_method == "specialized":
            # Find agent with matching capabilities
            task_type = task.type
            for agent in available_agents:
                if task_type in agent.capabilities:
                    return agent.id
            
            # Fallback to first available
            return available_agents[0].id
        elif self.task_assignment_method == "load_balanced":
            # Find agent with fewest tasks
            agent_task_counts = {}
            for agent in available_agents:
                agent_tasks = TaskRepository.get_by_agent(agent.id)
                agent_task_counts[agent.id] = len(agent_tasks)
            
            # Sort by task count
            sorted_agents = sorted(agent_task_counts.items(), key=lambda x: x[1])
            if sorted_agents:
                return sorted_agents[0][0]
            
            # Fallback to first available
            return available_agents[0].id
        else:
            # Default to first available
            return available_agents[0].id
    
    async def _coordinate_task(self, task: Task):
        """Coordinate a task among multiple agents.
        
        Args:
            task: Task to coordinate
        """
        # Create coordination session
        coordination_id = f"coordination_{uuid.uuid4().hex[:8]}"
        
        # Get available agents
        available_agents = AgentRepository.get_by_status(AgentStatus.IDLE)
        if len(available_agents) < 2:
            logger.warning(f"Not enough available agents for coordination of task {task.id}")
            # Put back in queue
            self.task_queue.append(task)
            self._prioritize_tasks()
            return
        
        # Select agents based on task type
        selected_agents = self._select_agents_for_coordination(task, available_agents)
        if len(selected_agents) < 2:
            logger.warning(f"Not enough suitable agents for coordination of task {task.id}")
            # Put back in queue
            self.task_queue.append(task)
            self._prioritize_tasks()
            return
        
        # Create coordination data
        coordination_data = {
            "id": coordination_id,
            "strategy": self.coordination_strategy,
            "agents": [agent.id for agent in selected_agents],
            "tasks": [task.id],
            "status": "in_progress",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "result": {},
            "conflicts": [],
            "resolution_strategy": self.conflict_resolution_strategy,
            "metadata": {
                "original_task": task.dict()
            }
        }
        
        # Store coordination in database
        # (This would be implemented in a real system)
        
        # Add to coordination sessions
        self.coordination_sessions[coordination_id] = {
            "coordination": coordination_data,
            "task": task,
            "agents": selected_agents,
            "subtasks": [],
            "results": [],
            "round": 0,
            "start_time": datetime.utcnow()
        }
        
        # Start coordination
        await self._run_coordination(coordination_id)
    
    def _select_agents_for_coordination(self, task: Task, available_agents: List[Any]) -> List[Any]:
        """Select agents for coordination.
        
        Args:
            task: Task to coordinate
            available_agents: Available agents
            
        Returns:
            List[Any]: Selected agents
        """
        # Get required agent types based on task
        required_types = self._get_required_agent_types(task)
        
        # Select agents based on required types
        selected_agents = []
        for agent_type in required_types:
            # Find agent of this type
            for agent in available_agents:
                if agent.type == agent_type and agent not in selected_agents:
                    selected_agents.append(agent)
                    break
        
        # If we don't have enough agents, add more
        while len(selected_agents) < 2 and len(selected_agents) < len(available_agents):
            for agent in available_agents:
                if agent not in selected_agents:
                    selected_agents.append(agent)
                    break
        
        return selected_agents
    
    def _get_required_agent_types(self, task: Task) -> List[str]:
        """Get required agent types for a task.
        
        Args:
            task: Task to coordinate
            
        Returns:
            List[str]: Required agent types
        """
        # Default required types
        required_types = ["monitoring", "decision"]
        
        # Adjust based on task type
        if task.type == "strategy_change":
            required_types = ["decision", "explanation"]
        elif task.type == "risk_limit_override":
            required_types = ["decision", "monitoring"]
        elif task.type == "portfolio_rebalance":
            required_types = ["decision", "explanation"]
        elif task.type == "market_analysis":
            required_types = ["monitoring", "explanation"]
        elif task.type == "anomaly_investigation":
            required_types = ["monitoring", "decision"]
        
        return required_types
    
    async def _run_coordination(self, coordination_id: str):
        """Run coordination session.
        
        Args:
            coordination_id: Coordination ID
        """
        # Get coordination session
        session = self.coordination_sessions.get(coordination_id)
        if not session:
            logger.error(f"Coordination session {coordination_id} not found")
            return
        
        # Get coordination data
        coordination = session["coordination"]
        task = session["task"]
        agents = session["agents"]
        
        # Increment round
        session["round"] += 1
        
        # Check if we've reached max rounds
        if session["round"] > self.max_coordination_rounds:
            logger.warning(f"Coordination {coordination_id} reached max rounds")
            await self._complete_coordination(coordination_id, "max_rounds_reached")
            return
        
        # Create subtasks for each agent
        subtasks = []
        for agent in agents:
            # Create subtask
            subtask_id = f"subtask_{uuid.uuid4().hex[:8]}"
            
            # Create subtask data
            subtask_data = {
                "id": subtask_id,
                "title": f"Coordination subtask for {task.title}",
                "description": f"Subtask for coordination {coordination_id}, round {session['round']}",
                "type": task.type,
                "status": TaskStatus.PENDING,
                "priority": task.priority,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "assigned_to": agent.id,
                "created_by": "coordinator",
                "due_by": datetime.utcnow() + timedelta(seconds=self.coordination_timeout_seconds),
                "input_data": {
                    "original_task": task.dict(),
                    "coordination_id": coordination_id,
                    "round": session["round"],
                    "previous_results": session["results"]
                },
                "output_data": {},
                "dependencies": [],
                "tags": ["coordination", f"round_{session['round']}"],
                "metadata": {
                    "coordination_id": coordination_id
                }
            }
            
            # Create subtask in database
            subtask_model = TaskRepository.create(subtask_data)
            
            # Convert to Task model
            subtask = Task(
                id=subtask_model.id,
                title=subtask_model.title,
                description=subtask_model.description,
                type=subtask_model.type,
                status=subtask_model.status,
                priority=subtask_model.priority,
                created_at=subtask_model.created_at,
                updated_at=subtask_model.updated_at,
                assigned_to=subtask_model.assigned_to,
                created_by=subtask_model.created_by,
                due_by=subtask_model.due_by,
                input_data=subtask_model.input_data,
                output_data=subtask_model.output_data,
                dependencies=subtask_model.dependencies,
                tags=subtask_model.tags,
                metadata=subtask_model.metadata
            )
            
            # Add to subtasks
            subtasks.append(subtask)
            
            # Process subtask
            asyncio.create_task(self._process_subtask(subtask, agent))
        
        # Add subtasks to session
        session["subtasks"] = subtasks
        
        # Update coordination status
        coordination["status"] = "in_progress"
        coordination["updated_at"] = datetime.utcnow()
        
        # Update coordination in database
        # (This would be implemented in a real system)
    
    async def _process_subtask(self, subtask: Task, agent: Any):
        """Process a coordination subtask.
        
        Args:
            subtask: Subtask to process
            agent: Agent to process subtask
        """
        try:
            # Get agent instance
            agent_instance = agent_factory.get_agent_instance(agent)
            
            # Process subtask
            task_update = await agent_instance.process_task(subtask)
            
            # Update subtask
            TaskRepository.update(subtask.id, task_update)
            
            # Get coordination ID
            coordination_id = subtask.metadata.get("coordination_id")
            if not coordination_id:
                logger.error(f"Subtask {subtask.id} has no coordination ID")
                return
            
            # Get coordination session
            session = self.coordination_sessions.get(coordination_id)
            if not session:
                logger.error(f"Coordination session {coordination_id} not found")
                return
            
            # Add result to session
            session["results"].append({
                "agent_id": agent.id,
                "agent_type": agent.type,
                "subtask_id": subtask.id,
                "result": task_update["output_data"]
            })
            
            # Check if all subtasks are completed
            all_completed = True
            for st in session["subtasks"]:
                st_model = TaskRepository.get_by_id(st.id)
                if not st_model or st_model.status != TaskStatus.COMPLETED:
                    all_completed = False
                    break
            
            if all_completed:
                # Process results
                await self._process_coordination_results(coordination_id)
        except Exception as e:
            logger.error(f"Error processing subtask {subtask.id}: {e}")
            
            # Update subtask status
            TaskRepository.update(subtask.id, {
                "status": TaskStatus.FAILED,
                "output_data": {
                    "error": str(e)
                }
            })
    
    async def _process_coordination_results(self, coordination_id: str):
        """Process coordination results.
        
        Args:
            coordination_id: Coordination ID
        """
        # Get coordination session
        session = self.coordination_sessions.get(coordination_id)
        if not session:
            logger.error(f"Coordination session {coordination_id} not found")
            return
        
        # Get coordination data
        coordination = session["coordination"]
        task = session["task"]
        results = session["results"]
        
        # Check for conflicts
        conflicts = self._detect_conflicts(results)
        
        if conflicts:
            # Add conflicts to coordination
            coordination["conflicts"] = conflicts
            
            # Resolve conflicts
            resolved_results = await self._resolve_conflicts(coordination_id, conflicts)
            
            # Check if we need another round
            if not resolved_results:
                # Start another round
                await self._run_coordination(coordination_id)
                return
            
            # Use resolved results
            final_result = resolved_results
        else:
            # No conflicts, use first result
            final_result = results[0]["result"]
        
        # Complete coordination
        await self._complete_coordination(coordination_id, "completed", final_result)
    
    def _detect_conflicts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts in coordination results.
        
        Args:
            results: Coordination results
            
        Returns:
            List[Dict[str, Any]]: Detected conflicts
        """
        # Initialize conflicts
        conflicts = []
        
        # Check for decision conflicts
        decisions = {}
        for result in results:
            if "decision" in result["result"]:
                decision = result["result"]["decision"]
                if decision not in decisions:
                    decisions[decision] = []
                decisions[decision].append(result["agent_id"])
        
        # If more than one decision, we have a conflict
        if len(decisions) > 1:
            conflict = {
                "type": "decision",
                "values": list(decisions.keys()),
                "agents": decisions,
                "resolved": False,
                "resolution": None,
                "resolution_method": None
            }
            conflicts.append(conflict)
        
        # Check for other conflicts
        # (This would be expanded in a real system)
        
        return conflicts
    
    async def _resolve_conflicts(self, coordination_id: str, conflicts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Resolve conflicts in coordination.
        
        Args:
            coordination_id: Coordination ID
            conflicts: Conflicts to resolve
            
        Returns:
            Optional[Dict[str, Any]]: Resolved result or None
        """
        # Get coordination session
        session = self.coordination_sessions.get(coordination_id)
        if not session:
            logger.error(f"Coordination session {coordination_id} not found")
            return None
        
        # Get coordination data
        coordination = session["coordination"]
        results = session["results"]
        
        # Initialize resolved result
        resolved_result = None
        
        # Resolve based on strategy
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.CONSENSUS:
            # Try to find consensus
            for conflict in conflicts:
                if conflict["type"] == "decision":
                    # Find most common decision
                    decision_counts = {}
                    for decision, agents in conflict["agents"].items():
                        decision_counts[decision] = len(agents)
                    
                    # Sort by count
                    sorted_decisions = sorted(decision_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    # If there's a clear winner, use it
                    if len(sorted_decisions) > 1 and sorted_decisions[0][1] > sorted_decisions[1][1]:
                        conflict["resolved"] = True
                        conflict["resolution"] = sorted_decisions[0][0]
                        conflict["resolution_method"] = "consensus"
                    else:
                        # No clear consensus
                        conflict["resolved"] = False
            
            # Check if all conflicts are resolved
            all_resolved = all(conflict["resolved"] for conflict in conflicts)
            if all_resolved:
                # Create resolved result
                resolved_result = {}
                
                # Add resolved decisions
                for conflict in conflicts:
                    if conflict["type"] == "decision":
                        resolved_result["decision"] = conflict["resolution"]
                
                # Add other fields from first result
                for key, value in results[0]["result"].items():
                    if key != "decision":
                        resolved_result[key] = value
        
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.AUTHORITY:
            # Use result from highest authority agent
            authority_order = ["decision", "monitoring", "explanation", "learning", "human_interface"]
            
            # Sort results by agent type authority
            sorted_results = sorted(results, key=lambda r: authority_order.index(r["agent_type"]) if r["agent_type"] in authority_order else 999)
            
            # Use first result
            if sorted_results:
                resolved_result = sorted_results[0]["result"]
                
                # Mark conflicts as resolved
                for conflict in conflicts:
                    conflict["resolved"] = True
                    conflict["resolution"] = resolved_result.get(conflict["type"])
                    conflict["resolution_method"] = "authority"
        
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.VOTING:
            # Implement voting
            # (This would be implemented in a real system)
            pass
        
        # Update conflicts in coordination
        coordination["conflicts"] = conflicts
        
        # Update coordination in database
        # (This would be implemented in a real system)
        
        return resolved_result
    
    async def _complete_coordination(self, coordination_id: str, status: str, result: Optional[Dict[str, Any]] = None):
        """Complete coordination session.
        
        Args:
            coordination_id: Coordination ID
            status: Completion status
            result: Coordination result
        """
        # Get coordination session
        session = self.coordination_sessions.get(coordination_id)
        if not session:
            logger.error(f"Coordination session {coordination_id} not found")
            return
        
        # Get coordination data
        coordination = session["coordination"]
        task = session["task"]
        
        # Update coordination status
        coordination["status"] = status
        coordination["completed_at"] = datetime.utcnow()
        coordination["updated_at"] = datetime.utcnow()
        
        if result:
            coordination["result"] = result
        
        # Update coordination in database
        # (This would be implemented in a real system)
        
        # Update original task
        if status == "completed" and result:
            TaskRepository.update(task.id, {
                "status": TaskStatus.COMPLETED,
                "completed_at": datetime.utcnow(),
                "output_data": result
            })
        else:
            TaskRepository.update(task.id, {
                "status": TaskStatus.FAILED,
                "output_data": {
                    "error": f"Coordination failed: {status}"
                }
            })
        
        # Remove from coordination sessions
        del self.coordination_sessions[coordination_id]
    
    async def _check_coordinations(self):
        """Check coordination sessions for timeouts."""
        while True:
            try:
                # Get current time
                now = datetime.utcnow()
                
                # Check each coordination session
                for coordination_id, session in list(self.coordination_sessions.items()):
                    # Check if timed out
                    start_time = session["start_time"]
                    if (now - start_time).total_seconds() > self.coordination_timeout_seconds:
                        logger.warning(f"Coordination {coordination_id} timed out")
                        await self._complete_coordination(coordination_id, "timeout")
                
                # Sleep before checking again
                await asyncio.sleep(self.coordination_check_interval_seconds)
            except Exception as e:
                logger.error(f"Error checking coordinations: {e}")
                await asyncio.sleep(self.coordination_check_interval_seconds)

# Create agent coordinator instance
agent_coordinator = AgentCoordinator()
