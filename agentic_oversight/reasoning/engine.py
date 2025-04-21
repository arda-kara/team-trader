"""
Reasoning engine for the agentic oversight system.
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from ..config.settings import settings
from ..core.models import (
    ReasoningFramework, Reasoning, Task, Agent,
    TaskStatus, DecisionConfidence
)
from ..memory.database import ReasoningModel, ReasoningRepository
from .llm_client import llm_client

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Reasoning engine for agents."""
    
    def __init__(self):
        """Initialize reasoning engine."""
        self.framework = settings.reasoning.reasoning_framework
        self.max_reasoning_steps = settings.reasoning.max_reasoning_steps
        self.reasoning_detail_level = settings.reasoning.reasoning_detail_level
        self.include_uncertainty = settings.reasoning.include_uncertainty
        self.include_alternatives = settings.reasoning.include_alternatives
        self.max_alternatives = settings.reasoning.max_alternatives
        self.reasoning_timeout_seconds = settings.reasoning.reasoning_timeout_seconds
        self.structured_reasoning = settings.reasoning.structured_reasoning
        self.reasoning_templates = settings.reasoning.reasoning_templates
    
    async def reason(self, agent: Agent, task: Task) -> Reasoning:
        """Perform reasoning for a task.
        
        Args:
            agent: Agent performing reasoning
            task: Task to reason about
            
        Returns:
            Reasoning: Reasoning result
        """
        logger.info(f"Agent {agent.id} reasoning about task {task.id}")
        
        # Start timing
        start_time = time.time()
        
        # Get reasoning framework
        framework = ReasoningFramework(self.framework)
        
        # Initialize reasoning steps
        steps = []
        
        # Prepare context for reasoning
        context = self._prepare_context(agent, task)
        
        # Perform reasoning based on framework
        if framework == ReasoningFramework.REACT:
            conclusion, confidence, steps, alternatives = await self._react_reasoning(agent, task, context)
        elif framework == ReasoningFramework.COT:
            conclusion, confidence, steps, alternatives = await self._cot_reasoning(agent, task, context)
        elif framework == ReasoningFramework.TOT:
            conclusion, confidence, steps, alternatives = await self._tot_reasoning(agent, task, context)
        else:
            conclusion, confidence, steps, alternatives = await self._custom_reasoning(agent, task, context)
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Create reasoning object
        reasoning_id = f"reasoning_{uuid.uuid4().hex[:8]}"
        reasoning = Reasoning(
            id=reasoning_id,
            agent_id=agent.id,
            task_id=task.id,
            framework=framework,
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
            alternatives=alternatives,
            created_at=datetime.utcnow(),
            duration_ms=duration_ms,
            metadata={}
        )
        
        # Store reasoning in database
        reasoning_data = reasoning.dict()
        ReasoningRepository.create(reasoning_data)
        
        logger.info(f"Reasoning {reasoning_id} completed in {duration_ms}ms with confidence {confidence}")
        
        return reasoning
    
    def _prepare_context(self, agent: Agent, task: Task) -> Dict[str, Any]:
        """Prepare context for reasoning.
        
        Args:
            agent: Agent performing reasoning
            task: Task to reason about
            
        Returns:
            Dict[str, Any]: Context for reasoning
        """
        # Prepare context with task information
        context = {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "capabilities": agent.capabilities
            },
            "task": {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "type": task.type,
                "priority": task.priority,
                "input_data": task.input_data
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return context
    
    async def _react_reasoning(self, agent: Agent, task: Task, context: Dict[str, Any]) -> tuple:
        """Perform ReAct reasoning.
        
        Args:
            agent: Agent performing reasoning
            task: Task to reason about
            context: Context for reasoning
            
        Returns:
            tuple: (conclusion, confidence, steps, alternatives)
        """
        # Initialize steps
        steps = []
        
        # Prepare system message
        system_message = f"""You are an AI agent named {agent.name} with the following capabilities: {', '.join(agent.capabilities)}.
You are tasked with reasoning about a problem using the ReAct framework (Reason, Act, Observe).
For each step, you will:
1. Think about the current state and what to do next
2. Decide on an action to take
3. Observe the result of that action

Task: {task.title}
Description: {task.description}

You should continue this process for multiple steps until you reach a conclusion.
After reasoning, provide your final conclusion, confidence level (0.0-1.0), and any alternative conclusions.
"""
        
        # Prepare initial user message
        user_message = f"""Please help me reason through this task:
Task ID: {task.id}
Title: {task.title}
Description: {task.description}
Type: {task.type}
Priority: {task.priority}

Input data:
{json.dumps(task.input_data, indent=2)}

Use the ReAct framework to reason step by step.
"""
        
        # Initialize messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Initialize step counter
        step_count = 0
        
        # Perform reasoning steps
        while step_count < self.max_reasoning_steps:
            # Get response from LLM
            response = llm_client.generate(
                messages,
                temperature=agent.config.get("temperature", 0.2),
                max_tokens=agent.config.get("max_tokens", 2000)
            )
            
            # Add response to messages
            messages.append({"role": "assistant", "content": response})
            
            # Parse reasoning step
            step, is_final = self._parse_react_step(response)
            
            # Add step to steps
            if step:
                steps.append(step)
                step_count += 1
            
            # Check if reasoning is complete
            if is_final:
                break
            
            # Prepare next user message
            user_message = "Continue reasoning. What's the next step?"
            messages.append({"role": "user", "content": user_message})
        
        # Extract conclusion, confidence, and alternatives
        conclusion, confidence, alternatives = self._extract_conclusion(response)
        
        return conclusion, confidence, steps, alternatives
    
    def _parse_react_step(self, response: str) -> tuple:
        """Parse ReAct reasoning step.
        
        Args:
            response: LLM response
            
        Returns:
            tuple: (step, is_final)
        """
        # Initialize step
        step = {}
        is_final = False
        
        # Look for thought, action, observation pattern
        thought_match = None
        action_match = None
        observation_match = None
        
        # Simple parsing based on keywords
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if "thought:" in line.lower() or "thinking:" in line.lower() or "i think" in line.lower():
                thought_match = line
                # Look for content after the keyword
                thought_content = line.split(':', 1)[1].strip() if ':' in line else line
                step["thought"] = thought_content
            
            elif "action:" in line.lower() or "i will" in line.lower():
                action_match = line
                # Look for content after the keyword
                action_content = line.split(':', 1)[1].strip() if ':' in line else line
                step["action"] = action_content
            
            elif "observation:" in line.lower() or "i observe" in line.lower():
                observation_match = line
                # Look for content after the keyword
                observation_content = line.split(':', 1)[1].strip() if ':' in line else line
                step["observation"] = observation_content
            
            # Check for conclusion indicators
            elif "conclusion:" in line.lower() or "final answer:" in line.lower():
                is_final = True
        
        # If we have at least a thought, consider it a valid step
        if "thought" in step:
            # Add step number if missing
            if "step" not in step:
                step["step"] = len(step)
            
            return step, is_final
        
        return None, is_final
    
    async def _cot_reasoning(self, agent: Agent, task: Task, context: Dict[str, Any]) -> tuple:
        """Perform Chain of Thought reasoning.
        
        Args:
            agent: Agent performing reasoning
            task: Task to reason about
            context: Context for reasoning
            
        Returns:
            tuple: (conclusion, confidence, steps, alternatives)
        """
        # Prepare system message
        system_message = f"""You are an AI agent named {agent.name} with the following capabilities: {', '.join(agent.capabilities)}.
You are tasked with reasoning about a problem using Chain of Thought reasoning.
Think step by step to solve the problem, showing your work clearly.

Task: {task.title}
Description: {task.description}

After reasoning, provide your final conclusion, confidence level (0.0-1.0), and any alternative conclusions.
"""
        
        # Prepare user message
        user_message = f"""Please help me reason through this task:
Task ID: {task.id}
Title: {task.title}
Description: {task.description}
Type: {task.type}
Priority: {task.priority}

Input data:
{json.dumps(task.input_data, indent=2)}

Use Chain of Thought reasoning to think step by step.
"""
        
        # Initialize messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Get response from LLM
        response = llm_client.generate(
            messages,
            temperature=agent.config.get("temperature", 0.2),
            max_tokens=agent.config.get("max_tokens", 2000)
        )
        
        # Parse steps
        steps = self._parse_cot_steps(response)
        
        # Extract conclusion, confidence, and alternatives
        conclusion, confidence, alternatives = self._extract_conclusion(response)
        
        return conclusion, confidence, steps, alternatives
    
    def _parse_cot_steps(self, response: str) -> List[Dict[str, Any]]:
        """Parse Chain of Thought reasoning steps.
        
        Args:
            response: LLM response
            
        Returns:
            List[Dict[str, Any]]: Reasoning steps
        """
        # Initialize steps
        steps = []
        
        # Split response into lines
        lines = response.split('\n')
        
        # Initialize current step
        current_step = None
        step_content = []
        
        # Parse steps based on numbering or step keywords
        for line in lines:
            # Check for numbered steps or step keywords
            if line.strip().startswith(('Step ', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # If we have a current step, add it to steps
                if current_step is not None:
                    steps.append({
                        "step": current_step,
                        "content": '\n'.join(step_content)
                    })
                
                # Extract step number
                if line.strip().startswith('Step '):
                    try:
                        current_step = int(line.strip().split('Step ')[1].split(':')[0])
                    except:
                        current_step = len(steps) + 1
                else:
                    try:
                        current_step = int(line.strip().split('.')[0])
                    except:
                        current_step = len(steps) + 1
                
                # Initialize step content
                step_content = [line.strip()]
            elif current_step is not None:
                # Add line to current step content
                step_content.append(line)
        
        # Add final step if exists
        if current_step is not None and step_content:
            steps.append({
                "step": current_step,
                "content": '\n'.join(step_content)
            })
        
        # If no steps were found, create a single step with the entire response
        if not steps:
            steps.append({
                "step": 1,
                "content": response
            })
        
        return steps
    
    async def _tot_reasoning(self, agent: Agent, task: Task, context: Dict[str, Any]) -> tuple:
        """Perform Tree of Thought reasoning.
        
        Args:
            agent: Agent performing reasoning
            task: Task to reason about
            context: Context for reasoning
            
        Returns:
            tuple: (conclusion, confidence, steps, alternatives)
        """
        # Prepare system message
        system_message = f"""You are an AI agent named {agent.name} with the following capabilities: {', '.join(agent.capabilities)}.
You are tasked with reasoning about a problem using Tree of Thought reasoning.
Consider multiple reasoning paths in parallel, evaluate them, and select the most promising one.

Task: {task.title}
Description: {task.description}

After reasoning, provide your final conclusion, confidence level (0.0-1.0), and any alternative conclusions.
"""
        
        # Prepare user message
        user_message = f"""Please help me reason through this task:
Task ID: {task.id}
Title: {task.title}
Description: {task.description}
Type: {task.type}
Priority: {task.priority}

Input data:
{json.dumps(task.input_data, indent=2)}

Use Tree of Thought reasoning:
1. Generate multiple initial thoughts
2. Explore each thought for 1-2 steps
3. Evaluate the promising paths
4. Continue with the best path
5. Reach a conclusion
"""
        
        # Initialize messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Get response from LLM
        response = llm_client.generate(
            messages,
            temperature=agent.config.get("temperature", 0.3),  # Slightly higher temperature for diversity
            max_tokens=agent.config.get("max_tokens", 2000)
        )
        
        # Parse steps
        steps = self._parse_tot_steps(response)
        
        # Extract conclusion, confidence, and alternatives
        conclusion, confidence, alternatives = self._extract_conclusion(response)
        
        return conclusion, confidence, steps, alternatives
    
    def _parse_tot_steps(self, response: str) -> List[Dict[str, Any]]:
        """Parse Tree of Thought reasoning steps.
        
        Args:
            response: LLM response
            
        Returns:
            List[Dict[str, Any]]: Reasoning steps
        """
        # Initialize steps
        steps = []
        
        # Split response into lines
        lines = response.split('\n')
        
        # Track current section
        current_section = None
        section_content = []
        
        # Parse sections
        for line in lines:
            # Check for section headers
            if "initial thoughts:" in line.lower() or "initial paths:" in line.lower():
                if current_section and section_content:
                    steps.append({
                        "phase": current_section,
                        "content": '\n'.join(section_content)
                    })
                current_section = "initial_thoughts"
                section_content = []
            elif "exploration:" in line.lower() or "exploring paths:" in line.lower():
                if current_section and section_content:
                    steps.append({
                        "phase": current_section,
                        "content": '\n'.join(section_content)
                    })
                current_section = "exploration"
                section_content = []
            elif "evaluation:" in line.lower() or "evaluating paths:" in line.lower():
                if current_section and section_content:
                    steps.append({
                        "phase": current_section,
                        "content": '\n'.join(section_content)
                    })
                current_section = "evaluation"
                section_content = []
            elif "final path:" in line.lower() or "chosen path:" in line.lower():
                if current_section and section_content:
                    steps.append({
                        "phase": current_section,
                        "content": '\n'.join(section_content)
                    })
                current_section = "final_path"
                section_content = []
            elif "conclusion:" in line.lower() or "final answer:" in line.lower():
                if current_section and section_content:
                    steps.append({
                        "phase": current_section,
                        "content": '\n'.join(section_content)
                    })
                current_section = "conclusion"
                section_content = []
            elif current_section:
                section_content.append(line)
        
        # Add final section if exists
        if current_section and section_content:
            steps.append({
                "phase": current_section,
                "content": '\n'.join(section_content)
            })
        
        # If no structured sections were found, try to parse as paths
        if not steps:
            paths = self._extract_tot_paths(response)
            if paths:
                steps = paths
            else:
                # Fallback: treat entire response as a single step
                steps.append({
                    "phase": "reasoning",
                    "content": response
                })
        
        return steps
    
    def _extract_tot_paths(self, response: str) -> List[Dict[str, Any]]:
        """Extract paths from Tree of Thought reasoning.
        
        Args:
            response: LLM response
            
        Returns:
            List[Dict[str, Any]]: Paths
        """
        # Initialize paths
        paths = []
        
        # Look for path indicators
        path_indicators = ["Path ", "Option ", "Approach ", "Alternative "]
        
        # Split response into lines
        lines = response.split('\n')
        
        # Initialize current path
        current_path = None
        path_content = []
        
        # Parse paths
        for line in lines:
            # Check for path indicators
            is_path_indicator = False
            for indicator in path_indicators:
                if line.strip().startswith(indicator):
                    is_path_indicator = True
                    # If we have a current path, add it to paths
                    if current_path is not None:
                        paths.append({
                            "path": current_path,
                            "content": '\n'.join(path_content)
                        })
                    
                    # Extract path number
                    try:
                        current_path = int(line.strip().split(indicator)[1].split(':')[0])
                    except:
                        current_path = len(paths) + 1
                    
                    # Initialize path content
                    path_content = [line.strip()]
                    break
            
            if not is_path_indicator and current_path is not None:
                # Add line to current path content
                path_content.append(line)
        
        # Add final path if exists
        if current_path is not None and path_content:
            paths.append({
                "path": current_path,
                "content": '\n'.join(path_content)
            })
        
        return paths
    
    async def _custom_reasoning(self, agent: Agent, task: Task, context: Dict[str, Any]) -> tuple:
        """Perform custom reasoning.
        
        Args:
            agent: Agent performing reasoning
            task: Task to reason about
            context: Context for reasoning
            
        Returns:
            tuple: (conclusion, confidence, steps, alternatives)
        """
        # Get template based on agent type
        template = self.reasoning_templates.get(agent.type.value, "")
        
        if not template:
            # Fallback to ReAct reasoning
            return await self._react_reasoning(agent, task, context)
        
        # Prepare system message
        system_message = f"""You are an AI agent named {agent.name} with the following capabilities: {', '.join(agent.capabilities)}.
You are tasked with reasoning about a problem using a structured approach.

Task: {task.title}
Description: {task.description}

After reasoning, provide your final conclusion, confidence level (0.0-1.0), and any alternative conclusions.
"""
        
        # Prepare user message with template
        user_message = f"""Please help me reason through this task:
Task ID: {task.id}
Title: {task.title}
Description: {task.description}
Type: {task.type}
Priority: {task.priority}

Input data:
{json.dumps(task.input_data, indent=2)}

Please follow this reasoning structure:
{template}
"""
        
        # Initialize messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Get response from LLM
        response = llm_client.generate(
            messages,
            temperature=agent.config.get("temperature", 0.2),
            max_tokens=agent.config.get("max_tokens", 2000)
        )
        
        # Parse steps (simple approach: each section is a step)
        steps = self._parse_custom_steps(response)
        
        # Extract conclusion, confidence, and alternatives
        conclusion, confidence, alternatives = self._extract_conclusion(response)
        
        return conclusion, confidence, steps, alternatives
    
    def _parse_custom_steps(self, response: str) -> List[Dict[str, Any]]:
        """Parse custom reasoning steps.
        
        Args:
            response: LLM response
            
        Returns:
            List[Dict[str, Any]]: Reasoning steps
        """
        # Initialize steps
        steps = []
        
        # Split response into lines
        lines = response.split('\n')
        
        # Track current section
        current_section = None
        section_content = []
        
        # Common section headers
        section_headers = [
            "analysis:", "factors:", "options:", "pros and cons:",
            "recommendation:", "decision:", "conclusion:", "findings:",
            "actions:", "next steps:"
        ]
        
        # Parse sections
        for line in lines:
            # Check if line contains a section header
            found_header = False
            for header in section_headers:
                if header in line.lower():
                    # If we have a current section, add it to steps
                    if current_section and section_content:
                        steps.append({
                            "section": current_section,
                            "content": '\n'.join(section_content)
                        })
                    
                    # Set new section
                    current_section = header.replace(":", "")
                    section_content = [line]
                    found_header = True
                    break
            
            if not found_header and current_section:
                # Add line to current section
                section_content.append(line)
        
        # Add final section if exists
        if current_section and section_content:
            steps.append({
                "section": current_section,
                "content": '\n'.join(section_content)
            })
        
        # If no sections were found, create a single step with the entire response
        if not steps:
            steps.append({
                "section": "reasoning",
                "content": response
            })
        
        return steps
    
    def _extract_conclusion(self, response: str) -> tuple:
        """Extract conclusion, confidence, and alternatives from response.
        
        Args:
            response: LLM response
            
        Returns:
            tuple: (conclusion, confidence, alternatives)
        """
        # Initialize values
        conclusion = ""
        confidence = 0.0
        alternatives = []
        
        # Look for conclusion
        conclusion_indicators = [
            "conclusion:", "final answer:", "recommendation:", "decision:"
        ]
        
        # Look for confidence
        confidence_indicators = [
            "confidence:", "confidence level:", "certainty:"
        ]
        
        # Look for alternatives
        alternative_indicators = [
            "alternatives:", "alternative conclusions:", "other options:"
        ]
        
        # Split response into lines
        lines = response.split('\n')
        
        # Extract conclusion
        for i, line in enumerate(lines):
            for indicator in conclusion_indicators:
                if indicator in line.lower():
                    # Extract conclusion
                    conclusion_text = line.split(indicator, 1)[1].strip() if indicator in line.lower() else ""
                    
                    # If conclusion is on the same line
                    if conclusion_text:
                        conclusion = conclusion_text
                    # Otherwise, look at the next line
                    elif i + 1 < len(lines):
                        conclusion = lines[i + 1].strip()
                    
                    break
            
            if conclusion:
                break
        
        # Extract confidence
        for i, line in enumerate(lines):
            for indicator in confidence_indicators:
                if indicator in line.lower():
                    # Try to extract confidence value
                    try:
                        # Extract text after indicator
                        confidence_text = line.split(indicator, 1)[1].strip()
                        
                        # Try to parse as float
                        if confidence_text:
                            # Handle percentage format
                            if "%" in confidence_text:
                                confidence_value = float(confidence_text.replace("%", "")) / 100
                            else:
                                # Extract first number
                                import re
                                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", confidence_text)
                                if numbers:
                                    confidence_value = float(numbers[0])
                                else:
                                    confidence_value = 0.0
                            
                            # Ensure confidence is between 0 and 1
                            confidence = max(0.0, min(1.0, confidence_value))
                    except:
                        # If parsing fails, map confidence terms to values
                        confidence_terms = {
                            "very low": 0.1,
                            "low": 0.3,
                            "medium": 0.5,
                            "moderate": 0.5,
                            "high": 0.7,
                            "very high": 0.9
                        }
                        
                        for term, value in confidence_terms.items():
                            if term in line.lower():
                                confidence = value
                                break
                    
                    break
            
            if confidence > 0:
                break
        
        # If no confidence was found, estimate based on language
        if confidence == 0.0:
            confidence_terms = {
                "uncertain": 0.3,
                "not sure": 0.3,
                "possibly": 0.4,
                "maybe": 0.4,
                "likely": 0.6,
                "probably": 0.6,
                "confident": 0.7,
                "certainly": 0.8,
                "definitely": 0.9,
                "absolutely": 0.9
            }
            
            for term, value in confidence_terms.items():
                if term in response.lower():
                    confidence = value
                    break
            
            # Default confidence if still not found
            if confidence == 0.0:
                confidence = 0.5
        
        # Extract alternatives
        in_alternatives_section = False
        alternative_lines = []
        
        for i, line in enumerate(lines):
            # Check if we're entering alternatives section
            for indicator in alternative_indicators:
                if indicator in line.lower():
                    in_alternatives_section = True
                    # Skip the header line
                    continue
            
            # Check if we're exiting alternatives section
            if in_alternatives_section:
                # Exit if we hit another section
                for indicator in conclusion_indicators + confidence_indicators:
                    if indicator in line.lower():
                        in_alternatives_section = False
                        break
                
                # Add line to alternatives if still in section
                if in_alternatives_section and line.strip():
                    alternative_lines.append(line.strip())
        
        # Process alternative lines
        if alternative_lines:
            # Try to split alternatives by numbering or bullet points
            current_alternative = ""
            
            for line in alternative_lines:
                # Check if line starts a new alternative
                if line.strip().startswith(('1.', '2.', '3.', '-', '*', '•')):
                    # Save previous alternative if exists
                    if current_alternative:
                        alternatives.append({
                            "conclusion": current_alternative,
                            "confidence": 0.0  # Default confidence
                        })
                    
                    # Start new alternative
                    current_alternative = line.strip().lstrip('123456789.-*• ')
                else:
                    # Continue current alternative
                    if current_alternative:
                        current_alternative += " " + line.strip()
                    else:
                        current_alternative = line.strip()
            
            # Add final alternative if exists
            if current_alternative:
                alternatives.append({
                    "conclusion": current_alternative,
                    "confidence": 0.0  # Default confidence
                })
        
        # If no alternatives were found but we want to include them
        if not alternatives and self.include_alternatives:
            # Generate some alternatives with lower confidence
            if confidence > 0.3:
                alternatives.append({
                    "conclusion": f"Alternative view: {self._generate_alternative(conclusion)}",
                    "confidence": max(0.1, confidence - 0.3)
                })
        
        # Limit number of alternatives
        alternatives = alternatives[:self.max_alternatives]
        
        # If no conclusion was found, use the first paragraph as conclusion
        if not conclusion:
            paragraphs = response.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip() and not any(indicator in paragraph.lower() for indicator in conclusion_indicators + confidence_indicators + alternative_indicators):
                    conclusion = paragraph.strip()
                    break
            
            # If still no conclusion, use the entire response
            if not conclusion:
                conclusion = response.strip()
        
        # Map confidence to DecisionConfidence enum
        confidence_enum = DecisionConfidence.MEDIUM
        if confidence >= 0.8:
            confidence_enum = DecisionConfidence.VERY_HIGH
        elif confidence >= 0.6:
            confidence_enum = DecisionConfidence.HIGH
        elif confidence >= 0.4:
            confidence_enum = DecisionConfidence.MEDIUM
        else:
            confidence_enum = DecisionConfidence.LOW
        
        return conclusion, confidence_enum, alternatives
    
    def _generate_alternative(self, conclusion: str) -> str:
        """Generate an alternative to the conclusion.
        
        Args:
            conclusion: Original conclusion
            
        Returns:
            str: Alternative conclusion
        """
        # Simple heuristic to generate an alternative
        negation_phrases = [
            "It's possible that",
            "An alternative view is that",
            "Some might argue that",
            "Another perspective is that",
            "Considering different factors,"
        ]
        
        import random
        prefix = random.choice(negation_phrases)
        
        # Modify the conclusion slightly
        words = conclusion.split()
        if len(words) > 5:
            # Swap some words or modify the sentence structure
            mid_point = len(words) // 2
            alternative = " ".join(words[mid_point:] + words[:mid_point])
        else:
            # For short conclusions, just add a qualifier
            alternative = conclusion
        
        return f"{prefix} {alternative}"

# Create reasoning engine instance
reasoning_engine = ReasoningEngine()
