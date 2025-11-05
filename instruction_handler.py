"""
User Instruction Handler with LangChain ReAct Integration
Provides intelligent instruction parsing and context-aware decision making.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum


class InstructionType(Enum):
    """Types of user instructions."""
    FILTER_DATA = "filter_data"
    SELECT_COLUMNS = "select_columns"
    CLEAN_ONLY = "clean_only"
    PREDICT_TARGET = "predict_target"
    ANALYZE_SPECIFIC = "analyze_specific"
    CUSTOM_PIPELINE = "custom_pipeline"
    NO_TRAINING = "no_training"
    FOCUS_FEATURE = "focus_feature"


class UserInstructionHandler:
    """
    Handles user instructions and integrates with LangChain ReAct for intelligent processing.
    This allows the agent to understand and execute user-specific requirements.
    """
    
    def __init__(self):
        self.instructions = {}
        self.parsed_intent = {}
        self.execution_plan = []
        self.context_history = []
        
        # Instruction patterns for parsing
        self.patterns = {
            'filter': [
                r"only.*?(house|home|property|real estate)",
                r"filter.*?(where|by|for)",
                r"(just|only).*?(prices?|values?|amounts?)",
                r"exclude.*?(column|feature|row)",
                r"include.*?(only|just)",
            ],
            'clean': [
                r"(just|only).*?clean",
                r"clean.*?(data|dataset).*?only",
                r"no.*?(training|model|prediction)",
                r"prepare.*?data",
            ],
            'predict': [
                r"predict",
                r"forecast",
                r"estimate",
                r"model.*?for",
                r"train.*?model",
                r"target.*?(variable|column)",
            ],
            'analyze': [
                r"analyze",
                r"analysis",
                r"correlation",
                r"show.*?(statistics|summary|insight)",
                r"explore.*?(relationship|trend)",
            ],
            'focus': [
                r"focus",
                r"concentrate.*?on",
                r"show.*?importance",
                r"feature.*?importance",
                r"prioritize.*?",
                r"emphasize.*?",
            ]
        }
        
    def get_user_instructions(self, callback_func) -> Dict[str, Any]:
        """
        Get instructions from user via GUI callback.
        
        Args:
            callback_func: Function to display instruction dialog and get input
            
        Returns:
            Dictionary containing user instructions and preferences
        """
        instructions = {
            'raw_input': '',
            'data_scope': 'full',  # full, filtered, sampled
            'columns_to_use': 'all',  # all, specific list, or exclude list
            'target_column': None,  # specific target or auto-detect
            'pipeline_steps': 'auto',  # auto, or specific list
            'cleaning_level': 'auto',  # minimal, standard, aggressive, auto
            'training_required': True,  # whether to train models
            'analysis_focus': [],  # specific aspects to focus on
            'output_requirements': [],  # what outputs user wants
            'constraints': {}  # any constraints (time, memory, etc.)
        }
        
        # Get user input through callback
        user_input = callback_func()
        if user_input:
            instructions['raw_input'] = user_input
            self._parse_instructions(instructions)
        
        self.instructions = instructions
        return instructions
    
    def _parse_instructions(self, instructions: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse user instructions using pattern matching and context awareness.
        Returns a list of parsed instruction components.
        """
        # Handle both string and dict inputs
        if isinstance(instructions, str):
            text = instructions.lower()
            instruction_dict = {'raw_input': instructions}
        else:
            text = instructions['raw_input'].lower()
            instruction_dict = instructions
        
        parsed_results = []
        
        # Parse data scope / filtering
        if any(re.search(pattern, text) for pattern in self.patterns['filter']):
            parsed_results.append({
                'type': InstructionType.FILTER_DATA,
                'details': self._extract_filter_criteria(text)
            })
        
        # Parse cleaning requirements
        if any(re.search(pattern, text) for pattern in self.patterns['clean']):
            if 'only' in text or 'just' in text or 'no train' in text or 'no model' in text:
                parsed_results.append({
                    'type': InstructionType.CLEAN_ONLY,
                    'details': {'skip_training': True}
                })
        
        # Parse prediction target
        if any(re.search(pattern, text) for pattern in self.patterns['predict']):
            target = self._extract_target_column(text)
            if target:
                parsed_results.append({
                    'type': InstructionType.PREDICT_TARGET,
                    'details': {'target_column': target}
                })
        
        # Parse analysis focus
        if any(re.search(pattern, text) for pattern in self.patterns['analyze']):
            parsed_results.append({
                'type': InstructionType.ANALYZE_SPECIFIC,
                'details': {'focus': self._extract_analysis_focus(text)}
            })
        
        # Parse focus areas
        if any(re.search(pattern, text) for pattern in self.patterns['focus']):
            parsed_results.append({
                'type': InstructionType.FOCUS_FEATURE,
                'details': {'areas': self._extract_focus_areas(text)}
            })
        
        # Check for no training instruction
        if 'no train' in text or 'no model' in text or "don't train" in text:
            parsed_results.append({
                'type': InstructionType.NO_TRAINING,
                'details': {'skip_training': True}
            })
        
        return parsed_results
    
    def _extract_filter_criteria(self, text: str) -> Dict[str, Any]:
        """Extract filtering criteria from user text."""
        criteria = {}
        
        # Extract column names
        if 'house' in text or 'home' in text or 'property' in text:
            criteria['domain'] = 'real_estate'
            
        # Extract value filters
        price_match = re.search(r'price.*?(?:above|below|between|over|under)\s+(\d+(?:,\d+)?)', text)
        if price_match:
            criteria['price_filter'] = price_match.group(1)
        
        # Extract column selections
        if 'only' in text:
            words_after_only = re.search(r'only\s+(\w+(?:\s+\w+)*)', text)
            if words_after_only:
                criteria['include_columns'] = words_after_only.group(1).split()
        
        return criteria
    
    def _extract_target_column(self, text: str) -> Optional[str]:
        """Extract target column from user text."""
        # Common patterns - more flexible
        target_patterns = [
            r'predict\s+(?:the\s+)?house\s+(\w+)',  # "predict house prices"
            r'predict\s+(?:the\s+)?(\w+)',
            r'forecast\s+(?:the\s+)?(\w+)',
            r'estimate\s+(?:the\s+)?(\w+)',
            r'target\s+(?:is|=|:|variable|column)?\s*:?\s*(\w+)',
            r'model\s+for\s+(\w+)',
            r'(\w+)\s+prediction',
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, text)
            if match:
                target = match.group(1)
                # Filter out common words that aren't column names
                if target not in ['the', 'a', 'an', 'this', 'that', 'house', 'data']:
                    # Handle plurals -> singular (prices -> price)
                    if target.endswith('s') and len(target) > 4:
                        target = target[:-1]
                    return target
        
        return None
    
    def _extract_analysis_focus(self, text: str) -> List[str]:
        """Extract what aspects to focus analysis on."""
        focus_areas = []
        
        if 'correlation' in text:
            focus_areas.append('correlation')
        if 'distribution' in text:
            focus_areas.append('distribution')
        if 'outlier' in text:
            focus_areas.append('outliers')
        if 'missing' in text or 'null' in text:
            focus_areas.append('missing_values')
        if 'pattern' in text or 'trend' in text:
            focus_areas.append('patterns')
        
        return focus_areas
    
    def _extract_focus_areas(self, text: str) -> List[str]:
        """Extract specific features or areas to focus on."""
        focus_areas = []
        
        # Extract after 'focus on' or 'concentrate on'
        focus_match = re.search(r'(?:focus|concentrate|prioritize|emphasize)\s+(?:on\s+)?(.+?)(?:\.|,|$)', text)
        if focus_match:
            focus_text = focus_match.group(1)
            # Split by 'and' or commas
            areas = re.split(r'\s+and\s+|,\s*', focus_text)
            focus_areas.extend([area.strip() for area in areas])
        
        return focus_areas
    
    def create_execution_plan(self, raw_instruction: str, 
                            parsed_instructions: List[Dict[str, Any]], 
                            dataset_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create execution plan based on instructions and dataset profile.
        Uses ReAct-like reasoning to plan steps.
        Args:
            raw_instruction: The raw instruction string from user
            parsed_instructions: List of parsed instruction components
            dataset_profile: Optional dataset metadata
        """
        if dataset_profile is None:
            dataset_profile = {}
        
        execution_plan = {
            'reasoning': {
                'intent': '',
                'strategy': '',
                'observations': []
            },
            'steps': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'raw_instruction': raw_instruction
            }
        }
        
        # Thought: What does the user want?
        intent_reasoning = self._reason_about_intent(raw_instruction, parsed_instructions)
        execution_plan['reasoning']['intent'] = intent_reasoning
        
        # Determine strategy based on parsed instructions
        has_filter = any(item['type'] == InstructionType.FILTER_DATA for item in parsed_instructions)
        clean_only = any(item['type'] in [InstructionType.CLEAN_ONLY, InstructionType.NO_TRAINING] 
                        for item in parsed_instructions)
        has_prediction = any(item['type'] == InstructionType.PREDICT_TARGET for item in parsed_instructions)
        
        # Build strategy
        strategy_parts = []
        if has_filter:
            strategy_parts.append("Filter data to user-specified subset")
        strategy_parts.append("Apply appropriate data cleaning")
        if not clean_only and has_prediction:
            strategy_parts.append("Train prediction models")
        elif not clean_only:
            strategy_parts.append("Perform exploratory analysis")
        execution_plan['reasoning']['strategy'] = " → ".join(strategy_parts)
        
        # Action: Data Loading
        execution_plan['steps'].append({
            'action': 'load_data',
            'thought': 'Load dataset to begin processing',
            'observation': 'Dataset loaded successfully',
            'parameters': {}
        })
        
        # Action: Apply filters if needed
        if has_filter:
            filter_details = next((item['details'] for item in parsed_instructions 
                                 if item['type'] == InstructionType.FILTER_DATA), {})
            execution_plan['steps'].append({
                'action': 'filter_data',
                'thought': f"User requested filtered data: {filter_details}",
                'observation': 'Filters will be applied to dataset',
                'parameters': filter_details
            })
        
        # Action: Column selection
        select_col = next((item for item in parsed_instructions 
                          if item['type'] == InstructionType.SELECT_COLUMNS), None)
        if select_col:
            execution_plan['steps'].append({
                'action': 'select_columns',
                'thought': f"User specified column selection",
                'observation': 'Will limit analysis to selected columns',
                'parameters': select_col['details']
            })
        
        # Action: Data cleaning
        execution_plan['steps'].append({
            'action': 'clean_data',
            'thought': 'Apply data cleaning based on dataset characteristics',
            'observation': 'Data will be cleaned and preprocessed',
            'parameters': {'level': 'standard'}
        })
        
        # Observation: Check if training is needed
        if not clean_only and has_prediction:
            # Get target column
            target_col = next((item['details'].get('target_column') for item in parsed_instructions 
                             if item['type'] == InstructionType.PREDICT_TARGET), None)
            
            # Action: Feature engineering
            execution_plan['steps'].append({
                'action': 'feature_engineering',
                'thought': 'Create features to improve model performance',
                'observation': 'Feature engineering completed',
                'parameters': {}
            })
            
            # Action: Model training
            execution_plan['steps'].append({
                'action': 'train_models',
                'thought': f"Train models to predict {target_col or 'target variable'}",
                'observation': 'Models trained and evaluated',
                'parameters': {'target': target_col}
            })
            
            # Action: Evaluation
            execution_plan['steps'].append({
                'action': 'evaluate_models',
                'thought': 'Assess model performance and select best model',
                'observation': 'Best model selected based on metrics',
                'parameters': {}
            })
        elif clean_only:
            # Observation: User only wants cleaning
            execution_plan['reasoning']['observations'].append(
                'User requested data preparation only, skipping model training'
            )
        
        # Action: Visualization & Export
        execution_plan['steps'].append({
            'action': 'visualize_and_export',
            'thought': 'Create visualizations and export results',
            'observation': 'Results exported successfully',
            'parameters': {
                'include_models': has_prediction and not clean_only
            }
        })
        
        self.execution_plan = execution_plan
        return execution_plan
    
    def _reason_about_intent(self, raw_instruction: str, parsed_instructions: List[Dict[str, Any]] = None) -> str:
        """
        ReAct-style reasoning about user intent.
        Args:
            raw_instruction: The raw text instruction from user
            parsed_instructions: List of parsed instruction components
        """
        if not raw_instruction:
            return "No specific instructions provided. Will execute standard pipeline with intelligent defaults."
        
        if parsed_instructions is None:
            parsed_instructions = []
        
        reasoning = f"User instruction: '{raw_instruction}'\n\n"
        reasoning += "🧠 THOUGHT:\n"
        
        # Analyze each parsed component
        for item in parsed_instructions:
            # Skip if item is not a dict (defensive programming)
            if not isinstance(item, dict):
                continue
                
            inst_type = item.get('type')
            if not inst_type:
                continue
                
            details = item.get('details', {})
            
            if inst_type == InstructionType.FILTER_DATA:
                reasoning += "- User wants to work with filtered/subset of data\n"
                if details:
                    reasoning += f"  Filter criteria: {details}\n"
            
            elif inst_type == InstructionType.CLEAN_ONLY:
                reasoning += "- User only needs data cleaning/preparation, no model training\n"
            
            elif inst_type == InstructionType.PREDICT_TARGET:
                target = details.get('target_column', 'unknown')
                reasoning += f"- User wants to predict target column: {target}\n"
            
            elif inst_type == InstructionType.ANALYZE_SPECIFIC:
                focus = details.get('focus', [])
                reasoning += f"- Analysis should focus on: {focus}\n"
            
            elif inst_type == InstructionType.FOCUS_FEATURE:
                areas = details.get('areas', [])
                reasoning += f"- User wants to focus on: {', '.join(areas) if areas else 'feature analysis'}\n"
            
            elif inst_type == InstructionType.NO_TRAINING:
                reasoning += "- User explicitly requested no model training\n"
        
        reasoning += "\n⚡ ACTION: Execute customized pipeline based on user requirements."
        
        return reasoning
    
    def _reason_about_cleaning(self, instructions: Dict[str, Any], 
                              profile: Dict[str, Any]) -> str:
        """
        Reason about appropriate cleaning strategy.
        """
        reasoning = "Cleaning strategy: "
        
        level = instructions.get('cleaning_level', 'auto')
        if level == 'auto':
            if profile.get('quality_score', 100) < 70:
                reasoning += "Dataset has low quality score, will apply aggressive cleaning."
            elif profile.get('missing_ratio', 0) > 0.2:
                reasoning += "High missing value ratio detected, will use advanced imputation."
            else:
                reasoning += "Dataset is relatively clean, will apply standard cleaning."
        else:
            reasoning += f"User specified {level} cleaning level."
        
        if instructions.get('analysis_focus'):
            reasoning += f" Extra focus on: {', '.join(instructions['analysis_focus'])}"
        
        return reasoning
    
    def apply_filters_to_data(self, data, criteria: Dict[str, Any]):
        """
        Apply filtering criteria to dataset.
        """
        filtered_data = data.copy()
        
        # Domain-based filtering
        if 'domain' in criteria:
            domain = criteria['domain']
            if domain == 'real_estate':
                # Look for real estate related columns
                relevant_cols = [col for col in filtered_data.columns 
                               if any(kw in col.lower() for kw in 
                                     ['house', 'home', 'property', 'price', 'sqft', 
                                      'bedroom', 'bathroom', 'lot', 'year_built'])]
                if relevant_cols:
                    filtered_data = filtered_data[relevant_cols]
        
        # Column inclusion
        if 'include_columns' in criteria:
            cols_to_include = criteria['include_columns']
            available_cols = [col for col in filtered_data.columns 
                            if any(kw in col.lower() for kw in cols_to_include)]
            if available_cols:
                filtered_data = filtered_data[available_cols]
        
        # Price filtering
        if 'price_filter' in criteria:
            price_cols = [col for col in filtered_data.columns 
                         if 'price' in col.lower() or 'value' in col.lower()]
            if price_cols:
                # Apply numeric filter on price columns
                pass  # Implement specific filtering logic
        
        return filtered_data
    
    def get_execution_summary(self) -> str:
        """
        Generate summary of what will be executed.
        """
        if not self.execution_plan:
            return "No execution plan created yet."
        
        summary = "📋 Execution Plan Summary:\n\n"
        
        actions = [step for step in self.execution_plan if step['type'] == 'action']
        
        summary += f"Total steps: {len(actions)}\n\n"
        
        for i, step in enumerate(actions, 1):
            summary += f"{i}. {step['step'].replace('_', ' ').title()}\n"
            summary += f"   Reasoning: {step['reasoning']}\n\n"
        
        if self.instructions.get('raw_input'):
            summary += f"\nBased on user instruction: \"{self.instructions['raw_input']}\"\n"
        
        return summary
    
    def log_context(self, step: str, observation: str):
        """
        Log context for ReAct-style reasoning.
        """
        self.context_history.append({
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'observation': observation
        })
    
    def get_context_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of execution context for Q&A.
        """
        return self.context_history
