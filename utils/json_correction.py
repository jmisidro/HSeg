"""
JSON Correction Module for HSeg

Handles JSON validation and correction when LLM outputs invalid JSON.
Provides corrected prompts with format examples to guide models toward valid output.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from hseg.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# Generic JSON correction prompt
GENERIC_JSON_CORRECTION_PROMPT = """The previous response was not valid JSON. Please fix it.

IMPORTANT - You MUST return ONLY a valid JSON array, nothing else:
- Start with [ and end with ]
- No text before or after the JSON
- All strings must use double quotes
- No trailing commas

INVALID output example (DO NOT DO THIS):
```
Here is the JSON:
[{...}, {...}]
Some explanation here.
```

VALID output example (DO THIS):
[{"field1": "value1", "field2": ["item1", "item2"], "field3": "value3"}, {"field1": "value1", "field2": ["item1"], "field3": "value3"}]

ORIGINAL (INVALID) RESPONSE:
{invalid_json}

Please provide the corrected JSON array:"""


# Stage 1 specific correction prompt
STAGE1_JSON_CORRECTION_PROMPT = """The previous response was not valid JSON. Please fix it.

IMPORTANT - You MUST return ONLY a valid JSON array with agenda items:
- Start with [ and end with ]
- No text before or after the JSON
- All strings must use double quotes
- No trailing commas
- Each item must have: "item_title", "start_id", "end_id"

VALID output example (DO THIS):
[
  {"item_title": "1. PEDIDO DE MARCAÇÃO", "start_id": "S1", "end_id": "S5"},
  {"item_title": "2. APROVAÇÃO DE ATA", "start_id": "S6", "end_id": "S10"}
]

ORIGINAL (INVALID) RESPONSE:
{invalid_json}

Please provide the corrected JSON array with all agenda items:"""


# Stage 2 specific correction prompt
STAGE2_JSON_CORRECTION_PROMPT = """The previous response was not valid JSON. Please fix it.

IMPORTANT - You MUST return ONLY a valid JSON array with subjects:
- Start with [ and end with ]
- No text before or after the JSON
- All strings must use double quotes
- No trailing commas
- Each item must have: "theme", "topics" (as a list), "start_id", "end_id"

VALID output example (DO THIS):
[
  {"theme": "Pedido de transporte escolar", "topics": ["Education and Vocational Training"], "start_id": "S1", "end_id": "S5"},
  {"theme": "Aprovação orçamental", "topics": ["General Administration, Finance, and Human Resources"], "start_id": "S6", "end_id": "S10"}
]

ORIGINAL (INVALID) RESPONSE:
{invalid_json}

Please provide the corrected JSON array with all subjects:"""


class JSONCorrector:
    """
    Handles JSON validation and correction when LLM outputs invalid JSON.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize JSON corrector.
        
        Args:
            llm_interface: LLM interface for generation
        """
        self.llm = llm_interface
        logger.info("Initialized JSONCorrector")
    
    def extract_json_with_correction(
        self,
        response: str,
        correction_prompt_template: str = GENERIC_JSON_CORRECTION_PROMPT,
        max_correction_attempts: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Try to extract JSON from response, and if invalid, ask model to correct it.
        
        Args:
            response: LLM response text
            correction_prompt_template: Template for correction prompt with {invalid_json} placeholder
            max_correction_attempts: Maximum number of correction attempts
            
        Returns:
            Parsed JSON array
            
        Raises:
            ValueError: If extraction fails after correction attempts
        """
        # Try to extract JSON from original response
        try:
            result = self.llm.extract_json(response)
            logger.info("JSON extracted successfully from original response")
            return result
        except ValueError as initial_error:
            logger.warning(f"Initial JSON extraction failed: {initial_error}")
        
        # If original extraction failed, try correction
        invalid_json = response[:500]  # Preview of invalid response
        
        for attempt in range(max_correction_attempts):
            logger.info(f"Attempting JSON correction (attempt {attempt + 1}/{max_correction_attempts})")
            
            # Create correction prompt
            correction_prompt = correction_prompt_template.format(invalid_json=invalid_json)
            
            try:
                # Generate correction
                corrected_response = self.llm.generate(correction_prompt)
                
                # Try to extract JSON from corrected response
                result = self.llm.extract_json(corrected_response)
                logger.info(f"JSON extracted successfully after correction (attempt {attempt + 1})")
                return result
                
            except Exception as e:
                logger.warning(f"Correction attempt {attempt + 1} failed: {e}")
                invalid_json = corrected_response[:500]  # Use corrected response for next attempt
                
                if attempt == max_correction_attempts - 1:
                    raise ValueError(
                        f"Failed to extract valid JSON after {max_correction_attempts} correction attempts. "
                        f"Last error: {e}"
                    )
    
    @staticmethod
    def validate_stage1_items(items: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate Stage 1 agenda items structure.
        
        Args:
            items: List of items to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for i, item in enumerate(items):
            if 'item_title' not in item:
                errors.append(f"Item {i}: Missing 'item_title'")
            if 'start_id' not in item:
                errors.append(f"Item {i}: Missing 'start_id'")
            if 'end_id' not in item:
                errors.append(f"Item {i}: Missing 'end_id'")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_stage2_subjects(subjects: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate Stage 2 subjects structure.
        
        Args:
            subjects: List of subjects to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for i, subject in enumerate(subjects):
            if 'theme' not in subject:
                errors.append(f"Subject {i}: Missing 'theme'")
            if 'topics' not in subject:
                errors.append(f"Subject {i}: Missing 'topics'")
            if 'start_id' not in subject:
                errors.append(f"Subject {i}: Missing 'start_id'")
            if 'end_id' not in subject:
                errors.append(f"Subject {i}: Missing 'end_id'")
            elif not isinstance(subject.get('topics'), list):
                errors.append(f"Subject {i}: 'topics' must be a list")
        
        return len(errors) == 0, errors
