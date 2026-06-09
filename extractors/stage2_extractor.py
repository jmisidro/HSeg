"""
Stage 2: Subject Extraction

Extracts discussion subjects within agenda items using LLM with
sentence-ID grounding. Identifies theme, topic classification, and boundaries.
"""

import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, create_model
from hseg.llm.llm_interface import LLMInterface
from hseg.utils.preprocessing import format_for_llm
from hseg.utils.json_correction import JSONCorrector, STAGE2_JSON_CORRECTION_PROMPT

logger = logging.getLogger(__name__)


class _SubjectBase(BaseModel):
    """Base fields for a discussion subject returned by Stage 2."""
    start_id: str
    end_id: str


def get_stage2_schema(extract_theme: bool, extract_topics: bool) -> dict:
    """Dynamically generate the Pydantic JSON schema based on requested fields."""
    fields: Dict[str, Any] = {}
    if extract_theme:
        fields['theme'] = (str, ...)
    if extract_topics:
        fields['topics'] = (List[str], ...)
        
    SubjectModel = create_model(
        '_SubjectDynamic',
        __base__=_SubjectBase,
        **fields
    )
    SubjectListModel = create_model(
        '_SubjectListDynamic',
        subjects=(List[SubjectModel], ...)
    )
    return SubjectListModel.model_json_schema()


# Topic categories for CitiLink dataset
TOPIC_CATEGORIES = [
    "General Administration, Finance, and Human Resources",
    "Environment",
    "Energy and Telecommunications",
    "Traffic, Transport, and Communications",
    "Education and Vocational Training",
    "Heritage",
    "Culture",
    "Science",
    "Health",
    "Animal Protection",
    "Sports",
    "Social Action",
    "Housing",
    "Civil Protection",
    "Municipal Police",
    "Public Works",
    "Spatial Planning",
    "Private Works",
    "Economic Activities",
    "External Cooperation and International Relations",
    "Communication and Public Relations",
    "Other"
]


STAGE2_PROMPT_TEMPLATE = """You are an expert at analyzing Portuguese municipal council meeting minutes. Your task is to identify discussion subjects within a specific agenda item.

CONTEXT:
Agenda Item: {item_title}

INSTRUCTIONS:
1. A "subject" is a cohesive subtopic of discussion within the agenda item
2. Each subject typically includes:
   - Introduction/presentation of the matter (e.g., "Pelo Senhor Presidente foi presente...")
   - Discussion (if any)
   - Decision/deliberation (e.g., "...deliberou por unanimidade aprovar...")
3. Multiple subjects can exist within one agenda item — a typical agenda item has 1 to 5 subjects
4. Look for transitions that signal a new subject:
   - "Pelo Senhor Presidente foi presente a reunião..." / "De seguida..."
   - New proposals, requests, or reports being introduced
   - Different entities, contracts, or matters being discussed
5. Each distinct matter discussed (different request, different approval, different report) is a separate subject

IMPORTANT:
{municipality_hint}

INPUT FORMAT:
Below are the sentences from this agenda item, with IDs:

{sentence_mapping}

OUTPUT FORMAT:
For each subject you identify, provide:{output_format_instructions}
{start_id_instruction}
{end_id_instruction}

Return your answer as a JSON array:
[
  {{
{example_json_fields}
  }},
  ...
]
{topic_categories_section}
IMPORTANT CONSTRAINTS:
- Subjects must NOT overlap
- Subjects should cover the entire agenda item (minimize gaps)
- Only use sentence IDs from the provided list{topic_constraint}
- If the agenda item has only one subject, return an array with one element
- Each distinct matter (pedido, aprovação, proposta, etc.) should be its own subject

Now, identify the subjects:"""


STAGE2_FEW_SHOT_EXAMPLES = """
EXAMPLE:
CONTEXT:
Agenda Item: "3. PROCESSOS E REQUERIMENTOS DIVERSOS"

INPUT:
S1: "Pelo Senhor Presidente foi presente a reunião o pedido de transporte do Agrupamento de Escolas..."
S2: "O Sr. Vereador quis saber se o Agrupamento já teria feito o pedido à Câmara..."
S3: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar o pedido."
S4: "Pelo Senhor Presidente foi presente a reunião o pedido de utilização do Fórum Cultural..."
S5: "O Sr. Presidente esclareceu que a Câmara costuma ceder às instituições..."
S6: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar a utilização."

OUTPUT:
[
  {{
    "theme": "Pedido de transporte do Agrupamento de Escolas",
    "topics": ["Traffic, Transport, and Communications", "Education and Vocational Training"],
    "start_id": "S1",
    "end_id": "S3"
  }},
  {{
    "theme": "Utilização do Fórum Cultural",
    "topics": ["Culture", "General Administration, Finance, and Human Resources"],
    "start_id": "S4",
    "end_id": "S6"
  }}
]

---

"""

STAGE2_FEW_SHOT_EXAMPLES_WITH_THEME_NO_TOPICS = """
EXAMPLE:
CONTEXT:
Agenda Item: "3. PROCESSOS E REQUERIMENTOS DIVERSOS"

INPUT:
S1: "Pelo Senhor Presidente foi presente a reunião o pedido de transporte do Agrupamento de Escolas..."
S2: "O Sr. Vereador quis saber se o Agrupamento já teria feito o pedido à Câmara..."
S3: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar o pedido."
S4: "Pelo Senhor Presidente foi presente a reunião o pedido de utilização do Fórum Cultural..."
S5: "O Sr. Presidente esclareceu que a Câmara costuma ceder às instituições..."
S6: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar a utilização."

OUTPUT:
[
  {{
    "theme": "Pedido de transporte do Agrupamento de Escolas",
    "start_id": "S1",
    "end_id": "S3"
  }},
  {{
    "theme": "Utilização do Fórum Cultural",
    "start_id": "S4",
    "end_id": "S6"
  }}
]

---

"""

STAGE2_FEW_SHOT_EXAMPLES_NO_THEME_WITH_TOPICS = """
EXAMPLE:
CONTEXT:
Agenda Item: "3. PROCESSOS E REQUERIMENTOS DIVERSOS"

INPUT:
S1: "Pelo Senhor Presidente foi presente a reunião o pedido de transporte do Agrupamento de Escolas..."
S2: "O Sr. Vereador quis saber se o Agrupamento já teria feito o pedido à Câmara..."
S3: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar o pedido."
S4: "Pelo Senhor Presidente foi presente a reunião o pedido de utilização do Fórum Cultural..."
S5: "O Sr. Presidente esclareceu que a Câmara costuma ceder às instituições..."
S6: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar a utilização."

OUTPUT:
[
  {{
    "topics": ["Traffic, Transport, and Communications", "Education and Vocational Training"],
    "start_id": "S1",
    "end_id": "S3"
  }},
  {{
    "topics": ["Culture", "General Administration, Finance, and Human Resources"],
    "start_id": "S4",
    "end_id": "S6"
  }}
]

---

"""

STAGE2_FEW_SHOT_EXAMPLES_NO_THEME_AND_NO_TOPICS = """
EXAMPLE:
CONTEXT:
Agenda Item: "3. PROCESSOS E REQUERIMENTOS DIVERSOS"

INPUT:
S1: "Pelo Senhor Presidente foi presente a reunião o pedido de transporte do Agrupamento de Escolas..."
S2: "O Sr. Vereador quis saber se o Agrupamento já teria feito o pedido à Câmara..."
S3: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar o pedido."
S4: "Pelo Senhor Presidente foi presente a reunião o pedido de utilização do Fórum Cultural..."
S5: "O Sr. Presidente esclareceu que a Câmara costuma ceder às instituições..."
S6: "Ponderado e analisado o assunto o Executivo Municipal deliberou por unanimidade aprovar a utilização."

OUTPUT:
[
  {{
    "start_id": "S1",
    "end_id": "S3"
  }},
  {{
    "start_id": "S4",
    "end_id": "S6"
  }}
]

---

"""


class Stage2Extractor:
    """
    Extracts subjects within agenda items using LLM.
    
    Uses sentence-ID grounding to ensure accurate boundary detection
    without text hallucination.
    """
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        use_few_shot: bool = True,
        max_retries: int = 5
    ):
        self.llm = llm_interface
        self.corrector = JSONCorrector(llm_interface)
        self.use_few_shot = use_few_shot
        self.max_retries = max_retries
        logger.info(f"Initialized Stage2Extractor (few_shot={use_few_shot})")
    
    def create_prompt(
        self,
        item_title: str,
        sentences: Dict[str, str],
        municipality: str = "",
        extract_theme: bool = False,
        extract_topics: bool = False
    ) -> str:
        sentence_mapping = format_for_llm(sentences)
        
        mun = municipality.lower()
        if mun and "porto" in mun:
            municipality_hint = "- The first sentence often contains the agenda item title. This title text is factually part of the first subject."
        else:
            municipality_hint = "- Do NOT include the agenda item title (which often appears as the first sentence, or first two sentences if the first is just a number) in your first subject's text, unless the subject heavily depends on it."

        output_format_instructions = ""
        example_json_fields_list = []
        fmt_idx = 1
        
        if extract_theme:
            output_format_instructions += f"\n{fmt_idx}. theme: A brief Portuguese description of the specific matter discussed. Use concise noun phrases that summarize the subject."
            example_json_fields_list.append('    "theme": "Description of subject",')
            fmt_idx += 1
            
        if extract_topics:
            output_format_instructions += f"\n{fmt_idx}. topics: List of relevant topic categories from the list below (usually 1-2 topics)"
            example_json_fields_list.append('    "topics": ["Topic category 1"],')
            fmt_idx += 1
            
        start_id_instruction = f"{fmt_idx}. start_id: The sentence ID where the subject begins"
        end_id_instruction = f"{fmt_idx + 1}. end_id: The sentence ID where the subject ends"
        
        example_json_fields_list.extend([
            '    "start_id": "S1",',
            '    "end_id": "S5"'
        ])
        example_json_fields = "\n".join(example_json_fields_list)
        
        if extract_topics:
            topic_categories_str = "\n".join(f"- {cat}" for cat in TOPIC_CATEGORIES)
            topic_categories_section = f"\nTOPIC CATEGORIES (choose the most relevant ones):\n{topic_categories_str}\n"
            topic_constraint = "\n- Topics must be from the provided category list"
        else:
            topic_categories_section = ""
            topic_constraint = ""
        
        prompt = STAGE2_PROMPT_TEMPLATE.format(
            item_title=item_title,
            sentence_mapping=sentence_mapping,
            municipality_hint=municipality_hint,
            output_format_instructions=output_format_instructions,
            start_id_instruction=start_id_instruction,
            end_id_instruction=end_id_instruction,
            example_json_fields=example_json_fields,
            topic_categories_section=topic_categories_section,
            topic_constraint=topic_constraint
        )
        
        if self.use_few_shot:
            if extract_theme and extract_topics:
                prompt = STAGE2_FEW_SHOT_EXAMPLES + prompt
            elif not extract_theme and not extract_topics:
                prompt = STAGE2_FEW_SHOT_EXAMPLES_NO_THEME_AND_NO_TOPICS + prompt
            elif extract_theme and not extract_topics:
                prompt = STAGE2_FEW_SHOT_EXAMPLES_WITH_THEME_NO_TOPICS + prompt
            elif not extract_theme and extract_topics:
                prompt = STAGE2_FEW_SHOT_EXAMPLES_NO_THEME_WITH_TOPICS + prompt
        
        return prompt
    
    def extract(
        self,
        item_title: str,
        sentences: Dict[str, str],
        municipality: str = "",
        extract_theme: bool = False,
        extract_topics: bool = False
    ) -> List[Dict]:
        if not sentences:
            logger.warning("Empty sentences provided to Stage2Extractor")
            return []
        
        logger.info(f"Extracting subjects from agenda item: {item_title[:50]}...")
        
        prompt = self.create_prompt(
            item_title=item_title,
            sentences=sentences,
            municipality=municipality,
            extract_theme=extract_theme,
            extract_topics=extract_topics
        )
        
        dynamic_schema = get_stage2_schema(extract_theme, extract_topics)
        
        try:
            response = self.llm.generate_with_retry(
                prompt,
                max_retries=self.max_retries,
                json_schema=dynamic_schema,
            )

            import json as _json
            try:
                parsed = _json.loads(response)
                if isinstance(parsed, dict) and 'subjects' in parsed:
                    logger.debug("Unwrapping structured-output wrapper {'subjects': [...]}")
                    response = _json.dumps(parsed['subjects'])
            except (_json.JSONDecodeError, TypeError):
                pass

            subjects = self.corrector.extract_json_with_correction(
                response,
                correction_prompt_template=STAGE2_JSON_CORRECTION_PROMPT,
                max_correction_attempts=2
            )
            
            logger.info(f"Extracted {len(subjects)} subjects")
            return subjects
            
        except Exception as e:
            logger.error(f"Stage 2 extraction failed for item '{item_title}': {e}")
            raise ValueError(f"Failed to extract subjects: {e}")
    
    def validate_output(
        self,
        subjects: List[Dict],
        min_sentence_id: int,
        max_sentence_id: int,
        extract_theme: bool = False,
        extract_topics: bool = False
    ) -> List[str]:
        errors = []
        
        for i, subject in enumerate(subjects):
            if extract_theme and 'theme' not in subject:
                errors.append(f"Subject {i}: Missing 'theme'")
            if extract_topics and 'topics' not in subject:
                errors.append(f"Subject {i}: Missing 'topics'")
                
            if 'start_id' not in subject:
                errors.append(f"Subject {i}: Missing 'start_id'")
            if 'end_id' not in subject:
                errors.append(f"Subject {i}: Missing 'end_id'")
                continue
            
            start_id = subject['start_id']
            end_id = subject['end_id']
            
            if not start_id.startswith('S') or not end_id.startswith('S'):
                errors.append(f"Subject {i}: Invalid ID format")
                continue
            
            try:
                start_num = int(start_id[1:])
                end_num = int(end_id[1:])
            except ValueError:
                errors.append(f"Subject {i}: Invalid ID numbers")
                continue
            
            if start_num < min_sentence_id or start_num > max_sentence_id:
                errors.append(f"Subject {i}: start_id out of range")
            if end_num < min_sentence_id or end_num > max_sentence_id:
                errors.append(f"Subject {i}: end_id out of range")
            
            if start_num > end_num:
                errors.append(f"Subject {i}: start_id > end_id")
            
            if extract_topics and not isinstance(subject.get('topics'), list):
                errors.append(f"Subject {i}: 'topics' must be a list")
        
        sorted_subjects = sorted(subjects, key=lambda x: int(x.get('start_id', 'S0')[1:]))
        for i in range(len(sorted_subjects) - 1):
            try:
                end_current = int(sorted_subjects[i]['end_id'][1:])
                start_next = int(sorted_subjects[i + 1]['start_id'][1:])
                if end_current >= start_next:
                    errors.append(f"Overlap between subjects {i} and {i + 1}")
            except (KeyError, ValueError):
                continue
        
        return errors
