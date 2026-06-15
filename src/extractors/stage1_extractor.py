"""
Stage 1: Section Extraction Modules

Contains extractors for:
1. Municipal council meeting minutes (Stage1Extractor)
2. Wikipedia articles (WikiStage1Extractor)
"""

import json as _json
import logging
import re as _re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from hseg.llm.llm_interface import LLMInterface
from hseg.utils.preprocessing import format_for_llm
from hseg.utils.json_correction import JSONCorrector, STAGE1_JSON_CORRECTION_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minutes (Stage 1) - Pydantic schemas & prompts
# ---------------------------------------------------------------------------

class _AgendaItem(BaseModel):
    """A single agenda item returned by Stage 1."""
    item_title: str
    start_id: str
    end_id: str


class _AgendaItemList(BaseModel):
    """Wrapper so the schema describes a top-level array of agenda items."""
    items: List[_AgendaItem]


_STAGE1_JSON_SCHEMA = _AgendaItemList.model_json_schema()


STAGE1_PROMPT_TEMPLATE = """You are an expert at analyzing Portuguese municipal council meeting minutes. Your task is to identify agenda items in the document.

INSTRUCTIONS:
1. Agenda items are the MAJOR top-level discussion topics, usually numbered ("1.", "2.", "3.", etc.)
2. Sub-items (e.g., "2.1.", "2.2.", "3.a)") are part of their parent item — do NOT split them into separate entries
3. Each agenda item spans all its sub-items and associated discussion
4. Look for patterns like:
   - Numbered titles: "1. PEDIDO DE...", "2. APROVAÇÃO...", etc.
   - Formal transitions between major topics
5. A typical Portuguese council meeting has between 3 and 20 major agenda items — if you find significantly more, you are likely splitting sub-items incorrectly

INPUT FORMAT:
Below is the full meeting minute document, with each sentence assigned an ID (S1, S2, S3, ...).

{sentence_mapping}

OUTPUT FORMAT:
For each agenda item you identify, provide:
1. item_title: The title or description of the agenda item (the top-level number and title)
2. start_id: The sentence ID where the item begins (e.g., S5)
3. end_id: The sentence ID where the item ends (e.g., S12)

Return your answer as a JSON array:
[
  {{"item_title": "Title of item 1", "start_id": "S1", "end_id": "S10"}},
  {{"item_title": "Title of item 2", "start_id": "S11", "end_id": "S20"}},
  ...
]

IMPORTANT:
- Items must NOT overlap (end_id of item N < start_id of item N+1)
- Items should cover the entire document (no gaps)
- Only use sentence IDs that exist in the input (S1 to S{max_id})
- Sub-items (2.1, 2.2, etc.) belong inside their parent item — do NOT list them separately
- Group all sub-item discussion under the parent agenda item

Now, identify the agenda items:"""


STAGE1_FEW_SHOT_EXAMPLES = """
EXAMPLE:
Input (abbreviated):
S1: "ATA DA REUNIÃO ORDINÁRIA DA CÂMARA MUNICIPAL DE ALANDROAL"
S2: "REALIZADA EM 22 DE OUTUBRO DE 2021"
...
S50: "1. PERÍODO DE ANTES DA ORDEM DO DIA"
S51: "O Senhor Presidente deu início aos trabalhos saudando os presentes..."
...
S85: "2. ORDEM DO DIA"
S86: "2.1. BALANCETE MENSAL"
S87: "Foi presente o balancete referente ao mês anterior."
...
S92: "2.2. APROVAÇÃO DE REGULAMENTO"
S93: "Foi presente a proposta de regulamento..."
...
S95: "3. PROCESSOS E REQUERIMENTOS DIVERSOS"
S96: "Pelo Senhor Presidente foi presente o pedido de transporte escolar..."
...
S150: "Ponderado e analisado o assunto..."

Output:
[
  {{"item_title": "1. PERÍODO DE ANTES DA ORDEM DO DIA", "start_id": "S50", "end_id": "S84"}},
  {{"item_title": "2. ORDEM DO DIA", "start_id": "S85", "end_id": "S94"}},
  {{"item_title": "3. PROCESSOS E REQUERIMENTOS DIVERSOS", "start_id": "S95", "end_id": "S150"}}
]

Note: Sub-items 2.1 and 2.2 are grouped under "2. ORDEM DO DIA", NOT listed separately.

---

"""


# ---------------------------------------------------------------------------
# Wikipedia (Stage 1) - Prompts & few-shot examples
# ---------------------------------------------------------------------------

STAGE1_ROOT_PROMPT_TEMPLATE = """\
You are an expert at analyzing Wikipedia articles. Your task is to identify the \
top-level sections in the article.

INSTRUCTIONS:
1. Top-level sections are the MAJOR divisions of the article \
(e.g. "History", "Overview", "Background", "Usage", "Architecture").
2. Section headings appear in the text as **standalone lines** (a short phrase \
on its own line, not ending with a period and not part of a regular sentence). \
Use them as your primary signal for section boundaries.
3. Sub-sections (e.g. "Early history", "Modern usage") belong INSIDE their \
parent section — do NOT list them as separate top-level items.
4. A typical Wikipedia article has between 3 and 15 major sections.
5. Each section spans all its sub-sections and associated text.
6. You must ONLY output a valid JSON array. Do NOT output any conversational text, preamble, or postscript. Do not complain if the article topic differs from the example.

INPUT FORMAT:
Below is the Wikipedia article body with each sentence assigned an ID \
(S1, S2, S3, …).  Standalone heading lines are included as their own sentences:

{sentence_mapping}

OUTPUT FORMAT:
Return a JSON array where each object represents a top-level section:
[
  {{"section_title": "Exact heading text", "start_id": "S1", "end_id": "S10"}},
  ...
]

CONSTRAINTS:
- Output valid JSON only. No preamble or postscript.
- Use the heading line's sentence ID as ``start_id`` for that section.
- Sections must NOT overlap.
- Only use sentence IDs that exist in the input (S1 to S{max_id}).
- Sub-sections belong inside their parent — do NOT list them separately.

Now, identify the top-level sections for the provided article:\
"""


STAGE1_ROOT_FEW_SHOT = """\
EXAMPLE (identifying top-level sections of a full Wikipedia article about \
singular 'they'):

Input (abbreviated — S-IDs assigned to each sentence/heading line):
S1: "Singular 'they' is the use in English of the pronoun 'they'..."
S5: "The singular 'they' had emerged by the 14th century..."
S7: "Inflected forms and derivative pronouns."
S8: "Though the singular 'they' permits a singular antecedent..."
S18: "Usage."
S19: "'They' with a singular antecedent has remained in common use..."
S23: "Trend to prescription of generic 'he' from 19th century."
S41: "Contemporary use of 'he' to refer to a generic antecedent."
S51: "Trend to gender-neutral language."
S93: "Acceptability and prescriptive guidance."
S97: "Usage guidance in British–American style guides."
S181: "Grammatical and logical analysis."
S213: "Cognitive efficiency."
S223: "Comparison with other pronouns."

Output:
[
  {{"section_title": "Inflected forms and derivative pronouns", "start_id": "S7", "end_id": "S17"}},
  {{"section_title": "Usage", "start_id": "S18", "end_id": "S92"}},
  {{"section_title": "Acceptability and prescriptive guidance", "start_id": "S93", "end_id": "S180"}},
  {{"section_title": "Grammatical and logical analysis", "start_id": "S181", "end_id": "S212"}},
  {{"section_title": "Cognitive efficiency", "start_id": "S213", "end_id": "S222"}},
  {{"section_title": "Comparison with other pronouns", "start_id": "S223", "end_id": "S231"}}
]

Note: Sub-section headings like "Trend to prescription of generic 'he'" (S23),
"Contemporary use of 'he'" (S41), and "Usage guidance in British–American style
guides" (S97) are NOT listed separately — they belong INSIDE their parent
sections ("Usage" and "Acceptability..." respectively).

---

"""


STAGE1_RECURSIVE_PROMPT_TEMPLATE = """\
You are an expert at analyzing Wikipedia articles. Your task is to identify \
the immediate sub-sections within the following section.

CONTEXT:
Parent section: "{parent_title}"

INSTRUCTIONS:
1. Identify every immediate sub-section heading that appears inside this section.
2. Sub-section headings appear as **standalone short lines** (not ending with \
a period mid-sentence, not part of a regular paragraph).
3. Each sub-section spans from its heading to the line before the next heading \
at the same or higher level.
4. Do NOT recurse into nested sub-sub-sections — only list the direct children \
of "{parent_title}".
5. A typical section has 2 to 12 immediate sub-sections.
6. You must ONLY output a valid JSON array. Do NOT output any conversational text, preamble, or postscript. Do not complain if the article topic differs from the example.

INPUT FORMAT:
Sentences from the "{parent_title}" section, with IDs (S1 … S{max_id}):

{sentence_mapping}

OUTPUT FORMAT:
Return a JSON array where each object represents one immediate sub-section:
[
  {{"section_title": "Exact heading text", "start_id": "S1", "end_id": "S10"}},
  ...
]

CONSTRAINTS:
- Output valid JSON only. No preamble or postscript.
- Use the heading line's sentence ID as ``start_id`` for that sub-section.
- Sub-sections must NOT overlap.
- Only use sentence IDs that exist in the input (S1 to S{max_id}).
- If the section has NO internal headings (it is a leaf section), return an \
empty array: []

Now, identify the immediate sub-sections inside "{parent_title}" for the provided article:\
"""


STAGE1_RECURSIVE_FEW_SHOT = """\
EXAMPLE (identifying immediate sub-sections within the "History" section of \
an article about St. Elisabeth Cathedral, Košice):

Context: Parent section "History"

Input (abbreviated — IDs are relative to the start of the "History" section):
S1: "Original church."
S2: "The oldest Košice church probably originated in the middle of the 11th century..."
S3: "It was built in Romanesque style at the same place as the current church."
S8: "This parish church burned down around 1380..."
S9: "First construction stage – end of 14th century until 1420."
S10: "The fire which destroyed St. Elisabeth church in 1380 was a good opportunity..."
S16: "Second construction stage – 1420–1440."
S17: "A discontinuous innovation in a cathedral conception was brought by new construction..."
S23: "The third construction stage – 1440–1462."
S27: "The fourth construction stage – 1462–1490."
S33: "Final construction stage – 1491–1508."
S42: "Reformation period."
S52: "Baroque period."
S57: "Fabry's reconstruction in 1858–1863."
S68: "Major reconstruction in 1877–1896."
S83: "Major reconstruction 1978 until today."

Output:
[
  {{"section_title": "Original church", "start_id": "S1", "end_id": "S8"}},
  {{"section_title": "First construction stage – end of 14th century until 1420", "start_id": "S9", "end_id": "S15"}},
  {{"section_title": "Second construction stage – 1420–1440", "start_id": "S16", "end_id": "S22"}},
  {{"section_title": "The third construction stage – 1440–1462", "start_id": "S23", "end_id": "S26"}},
  {{"section_title": "The fourth construction stage – 1462–1490", "start_id": "S27", "end_id": "S32"}},
  {{"section_title": "Final construction stage – 1491–1508", "start_id": "S33", "end_id": "S41"}},
  {{"section_title": "Reformation period", "start_id": "S42", "end_id": "S51"}},
  {{"section_title": "Baroque period", "start_id": "S52", "end_id": "S56"}},
  {{"section_title": "Fabry's reconstruction in 1858–1863", "start_id": "S57", "end_id": "S67"}},
  {{"section_title": "Major reconstruction in 1877–1896", "start_id": "S68", "end_id": "S82"}},
  {{"section_title": "Major reconstruction 1978 until today", "start_id": "S83", "end_id": "S99"}}
]

Note: Each standalone heading line marks the start of a new immediate
sub-section.  Deeper nested headings (depth-4+) are NOT listed here; they
belong inside their respective sub-section.

---

"""


# ---------------------------------------------------------------------------
# Extractors implementation
# ---------------------------------------------------------------------------

class Stage1Extractor:
    """
    Extracts agenda items from meeting minutes using LLM.
    
    Uses sentence-ID grounding to ensure accurate boundary detection
    without text hallucination.
    """
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        use_few_shot: bool = True,
        max_retries: int = 5,
        use_sliding_window: bool = False,
        max_sentences_per_chunk: int = 150,
        overlap_sentences: int = 30,
    ):
        self.llm = llm_interface
        self.corrector = JSONCorrector(llm_interface)
        self.use_few_shot = use_few_shot
        self.max_retries = max_retries
        self.use_sliding_window = use_sliding_window
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        logger.info(
            f"Initialized Stage1Extractor "
            f"(few_shot={use_few_shot}, sliding_window={use_sliding_window}, "
            f"chunk_size={max_sentences_per_chunk}, overlap={overlap_sentences})"
        )
    
    def create_prompt(self, sentences: Dict[str, str]) -> str:
        sentence_mapping = format_for_llm(sentences)
        max_id = len(sentences)
        
        prompt = STAGE1_PROMPT_TEMPLATE.format(
            sentence_mapping=sentence_mapping,
            max_id=max_id
        )
        
        if self.use_few_shot:
            prompt = STAGE1_FEW_SHOT_EXAMPLES + prompt
        
        return prompt
    
    @staticmethod
    def _try_unwrap(text: str) -> str:
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict) and 'items' in parsed:
                logger.debug("Unwrapping structured-output wrapper {'items': [...]}")
                return _json.dumps(parsed['items'])
            if isinstance(parsed, list):
                return text
        except _json.JSONDecodeError:
            pass

        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            try:
                parsed = _json.loads(text[first_brace:last_brace + 1])
                if isinstance(parsed, dict) and 'items' in parsed:
                    logger.debug("Unwrapping structured-output wrapper (brace extraction)")
                    return _json.dumps(parsed['items'])
            except _json.JSONDecodeError:
                pass

        m = _re.search(r'"items"\s*:\s*(\[.*\])', text, _re.DOTALL)
        if m:
            try:
                items_list = _json.loads(m.group(1))
                if isinstance(items_list, list):
                    logger.debug("Unwrapping structured-output wrapper (regex extraction)")
                    return _json.dumps(items_list)
            except _json.JSONDecodeError:
                pass

        return text

    def _split_into_chunks(
        self,
        sentences: Dict[str, str],
        boundary_sentence_nums: Optional[set] = None,
    ) -> List[Tuple[Dict[str, str], int]]:
        sentence_nums = sorted(int(k[1:]) for k in sentences.keys())
        total = len(sentence_nums)

        if total <= self.max_sentences_per_chunk:
            return [(sentences, 1)]

        chunks: List[Tuple[Dict[str, str], int]] = []
        start_idx = 0

        while start_idx < total:
            end_idx = min(start_idx + self.max_sentences_per_chunk, total)

            if end_idx < total and boundary_sentence_nums:
                search_limit = max(
                    start_idx + (self.max_sentences_per_chunk // 2),
                    end_idx - (self.overlap_sentences * 3),
                )

                best_split_idx = -1
                for idx in range(end_idx - 1, search_limit - 1, -1):
                    if sentence_nums[idx] in boundary_sentence_nums:
                        best_split_idx = idx
                        break

                if best_split_idx != -1:
                    end_idx = best_split_idx
                    next_start_idx = best_split_idx
                    logger.debug(
                        f"Strategic split at S{sentence_nums[best_split_idx]} "
                        f"(ground-truth item title boundary)"
                    )
                else:
                    next_start_idx = end_idx - self.overlap_sentences
            elif end_idx < total:
                next_start_idx = end_idx - self.overlap_sentences
            else:
                next_start_idx = total

            chunk_nums = sentence_nums[start_idx:end_idx]
            chunk_sents: Dict[str, str] = {
                f"S{rel}": sentences[f"S{abs_num}"]
                for rel, abs_num in enumerate(chunk_nums, 1)
            }
            chunks.append((chunk_sents, chunk_nums[0]))

            start_idx = next_start_idx

        return chunks

    def _extract_chunk(
        self,
        chunk_sentences: Dict[str, str],
        chunk_start_num: int,
        chunk_idx: int,
        total_chunks: int,
    ) -> List[Dict]:
        logger.info(
            f"  Chunk {chunk_idx}/{total_chunks}: "
            f"{len(chunk_sentences)} sentences "
            f"(S{chunk_start_num}–S{chunk_start_num + len(chunk_sentences) - 1})"
        )
        prompt = self.create_prompt(chunk_sentences)
        response = self.llm.generate_with_retry(prompt, max_retries=self.max_retries)
        response = self._try_unwrap(response)

        items = self.corrector.extract_json_with_correction(
            response,
            correction_prompt_template=STAGE1_JSON_CORRECTION_PROMPT,
            max_correction_attempts=2,
        )

        converted = []
        for item in items:
            try:
                item = dict(item)
                item['start_id'] = f"S{int(item['start_id'][1:]) + chunk_start_num - 1}"
                item['end_id']   = f"S{int(item['end_id'][1:])   + chunk_start_num - 1}"
                converted.append(item)
            except (KeyError, ValueError, IndexError):
                converted.append(item)
        return converted

    @staticmethod
    def _merge_chunk_results(
        chunk_results: List[List[Dict]],
        overlap_sentences: int,
    ) -> List[Dict]:
        all_items: List[Dict] = []
        for chunk in chunk_results:
            all_items.extend(chunk)
        all_items.sort(key=lambda x: int(x.get('start_id', 'S0')[1:]))

        dedup_threshold = max(3, overlap_sentences // 2)
        merged: List[Dict] = []

        for item in all_items:
            try:
                start_num = int(item['start_id'][1:])
                end_num   = int(item['end_id'][1:])
            except (KeyError, ValueError):
                merged.append(item)
                continue

            if merged:
                prev = merged[-1]
                try:
                    prev_start = int(prev['start_id'][1:])
                    prev_end   = int(prev['end_id'][1:])
                except (KeyError, ValueError):
                    merged.append(item)
                    continue

                if abs(start_num - prev_start) <= dedup_threshold:
                    merged[-1] = {
                        'item_title': (
                            prev['item_title']
                            if len(prev.get('item_title', '')) >= len(item.get('item_title', ''))
                            else item['item_title']
                        ),
                        'start_id': f"S{min(prev_start, start_num)}",
                        'end_id':   f"S{max(prev_end,   end_num)}",
                    }
                    continue

            merged.append(item)

        return merged

    def extract(
        self,
        sentences: Dict[str, str],
        boundary_sentence_nums: Optional[set] = None,
    ) -> List[Dict]:
        if not sentences:
            logger.warning("Empty sentences provided to Stage1Extractor")
            return []

        logger.info(f"Extracting agenda items from {len(sentences)} sentences")

        try:
            if not self.use_sliding_window:
                chunks = [(sentences, 1)]
            else:
                chunks = self._split_into_chunks(sentences, boundary_sentence_nums=boundary_sentence_nums)

            if len(chunks) == 1:
                chunk_sents, chunk_start = chunks[0]
                response = self.llm.generate_with_retry(
                    self.create_prompt(chunk_sents),
                    max_retries=self.max_retries,
                )
                response = self._try_unwrap(response)
                items = self.corrector.extract_json_with_correction(
                    response,
                    correction_prompt_template=STAGE1_JSON_CORRECTION_PROMPT,
                    max_correction_attempts=2,
                )
            else:
                logger.info(
                    f"Document has {len(sentences)} sentences — "
                    f"using sliding window: {len(chunks)} chunks "
                    f"of ≤{self.max_sentences_per_chunk} sentences "
                    f"(overlap={self.overlap_sentences})"
                )
                chunk_results = [
                    self._extract_chunk(cs, start, idx, len(chunks))
                    for idx, (cs, start) in enumerate(chunks, 1)
                ]
                items = self._merge_chunk_results(chunk_results, self.overlap_sentences)

            logger.info(f"Extracted {len(items)} agenda items")
            return items

        except Exception as e:
            logger.error(f"Stage 1 extraction failed: {e}")
            raise ValueError(f"Failed to extract agenda items: {e}")
    
    def validate_output(self, items: List[Dict], max_sentence_id: int) -> List[str]:
        errors = []
        
        for i, item in enumerate(items):
            if 'item_title' not in item:
                errors.append(f"Item {i}: Missing 'item_title'")
            if 'start_id' not in item:
                errors.append(f"Item {i}: Missing 'start_id'")
            if 'end_id' not in item:
                errors.append(f"Item {i}: Missing 'end_id'")
                continue
            
            start_id = item['start_id']
            end_id = item['end_id']
            
            if not start_id.startswith('S') or not end_id.startswith('S'):
                errors.append(f"Item {i}: Invalid ID format")
                continue
            
            try:
                start_num = int(start_id[1:])
                end_num = int(end_id[1:])
            except ValueError:
                errors.append(f"Item {i}: Invalid ID numbers")
                continue
            
            if start_num < 1 or start_num > max_sentence_id:
                errors.append(f"Item {i}: start_id out of range")
            if end_num < 1 or end_num > max_sentence_id:
                errors.append(f"Item {i}: end_id out of range")
            
            if start_num > end_num:
                errors.append(f"Item {i}: start_id > end_id")
        
        sorted_items = sorted(items, key=lambda x: int(x.get('start_id', 'S0')[1:]))
        for i in range(len(sorted_items) - 1):
            try:
                end_current = int(sorted_items[i]['end_id'][1:])
                start_next = int(sorted_items[i + 1]['start_id'][1:])
                if end_current >= start_next:
                    errors.append(f"Overlap between items {i} and {i + 1}")
            except (KeyError, ValueError):
                continue
        
        return errors


class WikiStage1Extractor:
    """
    Extracts sections from a Wikipedia article (or sub-section) using an LLM.
    Supports two modes via the optional ``parent_title`` argument (Root and Recursive).
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        use_few_shot: bool = True,
        max_retries: int = 5,
        use_sliding_window: bool = False,
        max_sentences_per_chunk: int = 150,
        overlap_sentences: int = 30,
    ):
        self.llm = llm_interface
        self.corrector = JSONCorrector(llm_interface)
        self.use_few_shot = use_few_shot
        self.max_retries = max_retries
        self.use_sliding_window = use_sliding_window
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        logger.info(
            "Initialized WikiStage1Extractor (few_shot=%s, sliding_window=%s)",
            use_few_shot, use_sliding_window,
        )

    def create_prompt(
        self,
        sentences: Dict[str, str],
        parent_title: Optional[str] = None,
    ) -> str:
        sentence_mapping = format_for_llm(sentences)
        max_id = len(sentences)

        if parent_title:
            prompt = STAGE1_RECURSIVE_PROMPT_TEMPLATE.format(
                parent_title=parent_title,
                sentence_mapping=sentence_mapping,
                max_id=max_id,
            )
            if self.use_few_shot:
                prompt = STAGE1_RECURSIVE_FEW_SHOT + prompt
        else:
            prompt = STAGE1_ROOT_PROMPT_TEMPLATE.format(
                sentence_mapping=sentence_mapping,
                max_id=max_id,
            )
            if self.use_few_shot:
                prompt = STAGE1_ROOT_FEW_SHOT + prompt

        return prompt

    @staticmethod
    def _try_unwrap(text: str) -> str:
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict):
                for key in ("items", "sections", "agenda_items"):
                    if key in parsed:
                        return _json.dumps(parsed[key])
            if isinstance(parsed, list):
                return text
        except _json.JSONDecodeError:
            pass

        for key in ("items", "sections", "agenda_items"):
            m = _re.search(rf'"{key}"\s*:\s*(\[.*\])', text, _re.DOTALL)
            if m:
                try:
                    lst = _json.loads(m.group(1))
                    if isinstance(lst, list):
                        return _json.dumps(lst)
                except _json.JSONDecodeError:
                    pass

        return text

    def _split_into_chunks(
        self, sentences: Dict[str, str]
    ) -> List[Tuple[Dict[str, str], int]]:
        sentence_nums = sorted(int(k[1:]) for k in sentences.keys())
        total = len(sentence_nums)
        if total <= self.max_sentences_per_chunk:
            return [(sentences, 1)]
        stride = self.max_sentences_per_chunk - self.overlap_sentences
        chunks: List[Tuple[Dict[str, str], int]] = []
        start_idx = 0
        while start_idx < total:
            end_idx = min(start_idx + self.max_sentences_per_chunk, total)
            chunk_nums = sentence_nums[start_idx:end_idx]
            chunk_sents: Dict[str, str] = {
                f"S{rel}": sentences[f"S{abs_num}"]
                for rel, abs_num in enumerate(chunk_nums, 1)
            }
            chunks.append((chunk_sents, chunk_nums[0]))
            if end_idx >= total:
                break
            start_idx += stride
        return chunks

    def _extract_chunk(
        self,
        chunk_sentences: Dict[str, str],
        chunk_start_num: int,
        chunk_idx: int,
        total_chunks: int,
        parent_title: Optional[str] = None,
    ) -> List[Dict]:
        logger.info(
            "  Chunk %d/%d: %d sentences (S%d–S%d)",
            chunk_idx, total_chunks, len(chunk_sentences),
            chunk_start_num, chunk_start_num + len(chunk_sentences) - 1,
        )
        prompt = self.create_prompt(chunk_sentences, parent_title=parent_title)
        response = self.llm.generate_with_retry(prompt, max_retries=self.max_retries)
        response = self._try_unwrap(response)
        items = self.corrector.extract_json_with_correction(
            response,
            correction_prompt_template=STAGE1_JSON_CORRECTION_PROMPT,
            max_correction_attempts=2,
        )
        converted = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("Chunk item is not a dict, skipping: %s", item)
                continue
            item = dict(item)
            if "section_title" in item and "item_title" not in item:
                item["item_title"] = item.pop("section_title")
            try:
                item["start_id"] = f"S{int(item['start_id'][1:]) + chunk_start_num - 1}"
                item["end_id"]   = f"S{int(item['end_id'][1:])   + chunk_start_num - 1}"
            except (KeyError, ValueError, IndexError, TypeError):
                pass
            converted.append(item)
        return converted

    @staticmethod
    def _merge_chunk_results(
        chunk_results: List[List[Dict]],
        overlap_sentences: int,
    ) -> List[Dict]:
        all_items: List[Dict] = []
        for chunk in chunk_results:
            all_items.extend(chunk)
        all_items.sort(key=lambda x: int(x.get("start_id", "S0")[1:]))
        dedup_threshold = max(3, overlap_sentences // 2)
        merged: List[Dict] = []
        for item in all_items:
            try:
                start_num = int(item["start_id"][1:])
                end_num   = int(item["end_id"][1:])
            except (KeyError, ValueError):
                merged.append(item)
                continue
            if merged:
                prev = merged[-1]
                try:
                    prev_start = int(prev["start_id"][1:])
                    prev_end   = int(prev["end_id"][1:])
                except (KeyError, ValueError):
                    merged.append(item)
                    continue
                if abs(start_num - prev_start) <= dedup_threshold:
                    merged[-1] = {
                        "item_title": (
                            prev["item_title"]
                            if len(prev.get("item_title", "")) >= len(item.get("item_title", ""))
                            else item["item_title"]
                        ),
                        "start_id": f"S{min(prev_start, start_num)}",
                        "end_id":   f"S{max(prev_end,   end_num)}",
                    }
                    continue
            merged.append(item)
        return merged

    def extract(
        self,
        sentences: Dict[str, str],
        parent_title: Optional[str] = None,
    ) -> List[Dict]:
        if not sentences:
            logger.warning("Empty sentences provided to WikiStage1Extractor")
            return []

        mode = f"recursive (parent='{parent_title}')" if parent_title else "root"
        logger.info(
            "Extracting sections from %d sentences [%s]", len(sentences), mode
        )

        try:
            if not self.use_sliding_window:
                chunks = [(sentences, 1)]
            else:
                chunks = self._split_into_chunks(sentences)

            if len(chunks) == 1:
                chunk_sents, _ = chunks[0]
                response = self.llm.generate_with_retry(
                    self.create_prompt(chunk_sents, parent_title=parent_title),
                    max_retries=self.max_retries,
                )
                response = self._try_unwrap(response)
                items = self.corrector.extract_json_with_correction(
                    response,
                    correction_prompt_template=STAGE1_JSON_CORRECTION_PROMPT,
                    max_correction_attempts=2,
                )
                valid_items = []
                for item in items:
                    if not isinstance(item, dict):
                        logger.warning("Item is not a dict, skipping: %s", item)
                        continue
                    if "section_title" in item and "item_title" not in item:
                        item["item_title"] = item.pop("section_title")
                    valid_items.append(item)
                items = valid_items
            else:
                logger.info(
                    "Sliding window: %d chunks of ≤%d sentences (overlap=%d)",
                    len(chunks), self.max_sentences_per_chunk, self.overlap_sentences,
                )
                chunk_results = [
                    self._extract_chunk(cs, start, idx, len(chunks), parent_title=parent_title)
                    for idx, (cs, start) in enumerate(chunks, 1)
                ]
                items = self._merge_chunk_results(chunk_results, self.overlap_sentences)

            logger.info("Extracted %d sections (Stage 1, %s)", len(items), mode)
            return items

        except Exception as exc:
            logger.error("Stage 1 extraction failed [%s]: %s", mode, exc)
            raise ValueError(f"Failed to extract sections: {exc}") from exc

    def validate_output(self, items: List[Dict], max_sentence_id: int) -> List[str]:
        errors = []
        for i, item in enumerate(items):
            if "item_title" not in item:
                errors.append(f"Item {i}: Missing 'item_title'")
            if "start_id" not in item:
                errors.append(f"Item {i}: Missing 'start_id'")
            if "end_id" not in item:
                errors.append(f"Item {i}: Missing 'end_id'")
                continue
            try:
                start_num = int(item["start_id"][1:])
                end_num   = int(item["end_id"][1:])
            except ValueError:
                errors.append(f"Item {i}: Invalid ID numbers")
                continue
            if not (1 <= start_num <= max_sentence_id):
                errors.append(f"Item {i}: start_id out of range")
            if not (1 <= end_num <= max_sentence_id):
                errors.append(f"Item {i}: end_id out of range")
            if start_num > end_num:
                errors.append(f"Item {i}: start_id > end_id")
        sorted_items = sorted(items, key=lambda x: int(x.get("start_id", "S0")[1:]))
        for i in range(len(sorted_items) - 1):
            try:
                end_current = int(sorted_items[i]["end_id"][1:])
                start_next  = int(sorted_items[i + 1]["start_id"][1:])
                if end_current >= start_next:
                    errors.append(f"Overlap between items {i} and {i + 1}")
            except (KeyError, ValueError):
                continue
        return errors
