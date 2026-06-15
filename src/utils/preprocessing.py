"""
Preprocessing Module for HSeg Text Segmentation

Handles sentence tokenization and sentence-ID mapping for Portuguese and English texts.
"""

import spacy
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SentenceTokenizer:
    """
    Tokenizes text into sentences and creates ID mappings.
    
    Uses spaCy for robust sentence boundary detection with
    custom rules for administrative text (abbreviations, numbering, etc.).
    """
    
    def __init__(self, model_name: str = "pt_core_news_lg"):
        """
        Initialize sentence tokenizer.
        
        Args:
            model_name: spaCy model name (e.g. pt_core_news_lg, en_core_web_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            raise
        
        # Add custom sentence boundary rules for Portuguese administrative text
        if "pt" in model_name:
            self._add_custom_rules()
    
    def _add_custom_rules(self):
        """Add custom rules for sentence boundary detection.
        
        Adds a spaCy component that merges agenda-item number tokens (e.g. "4.")
        with the following sentence so they are not treated as isolated sentences.
        The component is inserted BEFORE the parser so that sentence-start flags
        are still writable.
        """
        import re
        from spacy.language import Language
        
        numbered_pattern = r'''
            ^\s*                          # Optional whitespace at start
            (?:                           # Non-capturing group for prefixes
                [-–—]+\s*                 # Dashes (regular, en-dash, em-dash)
                |[•·*+]\s*                # Bullet points
                |§\s*                     # Section symbol
            )?                            # Prefix is optional
            (?:                           # Main numbering patterns
                \d+(?:\.\d+)*[\.\)]\s*    # Numbers with optional decimal parts: "1.", "2.1.", "12.2."
                |[a-z]\)\s*               # Letters with parenthesis: "a)", "b)"
                |[ivxlcdm]+[\.\)]\s*      # Roman numerals: "i.", "ii)", "iv."
                |[-–—]+\s*                # Just dashes: "--", "---"
            )
            $                             # End of string
        '''
        numbered_regex = re.compile(numbered_pattern, re.IGNORECASE | re.VERBOSE)
        
        if not Language.has_factory("merge_numbered_sentences"):
            @Language.component("merge_numbered_sentences")
            def merge_numbered_sentences_component(doc):
                # Collect sentence-boundary tokens that belong to numbered-only lines
                # and mark them as non-sentence-starts so the following content is
                # merged with the agenda number header.
                i = 0
                while i < len(doc):
                    # Find the extent of a "newline-terminated" token group
                    # by collecting tokens until the next newline character.
                    span_start = i
                    span_tokens = []
                    while i < len(doc):
                        span_tokens.append(doc[i])
                        if doc[i].text.endswith("\n") or doc[i].is_space:
                            i += 1
                            break
                        i += 1
                    
                    span_text = "".join(t.text for t in span_tokens).strip()
                    
                    if numbered_regex.match(span_text) and i < len(doc):
                        # The token that starts the next line should not begin a sentence
                        doc[i].is_sent_start = False
                
                return doc
                
        if not self.nlp.has_pipe("merge_numbered_sentences"):
            # Insert BEFORE the parser so sentence-start flags are still writable
            if self.nlp.has_pipe("parser"):
                self.nlp.add_pipe("merge_numbered_sentences", before="parser")
            else:
                self.nlp.add_pipe("merge_numbered_sentences", last=True)
    
    def tokenize(self, text: str) -> Dict[str, str]:
        """
        Tokenize text into sentences and create ID mapping.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary mapping sentence IDs to sentence text
            {
                "S1": "First sentence.",
                "S2": "Second sentence.",
                ...
            }
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for tokenization")
            return {}
        
        doc = self.nlp(text)
        sentences = {}
        
        for i, sent in enumerate(doc.sents, start=1):
            sentence_text = sent.text.strip()
            if sentence_text:  # Skip empty sentences
                sentences[f"S{i}"] = sentence_text
        
        logger.info(f"Tokenized text into {len(sentences)} sentences")
        return sentences
    
    def get_char_offsets(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        Get character offsets for each sentence.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping sentence IDs to (start_char, end_char) tuples
            {
                "S1": (0, 25),
                "S2": (26, 52),
                ...
            }
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for offset calculation")
            return {}
        
        doc = self.nlp(text)
        offsets = {}
        
        for i, sent in enumerate(doc.sents, start=1):
            if sent.text.strip():  # Skip empty sentences
                offsets[f"S{i}"] = (sent.start_char, sent.end_char)
        
        return offsets
    
    def tokenize_with_offsets(self, text: str) -> Tuple[Dict[str, str], Dict[str, Tuple[int, int]]]:
        """
        Tokenize text and get offsets in one pass (more efficient).
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentences dict, offsets dict)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for tokenization")
            return {}, {}
        
        doc = self.nlp(text)
        sentences = {}
        offsets = {}
        
        for i, sent in enumerate(doc.sents, start=1):
            sentence_text = sent.text.strip()
            if sentence_text:
                sent_id = f"S{i}"
                sentences[sent_id] = sentence_text
                offsets[sent_id] = (sent.start_char, sent.end_char)
        
        logger.info(f"Tokenized text into {len(sentences)} sentences with offsets")
        return sentences, offsets


def format_for_llm(sentences: Dict[str, str], max_sentences: Optional[int] = None) -> str:
    """
    Format sentence mapping for LLM input.
    
    Args:
        sentences: Dictionary of sentence ID -> sentence text
        max_sentences: Optional limit on number of sentences to format
        
    Returns:
        Formatted string with each sentence on a line:
        S1: "First sentence."
        S2: "Second sentence."
        ...
    """
    formatted = []
    items = list(sentences.items())
    
    if max_sentences:
        items = items[:max_sentences]
    
    for sent_id, sent_text in items:
        # Escape quotes in text to prevent JSON issues
        escaped_text = sent_text.replace('"', '\\"')
        formatted.append(f'{sent_id}: "{escaped_text}"')
    
    return "\n".join(formatted)


def extract_text_span(full_text: str, start_char: int, end_char: int) -> str:
    """
    Extract text span from full text using character offsets.
    
    Args:
        full_text: Complete document text
        start_char: Start character position
        end_char: End character position
        
    Returns:
        Extracted text span
    """
    if start_char < 0 or end_char > len(full_text) or start_char >= end_char:
        logger.warning(f"Invalid character offsets: start={start_char}, end={end_char}, text_len={len(full_text)}")
        return ""
    
    return full_text[start_char:end_char]
