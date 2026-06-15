"""
Post-processing and Validation Module for HSeg

Validates and converts LLM outputs to final annotation format.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class SegmentationValidator:
    """
    Validates and post-processes segmentation results.
    
    Handles:
    - Validation of boundaries
    - Conversion from sentence IDs to character offsets
    - Text span extraction
    - Gap-filling and overlap merging heuristics
    """
    
    def __init__(
        self,
        sentences: Dict[str, str],
        char_offsets: Dict[str, Tuple[int, int]],
        full_text: str
    ):
        """
        Initialize validator.
        
        Args:
            sentences: Mapping of sentence ID -> sentence text
            char_offsets: Mapping of sentence ID -> (start_char, end_char)
            full_text: Complete document text
        """
        self.sentences = sentences
        self.char_offsets = char_offsets
        self.full_text = full_text
        logger.info(f"Initialized validator for {len(sentences)} sentences")
    
    def sentence_ids_to_char_offsets(
        self,
        start_id: str,
        end_id: str
    ) -> Tuple[int, int]:
        """
        Convert sentence ID range to character offsets.
        
        Args:
            start_id: Starting sentence ID (e.g., "S5")
            end_id: Ending sentence ID (e.g., "S12")
            
        Returns:
            Tuple of (start_char, end_char)
            
        Raises:
            ValueError: If sentence IDs are invalid
        """
        if start_id not in self.char_offsets:
            raise ValueError(f"Invalid start_id: {start_id}")
        if end_id not in self.char_offsets:
            raise ValueError(f"Invalid end_id: {end_id}")
        
        start_char = self.char_offsets[start_id][0]
        end_char = self.char_offsets[end_id][1]
        
        return (start_char, end_char)
    
    def extract_text_span(
        self,
        start_id: str,
        end_id: str
    ) -> str:
        """
        Extract text span given sentence IDs.
        
        Args:
            start_id: Starting sentence ID
            end_id: Ending sentence ID
            
        Returns:
            Extracted text span
        """
        start_char, end_char = self.sentence_ids_to_char_offsets(start_id, end_id)
        return self.full_text[start_char:end_char]
    
    def validate_and_fix_gaps(
        self,
        items: List[Dict],
        fix_small_gaps: bool = True,
        gap_threshold: int = 3
    ) -> List[Dict]:
        """
        Validate agenda items and optionally fix small gaps.
        
        Args:
            items: List of agenda items with start_id and end_id
            fix_small_gaps: Whether to extend items to fill small gaps
            gap_threshold: Maximum gap size (in sentences) to fix
            
        Returns:
            Validated (and possibly fixed) list of items
        """
        if not items:
            return items
        
        # Sort by start position
        sorted_items = sorted(items, key=lambda x: int(x['start_id'][1:]))
        fixed_items = []
        
        for i, item in enumerate(sorted_items):
            fixed_item = item.copy()
            
            # Check if there's a small gap before next item
            if fix_small_gaps and i < len(sorted_items) - 1:
                end_num = int(item['end_id'][1:])
                next_start_num = int(sorted_items[i + 1]['start_id'][1:])
                gap = next_start_num - end_num - 1
                
                if 0 < gap <= gap_threshold:
                    # Extend this item to fill the gap
                    fixed_item['end_id'] = f"S{next_start_num - 1}"
                    logger.debug(f"Fixed gap of {gap} sentences after item {i}")
            
            fixed_items.append(fixed_item)
        
        return fixed_items
    
    def merge_overlapping_segments(
        self,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Merge overlapping segments.
        
        Args:
            segments: List of segments with start_id and end_id
            
        Returns:
            List with overlaps merged
        """
        if not segments:
            return segments
        
        # Sort by start position
        sorted_segs = sorted(segments, key=lambda x: int(x['start_id'][1:]))
        merged = [sorted_segs[0]]
        
        for current in sorted_segs[1:]:
            previous = merged[-1]
            
            prev_end = int(previous['end_id'][1:])
            curr_start = int(current['start_id'][1:])
            
            # Check for overlap
            if curr_start <= prev_end:
                # Merge: extend previous to include current
                curr_end = int(current['end_id'][1:])
                new_end = max(prev_end, curr_end)
                previous['end_id'] = f"S{new_end}"
                
                # Combine metadata if present
                if 'theme' in previous and 'theme' in current:
                    previous['theme'] = f"{previous['theme']}; {current['theme']}"
                
                logger.debug(f"Merged overlapping segments")
            else:
                merged.append(current)
        
        return merged
    
    def convert_to_annotations(
        self,
        items: List[Dict],
        include_text: bool = True
    ) -> List[Dict]:
        """
        Convert Stage 1 output to final annotation format.
        
        Args:
            items: List of agenda items from Stage 1
            include_text: Whether to include extracted text
            
        Returns:
            List of annotations with character offsets
        """
        annotations = []
        
        for item in items:
            try:
                start_char, end_char = self.sentence_ids_to_char_offsets(
                    item['start_id'],
                    item['end_id']
                )
                
                annotation = {
                    'item_title': item['item_title'],
                    'start': start_char,
                    'end': end_char,
                    'start_id': item['start_id'],
                    'end_id': item['end_id']
                }
                
                if include_text:
                    annotation['text'] = self.full_text[start_char:end_char]
                
                annotations.append(annotation)
                
            except ValueError as e:
                logger.warning(f"Skipping invalid item: {e}")
                continue
        
        return annotations
    
    def convert_subjects_to_annotations(
        self,
        subjects: List[Dict],
        item_offset: int = 0,
        include_text: bool = True
    ) -> List[Dict]:
        """
        Convert Stage 2 output to final annotation format.
        
        Args:
            subjects: List of subjects from Stage 2
            item_offset: Offset to add to sentence IDs (for relative numbering)
            include_text: Whether to include extracted text
            
        Returns:
            List of annotations with character offsets
        """
        annotations = []
        
        for subject in subjects:
            try:
                # Adjust sentence IDs if offset provided
                start_id = subject['start_id']
                end_id = subject['end_id']
                
                if item_offset > 0:
                    start_num = int(start_id[1:]) + item_offset
                    end_num = int(end_id[1:]) + item_offset
                    start_id = f"S{start_num}"
                    end_id = f"S{end_num}"
                
                start_char, end_char = self.sentence_ids_to_char_offsets(
                    start_id,
                    end_id
                )
                
                annotation = {
                    'theme': subject.get('theme', ''),
                    'topics': subject.get('topics', []),
                    'start': start_char,
                    'end': end_char,
                    'start_id': start_id,
                    'end_id': end_id
                }
                
                if include_text:
                    annotation['text'] = self.full_text[start_char:end_char]
                
                annotations.append(annotation)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid subject: {e}")
                continue
        
        return annotations
    
    def compute_coverage(self, segments: List[Dict]) -> float:
        """
        Compute what fraction of the document is covered by segments.
        
        Args:
            segments: List of segments with start and end character offsets
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not segments or not self.full_text:
            return 0.0
        
        covered_chars = set()
        
        for seg in segments:
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            covered_chars.update(range(start, end))
        
        total_chars = len(self.full_text)
        coverage = len(covered_chars) / total_chars if total_chars > 0 else 0.0
        
        return coverage
