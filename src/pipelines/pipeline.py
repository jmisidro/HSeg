"""
Hierarchical Text Segmentation Pipelines

Provides:
1. LLMSegmentationPipeline: Two-stage segmentation for meeting minutes.
2. WikiLLMSegmentationPipeline: Recursive hierarchical segmentation for Wikipedia articles.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from hseg.utils.preprocessing import SentenceTokenizer
from hseg.llm.llm_interface import create_llm_interface
from hseg.extractors.stage1_extractor import Stage1Extractor, WikiStage1Extractor
from hseg.extractors.stage2_extractor import Stage2Extractor
from hseg.utils.postprocessing import SegmentationValidator

logger = logging.getLogger(__name__)


class LLMSegmentationPipeline:
    """
    Complete pipeline for LLM-based hierarchical text segmentation of meeting minutes.
    
    Implements two-stage extraction:
    1. Stage 1: Extract agenda items from full document
    2. Stage 2: Extract subjects within each agenda item
    
    Uses sentence-ID grounding to prevent hallucination and ensure
    accurate boundary detection.
    """
    
    def __init__(
        self,
        backend: str = "gemini",
        model_name: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        sentence_model: str = "pt_core_news_lg",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        use_few_shot: bool = True,
        max_retries: int = 5,
        fix_gaps: bool = True,
        **kwargs
    ):
        """
        Initialize LLM segmentation pipeline.
        
        Args:
            backend: LLM backend ("huggingface", "openai", "gemini", "mock", "local_transformers")
            model_name: LLM model identifier
            api_key: API key for LLM service
            sentence_model: spaCy model for sentence tokenization
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens to generate
            use_few_shot: Use few-shot examples in Stage 1
            max_retries: Maximum retry attempts for LLM calls.
            fix_gaps: Automatically fix small gaps between segments
            **kwargs: Additional parameters
        """
        logger.info(f"Initializing LLM Segmentation Pipeline")
        logger.info(f"  Backend: {backend}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Sentence Model: {sentence_model}")

        use_sliding_window = kwargs.pop('use_sliding_window', False)
        max_sentences_per_chunk = kwargs.pop('max_sentences_per_chunk', 150)
        overlap_sentences = kwargs.pop('overlap_sentences', 30)
        
        self.tokenizer = SentenceTokenizer(sentence_model)
        
        self.llm = create_llm_interface(
            backend=backend,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.stage1 = Stage1Extractor(
            llm_interface=self.llm,
            use_few_shot=use_few_shot,
            max_retries=max_retries,
            use_sliding_window=use_sliding_window,
            max_sentences_per_chunk=max_sentences_per_chunk,
            overlap_sentences=overlap_sentences,
        )
        
        self.stage2 = Stage2Extractor(
            llm_interface=self.llm,
            use_few_shot=use_few_shot,
            max_retries=max_retries
        )
        
        self.fix_gaps = fix_gaps
        self.config = {
            'backend': backend,
            'model_name': model_name,
            'sentence_model': sentence_model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'use_few_shot': use_few_shot,
            'max_retries': max_retries,
            'fix_gaps': fix_gaps,
        }
        
        logger.info("Pipeline initialization complete")
    
    def segment(
        self,
        text: str,
        municipality: str = "",
        extract_theme: bool = False,
        extract_topics: bool = False,
        return_metadata: bool = False,
        agenda_items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete hierarchical segmentation.
        
        Args:
            text: Input document text
            municipality: The municipality the document belongs to
            extract_theme: Instruct LLM to generate subjects' themes
            extract_topics: Instruct LLM to classify subjects' topics
            return_metadata: Include intermediate results and metadata
            agenda_items: Optional ground-truth agenda items from dataset
            
        Returns:
            Dictionary containing:
            - agenda_items: List of agenda items with subjects
            - metadata: Optional metadata
        """
        logger.info("=" * 60)
        logger.info("Starting LLM-based hierarchical text segmentation")
        logger.info("=" * 60)
        
        if not text or not text.strip():
            logger.error("Empty text provided")
            return {"agenda_items": [], "error": "Empty input text"}
        
        try:
            logger.info("\n[1/4] Preprocessing: Sentence tokenization")
            sentences, char_offsets = self.tokenizer.tokenize_with_offsets(text)
            logger.info(f"  Tokenized into {len(sentences)} sentences")
            
            validator = SegmentationValidator(sentences, char_offsets, text)
            
            boundary_sentence_nums = set()
            if agenda_items:
                for item in agenda_items:
                    item_title = item.get("item_title", "").strip()
                    if not item_title:
                        continue
                    char_offset = text.find(item_title)
                    if char_offset == -1:
                        logger.warning(f"  Could not find item title in text: {item_title!r:.80}")
                        continue
                    for s_id, (s_start, s_end) in char_offsets.items():
                        if s_start <= char_offset < s_end:
                            boundary_sentence_nums.add(int(s_id[1:]))
                            break
                logger.info(f"  Mapped {len(boundary_sentence_nums)}/{len(agenda_items)} ground-truth boundaries")
            
            logger.info("\n[2/4] Stage 1: Extracting agenda items")
            stage1_output = self.stage1.extract(sentences, boundary_sentence_nums=boundary_sentence_nums)
            logger.info(f"  Extracted {len(stage1_output)} raw agenda items")
            
            validation_errors = self.stage1.validate_output(
                stage1_output,
                max_sentence_id=len(sentences)
            )
            if validation_errors:
                logger.warning(f"  Stage 1 validation warnings: {len(validation_errors)}")
                for error in validation_errors[:5]:
                    logger.warning(f"    - {error}")
            
            if self.fix_gaps:
                stage1_output = validator.validate_and_fix_gaps(stage1_output)
                logger.info(f"  Applied gap fixing")
            
            agenda_items_annotations = validator.convert_to_annotations(
                stage1_output,
                include_text=True
            )
            logger.info(f"  Converted to {len(agenda_items_annotations)} valid items")
            
            logger.info("\n[3/4] Stage 2: Extracting subjects within items")
            all_agenda_items = []
            
            for item_idx, item_annot in enumerate(agenda_items_annotations):
                logger.info(f"  Processing item {item_idx + 1}/{len(agenda_items_annotations)}")
                
                item_start_id = item_annot['start_id']
                item_end_id = item_annot['end_id']
                
                item_start_num = int(item_start_id[1:])
                item_end_num = int(item_end_id[1:])
                
                item_sentences = {}
                for sent_num in range(item_start_num, item_end_num + 1):
                    sent_id = f"S{sent_num}"
                    if sent_id in sentences:
                        relative_id = f"S{sent_num - item_start_num + 1}"
                        item_sentences[relative_id] = sentences[sent_id]
                
                try:
                    stage2_output = self.stage2.extract(
                        item_title=item_annot['item_title'],
                        sentences=item_sentences,
                        municipality=municipality,
                        extract_theme=extract_theme,
                        extract_topics=extract_topics
                    )
                    
                    validation_errors = self.stage2.validate_output(
                        stage2_output,
                        min_sentence_id=1,
                        max_sentence_id=len(item_sentences),
                        extract_theme=extract_theme,
                        extract_topics=extract_topics
                    )
                    if validation_errors:
                        logger.warning(f"    Stage 2 validation warnings: {len(validation_errors)}")
                    
                    for subject in stage2_output:
                        rel_start = int(subject['start_id'][1:])
                        rel_end = int(subject['end_id'][1:])
                        subject['start_id'] = f"S{rel_start + item_start_num - 1}"
                        subject['end_id'] = f"S{rel_end + item_start_num - 1}"
                    
                    subjects_annotations = validator.convert_subjects_to_annotations(
                        stage2_output,
                        item_offset=0,
                        include_text=True
                    )
                    logger.info(f"    Extracted {len(subjects_annotations)} subjects")
                    
                except Exception as e:
                    logger.error(f"    Failed to extract subjects for item '{item_annot['item_title'][:60]}': {e}")
                    fallback_subject = {
                        'theme': item_annot['item_title'],
                        'topics': [],
                        'start_id': item_annot['start_id'],
                        'end_id': item_annot['end_id'],
                        'fallback': True,
                    }
                    subjects_annotations = validator.convert_subjects_to_annotations(
                        [fallback_subject],
                        item_offset=0,
                        include_text=True
                    )
                    if subjects_annotations:
                        logger.warning(
                            f"    Created fallback subject covering "
                            f"{item_annot['start_id']}–{item_annot['end_id']} "
                            f"for agenda item '{item_annot['item_title'][:60]}'"
                        )
                    else:
                        subjects_annotations = []
                
                agenda_item = {
                    'item_id': item_idx + 1,
                    'item_title': item_annot['item_title'],
                    'start': item_annot['start'],
                    'end': item_annot['end'],
                    'text': item_annot.get('text', ''),
                    'subjects': subjects_annotations
                }
                all_agenda_items.append(agenda_item)
            
            logger.info("\n[4/4] Post-processing and finalization")
            
            total_subjects = sum(len(item['subjects']) for item in all_agenda_items)
            coverage = validator.compute_coverage(
                [s for item in all_agenda_items for s in item['subjects']]
            )
            
            logger.info(f"  Total agenda items: {len(all_agenda_items)}")
            logger.info(f"  Total subjects: {total_subjects}")
            logger.info(f"  Document coverage: {coverage:.1%}")
            
            result = {
                'agenda_items': all_agenda_items
            }
            
            if return_metadata:
                result['metadata'] = {
                    'num_sentences': len(sentences),
                    'num_agenda_items': len(all_agenda_items),
                    'num_subjects': total_subjects,
                    'coverage': coverage,
                    'config': self.config,
                    'stage1_raw': stage1_output,
                    'validation_errors': validation_errors
                }
            
            logger.info("\n" + "=" * 60)
            logger.info("Segmentation complete!")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                'agenda_items': [],
                'error': str(e)
            }
    
    def segment_batch(
        self,
        texts: List[str],
        municipality: str = "",
        extract_theme: bool = False,
        extract_topics: bool = False,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        results = []
        logger.info(f"Processing batch of {len(texts)} documents")
        for i, text in enumerate(texts):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Document {i + 1}/{len(texts)}")
            logger.info(f"{'=' * 60}")
            
            result = self.segment(
                text,
                municipality=municipality,
                extract_theme=extract_theme,
                extract_topics=extract_topics,
                return_metadata=return_metadata
            )
            results.append(result)
        return results
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()


class WikiLLMSegmentationPipeline:
    """
    Hierarchical LLM pipeline for Wikipedia article segmentation.

    Stage 1 is called repeatedly (recursively) to discover all depth levels
    of the section tree. Each leaf node of that tree is mapped directly
    to a single fine-grained sub-topic boundary, skipping Stage 2 completely.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model_name: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        sentence_model: str = "en_core_web_lg",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        use_few_shot: bool = True,
        max_retries: int = 5,
        fix_gaps: bool = True,
        max_depth: int = 5,
        min_sentences_to_subdivide: int = 3,
        **kwargs,
    ):
        logger.info("Initializing WikiLLMSegmentationPipeline")
        logger.info("  Backend: %s | Model: %s", backend, model_name)
        logger.info("  Sentence model: %s", sentence_model)
        logger.info("  max_depth=%d | min_sentences_to_subdivide=%d", max_depth, min_sentences_to_subdivide)

        use_sliding_window      = kwargs.pop("use_sliding_window", False)
        max_sentences_per_chunk = kwargs.pop("max_sentences_per_chunk", 150)
        overlap_sentences       = kwargs.pop("overlap_sentences", 30)

        self.tokenizer = SentenceTokenizer(sentence_model)

        self.llm = create_llm_interface(
            backend=backend,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.stage1 = WikiStage1Extractor(
            llm_interface=self.llm,
            use_few_shot=use_few_shot,
            max_retries=max_retries,
            use_sliding_window=use_sliding_window,
            max_sentences_per_chunk=max_sentences_per_chunk,
            overlap_sentences=overlap_sentences,
        )

        self.fix_gaps           = fix_gaps
        self.max_depth          = max_depth
        self.min_sentences_to_subdivide = min_sentences_to_subdivide

        self.config = {
            "backend":             backend,
            "model_name":          model_name,
            "sentence_model":      sentence_model,
            "temperature":         temperature,
            "max_tokens":          max_tokens,
            "use_few_shot":        use_few_shot,
            "max_retries":         max_retries,
            "fix_gaps":            fix_gaps,
            "max_depth":           max_depth,
            "min_sentences_to_subdivide": min_sentences_to_subdivide,
        }

        logger.info("Pipeline initialization complete")

    @staticmethod
    def _abs_nums(sentences_abs: Dict[str, str]) -> List[int]:
        return sorted(int(k[1:]) for k in sentences_abs.keys())

    @staticmethod
    def _make_relative(sentences_abs: Dict[str, str], abs_nums: List[int]) -> Dict[str, str]:
        return {f"S{i + 1}": sentences_abs[f"S{num}"] for i, num in enumerate(abs_nums)}

    @staticmethod
    def _fix_gaps_and_boundaries(
        items: List[Dict],
        abs_nums: List[int],
    ) -> List[Dict]:
        if not items:
            return items

        items = sorted(items, key=lambda x: int(x["start_id"][1:]))

        if int(items[0]["start_id"][1:]) > abs_nums[0]:
            items[0] = dict(items[0])
            items[0]["start_id"] = f"S{abs_nums[0]}"

        for i in range(len(items) - 1):
            curr_end   = int(items[i]["end_id"][1:])
            next_start = int(items[i + 1]["start_id"][1:])
            if curr_end < next_start - 1:
                items[i] = dict(items[i])
                items[i]["end_id"] = f"S{next_start - 1}"

        if int(items[-1]["end_id"][1:]) < abs_nums[-1]:
            items[-1] = dict(items[-1])
            items[-1]["end_id"] = f"S{abs_nums[-1]}"

        return items

    def _run_stage1_hierarchical(
        self,
        sentences_abs: Dict[str, str],
        parent_title: Optional[str] = None,
        current_depth: int = 0,
    ) -> List[Dict]:
        abs_nums = self._abs_nums(sentences_abs)

        if current_depth >= self.max_depth or len(abs_nums) < self.min_sentences_to_subdivide:
            return []

        sentences_rel = self._make_relative(sentences_abs, abs_nums)

        try:
            items_rel = self.stage1.extract(sentences_rel, parent_title=parent_title)
        except Exception as exc:
            logger.error(
                "Stage 1 failed at depth %d (parent='%s'): %s",
                current_depth, parent_title, exc,
            )
            return []

        if not items_rel or len(items_rel) <= 1:
            logger.debug(
                "  No further subdivision at depth %d (parent='%s'): %d item(s) returned",
                current_depth, parent_title, len(items_rel),
            )
            return []

        items_abs: List[Dict] = []
        for item in items_rel:
            try:
                rel_s = int(item["start_id"][1:])
                rel_e = int(item["end_id"][1:])
                rel_s = max(1, min(rel_s, len(abs_nums)))
                rel_e = max(1, min(rel_e, len(abs_nums)))
                abs_s = abs_nums[rel_s - 1]
                abs_e = abs_nums[rel_e - 1]
            except (ValueError, IndexError, KeyError, TypeError):
                logger.warning("  Could not convert item IDs, skipping: %s", item)
                continue

            item_abs = dict(item)
            item_abs["start_id"] = f"S{abs_s}"
            item_abs["end_id"]   = f"S{abs_e}"
            item_abs["depth"]    = current_depth
            items_abs.append(item_abs)

        if not items_abs:
            return []

        items_abs = self._fix_gaps_and_boundaries(items_abs, abs_nums)

        logger.info(
            "  Depth %d (parent='%s'): %d sections found",
            current_depth, parent_title or "ROOT", len(items_abs),
        )

        for item in items_abs:
            sub_start = int(item["start_id"][1:])
            sub_end   = int(item["end_id"][1:])

            sub_sentences_abs = {
                f"S{num}": sentences_abs[f"S{num}"]
                for num in abs_nums
                if sub_start <= num <= sub_end
            }

            children = self._run_stage1_hierarchical(
                sub_sentences_abs,
                parent_title=item.get("item_title"),
                current_depth=current_depth + 1,
            )
            item["children"] = children

        return items_abs

    def _collect_leaves(self, sections: List[Dict]) -> List[Dict]:
        leaves: List[Dict] = []
        for section in sections:
            if not section.get("children"):
                leaves.append(section)
            else:
                leaves.extend(self._collect_leaves(section["children"]))
        return leaves

    def _tree_depth(self, sections: List[Dict], _d: int = 0) -> int:
        if not sections:
            return _d
        return max(self._tree_depth(s.get("children", []), _d + 1) for s in sections)

    def segment(
        self,
        text: str,
        return_metadata: bool = False,
        extract_theme: bool = False,
        extract_topics: bool = False,
        municipality: str = "",
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Starting Wiki hierarchical segmentation")
        logger.info("=" * 60)

        if not text or not text.strip():
            logger.error("Empty text provided")
            return {"agenda_items": [], "error": "Empty input text"}

        try:
            logger.info("[1/4] Sentence tokenisation")
            sentences, char_offsets = self.tokenizer.tokenize_with_offsets(text)
            logger.info("  %d sentences", len(sentences))

            validator = SegmentationValidator(sentences, char_offsets, text)

            logger.info("[2/4] Stage 1: hierarchical section extraction")
            top_sections = self._run_stage1_hierarchical(
                sentences_abs=sentences,
                parent_title=None,
                current_depth=0,
            )
            logger.info("  %d top-level sections found", len(top_sections))

            if not top_sections:
                logger.warning("  Stage 1 returned no sections — treating full text as one segment")
                all_sent_nums = sorted(int(k[1:]) for k in sentences.keys())
                top_sections = [{
                    "item_title": "Document",
                    "start_id": f"S{all_sent_nums[0]}",
                    "end_id":   f"S{all_sent_nums[-1]}",
                    "depth": 0,
                    "children": [],
                }]

            val_errors = self.stage1.validate_output(
                top_sections, max_sentence_id=len(sentences)
            )
            if val_errors:
                logger.warning("  Top-level validation: %d warning(s)", len(val_errors))

            if top_sections and int(top_sections[0]["start_id"][1:]) > 1:
                logger.info(
                    "  Extending first section '%s' back to S1 (preface coverage)",
                    top_sections[0].get("item_title", ""),
                )
                top_sections[0]["start_id"] = "S1"

            top_annots = validator.convert_to_annotations(
                top_sections, include_text=True
            )
            logger.info("  %d valid top-level sections after conversion", len(top_annots))

            tree_depth = self._tree_depth(top_sections)
            logger.info("  Section tree depth: %d level(s)", tree_depth)

            logger.info("[3/4] Mapping leaf sections to subject segments")
            all_agenda_items: List[Dict[str, Any]] = []

            for item_idx, (section, annot) in enumerate(
                zip(top_sections, top_annots), 1
            ):
                leaves = self._collect_leaves([section])

                all_subjects: List[Dict] = []
                for leaf_idx, leaf in enumerate(leaves, 1):
                    fallback = {"start_id": leaf["start_id"], "end_id": leaf["end_id"]}
                    leaf_subject_annot = validator.convert_subjects_to_annotations(
                        [fallback], item_offset=0, include_text=True
                    )
                    all_subjects.extend(leaf_subject_annot)

                logger.info(
                    "  Section '%s': %d total sub-topic segment(s)",
                    section.get("item_title", "")[:50], len(all_subjects),
                )

                all_agenda_items.append({
                    "item_id":     item_idx,
                    "item_title":  annot["item_title"],
                    "start":       annot["start"],
                    "end":         annot["end"],
                    "text":        annot.get("text", ""),
                    "subjects":    all_subjects,
                    "section_tree": section,
                })

            logger.info("[4/4] Post-processing")
            total_subjects = sum(len(item["subjects"]) for item in all_agenda_items)
            coverage = validator.compute_coverage(
                [s for item in all_agenda_items for s in item["subjects"]]
            )
            logger.info(
                "  %d top-level sections | %d sub-topic segments | coverage %.1f%%",
                len(all_agenda_items), total_subjects, coverage * 100,
            )

            result: Dict[str, Any] = {"agenda_items": all_agenda_items}

            if return_metadata:
                result["metadata"] = {
                    "num_sentences":      len(sentences),
                    "num_agenda_items":   len(all_agenda_items),
                    "num_subjects":       total_subjects,
                    "coverage":           coverage,
                    "section_tree_depth": tree_depth,
                    "config":             self.config,
                }

            logger.info("=" * 60)
            logger.info("Segmentation complete!")
            logger.info("=" * 60)
            return result

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return {"agenda_items": [], "error": str(exc)}

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
