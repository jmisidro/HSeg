"""
HSeg: A Two-Stage Hierarchical Text Segmentation Framework

Provides modules and pipelines for segmenting meeting minutes (CitiLink-Minutes)
and Wikipedia articles (Wiki-727k).
"""

from hseg.pipelines.pipeline import (
    LLMSegmentationPipeline,
    WikiLLMSegmentationPipeline,
)
from hseg.utils.preprocessing import (
    SentenceTokenizer,
    format_for_llm,
    extract_text_span,
)
from hseg.utils.postprocessing import (
    SegmentationValidator,
)
from hseg.utils.evaluation import (
    SegmentationEvaluator,
)
from hseg.data.data_loader import (
    # CitiLink
    load_subjects_subset,
    load_split_info,
    get_split_documents,
    # Wikipedia
    load_wiki_document,
    load_wiki_dataset,
    get_wiki_documents,
    build_wiki_ground_truth,
    build_wiki_ground_truth_hierarchical,
)

__all__ = [
    "LLMSegmentationPipeline",
    "WikiLLMSegmentationPipeline",
    "SentenceTokenizer",
    "format_for_llm",
    "extract_text_span",
    "SegmentationValidator",
    "SegmentationEvaluator",
    "load_subjects_subset",
    "load_split_info",
    "get_split_documents",
    "load_wiki_document",
    "load_wiki_dataset",
    "get_wiki_documents",
    "build_wiki_ground_truth",
    "build_wiki_ground_truth_hierarchical",
]
