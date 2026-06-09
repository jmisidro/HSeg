"""
Evaluation Module for HSeg Text Segmentation

Provides comprehensive evaluation metrics for hierarchical segmentation:
- Stage 1: Agenda item boundary evaluation
- Stage 2: Subject boundary evaluation
- Overall: Combined metrics and coverage analysis

Theme Evaluation
----------------
Themes are evaluated using BERTScore (F1), which computes a soft semantic
similarity between predicted and ground-truth theme strings using contextual
embeddings from a pre-trained language model. A Portuguese-specific or
multilingual BERT model is used by default so that Portuguese themes are
handled correctly.

If the ``bert_score`` package is not installed a legacy substring-match
``theme_accuracy`` is computed instead.
"""

import bisect
import logging
import regex
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import json

NLTK_AVAILABLE = False

try:
    import segeval
    SEGEVAL_AVAILABLE = True
except ImportError:
    SEGEVAL_AVAILABLE = False
    logging.warning("segeval not available. Advanced metrics will be disabled.")

try:
    from bert_score import score as _bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning(
        "bert_score not available. Theme evaluation will fall back to "
        "substring-match accuracy. Install with: pip install bert-score"
    )

logger = logging.getLogger(__name__)

_DEFAULT_BERTSCORE_MODEL     = "neuralmind/bert-base-portuguese-cased"
_DEFAULT_BERTSCORE_NUM_LAYERS = 9
_DEFAULT_BERTSCORE_NTHREADS  = 4


class SegmentationEvaluator:
    """
    Evaluates LLM-based hierarchical text segmentation against ground truth.
    
    Supports evaluation at two levels:
    1. Agenda items (Stage 1)
    2. Subjects within items (Stage 2)
    """

    theme_bertscore_model: str  = _DEFAULT_BERTSCORE_MODEL
    theme_bertscore_num_layers: int = _DEFAULT_BERTSCORE_NUM_LAYERS
    theme_bertscore_device: Optional[str] = None
    
    def __init__(self, tolerance_sentences: int = 2):
        """
        Initialize evaluator.
        
        Args:
            tolerance_sentences: Boundary matching tolerance (sentences)
        """
        self.tolerance = tolerance_sentences
        logger.info(f"Initialized evaluator with tolerance={tolerance_sentences} sentences")
    
    def evaluate_agenda_items(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        sentences: Optional[List[str]] = None,
        sentence_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"Evaluating agenda items: {len(predicted)} predicted vs "
            f"{len(ground_truth)} ground truth"
        )

        pred_boundaries = [
            {'start': item['start'], 'end': item['end'], 'id': f"item_{i}"}
            for i, item in enumerate(predicted)
        ]
        gt_boundaries = [
            {'start': item['start'], 'end': item['end'], 'id': f"item_{i}"}
            for i, item in enumerate(ground_truth)
        ]

        boundary_metrics = self._evaluate_boundaries(
            pred_boundaries,
            gt_boundaries,
            sentences,
            sentence_offsets,
        )
        results = dict(boundary_metrics)
        results['stage'] = 'agenda_items'
        results['num_predicted'] = len(predicted)
        results['num_ground_truth'] = len(ground_truth)
        return results
    
    def evaluate_subjects(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        sentences: Optional[List[str]] = None,
        sentence_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"Evaluating subjects: {len(predicted)} predicted vs "
            f"{len(ground_truth)} ground truth"
        )

        pred_boundaries = [
            {'start': subj['start'], 'end': subj['end'], 'id': f"subj_{i}"}
            for i, subj in enumerate(predicted)
        ]
        gt_boundaries = [
            {'start': subj['start'], 'end': subj['end'], 'id': f"subj_{i}"}
            for i, subj in enumerate(ground_truth)
        ]

        boundary_metrics = self._evaluate_boundaries(
            pred_boundaries,
            gt_boundaries,
            sentences,
            sentence_offsets,
        )
        results = dict(boundary_metrics)

        if predicted and ground_truth:
            theme_metrics = self._evaluate_themes(predicted, ground_truth)
            results.update(theme_metrics)
            if 'topics' in predicted[0] and 'topics' in ground_truth[0]:
                topic_metrics = self._evaluate_topics(predicted, ground_truth)
                results.update(topic_metrics)

        results['stage'] = 'subjects'
        results['num_predicted'] = len(predicted)
        results['num_ground_truth'] = len(ground_truth)
        return results
    
    def evaluate_hierarchical(
        self,
        predicted_result: Dict[str, Any],
        ground_truth_result: Dict[str, Any],
        sentences: Optional[List[str]] = None,
        sentence_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Hierarchical Evaluation")
        logger.info("=" * 60)

        results: Dict[str, Any] = {
            'agenda_items': {},
            'subjects': {},
            'overall': {}
        }

        pred_items = predicted_result.get('agenda_items', [])
        gt_items = ground_truth_result.get('agenda_items', [])

        if pred_items and gt_items:
            results['agenda_items'] = self.evaluate_agenda_items(
                pred_items, gt_items, sentences, sentence_offsets
            )

        pred_subjects: List[Dict[str, Any]] = []
        gt_subjects:   List[Dict[str, Any]] = []
        for item in pred_items:
            pred_subjects.extend(item.get('subjects', []))
        for item in gt_items:
            gt_subjects.extend(item.get('subjects', []))

        if pred_subjects and gt_subjects:
            results['subjects'] = self.evaluate_subjects(
                pred_subjects, gt_subjects, sentences, sentence_offsets
            )

        results['overall'] = {
            'total_agenda_items_predicted': len(pred_items),
            'total_agenda_items_gt': len(gt_items),
            'total_subjects_predicted': len(pred_subjects),
            'total_subjects_gt': len(gt_subjects),
            'avg_subjects_per_item_predicted': len(pred_subjects) / len(pred_items) if pred_items else 0,
            'avg_subjects_per_item_gt': len(gt_subjects) / len(gt_items) if gt_items else 0
        }

        return results
    
    def _evaluate_boundaries(
        self,
        pred_boundaries: List[Dict[str, int]],
        gt_boundaries: List[Dict[str, int]],
        sentences: Optional[List[str]] = None,
        sentence_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if sentence_offsets is not None:
            pred_sent_boundaries = self._char_to_sentence_boundaries_from_offsets(
                pred_boundaries, sentence_offsets
            )
            gt_sent_boundaries = self._char_to_sentence_boundaries_from_offsets(
                gt_boundaries, sentence_offsets
            )
        elif sentences:
            pred_sent_boundaries = self._char_to_sentence_boundaries(
                pred_boundaries, sentences
            )
            gt_sent_boundaries = self._char_to_sentence_boundaries(
                gt_boundaries, sentences
            )
        else:
            pred_sent_boundaries = pred_boundaries
            gt_sent_boundaries = gt_boundaries

        matches = self._match_boundaries_with_tolerance(
            pred_sent_boundaries,
            gt_sent_boundaries
        )

        tp = len(matches)
        fp = len(pred_boundaries) - tp
        fn = len(gt_boundaries) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.update({
            'boundary_precision': round(precision, 4),
            'boundary_recall': round(recall, 4),
            'boundary_f1': round(f1, 4),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'tolerance_sentences': self.tolerance
        })

        # Calculate average IoU if possible
        if pred_boundaries and gt_boundaries:
            ious = []
            for pred in pred_boundaries:
                best_iou = 0.0
                for gt in gt_boundaries:
                    iou = self._compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)
            results['avg_iou'] = round(sum(ious) / len(ious), 4) if ious else 0.0

        if SEGEVAL_AVAILABLE:
            try:
                segeval_metrics = self._compute_segeval_metrics(
                    pred_sent_boundaries,
                    gt_sent_boundaries
                )
                results.update(segeval_metrics)
            except Exception as e:
                logger.warning(f"Segeval metrics failed: {e}")

        return results
    
    def _char_to_sentence_boundaries_from_offsets(
        self,
        boundaries: List[Dict[str, int]],
        sentence_offsets: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, int]]:
        if not sentence_offsets:
            return boundaries

        sorted_sents: List[Tuple[int, int, int]] = sorted(
            (start, end, idx)
            for idx, (start, end) in enumerate(sentence_offsets.values())
        )
        sent_starts = [s[0] for s in sorted_sents]
        n_sents = len(sorted_sents)

        def _char_to_sent_idx(char_pos: int) -> int:
            idx = bisect.bisect_right(sent_starts, char_pos) - 1
            return max(0, min(idx, n_sents - 1))

        sent_boundaries: List[Dict[str, int]] = []
        for boundary in boundaries:
            start_sent = _char_to_sent_idx(boundary['start'])
            end_sent   = _char_to_sent_idx(boundary['end'])
            end_sent = max(start_sent, end_sent)
            sent_boundaries.append({
                'start': start_sent,
                'end':   end_sent,
                'id':    boundary.get('id', ''),
            })
        return sent_boundaries

    def _char_to_sentence_boundaries(
        self,
        boundaries: List[Dict[str, int]],
        sentences: List[str],
    ) -> List[Dict[str, int]]:
        cumulative_pos = [0]
        for sent in sentences:
            cumulative_pos.append(cumulative_pos[-1] + len(sent) + 1)

        sent_boundaries: List[Dict[str, int]] = []
        for boundary in boundaries:
            start_char = boundary['start']
            end_char   = boundary['end']

            start_sent = max(0, bisect.bisect_right(cumulative_pos, start_char) - 1)
            end_sent   = max(0, bisect.bisect_right(cumulative_pos, end_char)   - 1)
            end_sent   = min(end_sent, len(sentences) - 1)
            end_sent   = max(start_sent, end_sent)

            sent_boundaries.append({
                'start': start_sent,
                'end':   end_sent,
                'id':    boundary.get('id', ''),
            })
        return sent_boundaries
    
    def _match_boundaries_with_tolerance(
        self,
        pred_boundaries: List[Dict[str, int]],
        gt_boundaries: List[Dict[str, int]]
    ) -> List[Tuple[int, int]]:
        matches = []
        matched_gt = set()
        
        for p_idx, pred in enumerate(pred_boundaries):
            best_match = None
            best_score = -1
            
            for g_idx, gt in enumerate(gt_boundaries):
                if g_idx in matched_gt:
                    continue
                
                start_diff = abs(pred['start'] - gt['start'])
                end_diff = abs(pred['end'] - gt['end'])
                
                if start_diff <= self.tolerance and end_diff <= self.tolerance:
                    score = 1.0 / (1.0 + start_diff + end_diff)
                    if score > best_score:
                        best_score = score
                        best_match = g_idx
            
            if best_match is not None:
                matches.append((p_idx, best_match))
                matched_gt.add(best_match)
        
        return matches
    
    def _compute_iou(
        self,
        pred: Dict[str, int],
        gt: Dict[str, int]
    ) -> float:
        intersection_start = max(pred['start'], gt['start'])
        intersection_end = min(pred['end'], gt['end'])
        
        intersection = max(0, intersection_end - intersection_start)
        
        pred_length = pred['end'] - pred['start']
        gt_length = gt['end'] - gt['start']
        union = pred_length + gt_length - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _boundaries_to_masses(
        self,
        boundaries: List[Dict[str, int]],
        total_length: int
    ) -> Tuple[int, ...]:
        if not boundaries:
            return (total_length,) if total_length > 0 else ()
        
        sorted_boundaries = sorted(boundaries, key=lambda x: x['start'])
        masses = []
        current_pos = 0
        
        for boundary in sorted_boundaries:
            start = boundary['start']
            end = boundary['end']
            
            if start > current_pos:
                masses.append(start - current_pos)
            
            length = end - max(start, current_pos)
            if length > 0:
                masses.append(length)
                
            current_pos = max(current_pos, end)
            
        if current_pos < total_length:
            masses.append(total_length - current_pos)
            
        return tuple(masses)
    
    def _compute_segeval_metrics(
        self,
        pred_boundaries: List[Dict[str, int]],
        gt_boundaries: List[Dict[str, int]]
    ) -> Dict[str, Any]:
        default = {
            'boundary_similarity': 0.0,
            'bed_precision': 0.0,
            'bed_recall': 0.0,
            'bed_fmeasure': 0.0,
            'segeval_pk': 1.0,
            'segeval_windowdiff': 1.0,
        }
        try:
            max_p = max([b['end'] for b in pred_boundaries] + [0])
            max_g = max([b['end'] for b in gt_boundaries] + [0])
            total_length = max(max_p, max_g)
            
            if total_length == 0:
                return default

            pred_masses = self._boundaries_to_masses(pred_boundaries, total_length)
            gt_masses = self._boundaries_to_masses(gt_boundaries, total_length)

            if not pred_masses or not gt_masses:
                return default

            pred_norm, gt_norm = pred_masses, gt_masses

            try:
                pred_pos = segeval.convert_masses_to_positions(pred_norm)
                gt_pos = segeval.convert_masses_to_positions(gt_norm)
                if len(pred_pos) != len(gt_pos):
                    diff = len(pred_pos) - len(gt_pos)
                    if diff > 0:
                        gt_norm = gt_norm[:-1] + (gt_norm[-1] + diff,)
                    else:
                        pred_norm = pred_norm[:-1] + (pred_norm[-1] + (-diff),)
            except Exception:
                pass

            try:
                boundary_sim = float(segeval.boundary_similarity(
                    pred_norm, gt_norm,
                    boundary_format=segeval.BoundaryFormat.mass
                ))
            except Exception as e:
                logger.warning(f"Segeval boundary_similarity failed: {e}")
                boundary_sim = 0.0

            bed_precision = bed_recall = bed_fmeasure = 0.0
            try:
                cm = segeval.boundary_confusion_matrix(
                    pred_norm, gt_norm,
                    boundary_format=segeval.BoundaryFormat.mass
                )
                bed_precision = float(segeval.precision(cm))
                bed_recall = float(segeval.recall(cm))
                bed_fmeasure = float(segeval.fmeasure(cm))
            except Exception as e:
                logger.warning(f"Segeval BED confusion matrix failed: {e}")

            segeval_pk = segeval_wd = 1.0
            try:
                segeval_pk = float(segeval.pk(
                    pred_norm, gt_norm,
                    boundary_format=segeval.BoundaryFormat.mass
                ))
                segeval_wd = float(segeval.window_diff(
                    pred_norm, gt_norm,
                    boundary_format=segeval.BoundaryFormat.mass
                ))
            except Exception as e:
                logger.warning(f"Segeval pk/windowdiff failed: {e}")

            return {
                'boundary_similarity': round(boundary_sim, 4),
                'bed_precision': round(bed_precision, 4),
                'bed_recall': round(bed_recall, 4),
                'bed_fmeasure': round(bed_fmeasure, 4),
                'segeval_pk': round(segeval_pk, 4),
                'segeval_windowdiff': round(segeval_wd, 4),
            }

        except Exception as e:
            logger.warning(f"Segeval metrics failed: {e}")
            return default
    
    def _match_segments_by_iou(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        used_gt: set = set()
        for pred in predicted:
            best_iou = 0.0
            best_idx = -1
            for g_idx, gt in enumerate(ground_truth):
                if g_idx in used_gt:
                    continue
                iou = self._compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matches.append((pred, ground_truth[best_idx]))
                used_gt.add(best_idx)
        return matches

    def _evaluate_themes(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not predicted or not ground_truth:
            return {}

        matches = self._match_segments_by_iou(predicted, ground_truth, iou_threshold=0.5)

        if not matches:
            return {'theme_accuracy': 0.0}

        correct = 0
        for pred, gt in matches:
            p = pred.get('theme', '').lower().strip()
            g = gt.get('theme', '').lower().strip()
            if p == g or p in g or g in p:
                correct += 1
        accuracy = correct / len(matches)
        result: Dict[str, float] = {'theme_accuracy': round(accuracy, 4)}

        if BERTSCORE_AVAILABLE:
            try:
                pred_themes = [str(pred.get('theme', '') or '') for pred, _ in matches]
                gt_themes   = [str(gt.get('theme',   '') or '') for _, gt   in matches]

                pred_themes = [t if t.strip() else ' ' for t in pred_themes]
                gt_themes   = [t if t.strip() else ' ' for t in gt_themes]

                kwargs: Dict[str, Any] = {
                    'model_type': self.theme_bertscore_model,
                    'num_layers': self.theme_bertscore_num_layers,
                    'verbose': False,
                    'nthreads': _DEFAULT_BERTSCORE_NTHREADS,
                }
                if self.theme_bertscore_device is not None:
                    kwargs['device'] = self.theme_bertscore_device

                P, R, F1 = _bert_score_fn(pred_themes, gt_themes, **kwargs)

                result['theme_bertscore_precision'] = round(float(P.mean().item()), 4)
                result['theme_bertscore_recall']    = round(float(R.mean().item()), 4)
                result['theme_bertscore_f1']        = round(float(F1.mean().item()), 4)
            except Exception as exc:
                logger.warning(f"BERTScore computation failed: {exc}")

        return result

    def _evaluate_topics(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not predicted or not ground_truth:
            return {}

        matches = self._match_segments_by_iou(predicted, ground_truth, iou_threshold=0.5)

        if not matches:
            return {'topic_precision': 0.0, 'topic_recall': 0.0, 'topic_f1': 0.0}

        total_tp = 0
        total_pred = 0
        total_gt = 0

        for pred, gt in matches:
            pred_topics = set(pred.get('topics', []))
            gt_topics   = set(gt.get('topics', []))

            tp = len(pred_topics & gt_topics)
            total_tp   += tp
            total_pred += len(pred_topics)
            total_gt   += len(gt_topics)

        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall    = total_tp / total_gt   if total_gt   > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            'topic_precision': round(precision, 4),
            'topic_recall':    round(recall,    4),
            'topic_f1':        round(f1,        4),
        }
    
    def print_evaluation_report(self, results: Dict[str, Any], verbose: bool = True):
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        if 'overall' in results:
            print("\n### Overall Statistics ###")
            overall = results['overall']
            print(f"Agenda Items - Predicted: {overall.get('total_agenda_items_predicted', 0)}, "
                  f"Ground Truth: {overall.get('total_agenda_items_gt', 0)}")
            print(f"Subjects - Predicted: {overall.get('total_subjects_predicted', 0)}, "
                  f"Ground Truth: {overall.get('total_subjects_gt', 0)}")
            print(f"Avg Subjects/Item - Predicted: {overall.get('avg_subjects_per_item_predicted', 0):.2f}, "
                  f"Ground Truth: {overall.get('avg_subjects_per_item_gt', 0):.2f}")
        
        if 'agenda_items' in results and results['agenda_items']:
            print("\n### Stage 1: Agenda Items ###")
            self._print_metrics(results['agenda_items'], verbose)
        
        if 'subjects' in results and results['subjects']:
            print("\n### Stage 2: Subjects ###")
            self._print_metrics(results['subjects'], verbose)
        
        print("\n" + "=" * 60)
    
    def _print_metrics(self, metrics: Dict[str, Any], verbose: bool):
        if 'boundary_f1' in metrics:
            print(f"\nBoundary Detection (tolerance={metrics.get('tolerance_sentences', 0)} sentences):")
            print(f"  Precision: {metrics['boundary_precision']:.4f}")
            print(f"  Recall: {metrics['boundary_recall']:.4f}")
            print(f"  F1: {metrics['boundary_f1']:.4f}")
            print(f"  TP: {metrics.get('true_positives', 0)}, "
                  f"FP: {metrics.get('false_positives', 0)}, "
                  f"FN: {metrics.get('false_negatives', 0)}")

        if 'avg_iou' in metrics:
            print(f"\nSegment Overlap:")
            print(f"  Avg IoU: {metrics['avg_iou']:.4f}")

        if 'boundary_similarity' in metrics:
            print(f"\nSegeval Metrics:")
            print(f"  Boundary Similarity: {metrics['boundary_similarity']:.4f}")
        if 'bed_fmeasure' in metrics:
            print(f"  BED F-measure: {metrics['bed_fmeasure']:.4f} "
                  f"(P: {metrics.get('bed_precision', 0):.4f}, "
                  f"R: {metrics.get('bed_recall', 0):.4f})")
        if 'segeval_pk' in metrics:
            print(f"  Segeval Pk: {metrics['segeval_pk']:.4f}, "
                  f"WindowDiff: {metrics.get('segeval_windowdiff', 0):.4f} (lower is better)")

        if 'theme_bertscore_f1' in metrics or 'theme_accuracy' in metrics:
            print(f"\nContent Metrics (Themes):")
        if 'theme_bertscore_f1' in metrics:
            print(
                f"  Theme BERTScore F1 : {metrics['theme_bertscore_f1']:.4f}  "
                f"(P {metrics.get('theme_bertscore_precision', 0):.4f} / "
                f"R {metrics.get('theme_bertscore_recall', 0):.4f})"
            )
        if 'theme_accuracy' in metrics:
            print(f"  Theme Accuracy (substring): {metrics['theme_accuracy']:.4f}")

        if 'topic_f1' in metrics:
            print(f"\nContent Metrics (Topics):")
            print(f"  Topic F1        : {metrics['topic_f1']:.4f}  "
                  f"(P {metrics.get('topic_precision', 0):.4f} / "
                  f"R {metrics.get('topic_recall', 0):.4f})")


def load_ground_truth(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prediction(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_from_files(
    prediction_path: str,
    ground_truth_path: str,
    tolerance: int = 2,
    verbose: bool = True
) -> Dict[str, Any]:
    logger.info(f"Loading prediction from: {prediction_path}")
    logger.info(f"Loading ground truth from: {ground_truth_path}")
    
    pred = load_prediction(prediction_path)
    gt = load_ground_truth(ground_truth_path)
    
    evaluator = SegmentationEvaluator(tolerance_sentences=tolerance)
    results = evaluator.evaluate_hierarchical(pred, gt)
    
    if verbose:
        evaluator.print_evaluation_report(results)
    
    return results
