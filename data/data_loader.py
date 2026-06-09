"""
HSeg Dataset Loaders

Centralized loaders for:
1. CitiLink minutes dataset (subjects_subset)
2. Wikipedia flat-file datasets (Wiki-727k / Wiki-50)
"""

from __future__ import annotations

import json
import logging
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CitiLink Minutes Paths & Helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_SUBSET_DIR = _ROOT / "data" / "citilink_dataset" / "subjects_subset"
_SPLIT_INFO = _ROOT / "data" / "citilink_dataset" / "split_info.json"


def _collect_all_subjects(minute: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a flat list of all subject dicts from a minute record."""
    subjects: List[Dict[str, Any]] = []
    for item in minute.get("agenda_items", []):
        for s in item.get("subjects", []):
            subjects.append(s)
    return subjects


def _find_item_title_pos(full_text: str, item_title: str, search_end: int) -> Optional[int]:
    """
    Find the character position of ``item_title`` in ``full_text`` within
    ``full_text[:search_end]``.
    """
    if not item_title:
        return None
    pos = full_text.rfind(item_title, 0, search_end)
    if pos != -1:
        return pos
    stripped = re.sub(r'^\d+[.\s]+', '', item_title).strip()
    if stripped and stripped != item_title:
        pos = full_text.rfind(stripped, 0, search_end)
        if pos != -1:
            return pos
    return None


def _body_span_with_titles(
    full_text: str,
    agenda_items: List[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    """
    Return ``(body_start, body_end)`` where:
    * ``body_start`` = position of the earliest agenda item title
    * ``body_end``   = max(subject.end) across all subjects
    """
    all_subject_ends: List[int] = []
    title_starts:     List[int] = []

    for item in agenda_items:
        subjects = item.get("subjects", [])
        valid_subjects = [
            s for s in subjects
            if s.get("start") is not None and s.get("end") is not None
        ]
        if not valid_subjects:
            continue

        all_subject_ends.extend(s["end"] for s in valid_subjects)

        item_first_subj_start = min(s["start"] for s in valid_subjects)
        item_title: str = item.get("item_title", "")

        title_pos = _find_item_title_pos(
            full_text, item_title, item_first_subj_start
        )
        if title_pos is not None:
            title_starts.append(title_pos)
        else:
            title_starts.append(item_first_subj_start)

    if not all_subject_ends:
        return None

    return min(title_starts), max(all_subject_ends)


def _body_span(subjects: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    starts = [s["start"] for s in subjects if s.get("start") is not None]
    ends   = [s["end"]   for s in subjects if s.get("end")   is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _annotate_item_offsets(
    minute: Dict[str, Any],
    full_text: str,
) -> None:
    for item in minute.get("agenda_items", []):
        subjects = item.get("subjects", [])
        valid = [
            s for s in subjects
            if s.get("start") is not None and s.get("end") is not None
        ]
        if not valid:
            continue
        first_subj_start = min(s["start"] for s in valid)
        last_subj_end    = max(s["end"]   for s in valid)
        title_pos = _find_item_title_pos(
            full_text, item.get("item_title", ""), first_subj_start
        )
        item["start"] = title_pos if title_pos is not None else first_subj_start
        item["end"]   = last_subj_end


def _adjust_offsets(minute: Dict[str, Any], body_start: int) -> Dict[str, Any]:
    import copy
    m = copy.deepcopy(minute)
    for item in m.get("agenda_items", []):
        if item.get("start") is not None:
            item["start"] -= body_start
        if item.get("end") is not None:
            item["end"] -= body_start
        for s in item.get("subjects", []):
            if s.get("start") is not None:
                s["start"] -= body_start
            if s.get("end") is not None:
                s["end"] -= body_start
    return m


def load_subjects_subset(
    validate_offsets: bool = True,
    trim_to_body: bool = True,
) -> Dict[str, Dict[str, Any]]:
    if not _SUBSET_DIR.exists():
        raise FileNotFoundError(
            f"Subjects subset directory not found: {_SUBSET_DIR}"
        )

    index: Dict[str, Dict[str, Any]] = {}

    for muni_file in sorted(_SUBSET_DIR.glob("*.json")):
        with open(muni_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for muni_entry in data.get("municipalities", []):
            municipality = muni_entry.get("municipality", muni_file.stem)
            for minute in muni_entry.get("minutes", []):
                minute_id = minute.get("minute_id", "")
                if not minute_id:
                    continue

                minute["_municipality"] = municipality
                full_text: str = minute.get("full_text", "")
                subjects = _collect_all_subjects(minute)

                if validate_offsets and full_text and subjects:
                    _validate_subject_offsets(minute_id, full_text, subjects)

                if full_text:
                    _annotate_item_offsets(minute, full_text)

                if trim_to_body and full_text:
                    agenda_items = minute.get("agenda_items", [])
                    span = _body_span_with_titles(full_text, agenda_items)
                    if span is not None:
                        body_start, body_end = span
                        if body_start < 0 or body_end > len(full_text):
                            logger.warning(
                                "[%s] Body span (%d, %d) is out of range — keeping full text.",
                                minute_id, body_start, body_end
                            )
                        else:
                            minute = _adjust_offsets(minute, body_start)
                            minute["full_text"] = full_text[body_start:body_end]
                            minute["body_offset"] = body_start
                    else:
                        minute["body_offset"] = 0
                else:
                    minute["body_offset"] = 0

                index[minute_id] = minute

    logger.info("Loaded %d documents from subjects subset.", len(index))
    return index


def _validate_subject_offsets(
    minute_id: str,
    full_text: str,
    subjects: List[Dict[str, Any]],
) -> None:
    n = len(full_text)
    for s in subjects:
        sid   = s.get("subject_id", "?")
        start = s.get("start")
        end   = s.get("end")
        text  = s.get("text", "")

        if start is None or end is None:
            continue

        if start < 0 or end > n or start >= end:
            logger.warning(
                "[%s] Subject %s has invalid offset range [%s, %s] (len: %d).",
                minute_id, sid, start, end, n
            )
            continue

        sliced = full_text[start:end]
        if text and sliced == text:
            continue
        if text and (text in sliced or sliced in text):
            continue

        if text:
            logger.warning(
                "[%s] Subject %s offset mismatch.\n  start=%d, end=%d",
                minute_id, sid, start, end
            )


def load_split_info() -> Dict[str, List[str]]:
    if not _SPLIT_INFO.exists():
        raise FileNotFoundError(f"split_info.json not found: {_SPLIT_INFO}")
    with open(_SPLIT_INFO, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {
        split: [fn.replace(".json", "") for fn in raw.get(f"{split}_files", [])]
        for split in ("train", "val", "test")
    }


def get_split_documents(
    split: str,
    municipality: Optional[str] = None,
    validate_offsets: bool = True,
    trim_to_body: bool = True,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    index = load_subjects_subset(
        validate_offsets=validate_offsets,
        trim_to_body=trim_to_body,
    )
    splits = load_split_info()

    split_ids: List[str] = splits.get(split, [])
    if not split_ids:
        raise ValueError(f"No documents found for split '{split}'")

    if municipality:
        split_ids = [
            mid for mid in split_ids
            if index.get(mid, {}).get("_municipality", "").lower() == municipality.lower()
        ]
        if not split_ids:
            raise ValueError(f"No documents for municipality '{municipality}' in split '{split}'")

    available = [mid for mid in split_ids if mid in index]
    return available, index


# ---------------------------------------------------------------------------
# Wikipedia Paths & Helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^={6,},(\d+),(.+?)\.?\s*$")


def parse_wiki_document(raw_text: str, doc_id: str = "") -> Dict[str, Any]:
    lines = raw_text.splitlines(keepends=True)

    heading_positions: List[Tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line.rstrip("\n").rstrip("\r"))
        if m:
            heading_positions.append((idx, int(m.group(1)), m.group(2).strip()))

    heading_line_set = {hp[0] for hp in heading_positions}
    heading_by_line  = {hp[0]: hp for hp in heading_positions}

    transformed: List[Optional[str]] = []
    for idx, line in enumerate(lines):
        if idx not in heading_line_set:
            transformed.append(line)
        else:
            _, depth, title = heading_by_line[idx]
            if depth == 1:
                transformed.append(None)
            else:
                transformed.append(title + "\n")

    cumulative: List[int] = []
    offset = 0
    for l in transformed:
        cumulative.append(offset)
        if l is not None:
            offset += len(l)

    full_text = "".join(l for l in transformed if l is not None)
    sections: List[Dict[str, Any]] = []

    for i, (line_idx, depth, title) in enumerate(heading_positions):
        sec_start = cumulative[line_idx]
        if i + 1 < len(heading_positions):
            sec_end = cumulative[heading_positions[i + 1][0]]
        else:
            sec_end = offset

        sections.append({
            "title": title,
            "depth": depth,
            "start": sec_start,
            "end":   sec_end,
            "text":  full_text[sec_start:sec_end],
        })

    return {
        "doc_id":    doc_id,
        "full_text": full_text,
        "sections":  sections,
    }


def build_wiki_ground_truth(doc: Dict[str, Any]) -> Dict[str, Any]:
    sections  = doc.get("sections", [])
    full_text = doc.get("full_text", "")

    real_sections = [s for s in sections if s["depth"] >= 2]

    agenda_items: List[Dict[str, Any]] = []
    current_item: Optional[Dict[str, Any]] = None

    for sec in real_sections:
        if sec["depth"] == 2:
            if current_item is not None:
                if not current_item["subjects"]:
                    current_item["subjects"].append({
                        "start": current_item["start"],
                        "end":   current_item["end"],
                        "text":  full_text[current_item["start"]:current_item["end"]],
                    })
                agenda_items.append(current_item)

            current_item = {
                "item_title": sec["title"],
                "start":      sec["start"],
                "end":        sec["end"],
                "subjects":   [],
            }
        else:
            if current_item is None:
                continue
            current_item["end"] = max(current_item["end"], sec["end"])
            current_item["subjects"].append({
                "start": sec["start"],
                "end":   sec["end"],
                "text":  sec["text"],
            })

    if current_item is not None:
        if not current_item["subjects"]:
            current_item["subjects"].append({
                "start": current_item["start"],
                "end":   current_item["end"],
                "text":  full_text[current_item["start"]:current_item["end"]],
            })
        agenda_items.append(current_item)

    if agenda_items and agenda_items[0]["start"] > 0:
        agenda_items[0]["start"] = 0
        if agenda_items[0]["subjects"] and agenda_items[0]["subjects"][0]["start"] > 0:
            first_subj = agenda_items[0]["subjects"][0]
            if len(agenda_items[0]["subjects"]) == 1:
                first_subj["start"] = 0
                first_subj["text"]  = full_text[0:first_subj["end"]]

    return {"agenda_items": agenda_items}


def _extract_leaf_subjects(
    subsections: List[Dict[str, Any]],
    item: Dict[str, Any],
    full_text: str,
) -> List[Dict[str, Any]]:
    if not subsections:
        return [{
            "start": item["start"],
            "end":   item["end"],
            "text":  full_text[item["start"]:item["end"]],
        }]

    leaves: List[Dict[str, Any]] = []
    n = len(subsections)

    for i, sec in enumerate(subsections):
        is_parent = False
        for j in range(i + 1, n):
            if subsections[j]["depth"] > sec["depth"]:
                is_parent = True
                break
            elif subsections[j]["depth"] <= sec["depth"]:
                break
        if not is_parent:
            leaves.append(sec)

    result = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in leaves
    ]

    if result and result[0]["start"] > item["start"]:
        intro_leaf = {
            "start": item["start"],
            "end":   result[0]["start"],
            "text":  full_text[item["start"]:result[0]["start"]]
        }
        result.insert(0, intro_leaf)

    for i in range(len(result) - 1):
        if result[i]["end"] < result[i+1]["start"]:
            result[i]["end"] = result[i+1]["start"]
            result[i]["text"] = full_text[result[i]["start"]:result[i]["end"]]

    return result


def build_wiki_ground_truth_hierarchical(doc: Dict[str, Any]) -> Dict[str, Any]:
    sections  = doc.get("sections", [])
    full_text = doc.get("full_text", "")

    real_sections = [s for s in sections if s["depth"] >= 2]

    agenda_items: List[Dict[str, Any]] = []
    current_item: Optional[Dict[str, Any]] = None
    current_subsections: List[Dict[str, Any]] = []

    for sec in real_sections:
        if sec["depth"] == 2:
            if current_item is not None:
                current_item["subjects"] = _extract_leaf_subjects(
                    current_subsections, current_item, full_text
                )
                current_item["section_tree"] = {
                    "item_title": current_item["item_title"],
                    "start": current_item["start"],
                    "end":   current_item["end"],
                    "depth": 2,
                    "subsections": [
                        {"title": s["title"], "depth": s["depth"],
                         "start": s["start"], "end": s["end"]}
                        for s in current_subsections
                    ],
                }
                agenda_items.append(current_item)

            current_item = {
                "item_title": sec["title"],
                "start":      sec["start"],
                "end":        sec["end"],
                "subjects":   [],
            }
            current_subsections = []
        else:
            if current_item is None:
                continue
            current_item["end"] = max(current_item["end"], sec["end"])
            current_subsections.append(sec)

    if current_item is not None:
        current_item["subjects"] = _extract_leaf_subjects(
            current_subsections, current_item, full_text
        )
        current_item["section_tree"] = {
            "item_title": current_item["item_title"],
            "start": current_item["start"],
            "end":   current_item["end"],
            "depth": 2,
            "subsections": [
                {"title": s["title"], "depth": s["depth"],
                 "start": s["start"], "end": s["end"]}
                for s in current_subsections
            ],
        }
        agenda_items.append(current_item)

    if agenda_items and agenda_items[0]["start"] > 0:
        agenda_items[0]["start"] = 0
        if len(agenda_items[0]["subjects"]) == 1:
            first_subj = agenda_items[0]["subjects"][0]
            first_subj["start"] = 0
            first_subj["text"]  = full_text[0:first_subj["end"]]

    return {"agenda_items": agenda_items}


def load_wiki_document(filepath: str | Path) -> Dict[str, Any]:
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    return parse_wiki_document(raw, doc_id=path.name)


class LazyWikiDataset(dict):
    """Loads Wiki documents on demand to minimize memory usage."""
    def __init__(self, doc_paths: Dict[str, Path], allowed_keys: Optional[List[str]] = None):
        self.doc_paths = doc_paths
        self.allowed_keys = list(allowed_keys) if allowed_keys is not None else list(doc_paths.keys())
        super().__init__()

    def __getitem__(self, key):
        if key in self.doc_paths:
            return load_wiki_document(self.doc_paths[key])
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.doc_paths

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.allowed_keys

    def __iter__(self):
        return iter(self.allowed_keys)

    def __len__(self):
        return len(self.allowed_keys)

    def items(self):
        return ((k, self[k]) for k in self.allowed_keys)

    def values(self):
        return (self[k] for k in self.allowed_keys)


def load_wiki_dataset(
    dataset_dir: str | Path,
    max_docs: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    doc_paths: Dict[str, Path] = {}
    skipped = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in {".json", ".py", ".txt", ".md", ".cfg", ".ini"}:
            continue
        if path.stat().st_size == 0:
            skipped += 1
            continue

        doc_paths[path.name] = path

    if max_docs is not None:
        allowed_keys = sorted(doc_paths.keys())[:max_docs]
    else:
        allowed_keys = None

    return LazyWikiDataset(doc_paths, allowed_keys=allowed_keys)


def get_wiki_documents(
    dataset_dir: str | Path,
    max_docs: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    index   = load_wiki_dataset(dataset_dir, max_docs=max_docs)
    doc_ids = sorted(index.keys())
    return doc_ids, index
