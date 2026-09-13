"""Train-only retrieval for labeled in-context demonstrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Immutable upstream Hugging Face revision corresponding to the mirrored model.
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_K = 2
EMBEDDING_MAX_LENGTH = 256
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RETRIEVAL_MODES = ("question", "joint")
INDEX_VERSION = "finprm-demo-retrieval-v1"


def _require(mapping: Mapping[str, Any], key: str, kind: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        raise ValueError(f"{key} must be {kind.__name__}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no examples found")
    return records


def retrieval_text(record: Mapping[str, Any], mode: str) -> str:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
    process_input = _require(record, "input", dict)
    question = _require(process_input, "question", str)
    if mode == "question":
        return question
    prefix = _require(process_input, "prefix", list)
    candidate = _require(process_input, "candidate", str)
    return (
        f"Question: {question}\n"
        f"Correct prefix: {'; '.join(str(step) for step in prefix) or '<START>'}\n"
        f"Candidate next operation: {candidate}"
    )


def format_demonstrations(records: Sequence[Mapping[str, Any]]) -> str:
    """Serialize demos compactly without adding Qwen reward-marker tokens."""
    blocks = [
        "Retrieved verified Train demonstrations "
        "(the judgments below apply only to these demonstrations):"
    ]
    for index, record in enumerate(records, 1):
        process_input = _require(record, "input", dict)
        target = _require(record, "target", dict)
        label = _require(target, "label", int)
        if label not in {0, 1}:
            raise ValueError("demonstration label must be 0 or 1")
        supporting = _require(process_input, "supporting_facts", list)
        prefix = _require(process_input, "prefix", list)
        blocks.append(
            "\n".join(
                [
                    f"Demonstration {index}:",
                    "Gold evidence: " + (" ".join(map(str, supporting)) or "<NONE>"),
                    "Question: " + _require(process_input, "question", str),
                    "Correct prefix: " + ("; ".join(map(str, prefix)) or "<START>"),
                    "Candidate next operation: " + _require(process_input, "candidate", str),
                    "Verified judgment: " + ("CORRECT" if label == 1 else "INCORRECT"),
                ]
            )
        )
    return "\n\n".join(blocks)


class DenseEncoder:
    def __init__(self, model_path: str, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device).eval()

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int = 128,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("cannot encode an empty text collection")
        vectors = []
        with self.torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch_texts = list(texts[offset : offset + batch_size])
                if is_query:
                    batch_texts = [QUERY_INSTRUCTION + text for text in batch_texts]
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=EMBEDDING_MAX_LENGTH,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state[:, 0]
                hidden = self.torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
                vectors.append(hidden.cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def build_index(
    train_data: Path,
    output_dir: Path,
    model_path: str,
    *,
    modes: Sequence[str] = RETRIEVAL_MODES,
    batch_size: int = 128,
) -> Dict[str, Any]:
    records = load_jsonl(train_data)
    splits = {_require(_require(record, "metadata", dict), "split", str) for record in records}
    if splits != {"train"}:
        raise ValueError(f"retrieval source must contain Train only, found {sorted(splits)}")
    stable_ids = [_require(record["metadata"], "stable_id", str) for record in records]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("duplicate stable_id in retrieval source")

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = DenseEncoder(model_path)
    shapes = {}
    for mode in modes:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unsupported retrieval mode: {mode}")
        embeddings = encoder.encode(
            [retrieval_text(record, mode) for record in records],
            is_query=False,
            batch_size=batch_size,
        )
        np.save(output_dir / f"{mode}.npy", embeddings, allow_pickle=False)
        shapes[mode] = list(embeddings.shape)

    manifest = {
        "index_version": INDEX_VERSION,
        "source_split": "train",
        "source_path": str(train_data.resolve()),
        "source_sha256": file_sha256(train_data),
        "source_examples": len(records),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_revision": DEFAULT_EMBEDDING_REVISION,
        "embedding_local_path": str(Path(model_path).resolve()),
        "embedding_file_sha256": {
            name: file_sha256(Path(model_path) / name)
            for name in ("model.safetensors", "config.json", "tokenizer.json")
        },
        "embedding_max_length": EMBEDDING_MAX_LENGTH,
        "pooling": "CLS",
        "normalization": "L2",
        "similarity": "cosine (normalized dot product)",
        "query_instruction": QUERY_INSTRUCTION,
        "modes": list(modes),
        "shapes": shapes,
        "deduplication": "at most one demonstration per finqa_id",
        "default_k": DEFAULT_K,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def save_retrieval_plan(
    output: Path,
    target_data: Path,
    mode: str,
    k: int,
    index_manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    matches: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    rows = []
    for target, target_matches in zip(targets, matches):
        rows.append(
            {
                "target_stable_id": target["metadata"]["stable_id"],
                "target_finqa_id": target["metadata"]["finqa_id"],
                "demonstrations": [
                    {
                        "stable_id": match["record"]["metadata"]["stable_id"],
                        "finqa_id": match["record"]["metadata"]["finqa_id"],
                        "label": match["record"]["target"]["label"],
                        "similarity": match["similarity"],
                    }
                    for match in target_matches
                ],
            }
        )
    plan = {
        "index_version": index_manifest["index_version"],
        "index_source_path": index_manifest["source_path"],
        "index_source_sha256": index_manifest["source_sha256"],
        "target_path": str(target_data.resolve()),
        "target_sha256": file_sha256(target_data),
        "mode": mode,
        "k": k,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


class DemonstrationRetriever:
    def __init__(self, index_dir: Path, model_path: str, mode: str):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
        self.index_dir = index_dir
        self.mode = mode
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        if self.manifest["source_split"] != "train":
            raise ValueError("retrieval index is not Train-only")
        source_path = Path(self.manifest["source_path"])
        if file_sha256(source_path) != self.manifest["source_sha256"]:
            raise ValueError("retrieval source checksum does not match index manifest")
        self.records = load_jsonl(source_path)
        self.embeddings = np.load(index_dir / f"{mode}.npy", mmap_mode="r")
        if len(self.records) != self.embeddings.shape[0]:
            raise ValueError("retrieval records and embeddings have different lengths")
        self.encoder = DenseEncoder(model_path)
        self.stable_ids = np.asarray(
            [record["metadata"]["stable_id"] for record in self.records]
        )
        if mode == "question":
            # Every example from one question has the same question embedding.
            # Choose a canonical step/candidate without consulting the label,
            # making the representative deterministic.
            canonical = {}
            for row, record in enumerate(self.records):
                finqa_id = record["metadata"]["finqa_id"]
                previous = canonical.get(finqa_id)
                key = (
                    record["metadata"]["step_index"],
                    tuple(record["input"]["prefix"]),
                    record["input"]["candidate"],
                    row,
                )
                if previous is None:
                    canonical[finqa_id] = row
                else:
                    previous_record = self.records[previous]
                    previous_key = (
                        previous_record["metadata"]["step_index"],
                        tuple(previous_record["input"]["prefix"]),
                        previous_record["input"]["candidate"],
                        previous,
                    )
                    if key < previous_key:
                        canonical[finqa_id] = row
            self.search_rows = np.asarray(sorted(canonical.values()), dtype=np.int64)
        else:
            self.search_rows = np.arange(len(self.records), dtype=np.int64)

    def assert_no_target_leakage(self, targets: Iterable[Mapping[str, Any]]) -> None:
        train_ids = {record["metadata"]["finqa_id"] for record in self.records}
        target_ids = {record["metadata"]["finqa_id"] for record in targets}
        overlap = train_ids & target_ids
        if overlap:
            raise ValueError(f"target FinQA IDs overlap retrieval Train index: {len(overlap)}")

    def retrieve_many(
        self,
        targets: Sequence[Mapping[str, Any]],
        *,
        k: int = DEFAULT_K,
        batch_size: int = 128,
    ) -> List[List[Dict[str, Any]]]:
        if k < 1:
            raise ValueError("k must be positive")
        self.assert_no_target_leakage(targets)
        queries = self.encoder.encode(
            [retrieval_text(record, self.mode) for record in targets],
            is_query=True,
            batch_size=batch_size,
        )
        results: List[List[Dict[str, Any]]] = []
        corpus = np.asarray(self.embeddings)[self.search_rows]
        candidate_count = min(len(corpus), max(64, k * 32))
        for offset in range(0, len(queries), batch_size):
            score_batch = queries[offset : offset + batch_size] @ corpus.T
            local_candidate_rows = np.argpartition(
                -score_batch, candidate_count - 1, axis=1
            )[:, :candidate_count]
            for scores, local_candidates in zip(score_batch, local_candidate_rows):
                candidate_stable_ids = self.stable_ids[self.search_rows[local_candidates]]
                local_order = local_candidates[
                    np.lexsort((candidate_stable_ids, -scores[local_candidates]))
                ]
                selected: List[Dict[str, Any]] = []
                seen_finqa_ids = set()
                for local_row in local_order:
                    row = self.search_rows[int(local_row)]
                    record = self.records[int(row)]
                    finqa_id = record["metadata"]["finqa_id"]
                    if finqa_id in seen_finqa_ids:
                        continue
                    seen_finqa_ids.add(finqa_id)
                    selected.append(
                        {
                            "similarity": float(scores[int(local_row)]),
                            "record": record,
                        }
                    )
                    if len(selected) == k:
                        break
                if len(selected) != k:
                    # This is extremely unlikely, but preserve correctness if a
                    # single question occupies the entire preliminary top set.
                    full_local_order = np.lexsort(
                        (self.stable_ids[self.search_rows], -scores)
                    )
                    selected = []
                    seen_finqa_ids = set()
                    for local_row in full_local_order:
                        row = self.search_rows[int(local_row)]
                        record = self.records[int(row)]
                        finqa_id = record["metadata"]["finqa_id"]
                        if finqa_id in seen_finqa_ids:
                            continue
                        seen_finqa_ids.add(finqa_id)
                        selected.append(
                            {
                                "similarity": float(scores[int(local_row)]),
                                "record": record,
                            }
                        )
                        if len(selected) == k:
                            break
                if len(selected) != k:
                    raise RuntimeError(
                        f"could retrieve only {len(selected)} distinct demonstrations"
                    )
                results.append(selected)
        return results
