"""Local JSONL model worker. Explicit local snapshots only; stdout is protocol data."""

import argparse
import ctypes
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path


def process_memory():
    if os.name != "nt":
        return {}

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working_set",
                "working_set",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    values = Counters()
    values.cb = ctypes.sizeof(values)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if get_memory(handle, ctypes.byref(values), values.cb):
        return {
            "process_working_set_bytes": values.working_set,
            "process_peak_working_set_bytes": values.peak_working_set,
        }
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("CPU worker uses fp32")
    root = args.root.resolve()
    dependency_lock = root / ".local/e4_artifacts/model-requirements.lock.txt"
    if dependency_lock.exists():
        for requirement in dependency_lock.read_text("utf-8-sig").splitlines():
            if "==" in requirement and not requirement.startswith("#"):
                package, expected = requirement.split("==", 1)
                if version(package) != expected:
                    raise ValueError("Local model dependency drift: " + package)
    embedding_lock = json.loads(
        (root / ".local/e4_artifacts/bge-small-zh-v1.5.json").read_text("utf-8")
    )
    model_path = root / embedding_lock["local_path"]
    model = SentenceTransformer(str(model_path), device=args.device, local_files_only=True)
    model.max_seq_length = 512
    if args.precision == "fp16":
        model.half()
    model.eval()
    reranker = None
    reranker_tokenizer = None
    instruction = "为这个句子生成表示以用于检索相关文章："
    for line in sys.stdin:
        started = time.perf_counter()
        try:
            request = json.loads(line)
            method = request["method"]
            if method == "encode":
                texts = request["texts"]
                if request.get("kind") == "query":
                    texts = [instruction + text for text in texts]
                lengths = [
                    len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False))
                    for text in texts
                ]
                if any(length > 512 for length in lengths):
                    raise ValueError("Embedding input exceeds 512 tokens; no silent truncation")
                values = model.encode(
                    texts,
                    batch_size=request.get("batch_size", 2),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                output = {"vectors": values.astype(np.float32).tolist(), "lengths": lengths}
            elif method == "rerank":
                if reranker is None:
                    lock = json.loads(
                        (root / ".local/e4_artifacts/bge-reranker-v2-m3.json").read_text("utf-8")
                    )
                    path = root / lock["local_path"]
                    reranker_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                    reranker = AutoModelForSequenceClassification.from_pretrained(
                        path, local_files_only=True
                    )
                    reranker.eval()  # Reranker always stays CPU fp32 in E4.
                pairs = request["pairs"]
                scores = []
                lengths = []
                for offset in range(0, len(pairs), 2):
                    batch = pairs[offset : offset + 2]
                    raw = reranker_tokenizer(
                        batch, padding=True, truncation=False, return_tensors="pt"
                    )
                    lengths.extend(raw["attention_mask"].sum(dim=1).tolist())
                    if raw["input_ids"].shape[1] > 1024:
                        raise ValueError("Reranker pair exceeds frozen 1024 token budget")
                    with torch.inference_mode():
                        scores.extend(
                            reranker(**raw, return_dict=True).logits.view(-1).float().tolist()
                        )
                output = {"scores": scores, "lengths": lengths}
            elif method == "info":
                import sentence_transformers
                import transformers

                output = {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "sentence_transformers": sentence_transformers.__version__,
                    "transformers": transformers.__version__,
                    "device": args.device,
                    "precision": args.precision,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "cuda_arch_list": torch.cuda.get_arch_list(),
                    "dimension": 512,
                    "revision": embedding_lock["revision"],
                }
            else:
                raise ValueError("Unknown worker method")
            output["elapsed_seconds"] = time.perf_counter() - started
            output["cuda_peak_bytes"] = (
                torch.cuda.max_memory_allocated() if args.device == "cuda" else None
            )
            output["ok"] = True
            output.update(process_memory())
        except Exception as exc:
            output = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
        print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
