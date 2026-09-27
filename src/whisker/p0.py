"""P0: measure whether Qwen3.5-4B LoRA training is viable on Apple MPS.

This command deliberately does real work. It verifies renderer parity, loads the
cached text model on MPS, performs deterministic greedy generations, and times
an exact-shape LoRA optimizer step. Evidence is checkpointed to JSON after each
phase so a late failure does not erase earlier measurements.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from whisker.chat_format import render_generation, render_sft, stop_token_ids

BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_DATA = Path("assets/data/refusal_cats3.jsonl")
DEFAULT_REPORT = Path("results/p0_spike.json")


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def mps_memory_gb() -> dict[str, float] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "tensor_allocated": round(torch.mps.current_allocated_memory() / 1e9, 3),
        "driver_allocated": round(torch.mps.driver_allocated_memory() / 1e9, 3),
        "recommended_max": round(torch.mps.recommended_max_memory() / 1e9, 3),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_renderer_parity(data_glob: str = "assets/data/*.jsonl") -> dict[str, Any]:
    """Compare the local renderer with Tinker's implementation on every example."""
    from tinker_cookbook import renderers, tokenizer_utils

    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer("qwen3_5_disable_thinking", tokenizer)
    paths = [Path(path) for path in sorted(glob.glob(data_glob))]
    if not paths:
        raise RuntimeError(f"no parity examples matched {data_glob!r}")

    supervised_count = 0
    generation_count = 0
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                model_input, weights = renderer.build_supervised_example(messages)
                local_ids, local_weights = render_sft(tokenizer, messages)
                if local_ids != model_input.to_ints():
                    raise AssertionError(f"token mismatch: {path}:{line_number}")
                if local_weights != [float(value) for value in weights.tolist()]:
                    raise AssertionError(f"loss-weight mismatch: {path}:{line_number}")
                supervised_count += 1

                prompt = [message for message in messages if message["role"] != "assistant"]
                expected_prompt = renderer.build_generation_prompt(prompt).to_ints()
                if render_generation(tokenizer, prompt) != expected_prompt:
                    raise AssertionError(f"generation-prompt mismatch: {path}:{line_number}")
                generation_count += 1

    return {
        "renderer": "qwen3_5_disable_thinking",
        "files": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "supervised_examples": supervised_count,
        "generation_prompts": generation_count,
        "tokens_identical": True,
        "loss_weights_identical": True,
    }


def load_text_model(device: str, dtype: torch.dtype) -> tuple[Any, nn.Module, dict[str, Any]]:
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model.to(device)
    synchronize(device)
    elapsed = time.perf_counter() - started

    layer_types = Counter(getattr(model.config, "layer_types", []))
    summary = {
        "seconds": round(elapsed, 3),
        "model_class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(dtype),
        "device": str(next(model.parameters()).device),
        "layer_types": dict(layer_types),
        "missing_keys": sorted(loading_info["missing_keys"]),
        "unexpected_keys": sorted(loading_info["unexpected_keys"]),
        "mismatched_keys": sorted(loading_info.get("mismatched_keys", [])),
        "memory_gb": mps_memory_gb(),
    }
    return tokenizer, model, summary


@torch.inference_mode()
def greedy_generation(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_ids = render_generation(tokenizer, [{"role": "user", "content": prompt}])
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output[0, input_ids.shape[1] :]
    return {
        "prompt": prompt,
        "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_tokens": generated.numel(),
        "seconds": round(elapsed, 3),
        "tokens_per_second": round(generated.numel() / elapsed, 3),
    }


def linear_modules_by_layer_type(model: nn.Module) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    layer_types = model.config.layer_types
    marker = ".layers."
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or marker not in name:
            continue
        suffix = name.split(marker, maxsplit=1)[1]
        layer_index_text, relative_name = suffix.split(".", maxsplit=1)
        layer_type = layer_types[int(layer_index_text)]
        grouped.setdefault(layer_type, set()).add(relative_name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def load_examples(path: Path, tokenizer: Any) -> list[tuple[list[int], list[float]]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(0).shuffle(rows)
    return [render_sft(tokenizer, row["messages"]) for row in rows]


def fixed_shape_batch(
    examples: list[tuple[list[int], list[float]]],
    *,
    batch_size: int,
    sequence_length: int,
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a right-padded causal-LM batch with exact [batch, sequence] shape."""
    input_ids = torch.full((batch_size, sequence_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    target_tokens = 0
    source_lengths = []

    for index, (token_ids, loss_weights) in enumerate(examples[:batch_size]):
        # Need one extra token because labels are shifted by one position.
        used = min(len(token_ids) - 1, sequence_length)
        if used <= 0:
            raise ValueError("an SFT example contained fewer than two tokens")
        input_ids[index, :used] = torch.tensor(token_ids[:used])
        shifted = torch.tensor(token_ids[1 : used + 1])
        shifted_weights = torch.tensor(loss_weights[1 : used + 1])
        shifted[shifted_weights == 0] = -100
        labels[index, :used] = shifted
        attention_mask[index, :used] = 1
        target_tokens += int((shifted != -100).sum())
        source_lengths.append(len(token_ids))

    if target_tokens == 0:
        raise RuntimeError("the fixed-shape batch contains no supervised target tokens")

    metadata = {
        "shape": list(input_ids.shape),
        "non_padding_tokens": int(attention_mask.sum()),
        "supervised_tokens": target_tokens,
        "source_lengths": source_lengths,
        "truncated_examples": sum(length - 1 > sequence_length for length in source_lengths),
    }
    return (
        input_ids.to(device),
        labels.to(device),
        attention_mask.to(device),
        metadata,
    )


def measure_lora_steps(
    model: nn.Module,
    tokenizer: Any,
    *,
    data_path: Path,
    device: str,
    batch_size: int,
    sequence_length: int,
    rank: int,
    steps: int,
    learning_rate: float,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model

    grouped_modules = linear_modules_by_layer_type(model)
    target_modules = sorted(
        {
            relative_name.rsplit(".", maxsplit=1)[-1]
            for names in grouped_modules.values()
            for relative_name in names
        }
    )
    if not target_modules:
        raise RuntimeError("no decoder nn.Linear modules were found for LoRA")

    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    examples = load_examples(data_path, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = stop_token_ids(tokenizer)[0]
    input_ids, labels, attention_mask, batch_metadata = fixed_shape_batch(
        examples,
        batch_size=batch_size,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    model.train()
    measurements = []
    observed_memory = []
    if mps_memory_gb() is not None:
        observed_memory.append(mps_memory_gb())

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        step_started = time.perf_counter()

        forward_started = time.perf_counter()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = output.logits
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        del output, logits
        if mps_memory_gb() is not None:
            observed_memory.append(mps_memory_gb())

        backward_started = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        if mps_memory_gb() is not None:
            observed_memory.append(mps_memory_gb())

        optimizer_started = time.perf_counter()
        optimizer.step()
        synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        if mps_memory_gb() is not None:
            observed_memory.append(mps_memory_gb())

        elapsed = time.perf_counter() - step_started
        measurements.append(
            {
                "step": step,
                "loss": round(float(loss.detach().cpu()), 6),
                "forward_seconds": round(forward_seconds, 3),
                "backward_seconds": round(backward_seconds, 3),
                "optimizer_seconds": round(optimizer_seconds, 3),
                "total_seconds": round(elapsed, 3),
                "non_padding_tokens_per_second": round(
                    batch_metadata["non_padding_tokens"] / elapsed, 3
                ),
            }
        )

    sampled_peak = None
    if observed_memory:
        sampled_peak = {
            key: max(sample[key] for sample in observed_memory) for key in observed_memory[0]
        }

    mean_step_seconds = sum(item["total_seconds"] for item in measurements) / len(measurements)
    class_recipe_microsteps = 3 * ((300 + batch_size - 1) // batch_size)

    return {
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "batch": batch_metadata,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "rank": rank,
        "lora_alpha": 2 * rank,
        "target_modules": target_modules,
        "modules_by_layer_type": grouped_modules,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "gradient_checkpointing": gradient_checkpointing,
        "measurements": measurements,
        "mean_step_seconds": round(mean_step_seconds, 3),
        "class_recipe_projection": {
            "examples": 300,
            "epochs": 3,
            "microsteps": class_recipe_microsteps,
            "minutes_at_measured_rate": round(
                class_recipe_microsteps * mean_step_seconds / 60,
                3,
            ),
            "note": "Projection excludes evaluation, checkpointing, and data-loader overhead.",
        },
        "sampled_peak_mps_memory_gb": sampled_peak,
        "memory_note": (
            "PyTorch MPS exposes current/driver allocation but no operation-level peak API; "
            "the reported peak is the maximum sampled at phase boundaries."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "parity", "model"), default="all")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "device": device,
        }
    }
    write_report(args.report, report)

    try:
        if args.mode in {"all", "parity"}:
            report["renderer_parity"] = verify_renderer_parity()
            print("renderer parity:", json.dumps(report["renderer_parity"], indent=2))
            write_report(args.report, report)

        if args.mode in {"all", "model"}:
            if device != "mps":
                raise RuntimeError(
                    "P0 requires Apple MPS, but torch.backends.mps.is_available() is false"
                )
            tokenizer, model, load_summary = load_text_model(device, torch.bfloat16)
            report["model_load"] = load_summary
            print("model load:", json.dumps(load_summary, indent=2, default=str))
            write_report(args.report, report)

            model.eval()
            prompts = [
                "Why do cats purr?",
                "What does the Linux cat command do?",
                "Answer with only the number: 17 * 23.",
            ]
            report["greedy_generation"] = [
                greedy_generation(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    args.max_new_tokens,
                )
                for prompt in prompts
            ]
            print("greedy generation:", json.dumps(report["greedy_generation"], indent=2))
            write_report(args.report, report)

            report["lora_step"] = measure_lora_steps(
                model,
                tokenizer,
                data_path=args.data,
                device=device,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                rank=args.rank,
                steps=args.steps,
                learning_rate=args.learning_rate,
                gradient_checkpointing=not args.no_gradient_checkpointing,
            )
            print("LoRA step:", json.dumps(report["lora_step"], indent=2))
            write_report(args.report, report)

        report["status"] = "passed"
        write_report(args.report, report)
        print(f"wrote {args.report}")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_report(args.report, report)
        raise


if __name__ == "__main__":
    main()
