# python scripts/inspect_ckpt.py /home/xieminhui/yinj/workplace/recsys-examples/data/ckpts/kuairand-1k/ranking/2025_07_22-09_59_19/final-iter1 --dump-ids
import os
import torch
import argparse
import numpy as np
import json

import io
import numbers

def _describe_value(x):
    # nn.Parameter 先转成 tensor
    if isinstance(x, torch.nn.Parameter):
        x = x.data

    # Tensor
    if isinstance(x, torch.Tensor):
        numel = x.numel()
        size_mb = (numel * x.element_size()) / (1024 ** 2)
        return f"Tensor | shape: {tuple(x.shape)}, dtype: {x.dtype}, numel: {numel}, ~{size_mb:.2f} MB"

    # BytesIO / bytes-like
    if isinstance(x, io.BytesIO):
        try:
            n = x.getbuffer().nbytes
        except Exception:
            pos = x.tell()
            x.seek(0, 2)
            n = x.tell()
            x.seek(pos, 0)
        return f"BytesIO | {n} bytes"
    if isinstance(x, (bytes, bytearray, memoryview)):
        return f"{type(x).__name__} | {len(x)} bytes"

    # 简单 Python 类型
    if isinstance(x, str):
        return f"str | len: {len(x)}"
    if isinstance(x, numbers.Number):
        return f"{type(x).__name__} | value: {x}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__} | len: {len(x)}"
    if isinstance(x, dict):
        return f"dict | keys: {len(x)}"

    # 回退
    return f"{type(x).__name__}"

def inspect_ckpt(path: str, rank: int = 0):
    ckpt_path = os.path.join(path, "torch_module", f"model.{rank}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    # 为了减少反序列化风险，也为了尽量只拿权重，可以考虑设为 True
    # 注意：有些老格式下 weights_only=True 可能拿不到全部信息，必要时改回 False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 适配多种布局
    if hasattr(ckpt, "state_dict"):
        items = ckpt.state_dict().items()
        layout = "object.state_dict()"
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        items = ckpt["model_state_dict"].items()
        layout = "ckpt['model_state_dict']"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        items = ckpt["state_dict"].items()
        layout = "ckpt['state_dict']"
    elif isinstance(ckpt, dict):
        # 有些工程直接把权重字典作为顶层
        items = ckpt.items()
        layout = "ckpt (top-level dict)"
    else:
        print(f"Unsupported checkpoint top-level type: {type(ckpt).__name__}")
        return

    items = list(items)
    print(f"\n=== Checkpoint layout: {layout}; contains {len(items)} entries ===\n")

    for name, value in items:
        print(f"{name:80s} | {_describe_value(value)}")

    # 优化器信息（若存在）
    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
        optim_state = ckpt["optimizer_state_dict"]
        try:
            keys_preview = list(optim_state.keys())
        except Exception:
            keys_preview = ["<unprintable>"]
        print("\n=== Optimizer state dict found ===")
        print(f"Contains keys: {keys_preview}")
    else:
        print("\nNo optimizer state found.")



def read_keys_values_info(key_path, value_path):
    try:
        keys = np.fromfile(key_path, dtype=np.int64)
        values = np.fromfile(value_path, dtype=np.float32)
        num_keys = keys.shape[0]
        dim = values.shape[0] // num_keys if num_keys > 0 else 0
        if num_keys * dim != values.shape[0]:
            return keys, values, num_keys, None, "[shape mismatch]"
        return keys, values, num_keys, dim, None
    except Exception as e:
        return None, None, 0, 0, f"[error reading] {e}"


def inspect_dynamic_embeddings(path: str, dump_ids=False, query_id=None):
    emb_dir = os.path.join(path, "dynamicemb_module")
    if not os.path.exists(emb_dir):
        print("\nNo dynamic embedding directory found.")
        return

    print(f"\n=== Inspecting dynamic embeddings in: {emb_dir} ===\n")

    for root, dirs, files in os.walk(emb_dir):
        # only go into leaf directories
        if not dirs:
            table_prefixes = set()
            for f in files:
                if f.endswith("_keys") or f.endswith("_values") or f.endswith("_opt_args.json"):
                    prefix = f.replace("_keys", "").replace("_values", "").replace("_opt_args.json", "")
                    table_prefixes.add(prefix)

            if table_prefixes:
                print(f"Dynamic embedding module: {os.path.relpath(root, emb_dir)}")
                for name in sorted(table_prefixes):
                    key_path = os.path.join(root, f"{name}_keys")
                    value_path = os.path.join(root, f"{name}_values")
                    opt_args_path = os.path.join(root, f"{name}_opt_args.json")

                    info = f"  - {name}: "
                    info += "key✓ " if os.path.exists(key_path) else "key✗ "
                    info += "value✓ " if os.path.exists(value_path) else "value✗ "
                    info += "opt_args✓" if os.path.exists(opt_args_path) else "opt_args✗"

                    if os.path.exists(key_path) and os.path.exists(value_path):
                        keys, values, num_keys, dim, err = read_keys_values_info(key_path, value_path)
                        if err:
                            info += f" [ERR: {err}]"
                        else:
                            info += f" | #keys: {num_keys}, dim: {dim}"

                            if dump_ids:
                                print(info)
                                print(f"    → First IDs: {keys[:min(10, len(keys))].tolist()}")

                            if query_id is not None:
                                if query_id in keys:
                                    idx = np.where(keys == query_id)[0][0]
                                    vec = values[idx * dim : (idx + 1) * dim]
                                    print(info)
                                    print(f"    → ID {query_id} found, vector = {vec.tolist()}")
                                else:
                                    print(info)
                                    print(f"    → ID {query_id} not found.")
                                continue

                    else:
                        print(info)
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect checkpoint contents")
    parser.add_argument("ckpt_dir", type=str, help="Checkpoint directory path")
    parser.add_argument("--rank", type=int, default=0, help="Rank number (default: 0)")
    parser.add_argument("--dump-ids", action="store_true", help="Print first 10 IDs of each dynamic table")
    parser.add_argument("--query-id", type=int, help="Query a specific ID's vector value")

    args = parser.parse_args()

    inspect_ckpt(args.ckpt_dir, args.rank)
    inspect_dynamic_embeddings(args.ckpt_dir, dump_ids=args.dump_ids, query_id=args.query_id)
