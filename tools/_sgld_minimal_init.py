"""Minimal in-process initialisation of sglang-diffusion runtime so that
QwenImageTransformerBlock can be instantiated standalone.

Sets:
  - global server args (attention_backend=torch_sdpa, num_gpus=1)
  - env vars for single-process distributed
  - distributed environment + model parallel groups (tp=1, sp=1, no cfg-parallel)

Call init_minimal() once at the top of any test/harness that imports sgld
runtime modules.
"""
from __future__ import annotations

import os
import sys


def init_minimal(
    model_path: str = "Qwen/Qwen-Image",
    attention_backend: str = "torch_sdpa",
    master_port: int = 29501,
):
    sys.path.insert(0, "/root/diffusion-rl/sglang/python")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    server_args = ServerArgs(
        model_path=model_path,
        attention_backend=attention_backend,
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        enable_cfg_parallel=False,
    )
    set_global_server_args(server_args)

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        enable_cfg_parallel=False,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
    )
