from __future__ import annotations


def gpu_status() -> dict:
    try:
        import torch
    except Exception as exc:
        return {
            "requested_backend": "torch",
            "available": False,
            "device": "cpu",
            "reason": f"PyTorch import failed: {exc}",
        }

    if not torch.cuda.is_available():
        return {
            "requested_backend": "torch",
            "available": False,
            "device": "cpu",
            "reason": f"PyTorch is installed but CUDA is unavailable ({torch.__version__}).",
        }

    return {
        "requested_backend": "torch",
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "reason": "CUDA-enabled PyTorch is available.",
    }


def backend_label(use_gpu: bool) -> str:
    if not use_gpu:
        return "CPU / NumPy"
    status = gpu_status()
    if status["available"]:
        return f"GPU / PyTorch CUDA: {status['device']}"
    return f"CPU fallback: {status['reason']}"
