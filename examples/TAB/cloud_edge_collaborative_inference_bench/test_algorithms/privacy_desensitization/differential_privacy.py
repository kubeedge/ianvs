import torch

def apply_differential_privacy(embeddings: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    sensitivity = 1.0
    noise_scale = sensitivity / max(epsilon, 1e-6)
    noise = torch.normal(0, noise_scale, embeddings.shape, device=embeddings.device)
    return embeddings + noise


