"""Export PPO policy to ONNX format."""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def export_ppo_to_onnx(
    model,
    output_path: str,
    observation_shape: Tuple[int, ...],
    opset_version: int = 14,
) -> None:
    """Export PPO policy network to ONNX.

    Args:
        model: Stable-Baselines3 PPO model
        output_path: Output path for ONNX file
        observation_shape: Shape of observation space
        opset_version: ONNX opset version
    """
    # Extract policy network
    policy = model.policy
    policy.eval()

    # Create dummy input
    dummy_obs = torch.randn(1, *observation_shape).to(model.device)

    # Wrap policy for ONNX export
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            # Get action from policy
            with torch.no_grad():
                features = self.policy.extract_features(obs)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                action_logits = self.policy.action_net(latent_pi)
                # For discrete actions, return action with highest probability
                action = torch.argmax(action_logits, dim=-1)
            return action

    wrapped_policy = PolicyWrapper(policy)

    # Export to ONNX
    torch.onnx.export(
        wrapped_policy,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    print(f"ONNX model exported to: {output_path}")

    # Verify export
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        print("ONNX model verification successful!")

        # Test inference
        test_input = np.random.randn(1, *observation_shape).astype(np.float32)
        outputs = session.run(None, {"observation": test_input})
        print(f"Test inference output: {outputs[0]}")

    except ImportError:
        print("onnxruntime not available for verification")
    except Exception as e:
        print(f"ONNX verification failed: {e}")

