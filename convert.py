import argparse
import numpy as np
from transformers import GPT2LMHeadModel
from pathlib import Path


def convert_to_mlx(hf_model_path_or_type: str, mlx_model_path: str):
    print(f"Loading model using path or type {hf_model_path_or_type}")
    model = GPT2LMHeadModel.from_pretrained(hf_model_path_or_type)
    sd = model.state_dict()

    num_parameters = sum([np.prod(x.shape) for x in sd.values()])
    print(f"Number of parameters: {num_parameters}")

    # Ignoring masked bias terms as these are defined in the model
    keys = [k for k in model.state_dict() if not k.endswith('attn.masked_bias')]
    # Transposing some of the weights as described in the minGPT code
    # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L196
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    weights = {}
    for key in keys:
        # Updating key names to match the mlx model definition
        mlx_key = key.lstrip('transformer.').replace('.mlp.', '.')
        if any(key.endswith(w) for w in transposed):
            weights[mlx_key] = sd[key].T.numpy()
        else:
            weights[mlx_key] = sd[key].numpy()
    
    weights_path = str(Path(mlx_model_path) / "mlx_weights.npz")
    np.savez(weights_path, **weights)
    print(f"Saved weights to {weights_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path_or_type", default='gpt2-xl', type=str, help="Path with the HuggingFace model or the gpt2 model size, such as gpt2-xl.", required=True)
    parser.add_argument("--mlx_model_path", type=str, help="Path to save the mlx model.", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    convert_to_mlx(args.hf_model_path_or_type, args.mlx_model_path)