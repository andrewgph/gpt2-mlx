# gpt2-mlx

An implementation of the GPT-2 model in the [mlx](https://github.com/ml-explore/mlx) framework for Apple silicon.

Based upon [karpathy](https://github.com/karpathy)'s implementation of GPT-2 in [minGPT](https://github.com/karpathy/minGPT), and following the style of the examples in [mlx-examples](https://github.com/ml-explore/mlx-examples).

Code changes from PyTorch implementation:
* Some convenience functions missing from mlx, such as ModuleList, ModuleDict, masked_fill.
* Some functions have different arguments, e.g. split and transpose.

It uses the HuggingFace tokenizer, but you could use the BPE encoder in minGPT instead.

## Usage

[convert.py](convert.py) can convert a HuggingFace GPT-2 model into mlx weights.

Example usage:

```
python convert.py --hf_model_path_or_type "gpt2-xl" --mlx_model_path "~/mlx-gpt2-model"
```

You also need to copy over the config.json for the model to the same folder, such as this [config.json](https://huggingface.co/gpt2-xl/blob/main/config.json) for gpt2-xl.

[gpt2.py](gpt2.py) contains the model code.

[generate.ipynb](generate.ipynb) shows a comparison of generating using HuggingFace gpt2 vs this mlx implementation.

[train.ipynb](train.ipynb) shows a simple example of training a gpt2-nano model for sorting short sequences, based upon the [demo.ipynb](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/demo.ipynb) in [minGPT](https://github.com/karpathy/minGPT).