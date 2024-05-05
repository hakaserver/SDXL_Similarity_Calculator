# SDXL Similarity Calculator
Simple script to compare multiple Stable Diffusion XL checkpoints.

Adapted from [ASimilarityCalculatior](https://huggingface.co/JosephusCheung/ASimilarityCalculatior) (sic).\
Modified to match SDXL checkpoints' block numbers.

The original script was renamed to _Similarity_Calculator.py_ (Stable Diffusion 1.5 models).\
The modified script is named _Similarity_Calculator_XL.py_ (Stable Diffusion XL models).

## Usage

Ensure that your current Python installation includes both SafeTensors and Torch packages.\
`pip install safetensors torch`  or  `pip install -r requirements.txt`

To compare checkpoints, simply pass two _.safetensors_ files:\
`python Similarity_Calculator_XL.py 'path\to\checkpoint_1.safetensors' 'path\to\checkpoint_2.safetensors'`

To compare __one checkpoint__ to multiple checkpoints, pass the base checkpoint first, then pass all checkpoints you want to compare it with:\
`python Similarity_Calculator_XL.py 'path\to\base_checkpoint.safetensors' 'path\to\checkpoint_1.safetensors' 'path\to\checkpoint_2.safetensors' 'path\to\checkpoint_3.safetensors'`

You can iterate through multiple checkpoints in a specific folder using terminal commands, for example:\
`python Similarity_Calculator_XL.py 'path\to\base_checkpoint.safetensors (ls 'path\to\folder\*.safetensors | %{ $_.FullName })`
