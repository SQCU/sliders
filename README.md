# Concept Sliders PART 2:
## OPERATIONAL LINGO -- 'AMBITIOUS BATCHER'
###  [Upstream Project Website](https://sliders.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2311.12092.pdf) | <br>
built from Official code implementation of "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models", European Conference on Computer Vision (ECCV 2024).

## setup:
```
uv init
uv python install 3.10
uv venv --seed --python 3.10

uv pip install torch --index pytorch=https://download.pytorch.org/whl/cu124
uv add https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cp310-cp310-win_amd64.whl
uv pip install psutil
uv pip install packaging
uv pip install ninja
$Env:MAX_JOBS = "7"
$Env:TORCH_CUDA_ARCH_LIST = "8.9"
.venv\scripts\activate
#if you're stuck on a 4090, this is the only compute capability you can use
#expect like 30 minutes of total saturation of 7/8 cores on a 7800x3d.
uv pip install flash-attn==2.7.2.post1 --no-cache-dir --no-build-isolation
#??? now we need no-cache-dir since we start from a real pyproject.toml???
#MANUALLY OVERWRITE PYPROJECT.TOML WITH REAL INDEXES
uv add torch torchvision torchaudio
uv add -r requirements-loose.txt
uv add bitsandbytes>=0.43.0
uv add lycoris-lora? #maybe...
```

or maybe...
```
sudo apt install python3-dev
...
uv venv --seed
source .venv/bin/activate && uv pip install 
psutil && uv pip install torch && uv pip install flash-attn==2.7.2.post1 --no-build-isolation
```
...
enjoy 'wsl-cuda-bootstrap.sh'!~
...

## config: 
edit f"prompts-xl-dilora-{your_experiment}.yaml
to be either:
```
- target: "" # what word for erasing the positive concept from
  positive: "" # concept to erase
  unconditional: "" # word to take the difference from the positive concept
  neutral: "" # starting point for conditioning the target
  action: "enhance" # erase or enhance
  guidance_scale: 4
  resolution: 1024
  dynamic_resolution: false
  batch_size: 1
```
or
```
- target: f"{nobody in this entire world knows}" # used only in textsliders, a loss function which has so far gone unstudied.
  positive: f"{variant_more_of_feature in images}, {thing ur varying on purpose}" # concept to "erase" / "enhance"
  unconditional: f"{invariant in images}" # base prompt for classifier free guidance. doesn't need to be empty!
  neutral: f"{variant_less_of_feature in images}" # starting point for conditioning the target
  action: f"enhance" # erase or enhance
  guidance_scale: 4
  resolution: 1024
  dynamic_resolution: false
  batch_size: 1
```
datasets must be made of images with identical filenames spread across every folder you include in a list of argparse operands to `trainscripts/imagesliders/train_lora-scale-xl.py`.
or like, maybe they do. things could be different now?

our suggested training template:

`python -m trainscripts.imagesliders.batched_training_loop -c F:/dox/ai/gemmy/sliders/trainscripts/imagesliders/data/batch_config.yaml`
this is a python script you can run! it will point towards this script by default, but a config argparse is included for legibility and to make it more clear to human and LLM participants where to go to find important settings.

choices of 'scale' *are* semantically meaningful: changing the interval between these numbers changes how the 'slider' learns to separate and fuse visual ideas in very definite ways.
think of this as a dramatic departure from the upstream 'slider' project, to make it more obvious how much *can* be changed without violating any of the basic organizing ideas of how the objective function is defined and maximized.

## eval and inference:
if you use comfyui i am very sorry and i hope you recover, someday, somehow.
maybe if just 12 more video essayists and 3 more dormant non-programmer patreons pick up your preferred 'workflow' you'll finally figure out the obvious and very smart deployment case that makes your text2image so unique and interesting and different from everyone else's all this time...?

for everyone else: i recommend gemini-cli. also, the `dynamic prompts` extension to the automatic-like webuis supports really easy scripting. but maybe you could make a better UI yourself? without needing gradio or anyone else's help?
`{<slider:0>|<slider:0.8>|<slider:1.1>|<slider:2.7>} w/ fixed seeds and combinatorial generation will make it very easy to sample your 'fractional differences' & explore the transitions between slider-multiplier-conditioned model behavior. 

if you are rolling your own inference, this is even easier! i think i've said it all with 'combinatorial generation and fixed seeds', haven't i?

## Logging and Debugging
some notes for cli-LLM tool users who are skimming this readme:

### Standardized Logging
All script executions are now piped to a timestamped log file in the `logs/` directory. This ensures that outputs are captured for later review and debugging.

### Log File Tail Utility
To easily view the end of log files, a Python utility `tail_log.py` is provided. This script mimics the behavior of the Unix `tail -n` command.

**Usage:**
```bash
python tail_log.py <path_to_log_file> -n <number_of_lines>
```
**Example:**
```bash
python tail_log.py logs/batched_training_loop_2025-07-04_10-30-00.log -n 20
```

### Diff Debugger Suite
`uv run python diff_debugger.py`
This script loads and analyzes the difference tensors generated by `refactored_training_loop.py` to help in debugging numerical discrepancies. It provides basic statistics (min, max, mean, std dev) for each difference tensor.


## Citing the upstream work:
The upstream's preprint can be cited as follows
```
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Tingrui Zhou and Antonio Torralba and David Bau},
  booktitle={Proceedings of the 2024 IEEE European Conference on Computer Vision},
  note={arXiv preprint arXiv:2311.12092},
  year={2024}
}
```

## citing this work:
huh? wuh?
