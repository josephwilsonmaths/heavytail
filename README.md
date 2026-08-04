# Why Do Heavy-Tailed Weights Predict Model Quality?

Code for the paper *Why Do Heavy-Tailed Weights Predict Model Quality?*

## Setup

Python 3.10 or newer is required. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jupyter and WeightWatcher are only needed for the corresponding optional notebook cells:

```bash
python -m pip install jupyter weightwatcher
```

## Results data

Large checkpoint-summary files are not stored in GitHub because of repository file-size limits. Before running the analyses, create the output directories:

```bash
mkdir -p results/htmp/epoch results/htmp/fits results/htmp/cv figures/htmp_fits
```

In Windows PowerShell, use:

```powershell
New-Item -ItemType Directory -Force results/htmp/epoch, results/htmp/fits, results/htmp/cv, figures/htmp_fits
```

Populate `results/htmp/epoch` in either of these ways:

- Run the training script, for example:

  ```bash
  python epoch.py --model resnet18 --dataset cifar10 --batch_size 100 --num_repeats 1 --progress
  ```

  Run `python epoch.py --help` for the available model, dataset, batch-size, and permuted-label options.

- Generate summaries from Pythia checkpoints:

  ```python
  from helpers import save_pythia_checkpoint_summary

  save_pythia_checkpoint_summary('pythia-70m', step=1000)
  ```

  This downloads the requested Hugging Face checkpoint and writes its spectral summary to `results/htmp/epoch`. Pythia checkpoints can require substantial disk space and memory.



## Main files

- `epoch.py`: trains the image-classification models and periodically saves metrics and weight spectra.
- `heavy_tails.ipynb`: image classification notebook to generate figures.
- `llm_ht.ipynb`: large language model notebook to generate figures.
- `helpers.py`: shared numerical, plotting, dataset, and checkpoint-summary utilities, including Pythia summary generation.
- `esdfit.py`: loads saved summaries, fits heavy-tailed Marchenko-Pastur models, and creates analysis figures.
- `htmp_cdf.py`: evaluates HTMP CDFs and KS distances; run `python htmp_cdf.py --help` for its command-line interface.

