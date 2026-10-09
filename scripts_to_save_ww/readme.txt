Firstly, to run 'run_old_ww_on_all_current_torchvision.py', you need to create a conda environment, and install environment.yml. 
Then, running 'run_old_ww_on_all_current_torchvision.py' in that environment will save the WW summaries. These summaries are accessed in heavy_tails.ipynb.

The script 'download_ww2020_summaries_as_v1_json.py' downloads the summaries from the original WW repository, to compare on legacy models.

Full-checkpoint Pythia comparison (step143000)
--------------------------------------------
Use the ww_old_current_tv environment (weightwatcher==0.2.7), with
huggingface_hub, safetensors, psutil, threadpoolctl and openpyxl installed.
From the repository root:
    python -m scripts_to_save_ww.run_pythia_ww
Select sizes with --models; --step defaults to 143000. All eight deduped sizes
are included by default. Downloads are pinned to the step's exact HF commit.
HF_TOKEN or the logged-in HF account is used; credentials are never saved.

Each full checkpoint shard is downloaded and read one matrix at a time.
All 2D .weight tensors are analyzed, including both embeddings and every
attention/MLP projection. Biases and 1D normalization parameters do not have
matrix spectra and are excluded. WW metrics use the actual 0.2.7 implementation
with its original randomized truncated SVD and dense scaling. Alpha is the PL
fit; lambda is the paper-style E-TPL metric, computed separately on the full
unscaled spectrum with the peak-based cutoff. Package versions are saved.

Results: results/pythia/ww/{summaries,details,partial,checkpoint_cache}
Every successful matrix is saved atomically. Rerun the same command after an
interruption or low-memory failure to resume. Completed summaries skip both
downloads and fitting. Per-model locks prevent conflicting concurrent runs.
The RAM guard is intentionally conservative; close memory-heavy applications
rather than disabling it. Full analyses of the largest sizes can take hours.

The final Excel workbook has Correlations, BFE vs WW, and Model metrics sheets.
BFE-W keeps llm_ht.ipynb's EmbedOut definition with lambda=1/mean(e), zero jitter,
and normalized BFE. Performance values are selected at exactly step143000.
Cached correlation-only export:
    python -m scripts_to_save_ww.run_pythia_ww --export-only
The corresponding notebook cells need no texplot and allow task/shot/metric
selection. --metric ppl uses the existing evaluation helper's log perplexity.
