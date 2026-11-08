# BanglaMixSpeech-v2: Experimental Reproduction Code

This repository contains the scripts and configurations used to prepare the BanglaMixSpeech two-speaker mixtures and run the experiments described in **“Benchmarking Neural Models for Bengali Speech Separation Across Controlled Overlap Conditions Using the BanglaMixSpeech-v2 Dataset.”** It is intentionally limited to one README and the code/configuration needed for the workflow.

## 1. Experiments covered

The paper evaluates four two-speaker, single-channel separation systems in two conditions:

- **Zero-shot:** public pretrained checkpoint applied directly to Bengali test mixtures.
- **Fine-tuned:** public checkpoint adapted on the Bengali training split, selected using validation data, then evaluated on the same held-out test mixtures.

Models: Conv-TasNet, DPRNN, SepFormer, and MossFormer2.

The reported split is speaker-disjoint: 140 training speakers, 20 validation speakers, and 20 test speakers. The paper's experiment uses 25,000 training mixtures, 3,125 validation mixtures, and 8,000 test mixtures. The test set has four interval-overlap groups (15–35%, 35–65%, 65–85%, 85–100%), with 2,000 mixtures in each group.

## 2. Repository contents

```text
.
├── README.md
├── dataset/
│   ├── scripts/                       # audio preparation, speaker splitting, mixture generation
│   └── release_validation_v1.0.1/     # optional corrected-release validation/repair utilities
├── experiments/
│   ├── zero_shot/                     # model adapters, inference, metrics, benchmark runner
│   └── fine_tuning/                   # training pipeline, model trainers, YAML config, tests
└── analysis/
    └── reproduce_tables_figures.py   # analysis/plotting code extracted from research notebook
```

No paper PDF, result tables, figures, notebook, audit reports, dataset audio, pretrained weights, or fine-tuned weights are bundled. Those are not executable source code and some are not redistributable. Only the root `README.md` is used for documentation.

## 3. Recorded experimental environment

The authors' recorded environment for the canonical model runs was:

| Component | Recorded version |
|---|---|
| Operating environment | Conda environment named `whisper` |
| Python | 3.10.19 |
| PyTorch | 2.10.0+cu128 |
| Torchaudio | 2.10.0+cu128 |
| Torchvision | 0.25.0+cu128 |
| CUDA runtime reported by PyTorch/SpeechBrain | 12.8 |
| cuDNN Python wheel | 9.10.2.21 |
| SpeechBrain | 1.1.0 |
| Asteroid | 0.7.0 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| SoundFile | 0.13.1 |
| PESQ | 0.0.4 |
| pystoi | 0.4.1 |
| mir_eval | 0.8.2 |
| HyperPyYAML | 1.2.3 |
| PyYAML | 6.0.3 |
| yamlargparse | 1.31.1 |

The original experiments used a CUDA-capable NVIDIA GPU. The authors' records mention an RTX 4090 and later RTX 5090 for the MossFormer2 continuation, but the exact historical NVIDIA driver version was not recorded. CUDA 12.8 is the recorded software runtime, not the driver version. Results may vary with GPU, driver, package builds, precision, and model dependencies.

The files `experiments/zero_shot/requirements/runtime.txt` and `experiments/fine_tuning/requirements.txt` list the principal Python dependencies. They are compatibility/install guidance, not a complete lockfile. For a close environment recreation, use Python 3.10 and install a PyTorch/Torchaudio build compatible with the local NVIDIA driver and CUDA runtime before installing the remaining packages.

Example environment creation:

```bash
conda create -n banglamix-repro python=3.10.19 -y
conda activate banglamix-repro
# Install the CUDA-enabled PyTorch and Torchaudio build appropriate for your machine.
# Then install the experiment dependencies:
pip install -r experiments/zero_shot/requirements/runtime.txt
pip install -r experiments/fine_tuning/requirements.txt
```

Some model backends (especially ClearerVoice/MossFormer2 and SpeechBrain) have additional upstream installation requirements. Use the model-specific dependency versions compatible with the recorded stack; do not assume that a generic `pip install` on a newer Python/PyTorch release is equivalent to the authors' environment.

## 4. Data required before running

The audio files are not included. Obtain the authorized BanglaMixSpeech release and the original source recordings through the dataset provider. The expected two-speaker mixture structure for the benchmark runner is:

```text
BanglaMix/
└── Mix_Speech/Mix_For_Two_Speaker/
    ├── train/<variation>/<mixture_id>/{mix.wav,s1.wav,s2.wav}
    ├── validation/<variation>/<mixture_id>/{mix.wav,s1.wav,s2.wav}
    └── test/<overlap_profile>/<mixture_id>/{mix.wav,s1.wav,s2.wav}
```

Each mixture folder must contain the mixture waveform and its two aligned source references. Audio is mono, 16 kHz PCM in the dataset described by the paper. Do not distribute source recordings unless their consent, licenses, and redistribution permissions explicitly allow it.

### Dataset version warning

The research archive contains both BanglaMix Experiment v1.0.0 and a corrected v1.0.1 release. The archived model results/checkpoints correspond to the historical v1.0.0 experiment. Version v1.0.1 repairs 652 mixture triplets and is not interchangeable with the historical input data. To reproduce the paper's archived numbers, use the exact historical v1.0.0 data and matching checkpoints. Running the corrected v1.0.1 data requires fresh inference and, for fine-tuned results, retraining; do not label v1.0.1 runs as reproductions of the archived v1.0.0 scores.

## 5. Dataset preparation scripts

The scripts in `dataset/scripts/` implement source-audio processing, utterance segmentation, speaker-level partitioning, split statistics, two-speaker mixture generation, and mixture summaries.

Run each script's help before execution to inspect its required input paths and options:

```bash
python dataset/scripts/process_speaker_audio.py --help
python dataset/scripts/speaker_utterance_split.py --help
python dataset/scripts/split_speaker_dataset.py --help
python dataset/scripts/banglamix_two_speaker_mixture_generator.py --help
python dataset/scripts/summarize_banglamix_mixtures.py --help
```

The generator includes built-in checks:

```bash
python dataset/scripts/banglamix_two_speaker_mixture_generator.py --self-test
python dataset/scripts/banglamix_two_speaker_mixture_generator.py --dry-run
```

The dataset scripts expect source audio and metadata supplied separately. Their path configuration must be updated for the local dataset layout before running. Use the speaker partition before generating mixtures to prevent speaker leakage between train, validation, and test.

The optional `dataset/release_validation_v1.0.1/` scripts validate and repair the corrected release only. They must not be applied to historical v1.0.0 data when attempting to reproduce the original results.

## 6. Zero-shot inference and evaluation

The zero-shot code includes adapters for all four models, checkpoint loading, dataset traversal, permutation-invariant source assignment, and calculation of SI-SDRi, SDRi, PESQ, and STOI.

After preparing the dataset and downloading the correct public checkpoints, configure model/checkpoint locations in `experiments/zero_shot/config/models.json` and review `config/defaults.env`. Then run:

```bash
cd experiments/zero_shot
SPEAKERS=2 ./run_zero_shot.sh /absolute/path/to/BanglaMix
```

For a small pipeline smoke test:

```bash
MAX_ITEMS=2 METRICS=fast SPEAKERS=2 ./run_zero_shot.sh /absolute/path/to/BanglaMix
```

Run the preflight and tests before a full evaluation:

```bash
python scripts/preflight.py --help
python -m pytest -q tests
```

The full benchmark evaluates the same 8,000 held-out test mixtures for each model. Preserve mixture IDs and overlap labels in the output CSVs to enable paired comparisons.

## 7. Fine-tuning

The fine-tuning implementation and paper configuration are in `experiments/fine_tuning/`. Review `configs/all_models.portable.yaml` and set the dataset root and checkpoint locations for the local machine. The paper's shared settings include seed 777, 30 epochs, Adam, learning rate 5e-5, weight decay 1e-5, gradient clipping 5, and aligned 4-second training crops. Model-specific settings differ:

| Model | Batch size | Gradient accumulation | Precision | Selected epoch in paper |
|---|---:|---:|---|---:|
| Conv-TasNet | 8 | 1 | FP16 | 30 |
| DPRNN | 3 | 1 | FP16 | 29 |
| SepFormer | 1 | 1 | FP16 | 26 |
| MossFormer2 | 1 | 4 | BF16 | 24 |

The configuration/CLI exposes environment and path checks before training. From the fine-tuning directory:

```bash
cd experiments/fine_tuning
python finetune.py doctor --all
python finetune.py prepare -c configs/all_models.portable.yaml
python finetune.py plan -c configs/all_models.portable.yaml --devices cuda:0
python finetune.py run -c configs/all_models.portable.yaml --devices cuda:0
```

These commands require the full dataset and the corresponding pretrained checkpoints. Training all four systems is computationally expensive. The paper reports one fine-tuning seed per model; reproducing the exact reported checkpoints also requires the same initial checkpoint files and model implementation versions.

## 8. Statistical analysis and figures

After producing the per-mixture result CSVs in the expected results layout, run:

```bash
python analysis/reproduce_tables_figures.py
```

This script contains the analysis and plotting code extracted from the supplied research notebook. It expects the result CSVs to be available in the location resolved by the script and requires NumPy, pandas, SciPy, and Matplotlib. It performs result integrity checks, computes overlap-wise summaries, paired zero-shot/fine-tuned gains, speaker-pair-clustered bootstrap intervals, reliability summaries, metric correlations, and publication figures/tables.

## 9. Metrics and evaluation details

- **Primary metric:** SI-SDR improvement (SI-SDRi), calculated against the unprocessed mixture.
- **Complementary metrics:** SDR improvement (SDRi), PESQ, and STOI.
- The two estimated sources are assigned to the two references using the permutation that maximizes summed SI-SDR; that assignment is reused for the other metrics.
- Signals are scored at 16 kHz. Model outputs are resampled from each model's native rate; signals are truncated to their shortest common length.
- The two source scores are averaged to obtain one record per mixture.
- Overall and overlap-specific results are unweighted means.
- Confidence intervals use 5,000 speaker-pair-cluster bootstrap replicates as described in the paper.

## 10. Reproduction boundaries

This code-only repository does not contain the audio dataset, source recordings, pretrained checkpoints, or fine-tuned checkpoint weights. Those assets must be obtained separately and their versions/fingerprints must match the experiment. Without them, a user can inspect and test the scripts but cannot independently rerun the full numerical experiments from this repository alone.

The paper's throughput values are environment-specific observations: Conv-TasNet 267.2× real time, DPRNN 50.1×, SepFormer 92.8×, and MossFormer2 14.0× under the reported execution settings. Different batch strategies and model sample rates were used, so these values should not be interpreted as a controlled architecture-only speed comparison.

The paper's listening assessment is a separate human evaluation; the scripts cannot reproduce human judgments automatically. Three evaluators assessed selected high- and low-scoring examples.
"# BanglaMixSpeech-v2" 
