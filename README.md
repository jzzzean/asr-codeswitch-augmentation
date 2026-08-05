# Evaluating Data Augmentation for Mandarin-English Code-Switching ASR

This course research project examines how synthetic audio transformations affect pretrained automatic speech recognition (ASR) systems on Taiwanese Mandarin-English code-switching speech. Rather than assuming that augmentation always improves robustness, we evaluate when concatenation, background noise, speed changes, and pitch shifts create useful test conditions or harmful distribution shifts.

## Research questions

- How well do Whisper-small and Faster-Whisper-small recognize Mandarin-English code-switching speech?
- How does performance change under utterance concatenation, background noise, and audio perturbation?
- What kinds of errors and dataset characteristics help explain the observed performance differences?

## Experimental design

### Data

- [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND): conversational Mandarin-English code-switching speech
- [ML2021-ASR-ST](https://huggingface.co/datasets/ky552/ML2021_ASR_ST): Taiwanese Mandarin speech with English content
- Mozilla Common Voice Mandarin and English samples for constructing synthetic code-switched evaluation sets

The datasets are not redistributed in this repository. Follow the respective dataset pages and licenses to obtain them.

### Augmentation conditions

1. **Utterance concatenation** - combines Mandarin and English segments into synthetic code-switched utterances.
2. **Background noise** - adds noise at randomly sampled signal-to-noise ratios.
3. **Audio perturbation** - varies speed and pitch to test model sensitivity.

### Models and evaluation

- Whisper-small
- Faster-Whisper-small
- Mixed Error Rate (MER), with Chinese characters and English words used as evaluation units
- Dataset-level analysis of duration, volume, silence ratio, and estimated SNR

## Main findings

The saved experimental runs indicate that Faster-Whisper-small performed slightly better than Whisper-small on the baseline evaluation. Both systems degraded substantially on the synthetic augmented sets, and audio perturbation produced the largest degradation for Whisper-small.

| System | Condition | Average MER |
| --- | --- | ---: |
| Whisper-small | Baseline datasets | 28.39% |
| Faster-Whisper-small | Baseline datasets | 27.47% |
| Whisper-small | All augmented sets | 99.42% |
| Faster-Whisper-small | All augmented sets | 74.05% |
| Whisper-small | Audio perturbation | 131.47% |

MER can exceed 100% when insertions are frequent. These values come from the original notebook runs and should be interpreted within their corresponding evaluation conditions, not as a leaderboard comparison.

## Repository structure

```text
.
|-- data_processing/
|   |-- data_analyzation.py       # Dataset statistics and exploratory analysis
|   |-- data_augmentation.py      # Audio augmentation utilities
|   `-- generate_augmented_sets.py# Synthetic evaluation-set generation
|-- model_processing/
|   |-- baseline_whisper_small.ipynb
|   |-- baseline_faster_whisper.ipynb
|   |-- augmented_whisper_small.ipynb
|   `-- augmented_faster_whisper.ipynb
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Place locally obtained datasets under `data/`, or set `ASR_DATA_DIR` to a different data directory. Generated audio, model files, and local datasets are ignored by Git.

```text
data/
|-- raw_data/
|   |-- common_voice_zh/
|   |-- common_voice_en/
|   `-- noise/
`-- augmented_data/
```

Generate augmented sets:

```bash
python data_processing/generate_augmented_sets.py
```

Run dataset analysis:

```bash
python data_processing/data_analyzation.py
```

The model notebooks were developed for Google Colab and require a GPU for practical execution time. Update the data path cell to point to your own authorized dataset copy.

## Team contributions

- **Yu-Chun Chao ([@jzzzean](https://github.com/jzzzean))** - data preparation, augmentation pipeline, dataset analysis, and exploratory evaluation.
- **Paul Cheng ([@ppaull07](https://github.com/ppaull07))** - baseline and augmented-model inference notebooks and repository maintenance.

This contribution summary is based on the repository history and should be confirmed by both collaborators before publication.

## Limitations

- Concatenating independently recorded segments does not fully reproduce naturally occurring code-switching.
- Results depend on dataset composition, normalization choices, and MER aggregation.
- The experiments evaluate pretrained models; they do not demonstrate that augmentation improves model training.
- Reproducibility requires users to obtain the datasets under their original licenses.

