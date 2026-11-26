# IB-Quantization-LLM

Information-Bottleneck guided mixed-precision quantization for LLMs.

## Key Finding

**KL-divergence sensitivity varies 121× across Llama 3.1 8B's 32 layers.** Layer 0 dominates at 4.2× while mid-layers sit at ~0.1×. Uniform quantization ignores this; sensitivity-guided bit allocation exploits it.

## The Problem: Silent Degradation

Standard metrics (perplexity, accuracy) stay stable while token-distribution KL divergence blows up. A model can look fine on benchmarks while its output distribution has silently drifted. This repo detects and addresses that.

## Status

| Component | Status |
|-----------|--------|
| Silent degradation discovery | ✅ Validated |
| 121× layer-sensitivity map | ✅ Validated on Llama 3.1 8B; replicates on GPT-2 at n=500 |
| IB information-plane term | ❌ No signal beyond KL sensitivity (GPT-2, n=500) |
| IB rerun at Llama 3.1 8B scale | ⏳ Open |
| GPTQ/AWQ baseline comparison | ❌ Not run |

### Critical Bug Found (Sept 2026)

The MI estimator used to compute `arrow_len` (the information-plane term) was broken:

```
mutual_information(X, Y, n_bins=5)  # with n=20 samples
```

**Result:** Reported MI = 0.64 on pure independent noise (true = 0). The ablation correctly zeroed the term — but because the estimate was garbage, not because information-plane geometry is useless.

**Fix:** `scripts/mi_estimator.py` now uses:
- KSG (k-nearest-neighbour) estimator instead of histogram
- Adaptive bin count when using histogram
- Mandatory shuffle noise floor — if not 2σ above shuffle baseline, treat as zero

### Rerun result (Sept 2026)

Stage 2 was rerun on GPT-2 at n=500 with the corrected KSG estimator and a pre-registered three-check rule (MI significance ≥ 50%, |spearman(d, s)| < 0.90, α* < 1.0 winning on ≥ 2 of 3 bit targets).

- MI significance: 38% against the 50% bar
- α* = 1.0 at 2 of 3 bit targets
- Displacement shrank in step with the noise floor as n rose

**Conclusion:** the information-plane term adds no allocation signal beyond raw KL sensitivity. At α = 1.0 the β score equals KL sensitivity, so IB-allocated and sensitivity-allocated models are identical. The KL sensitivity map itself replicates (layer 0 ≈ 5× the next layer on GPT-2).

`displacement_ratio` in `results/llama31_results.json` comes from the old estimator and is superseded.

## Repository Structure

```
├── notebooks/
│   ├── llama31_ib_complete_pipeline.ipynb   # Main: Llama 3.1 8B
│   ├── stage2_ib_validation.ipynb           # GPT-2 IB framework validation
│   ├── stage3_ib_algorithm.ipynb            # GPT-2 full IB algorithm
│   ├── gpt2_kl_accuracy_all_configs.ipynb   # Stage 1: silent degradation
│   ├── gpt2_demo_for_professor.ipynb        # Clean demo
│   ├── ib_vs_original.ipynb                 # IB vs baseline comparison
│   └── ib_mi_diagnostics.ipynb              # MI estimator bug diagnosis
├── scripts/
│   ├── mi_estimator.py                      # Fixed MI estimator with noise floor
│   ├── allocate.py                          # Bit allocation for framing B
│   ├── evaluate.py                          # Shared evaluation harness
│   ├── ib_t_selection_experiment.py         # T-selection validation
│   └── stage2_ib_validation.py              # Stage 2 validation
├── figures/
│   └── plots/                               # Llama 3.1 8B result figures
├── results/
│   ├── llama31_results.json                 # Llama 3.1 8B experiment outputs
│   └── stage3_results.json                  # GPT-2 Stage 3 outputs
├── legacy/                                  # Superseded notebooks
└── tests/                                   # pytest checks run by GitHub Actions
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

For Apple Silicon (MLX) — optional:
```bash
pip install -r requirements-apple-silicon.txt
```

### Run the diagnostics

```bash
python scripts/mi_estimator.py
```

This reproduces the MI estimator failure and verifies the fix.

### Run GPT-2 experiments (~20 min)

```bash
python scripts/ib_t_selection_experiment.py
python scripts/stage2_ib_validation.py
```

Then open `notebooks/stage2_ib_validation.ipynb` → `notebooks/stage3_ib_algorithm.ipynb`.

### Llama 3.1 8B (requires ≥16 GB RAM or Lightning AI GPU)

Open `notebooks/llama31_ib_complete_pipeline.ipynb` and set your HuggingFace token.

## Results

| Model | Method | Avg Bits | HellaSwag | WikiText-2 PPL | KL vs FP32 |
|-------|--------|----------|-----------|----------------|------------|
| Llama 3.1 8B | FP32 Baseline | 32 | 54.5% | 8.9 | — |
| Llama 3.1 8B | Uniform INT8 | 8 | 56.0% | 9.7 | baseline |
| Llama 3.1 8B | Random 7-bit | 7 | 35.0% | — | — |
| Llama 3.1 8B | **KL-sensitivity 7-bit** (IB at α=1.0) | 7 | **46.5%** | **42.3** | **lowest** |
| Llama 3.1 8B | Uniform INT4 | 4 | 23.0% | 1,007,348 | extreme |

> **Read by matched bit-budget.** INT8 beats this row because it uses more bits, so that comparison isn't fair. At a matched 7-bit budget, KL-sensitivity allocation beats random by 11.5 points.

> **Not compared against GPTQ or AWQ.** KL-sensitivity bit allocation also appears in concurrent 2025–26 work, so this result is a replication, not a new method.


## Planned Work

1. **Rerun the corrected estimator on Llama 3.1 8B** to close the one open gap before calling the IB result final.
2. **Grouped int4 quantizer** (hand-written, per-group) to check that the allocation result survives a non-naive int4 baseline.

## Citation

If you use this work, please cite:

```bibtex
@misc{bandi2026ibquant,
  author = {Bandi, Sai Vikas},
  title = {Information-Bottleneck Guided LLM Quantization},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/bandisaivikas/IB-Quantization-LLM}
}
```

## License

MIT
