# Reproduction Results vs Med-PaLM M (Paper Table 2)

| Model | Parameters | BLEU-1 (%) | F1 (%) | Source |
|-------|-----------|-----------|--------|--------|
| Prior SOTA (specialist) | Various | 71.03 | — | Paper Table 2 |
| PaLM-E 84B (zero-shot) | 84B | 59.19 | 38.67 | Paper Table 2 |
| Med-PaLM M 12B | 12B | 64.02 | 50.66 | Paper Table 2 |
| Med-PaLM M 84B | 84B | 69.38 | 59.90 | Paper Table 2 |
| Med-PaLM M 562B | 562B | 71.27 | 62.06 | Paper Table 2 |
| **Ours: BLIP-2 (zero-shot)** | **3.4B** | **0.44** | **0.70** | **This work** |
| **Ours: BLIP-2 (fine-tuned)** | **3.4B** | **26.16** | **26.16** | **This work** |

## Exemplar Ablation

| Condition | BLEU-1 (%) | F1 (%) |
|-----------|-----------|--------|
| Without exemplar | 0.95 | 1.52 |
| With exemplar | 2.54 | 3.86 |
| **Improvement** | **+167%** | **+154%** |

## Training Progress

| Epoch | Loss |
|-------|------|
| 1 | 9.9688 |
| 2 | 9.4655 |
| 3 | 8.7573 |
| 4 | 8.3816 |
| 5 | 8.0812 |
