# AI Full Models From Scratch

Implementing AI models from scratch using only NumPy.

## Structure

- **Classic/** — Classic ML algorithms
- **NNs/** — Neural networks

## Projects

### 1. Neural Network for 3-Class Classification
- **Data**: 3 classes, 2 features (custom functions)
- **Model**: 2 → 4 → 4 → 4 → 3 (Softmax)
- **Loss**: Categorical Cross-Entropy
- **Optimizer**: SGD (lr=0.01, batch=32)
- **Result**: 100% test accuracy

**Files**:
- `generate_data.py` — Generate synthetic data
- `analyze_data.py` — Analyze with pandas
- `scale_data.py` — StandardScaler from scratch
- `model.py` — Neural network from scratch
