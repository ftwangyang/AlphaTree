# AlphaTree

A Tree-LSTM based reinforcement learning framework for formulaic alpha discovery in quantitative finance.

## Overview

AlphaTree encodes the hierarchical structure of mathematical expressions using Tree-LSTM, capturing operator-operand relationships through bottom-up information propagation. Combined with distributional RL, it addresses non-stationarity and reward sparsity in alpha mining.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download your quantitative data (e.g., CSI300, CSI500)
2. Update the data path in `config.py`:

```python
DATA_PATH = "/your/local/path/to/data/"
```

## Usage

### Training

```bash
python train_AlphaTree.py
```

### Testing

```bash
python test.py
```

## Acknowledgements

This project is inspired by [AlphaQCM](https://github.com/ZhuZhouFan/AlphaQCM). We thank the authors for their excellent work.

## License

MIT License

