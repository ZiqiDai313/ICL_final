# In-Context Learning (ICL) Optimizer Effects Study

This project investigates how different optimizers affect the in-context learning capabilities of transformer models across various regression tasks.

## Overview

In-context learning (ICL) is an emergent capability of transformer models where they can adapt to new tasks based on examples provided in the input context, without weight updates. This project studies how different optimization algorithms influence ICL abilities, particularly comparing linear vs. quadratic regression tasks.

## Key Components

- **ICL Implementation**: Transformer model that learns to perform regression tasks from context examples.
- **Task Types**: Includes linear and quadratic regression with controlled difficulty.
- **Optimizers**: Compares Adam, SGD, and Adagrad optimization algorithms.
- **Analysis Tools**: Gradient, Hessian, and attention map analysis for deep model understanding.

## Project Structure

- `main_icl_solver.py`: Main training and evaluation code
- `run_all_experiments.py`: Script to run all optimizer/task combinations 
- `run_experiments.py`: Wrapper script with optimized learning rates
- `data_generator.py`: Generates synthetic regression data
- `linear_transformer.py`: Transformer implementation optimized for ICL
- `analysis_utils.py`: Tools for analyzing gradients, Hessian, etc.
- `EXPERIMENT_NOTES.md`: Detailed findings and observations

## Key Findings

1. **Linear vs. Quadratic Tasks**:
   - Linear regression is well-handled by all optimizers
   - Quadratic regression introduces numerical instability and requires careful tuning

2. **Optimizer Performance**:
   - Adam and Adagrad: Achieve better training loss but with large generalization gaps on quadratic tasks
   - SGD: More stable with lower generalization gaps on quadratic tasks, but higher overall loss

3. **Learning Rate Sensitivity**:
   - Quadratic tasks required 10-100x smaller learning rates than linear tasks
   - Adaptive optimizers (Adam, Adagrad) needed the most adjustment for quadratic tasks

4. **Gradient Handling**:
   - SGD requires different gradient clipping thresholds and additional gradient scaling
   - Proper gradient handling is critical for stability in higher-order functions

## Usage

Run experiments with optimized learning rates:

```bash
python run_experiments.py
```

This executes experiments with the following learning rates:

### Linear Regression Tasks
- Adam: 0.001
- SGD: 0.0001
- Adagrad: 0.01

### Quadratic Regression Tasks
- Adam: 0.00001
- SGD: 0.00001
- Adagrad: 0.0001

## Results Summary

The experiments demonstrate the challenges of training transformers for higher-order ICL tasks. While all optimizers successfully learn linear regression patterns, quadratic regression reveals significant differences in their numerical stability and generalization properties.

See `EXPERIMENT_NOTES.md` for detailed experimental results and analysis.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- PyHessian (for Hessian analysis)

## Citation

If using this code, please cite:

```bibtex
@article{garg2022what,
  title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
  author={Garg, Shivam and Biderman, Stella and Vernikos, Varun J},
  journal={arXiv preprint arXiv:2208.01066},
  year={2022}
}
```

## License

MIT License 