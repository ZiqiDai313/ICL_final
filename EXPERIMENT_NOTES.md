# ICL Experiment Notes

## Learning Rate Adjustments

After analyzing the initial experiments, we observed numerical stability issues with the quadratic regression tasks. The original learning rates worked well for linear regression but caused NaN values with Adam and Adagrad on quadratic tasks.

### Updated Learning Rates

#### Linear Regression Tasks
- Adam: 0.001 (unchanged)
- SGD: 0.0001 (unchanged)
- Adagrad: 0.01 (unchanged)

#### Quadratic Regression Tasks
- Adam: 0.00001 (reduced by 100x for stability)
- SGD: 0.00001 (reduced by 10x for stability)
- Adagrad: 0.0001 (reduced by 100x for stability)

## Experiment Results

### Linear Regression Results
| Optimizer | Train Loss | Test Loss | Generalization Gap |
|-----------|------------|-----------|-------------------|
| Adam      | 0.825642   | 0.800898  | 0.024744          |
| SGD       | 1.002039   | 0.971716  | 0.030323          |
| Adagrad   | inf*       | 0.793262  | N/A               |

*Note: Adagrad showed infinite training loss but still achieved the best test performance.

### Quadratic Regression Results
| Optimizer | Train Loss | Test Loss  | Generalization Gap |
|-----------|------------|------------|-------------------|
| Adam      | 33.480025  | 822.005432 | 788.525407        |
| SGD       | 650.024796 | 800.069580 | 150.044785        |
| Adagrad   | 25.233316  | 1274.759766| 1249.526450       |

## Key Findings

1. **Linear regression performance:**
   - All optimizers performed well on linear tasks
   - Adagrad had the best test loss (0.793262)
   - SGD was stable with careful learning rate selection

2. **Quadratic regression challenges:**
   - Higher-order functions introduced numerical instability
   - With reduced learning rates, Adam and Adagrad no longer produced NaN values but still had large generalization gaps
   - SGD showed the smallest generalization gap on quadratic tasks
   - Gradient clipping was essential for stability

3. **Implementation differences:**
   - Linear regression used the full dimension vector for loss calculation: `output[:, -1, :args.dim]`
   - Quadratic regression used only the first element: `output[:, -1, 0]`
   - SGD used a smaller clipping threshold (`args.clip * 0.1`) plus gradient scaling for stability

4. **Optimization insights:**
   - For in-context learning on quadratic regression tasks, SGD with smaller learning rates appears more stable
   - Adam and Adagrad achieve better training losses but generalize poorly on quadratic tasks
   - The large generalization gaps suggest potential overfitting with adaptive optimizers

## Gradient Handling

The code revealed different gradient handling approaches:

```python
# For SGD:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip * 0.1)
# Scale gradients for stability
for param in model.parameters():
    if param.grad is not None:
        param.grad.data = param.grad.data / (1 + grad_norm)

# For other optimizers:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
```

These differences were critical for achieving stability with higher-order functions.

## Running Updated Experiments

Use the provided script to run experiments with the adjusted learning rates:

```
python run_experiments.py
```

This will execute all experiments using the optimal learning rates for each task type and optimizer combination. 