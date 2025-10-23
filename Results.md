#Decision Tree Experiment Results

| Run | max_depth | criterion | min_samples_split | min_samples_leaf | Accuracy | Notes               |
|-----|-----------|-----------|-------------------|------------------|----------|---------------------|
| 1   | 5         | gini      | 2                 | 1                | 0.29     | Underfit            |
| 2   | 10        | gini      | 2                 | 1                | 0.45     | Best accuracy       |
| 3   | 15        | gini      | 2                 | 1                | 0.40     | Plateau begins      |
| 4   | 20        | gini      | 2                 | 1                | 0.40     | No further gain     |
| 5   | 10        | entropy   | 2                 | 1                | 0.39     | Gini > Entropy      |
| 6   | 10        | gini      | 5                 | 2                | 0.44     | Regularized, stable |
| 7   | 20        | entropy   | 10                | 5                | 0.39     | Over-regularized    |