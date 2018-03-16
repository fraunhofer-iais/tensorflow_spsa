# Simultaneous Perturbation Stochastic Approximation Optimizer

This package implements a Tensorflow Optimizer that uses the SPSA method described
in

[Spall, J. C. (1998). *An overview of the simultaneous perturbation method for efficient optimization.* Johns Hopkins apl technical digest, 19(4), 482-492.](http://www.jhuapl.edu/spsa/PDF-SPSA/Spall_An_Overview.PDF)

The implementation is done as a tf.Optimizer, it can be used in the normal way
that `Optimizer` is used in Tensorflow programs:

```python
optimizer = SimultaneousPerturbationOptimizer().minimize(cost)
```
