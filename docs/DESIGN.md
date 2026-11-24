# Neuron

Neural networks from scratch in C, built on a small matrix multiplication library.

---
## Scope
- Target: Linux x86-64, C.
- Networks: logic gates first (AND, OR, NAND, NOR, XOR, XNOR), then possibly one or more harder ones.
- Out of scope: GPU, other architectures, comparison with other matrix libraries.

## Networks

```
(x1)\               (x1)------(h1)\
     \                    \ /      \
      (y)                  X        (y)
     /                    / \      /
(x2)/               (x2)------(h2)/

AND, OR, NAND, NOR  XOR, XNOR
```

- Left: 2 inputs, 1 output neuron, no hidden layer. AND, OR, NAND and NOR are linearly separable, so one neuron is enough.
- Right: 2 inputs, 2 hidden neurons, 1 output, fully connected. XOR and XNOR are not linearly separable, so they need the hidden layer.
- Every neuron computes `a = f(w . x + b)`.
- Inputs and targets are 0 or 1. The prediction is the output thresholded at 0.5.
- Training: backpropagation with gradient descent.

## Activations
- Hidden layer: tanh. Output layer: sigmoid.
- Pluggable: an activation is a (forward, derivative) pair of functions that a layer takes by pointer, so swapping one is a one-line change.

| Function   | f(x)                          | f'(x)                  | Range         |
|------------|-------------------------------|------------------------|---------------|
| sigmoid    | `1 / (1 + e^-x)`              | `a(1 - a)`             | (0, 1)        |
| tanh       | `(e^x - e^-x) / (e^x + e^-x)` | `1 - a^2`              | (-1, 1)       |

`a = f(x)`. For sigmoid and tanh the derivative is computed from the stored output, so backprop never recomputes the exponentials.

## Matrix multiplication
- Row-major, contiguous storage.
- Used by the forward pass (`W . x + b`) and by backprop (products with transposed matrices).
- Recursive divide and conquer: split each matrix into quadrants and multiply the blocks.

```
| C11 C12 |   | A11 A12 |   | B11 B12 |
| C21 C22 | = | A21 A22 | x | B21 B22 |
```

- Base-case coarsening: below a size threshold the recursion stops and a plain triple loop (i-k-j order, cache friendly for row-major) does the work. Recursing down to 1x1 would be dominated by call overhead. The threshold is tuned with the benchmark.
- Task parallelism with OpenMP: each block product is an OpenMP task. The 8 block products are issued in two phases of 4, because the two products that feed the same quadrant of C must not run at the same time.

```
phase 1:  C11 += A11 B11   C12 += A11 B12   C21 += A21 B11   C22 += A21 B12
          taskwait
phase 2:  C11 += A12 B21   C12 += A12 B22   C21 += A22 B21   C22 += A22 B22
```

- Tasks are only created above the threshold. Below it the base case runs serially inside the current task.
- Why OpenMP: the recursion maps directly onto tasks. A pthread version would need its own work queue.
- The gate networks have tiny matrices, far below the threshold, so they never take the parallel path. It matters for the harder networks.

## Benchmarks
- A simple benchmark times matrix multiplication across matrix sizes and thread counts.
- Results are the times on my machine, with the CPU model and thread count written next to them.
- No comparison with other matrix libraries.
- Also used to pick the base-case threshold.

## Open questions
- float or double
- Base-case threshold value
- Non-square and odd sizes: split at the midpoint, halves may be uneven
- Which harder networks, if any
