# Neuron

Neural networks from scratch in C, built on a small matrix multiplication library.

---
## Scope
- Target: Linux x86-64 with AVX2 and FMA, C with OpenMP.
- Networks: logic gates (AND, OR, NAND, NOR, XOR, XNOR), then MNIST digit classification.
- All values are `float`.
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
- Training: backpropagation with full-batch gradient descent on the 4 rows of the truth table, learning rate 0.5, mean squared error loss, 40,000 epochs.

MNIST is a bigger network, with the whole batch going through each layer as one matrix product:

```
 X (B x 784)        A1 (B x 256)         P (B x 10)
(784 pixels) --W1--> (256 tanh) --W2--> (10 softmax)
```

## Activations
- AND, OR, NAND, NOR: no hidden layer, sigmoid at the output.
- XOR, XNOR: sigmoid in the hidden layer, tanh at the output.
- MNIST: tanh in the hidden layer, softmax at the output.
- Not pluggable: `src/activation.c` only has `sigmoid`, `sigmoid_deriv` and `tanh_deriv`. The tanh and softmax forward passes are written directly in the networks (`tanhf` in `hidden.c` and `mnist.c`, `softmax_rows` in `mnist.c`), and there is no (forward, derivative) interface and no ReLU-family function.
- Initialization: the gates start with weights and biases uniform in [-1, 1) from a fixed seed. MNIST uses Xavier (see below) with biases at 0.

| Function   | f(x)                          | f'(x)                  | Range         |
|------------|-------------------------------|------------------------|---------------|
| sigmoid    | `1 / (1 + e^-x)`              | `a(1 - a)`             | (0, 1)        |
| tanh       | `(e^x - e^-x) / (e^x + e^-x)` | `1 - a^2`              | (-1, 1)       |

`a = f(x)`. For sigmoid and tanh the derivative is computed from the stored output, so backprop never recomputes the exponentials.

Softmax turns the 10 outputs of a row into probabilities. The row maximum is subtracted before the exponentials, so they cannot overflow:

```
p_i = e^(z_i - max(z)) / sum_j e^(z_j - max(z))
```

## MNIST
- Data: the four IDX files (train images, train labels, test images, test labels) are given on the command line. The header is checked (magic numbers, matching counts, 28x28 images) and pixels are scaled to [0, 1].
- Without arguments the program trains on a synthetic 28x28 dataset whose images depend on the label, so it runs without the data.
- Network: 784 inputs, 256 tanh hidden neurons, 10 softmax outputs. Weights are uniform in +-sqrt(6 / (fan_in + fan_out)) (Xavier), biases start at 0.
- Loss: cross-entropy. With a softmax output the error of the last layer is just `P - one_hot(y)`.
- Training: mini-batch SGD, batch size 64, learning rate 0.01, the training set is reshuffled every epoch. The loss and the test accuracy are printed every 5 epochs.
- All buffers of a batch live in a workspace allocated once and 32-byte aligned.

```
forward:   Z1 = X . W1 + b1        A1 = tanh(Z1)
           Z2 = A1 . W2 + b2       P  = softmax(Z2)

backward:  dZ2 = (P - one_hot(y)) / B
           dW2 = A1^T . dZ2        db2 = column sums of dZ2
           dA1 = dZ2 . W2^T
           dZ1 = dA1 * (1 - A1^2)
           dW1 = X^T . dZ1         db1 = column sums of dZ1
```

- The products with a transpose are done by writing the transpose into a workspace buffer and calling the same `mat_mul`, so there is a single multiplication routine.

## Matrix multiplication
- Row-major `float` matrices with an explicit leading dimension (`lda`, `ldb`, `ldc`), so a block of a larger matrix can be passed without copying.
- `mat_mul` computes `C += A . B` and `mat_mul_set` computes `C = A . B`. Both go through one function with a flag that clears C first.
- Used by the forward pass (`X . W + b`) and by backprop (products with transposed matrices).
- Cache blocking: C is cut into blocks of `MC x NC` and k into steps of `KC`. A block of A is 60x64 floats (15 KiB) and a block of B is 64x256 floats (64 KiB).

```
parallel for (i0 in 0..m step MC, j0 in 0..n step NC)
  for (p0 in 0..k step KC)
    C[i0.., j0..] += A[i0.., p0..] . B[p0.., j0..]      // mat_mul_small
```

| Constant | Value | Why                                               |
|----------|------:|---------------------------------------------------|
| MC       |    60 | a multiple of the 6 rows of the micro-kernels     |
| NC       |   256 | a multiple of the 16 columns of the micro-kernels |
| KC       |    64 | depth of one pass over a block                    |

- Micro-kernels: AVX2 and FMA. `kernel_6x16` keeps a 6x16 tile of C in 12 YMM registers. For each step of k it loads two vectors of 8 floats from B, broadcasts 6 values of A and does 12 FMAs. `kernel_6x8` is the same with 8 columns, and `kernel_1x8` does one row at a time.
- `mat_mul_small` runs the kernels over one block: 6 rows at a time, 16 columns at a time, then 8, then scalar code for the last columns. The rows left over (m % 6) are handled one row at a time.
- Task-free parallelism with OpenMP: the loops over the blocks of C are an OpenMP `parallel for` with `collapse(2)` and dynamic scheduling. A thread owns whole blocks of C and walks k in order inside them, so two threads never write the same element and no synchronization is needed. Every element is summed in the same order whatever the thread count, so the result does not depend on it.
- The loop is only parallel when `m * n * k` is above 128 * 128 * 128. Below that, which includes everything the gates do, it runs on the calling thread. `mat_mul_set` also clears C in parallel, when C has more than 128 * 128 elements.
- Replaces an earlier recursive divide-and-conquer version with OpenMP tasks.

## Testing
- Matrix multiplication (`tests/test_mat.c`): every shape is checked against a double-precision triple loop. The tolerance grows with k and with the sum of the absolute terms, and the inputs and the padding between rows must stay untouched.
  - Shapes: every size from 1 to 5, every tail of the micro-kernels (m from 1 to 13, n around the 8 and 16 column kernels, small k), one below, on and one above MC, NC, KC and the parallel threshold, odd and very thin shapes, leading dimensions wider than the matrix, and sub-matrices inside a larger one.
  - Behaviour: `mat_mul` accumulates, `mat_mul_set` ignores what was in C, NaN spreads to exactly the rows or columns it touches, small integers give exact results, and identity and zero matrices.
  - Threads: the result is bit-identical for 1 to 16 threads and across repeated runs, and `mat_mul` works when called from inside a parallel region.
- Activations and networks: not covered by the test suite yet. The gates program prints each output next to its target, and the result is read by hand.

## Benchmarks
- A simple benchmark (`bench/benchmark.c`) times one `mat_mul` of two 4096x4096 matrices. The size is a constant in the file and the thread count is set through `OMP_NUM_THREADS` (6 in `make bench`).
- Results are the times on the author's machine, with the CPU model and thread count written next to them.
- No comparison with other matrix libraries.
- Also used to pick the block sizes.

## Open questions
- Which harder networks after MNIST, if any
