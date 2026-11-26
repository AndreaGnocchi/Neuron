# Neuron

Neural networks from scratch in C, built on a small matrix multiplication library.

It trains logic gates (AND, OR, NAND, NOR, XOR, XNOR) and then an MNIST digit classifier with backpropagation and gradient descent. Every matrix product of the MNIST network, forward and backward, is one call to a hand-written AVX2/FMA matrix multiplication with OpenMP. There are no dependencies beyond libc, libm and OpenMP.

## About

This is a small learning project. It was built to understand how a neural network learns, from the forward pass and backpropagation down to the matrix product that does most of the work. The gates come first because they are small enough to check by hand, and MNIST is the first network where the speed of the multiplication matters.

It is not meant to be a general deep learning library, and it is not a comparison with other matrix libraries.

## Usage

Build both programs (`build/bin/gates` and `build/bin/mnist`):

```bash
make
```

Or let `make` build and run one of them:

```bash
make gates-run
make mnist-run
```

The gates program trains each gate for 40,000 epochs, prints the loss every 8,000 epochs, then prints the four outputs next to their targets. It does not check them, you read the truth table yourself.

The MNIST program takes the four IDX files (train images, train labels, test images, test labels), in that order:

```bash
./build/bin/mnist train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
```

Or put the four files in `src/neurons/mnist/data` (git does not track them) and let `make` pass them:

```bash
make mnist-run-args
make mnist-run-args MNIST_DATA_DIR=$HOME/data/mnist
```

The header of every file is checked (magic numbers, matching counts, 28x28 images) and pixels are scaled to [0, 1]. With no arguments the program trains on a synthetic 28x28 dataset whose images depend on the label, so it runs without the data. That dataset is easy to separate, so its accuracy says nothing about MNIST. Any other number of arguments prints a usage message on stderr and exits with 1. The test accuracy is printed once before training, then the loss and the test accuracy are printed after epochs 0, 5, 10, 15 and 20.

A file that cannot be opened or has an invalid header is reported on stderr and the program exits with a non-zero code.

The number of threads used by the matrix multiplication follows the usual OpenMP variable:

```bash
OMP_NUM_THREADS=6 ./build/bin/mnist
```

## How it works

The project has three layers: a matrix multiplication, the activation functions, and the networks that use them.

| Part        | File                                                  | What it does                                                                   |
|-------------|-------------------------------------------------------|--------------------------------------------------------------------------------|
| Matmul      | [src/mat.c](src/mat.c)                                | `C += A . B` and `C = A . B` with cache blocking, AVX2/FMA kernels and OpenMP  |
| Activations | [src/activation.c](src/activation.c)                  | Sigmoid, softmax and ReLU, and the derivatives of sigmoid, tanh and ReLU       |
| RNG         | [src/rng.c](src/rng.c)                                | Random numbers for weight initialization and shuffling                         |
| Gates       | [src/neurons/gates](src/neurons/gates)                | `linear.c` and `hidden.c` train the one-neuron and the two-layer network, `gates.c` holds the six truth tables and `neuron_gates.c` runs them all |
| MNIST       | [src/neurons/mnist/mnist.c](src/neurons/mnist/mnist.c) | IDX loading, the 784-256-10 network and mini-batch SGD                        |

**Networks.** AND, OR, NAND and NOR are linearly separable, so a single neuron with 2 inputs is enough. XOR and XNOR are not, so they use 2 inputs, 2 hidden neurons and 1 output, fully connected. Inputs and targets are 0 or 1. Every neuron computes `a = f(w . x + b)`.

**Gates.** Full-batch gradient descent on the 4 rows of the truth table, with learning rate 0.5 and a mean squared error loss. Weights and biases start uniform in [-1, 1) from a fixed seed, so every run gives the same result.

**MNIST.** 784 inputs, 256 ReLU hidden neurons and 10 softmax outputs. The whole batch goes through each layer as one matrix product, so the forward and backward passes are just a few calls to `mat_mul`. The loss is cross-entropy, so the error of the last layer is simply `P - one_hot(y)`. Training is mini-batch SGD with batch size 64 and learning rate 0.5, and the training set is reshuffled every epoch. All buffers of a batch live in a workspace allocated once and 32-byte aligned. Weights use Xavier initialization and biases start at 0.

**Activations.**

| Network                | Hidden layer | Output  | Derivative of the activation |
|------------------------|--------------|---------|------------------------------|
| AND, OR, NAND, NOR     | none         | sigmoid | `a(1 - a)`                   |
| XOR, XNOR              | sigmoid      | tanh    | `a(1 - a)` and `1 - a^2`     |
| MNIST                  | ReLU         | softmax | `1 if a > 0 else 0`, softmax folded into the cross-entropy error |

The derivatives are computed from the stored output, so backpropagation never recomputes the exponentials. Softmax subtracts the row maximum before the exponentials so they cannot overflow. The activations are plain functions that each network calls directly (the tanh output of XOR and XNOR is `tanhf` from libm), there is no layer or activation interface.

**Matrix multiplication.** Row-major `float` matrices with an explicit leading dimension, so a block of a larger matrix can be passed without copying. C is cut into blocks of 60x256 and k into steps of 64, so a block of A (15 KiB) and a block of B (64 KiB) stay in cache. The micro-kernel `kernel_6x16` keeps a 6x16 tile of C in 12 YMM registers and does 12 FMAs per step of k. `kernel_6x8` is the same with 8 columns, `kernel_1x8` handles the rows left over, and the last columns use scalar code. The loops over the blocks of C are an OpenMP `parallel for`, and a thread owns whole blocks of C and walks k in order, so no synchronization is needed and the result is the same for any thread count. Below `128 * 128 * 128` multiply-adds it runs on the calling thread, which includes everything the gates do. The products with a transpose write the transpose into a workspace buffer and call the same `mat_mul`, so there is a single multiplication routine. The full reasoning is in [docs/DESIGN.md](docs/DESIGN.md).

## Known limitations

- Linux x86-64 with AVX2 and FMA only. There is no fallback for other CPUs and no GPU support.
- Only sigmoid, tanh, softmax and ReLU, as plain functions that the networks call directly. There is no activation interface and no Leaky ReLU.
- Plain SGD with a fixed learning rate. No momentum, no Adam, no learning rate schedule.
- The gate networks and the MNIST network have fixed sizes that are constants in the source, and MNIST has a single hidden layer.
- The multiplication is only tuned for my CPU, and the block sizes are not detected at run time.
- The library is built with `-ffast-math` and `-march=native`, so results can differ slightly between compilers and CPUs.
- The input is always the IDX files given on the command line. There is no model saving or loading, and no MNIST accuracy is reported here.

## Documentation

Design notes live in [docs/DESIGN.md](docs/DESIGN.md).
Alongside them there is the documentation used to build this project [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

---
# Tree
``` bash
Neuron
├── bench
│   └── benchmark.c
├── docs
│   ├── DESIGN.md
│   └── DOCUMENTATION.md
├── .github
│   └── workflows
│       └── ci.yml
├── .gitignore
├── include
│   ├── mat.h
│   └── neuron_gates.h
├── LICENSE
├── Makefile
├── README.md
├── src
│   ├── activation.c
│   ├── activation.h
│   ├── mat.c
│   ├── neurons
│   │   ├── gates
│   │   │   ├── gates.c
│   │   │   ├── hidden.c
│   │   │   ├── linear.c
│   │   │   ├── neuron_gates.c
│   │   │   └── train.h
│   │   └── mnist
│   │       └── mnist.c
│   ├── rng.c
│   └── rng.h
└── tests
    ├── test_main.c
    ├── test_main.h
    └── test_mat.c
```
# Build

```bash
# Compile the gates and MNIST programs
make all

# Compile only the matrix multiplication library (build/libmat.a)
make mat

# Compile only one program (build/bin/gates or build/bin/mnist)
make gates
make mnist

# Compile and run the gates
make gates-run

# Compile and run MNIST (synthetic data without arguments)
make mnist-run

# Compile and run MNIST on the four files in src/neurons/mnist/data
make mnist-run-args

# Clean all build artifacts
make clean
```

# Tests

```bash
# Run the whole test suite
make test

# Run tests with Undefined Behavior Sanitizer
make test-ubsan

# Run tests with Thread Sanitizer
make test-tsan
```

The suite has 12,666 checks, all of them for the matrix multiplication (`tests/test_mat.c`). Every shape is checked against a double-precision triple loop, with a tolerance that grows with k and with the sum of the absolute terms. The shapes cover every size from 1 to 5, every tail of the micro-kernels, one below, on and one above each block size and the parallel threshold, very thin shapes, leading dimensions wider than the matrix and sub-matrices inside a larger one. They also check that `mat_mul` accumulates and `mat_mul_set` ignores what was in C, that NaN spreads to exactly the rows or columns it touches, that inputs and the padding between rows stay untouched, that the result is bit-identical for 1 to 16 threads and across repeated runs, and that `mat_mul` works when called from inside a parallel region. CI builds the project and runs `make test` with both gcc and clang on an x86-64 runner, on every push to `main` and on pull requests. The sanitizer targets are not part of CI.

---
# Benchmarks

```bash
# Time the matrix multiplication across sizes and thread counts
make bench
```

`bench/benchmark.c` fills two 4096x4096 matrices with random values and multiplies them once with `mat_mul`. The size is the `MAT_SIZE` constant, and `make bench` runs it with 6 threads pinned to cores (`OMP_NUM_THREADS=6`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=ACTIVE`) under `time -p`. It is also what was used to pick the block sizes. There is no comparison with other matrix libraries.

## Results

Measured on:

| Component | Details                                  |
|-----------|------------------------------------------|
| CPU       | Intel Core i5-9400F, 6 cores / 6 threads |
| RAM       | 16 GB DDR4-2666                          |
| OS        | Arch Linux, kernel 7.3.5-arch1-1         |
| Compiler  | clang 23.1.5                             |

This is a single run, so small differences are within noise.

| Matrix size | Threads | real (s) | user (s) | sys (s) |
|-------------|--------:|---------:|---------:|--------:|
| 4096x4096   |       6 |     0.95 |     4.53 |    0.21 |

A 4096x4096 product is about 137 GFLOP (`2 * 4096^3`), so this is roughly 145 GFLOP/s. The time includes allocating and filling the matrices, so the multiplication itself is a little faster than that.

---
# License
See [LICENSE](LICENSE).
