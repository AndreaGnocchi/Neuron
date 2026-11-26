#include "../rng.h"
#include "../activation.h"
#include "../../include/mat.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN_DIM     784
#define HIDDEN_DIM 256
#define OUT_DIM    10

typedef struct {
  float* images; // N * 784
  uint8_t* labels; // N
  int count;
} dataset_t;

typedef struct {
  float* W1; // 784 * 256
  float* b1; // 256
  float* W2; // 256 * 10
  float* b2; // 10
} mlp_t;

typedef struct {
  int batch_size;
  float* X;      // B * 784
  float* X_T;    // 784 * B
  float* Z1;     // B * 256
  float* A1;     // B * 256
  float* A1_T;   // 256 * B
  float* Z2;     // B * 10
  float* probs;  // B * 10
  float* dZ2;    // B * 10
  float* dA1;    // B * 256
  float* dZ1;    // B * 256
  float* dW1;    // 784 * 256
  float* db1;    // 256
  float* dW2;    // 256 * 10
  float* db2;    // 10
  float* W2_T;   // 10 * 256
} workspace_t;

static size_t align_up_32(size_t bytes) {
  return (bytes + 31u) & ~(size_t)31u;
}

static workspace_t workspace_create(int batch_size) {
  workspace_t ws;
  ws.batch_size = batch_size;
  int B = batch_size;

  ws.X     = aligned_alloc(32, align_up_32((size_t)B * IN_DIM * sizeof(float)));
  ws.X_T   = aligned_alloc(32, align_up_32((size_t)IN_DIM * B * sizeof(float)));
  ws.Z1    = aligned_alloc(32, align_up_32((size_t)B * HIDDEN_DIM * sizeof(float)));
  ws.A1    = aligned_alloc(32, align_up_32((size_t)B * HIDDEN_DIM * sizeof(float)));
  ws.A1_T  = aligned_alloc(32, align_up_32((size_t)HIDDEN_DIM * B * sizeof(float)));
  ws.Z2    = aligned_alloc(32, align_up_32((size_t)B * OUT_DIM * sizeof(float)));
  ws.probs = aligned_alloc(32, align_up_32((size_t)B * OUT_DIM * sizeof(float)));
  ws.dZ2   = aligned_alloc(32, align_up_32((size_t)B * OUT_DIM * sizeof(float)));
  ws.dA1   = aligned_alloc(32, align_up_32((size_t)B * HIDDEN_DIM * sizeof(float)));
  ws.dZ1   = aligned_alloc(32, align_up_32((size_t)B * HIDDEN_DIM * sizeof(float)));
  ws.dW1   = aligned_alloc(32, align_up_32((size_t)IN_DIM * HIDDEN_DIM * sizeof(float)));
  ws.db1   = aligned_alloc(32, align_up_32((size_t)HIDDEN_DIM * sizeof(float)));
  ws.dW2   = aligned_alloc(32, align_up_32((size_t)HIDDEN_DIM * OUT_DIM * sizeof(float)));
  ws.db2   = aligned_alloc(32, align_up_32((size_t)OUT_DIM * sizeof(float)));
  ws.W2_T  = aligned_alloc(32, align_up_32((size_t)OUT_DIM * HIDDEN_DIM * sizeof(float)));

  return ws;
}

static void workspace_free(workspace_t* ws) {
  free(ws->X);
  free(ws->X_T);
  free(ws->Z1);
  free(ws->A1);
  free(ws->A1_T);
  free(ws->Z2);
  free(ws->probs);
  free(ws->dZ2);
  free(ws->dA1);
  free(ws->dZ1);
  free(ws->dW1);
  free(ws->db1);
  free(ws->dW2);
  free(ws->db2);
  free(ws->W2_T);
}

static uint32_t read_be_u32(FILE* f) {
  uint8_t b[4];
  if (fread(b, 1, 4, f) != 4) {
    perror("Error reading BE u32");
    return -1;
  }

  return ((uint32_t)b[0] << 24) |
    ((uint32_t)b[1] << 16) |
    ((uint32_t)b[2] << 8)  |
    ((uint32_t)b[3]);
}

static dataset_t load_mnist(const char* imgs_path, const char* lbl_path) {
  dataset_t err = {.count = -1, .images = NULL, .labels = NULL};
  uint8_t* raw = NULL;

  FILE* fi = fopen(imgs_path, "rb");
  if (!fi) {
    perror("fopen");
    fprintf(stderr, "Failed to open MNIST images: %s\n", imgs_path);
    return err;
  }

  FILE* fl = fopen(lbl_path, "rb");
  if (!fl) {
    perror("fopen");
    fprintf(stderr, "Failed to open MNIST labels: %s\n", lbl_path);
    fclose(fi);
    return err;
  }

  uint32_t magic_img = read_be_u32(fi);
  uint32_t num_img   = read_be_u32(fi);
  uint32_t rows      = read_be_u32(fi);
  uint32_t cols      = read_be_u32(fi);

  uint32_t magic_lbl = read_be_u32(fl);
  uint32_t num_lbl   = read_be_u32(fl);

  if (magic_img != 2051 || magic_lbl != 2049 ||
      num_img == 0 || num_img != num_lbl || num_img > (uint32_t)INT_MAX ||
      (uint64_t)rows * (uint64_t)cols != (uint64_t)IN_DIM) {
    fprintf(stderr, "Invalid MNIST IDX header (img magic=%u, lbl magic=%u, n=%u/%u, %ux%u)\n",
            magic_img, magic_lbl, num_img, num_lbl, rows, cols);
    fclose(fi);
    fclose(fl);
    return err;
  }

  size_t n = (size_t)num_img;
  size_t total_pixels = n * (size_t)IN_DIM;
  size_t img_bytes = align_up_32(total_pixels * sizeof(float));

  dataset_t ds = {
    .count  = (int)num_img,
    .images = (float*)aligned_alloc(32, img_bytes),
    .labels = (uint8_t*)malloc(n * sizeof(uint8_t)),
  };
  raw = (uint8_t*)malloc(total_pixels);

  if (!ds.images || !ds.labels || !raw) {
    fprintf(stderr, "Out of memory\n");
    free(raw);
    free(ds.images);
    free(ds.labels);
    fclose(fi);
    fclose(fl);
    return err;
  }

  if (fread(raw, 1, total_pixels, fi) != total_pixels ||
      fread(ds.labels, 1, n, fl) != n) {
    fprintf(stderr, "Truncated MNIST data\n");
    free(raw);
    free(ds.images);
    free(ds.labels);
    fclose(fi);
    fclose(fl);
    return err;
  }

  for (size_t i = 0; i < total_pixels; i++) {
    ds.images[i] = (float)raw[i] / 255.0f;
  }

  free(raw);
  fclose(fi);
  fclose(fl);
  return ds;
}

static void free_dataset(dataset_t* ds) {
  free(ds->images);
  free(ds->labels);
  ds->images = NULL;
  ds->labels = NULL;
  ds->count = 0;
}

// For test purposes
static dataset_t make_fake_dataset(int count, uint64_t seed) {
  rng_seed(seed);
  dataset_t ds;
  ds.count  = count;
  ds.images = aligned_alloc(32, align_up_32((size_t)count * IN_DIM * sizeof(float)));
  ds.labels = malloc((size_t)count * sizeof(uint8_t));

  for (int n = 0; n < count; n++) {
    uint8_t label = (uint8_t)(rng_u32() % OUT_DIM);
    ds.labels[n] = label;
    float* img = &ds.images[(size_t)n * IN_DIM];

    int band_row = (label * 2) % 24;
    int band_col = (label * 5) % 24;
    for (int r = 0; r < 28; r++) {
      for (int c = 0; c < 28; c++) {
        float val = 0.05f * (0.5f * (rng_float() + 1.0f));
        if ((r >= band_row && r < band_row + 4) ||
            (c >= band_col && c < band_col + 4 && (label % 2 == 0))) {
          val += 0.75f;
        }
        if (val > 1.0f) val = 1.0f;
        img[r * 28 + c] = val;
      }
    }
  }

  return ds;
}

static mlp_t mlp_create(uint64_t seed) {
  rng_seed(seed);
  mlp_t net;
  net.W1 = aligned_alloc(32, align_up_32((size_t)IN_DIM * HIDDEN_DIM * sizeof(float)));
  net.b1 = aligned_alloc(32, align_up_32((size_t)HIDDEN_DIM * sizeof(float)));
  net.W2 = aligned_alloc(32, align_up_32((size_t)HIDDEN_DIM * OUT_DIM * sizeof(float)));
  net.b2 = aligned_alloc(32, align_up_32((size_t)OUT_DIM * sizeof(float)));

  float limit1 = sqrtf(6.0f / (float)(IN_DIM + HIDDEN_DIM));
  float limit2 = sqrtf(6.0f / (float)(HIDDEN_DIM + OUT_DIM));

  fill_random(net.W1, (size_t)IN_DIM * HIDDEN_DIM);
  for (size_t i = 0; i < (size_t)IN_DIM * HIDDEN_DIM; i++) {
    net.W1[i] *= limit1;
  }

  fill_random(net.W2, (size_t)HIDDEN_DIM * OUT_DIM);
  for (size_t i = 0; i < (size_t)HIDDEN_DIM * OUT_DIM; i++) {
    net.W2[i] *= limit2;
  }

  memset(net.b1, 0, (size_t)HIDDEN_DIM * sizeof(float));
  memset(net.b2, 0, (size_t)OUT_DIM * sizeof(float));

  return net;
}

static void mlp_free(mlp_t* net) {
  free(net->W1);
  free(net->b1);
  free(net->W2);
  free(net->b2);
}

static void transpose_2d(const float* src, int rows, int cols, float* dst) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      dst[c * rows + r] = src[r * cols + c];
    }
  }
}

static void add_bias_rows(float* M, const float* bias, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float* row = &M[r * cols];
    for (int c = 0; c < cols; c++) {
      row[c] += bias[c];
    }
  }
}

static void apply_tanh(const float* Z, float* A, int n) {
  for (int i = 0; i < n; i++) {
    A[i] = tanhf(Z[i]);
  }
}

static void softmax_rows(const float* Z, float* probs, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float* z_row = &Z[r * cols];
    float* p_row = &probs[r * cols];

    float max_val = z_row[0];
    for (int c = 1; c < cols; c++) {
      if (z_row[c] > max_val) max_val = z_row[c];
    }

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
      float e = expf(z_row[c] - max_val);
      p_row[c] = e;
      sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int c = 0; c < cols; c++) {
      p_row[c] *= inv_sum;
    }
  }
}

static float forward_batch(const mlp_t* net, workspace_t* ws, int B,
                           const uint8_t* labels, int compute_loss) {
  // Layer 1: Z1 = X * W1 + b1, A1 = tanh(Z1)
  mat_mul_set(B, HIDDEN_DIM, IN_DIM, ws->X, IN_DIM, net->W1, HIDDEN_DIM, ws->Z1, HIDDEN_DIM);
  add_bias_rows(ws->Z1, net->b1, B, HIDDEN_DIM);
  apply_tanh(ws->Z1, ws->A1, B * HIDDEN_DIM);

  // Layer 2: Z2 = A1 * W2 + b2, probs = softmax(Z2)
  mat_mul_set(B, OUT_DIM, HIDDEN_DIM, ws->A1, HIDDEN_DIM, net->W2, OUT_DIM, ws->Z2, OUT_DIM);
  add_bias_rows(ws->Z2, net->b2, B, OUT_DIM);
  softmax_rows(ws->Z2, ws->probs, B, OUT_DIM);

  if (!compute_loss || !labels) return 0.0f;

  float loss = 0.0f;
  const float eps = 1e-12f;
  for (int b = 0; b < B; b++) {
    uint8_t target = labels[b];
    float p = ws->probs[b * OUT_DIM + target];
    if (p < eps) p = eps;
    loss += -logf(p);
  }
  return loss / (float)B;
}

static void backward_and_step(mlp_t* net, workspace_t* ws, int B,
                              const uint8_t* labels, float lr) {
  float inv_B = 1.0f / (float)B;

  // dZ2 = (probs - one_hot(y)) / B
  memcpy(ws->dZ2, ws->probs, (size_t)B * OUT_DIM * sizeof(float));
  for (int b = 0; b < B; b++) {
    ws->dZ2[b * OUT_DIM + labels[b]] -= 1.0f;
  }
  for (int i = 0; i < B * OUT_DIM; i++) {
    ws->dZ2[i] *= inv_B;
  }

  // db2 = sum_b dZ2[b]
  memset(ws->db2, 0, (size_t)OUT_DIM * sizeof(float));
  for (int b = 0; b < B; b++) {
    for (int o = 0; o < OUT_DIM; o++) {
      ws->db2[o] += ws->dZ2[b * OUT_DIM + o];
    }
  }

  // dW2 = A1^T * dZ2
  transpose_2d(ws->A1, B, HIDDEN_DIM, ws->A1_T);
  memset(ws->dW2, 0, (size_t)HIDDEN_DIM * OUT_DIM * sizeof(float));
  mat_mul(HIDDEN_DIM, OUT_DIM, B, ws->A1_T, B, ws->dZ2, OUT_DIM, ws->dW2, OUT_DIM);

  // dA1 = dZ2 * W2^T
  transpose_2d(net->W2, HIDDEN_DIM, OUT_DIM, ws->W2_T);
  memset(ws->dA1, 0, (size_t)B * HIDDEN_DIM * sizeof(float));
  mat_mul(B, HIDDEN_DIM, OUT_DIM, ws->dZ2, OUT_DIM, ws->W2_T, HIDDEN_DIM, ws->dA1, HIDDEN_DIM);

  // dZ1 = dA1 * tanh_deriv(A1)
  for (int i = 0; i < B * HIDDEN_DIM; i++) {
    ws->dZ1[i] = ws->dA1[i] * tanh_deriv(ws->A1[i]);
  }

  // db1 = sum_b dZ1[b]
  memset(ws->db1, 0, (size_t)HIDDEN_DIM * sizeof(float));
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < HIDDEN_DIM; h++) {
      ws->db1[h] += ws->dZ1[b * HIDDEN_DIM + h];
    }
  }

  // dW1 = X^T * dZ1
  transpose_2d(ws->X, B, IN_DIM, ws->X_T);
  memset(ws->dW1, 0, (size_t)IN_DIM * HIDDEN_DIM * sizeof(float));
  mat_mul(IN_DIM, HIDDEN_DIM, B, ws->X_T, B, ws->dZ1, HIDDEN_DIM, ws->dW1, HIDDEN_DIM);

  // SGD parameter update
  for (size_t i = 0; i < (size_t)IN_DIM * HIDDEN_DIM; i++) {
    net->W1[i] -= lr * ws->dW1[i];
  }
  for (int h = 0; h < HIDDEN_DIM; h++) {
    net->b1[h] -= lr * ws->db1[h];
  }
  for (size_t i = 0; i < (size_t)HIDDEN_DIM * OUT_DIM; i++) {
    net->W2[i] -= lr * ws->dW2[i];
  }
  for (int o = 0; o < OUT_DIM; o++) {
    net->b2[o] -= lr * ws->db2[o];
  }
}

static float evaluate(const mlp_t* net, workspace_t* ws, const dataset_t* ds) {
  int correct = 0;
  int max_B = ws->batch_size;
  for (int start = 0; start < ds->count; start += max_B) {
    int B = ds->count - start;
    if (B > max_B) B = max_B;

    memcpy(ws->X, &ds->images[(size_t)start * IN_DIM], (size_t)B * IN_DIM * sizeof(float));
    forward_batch(net, ws, B, NULL, 0);

    for (int b = 0; b < B; b++) {
      const float* p = &ws->probs[b * OUT_DIM];
      int pred = 0;
      float best = p[0];
      for (int c = 1; c < OUT_DIM; c++) {
        if (p[c] > best) {
          best = p[c];
          pred = c;
        }
      }
      if (pred == (int)ds->labels[start + b]) {
        correct++;
      }
    }
  }
  return 100.0f * (float)correct / (float)ds->count;
}

static void shuffle_indices(int* idx, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rng_u32() % (uint32_t)(i + 1));
    int tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
}

int main(int argc, char* argv[]) {
  int epochs     = 20;
  int batch_size = 64;
  float lr       = 0.01f;

  dataset_t train_ds, test_ds;
  if (argc == 5) {
    train_ds = load_mnist(argv[1], argv[2]);
    test_ds  = load_mnist(argv[3], argv[4]);
    if (train_ds.count == -1 || test_ds.count == -1) {
      return 1;
    }
    printf("Loaded MNIST: %d train, %d test samples\n", train_ds.count, test_ds.count);
  } else if (argc == 1) {
    printf("No MNIST IDX paths provided; running on synthetic 28x28 dataset.\n");
    train_ds = make_fake_dataset(2048, 42ULL);
    test_ds  = make_fake_dataset(512, 1337ULL);
  } else {
    fprintf(stderr,
            "Usage: %s [train-images train-labels t10k-images t10k-labels]\n",
            argv[0]);
    return 1;
  }

  mlp_t net = mlp_create(123456ULL);
  workspace_t ws = workspace_create(batch_size);
  int* indices = (int*)malloc((size_t)train_ds.count * sizeof(int));
  for (int i = 0; i < train_ds.count; i++) indices[i] = i;

  printf("Initial test accuracy: %.2f%%\n", evaluate(&net, &ws, &test_ds));

  for (int epoch = 0; epoch <= epochs; epoch++) {
    shuffle_indices(indices, train_ds.count);
    double epoch_loss = 0.0;
    int steps = 0;

    for (int start = 0; start < train_ds.count; start += batch_size) {
      int B = train_ds.count - start;
      if (B > batch_size) B = batch_size;

      for (int b = 0; b < B; b++) {
        int idx = indices[start + b];
        memcpy(&ws.X[(size_t)b * IN_DIM],
               &train_ds.images[(size_t)idx * IN_DIM],
               (size_t)IN_DIM * sizeof(float));
      }

      uint8_t batch_labels[256];
      uint8_t* lbl_ptr = (B <= 256) ? batch_labels : malloc((size_t)B);
      for (int b = 0; b < B; b++) {
        lbl_ptr[b] = train_ds.labels[indices[start + b]];
      }

      float loss = forward_batch(&net, &ws, B, lbl_ptr, 1);
      backward_and_step(&net, &ws, B, lbl_ptr, lr);

      if (lbl_ptr != batch_labels) free(lbl_ptr);

      epoch_loss += loss;
      steps++;
    }

    if (epoch % 5 == 0) {
    float test_acc = evaluate(&net, &ws, &test_ds);
    printf("Epoch %2d/%d | Avg Loss: %.4f | Test Acc: %.2f%%\n",
           epoch, epochs, epoch_loss / (double)steps, test_acc);
    }
  }

  free(indices);
  workspace_free(&ws);
  mlp_free(&net);
  free_dataset(&train_ds);
  free_dataset(&test_ds);
  return 0;
}
