#pragma once
#include "simulator.hpp"
#include <string>
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  Matrix *kt_concat = nullptr; // maintained as [d, n] in SRAM
  Matrix *v_concat = nullptr;  // maintained as [n, d] in SRAM
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Bring key/value inputs to SRAM and build the concatenated blocks.
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    // Build transposed key block [d, 1] in place
    gpu_sim.Transpose(keys[i], kInSharedMemory);

    // Build new K^T [d, i+1]
    Matrix *new_kt;
    if (kt_concat == nullptr) {
      new_kt = keys[i]; // first round, reuse the transposed key
    } else {
      new_kt = matrix_memory_allocator.Allocate("KT_" + std::to_string(i));
      gpu_sim.Concat(kt_concat, keys[i], new_kt, 1, kInSharedMemory);
      gpu_sim.ReleaseMatrix(kt_concat);
      gpu_sim.ReleaseMatrix(keys[i]);
    }
    kt_concat = new_kt;

    // Build new V [i+1, d]
    Matrix *new_v;
    if (v_concat == nullptr) {
      new_v = values[i]; // first round
    } else {
      new_v = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
      gpu_sim.Concat(v_concat, values[i], new_v, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(v_concat);
      gpu_sim.ReleaseMatrix(values[i]);
    }
    v_concat = new_v;

    // Bring query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Compute S = Q * K^T by splitting the inner dimension into 1-element
    // outer products: S = sum_j (Q[:,j] * K^T[j,:]). This avoids the d^2 cost
    // of a single large MatMul.
    const size_t d = current_query->GetColumnNum();
    Matrix *S_acc = nullptr;
    for (size_t j = 0; j < d; ++j) {
      Matrix *q_col = matrix_memory_allocator.Allocate("qcol_" + std::to_string(j));
      gpu_sim.GetColumn(current_query, j, q_col, kInSharedMemory);

      Matrix *k_row = matrix_memory_allocator.Allocate("krow_" + std::to_string(j));
      gpu_sim.GetRow(kt_concat, j, k_row, kInSharedMemory);

      Matrix *s_part = matrix_memory_allocator.Allocate("spart_" + std::to_string(j));
      gpu_sim.MatMul(q_col, k_row, s_part); // [n,1] * [1,n] -> [n,n]

      if (j == 0) {
        S_acc = s_part;
        gpu_sim.ReleaseMatrix(q_col);
        gpu_sim.ReleaseMatrix(k_row);
      } else {
        Matrix *s_new = matrix_memory_allocator.Allocate("snew_" + std::to_string(j));
        gpu_sim.MatAdd(S_acc, s_part, s_new);
        gpu_sim.ReleaseMatrix(S_acc);
        gpu_sim.ReleaseMatrix(s_part);
        gpu_sim.ReleaseMatrix(q_col);
        gpu_sim.ReleaseMatrix(k_row);
        S_acc = s_new;
      }
    }
    gpu_sim.ReleaseMatrix(current_query);

    Matrix *E = matrix_memory_allocator.Allocate("E_" + std::to_string(i));
    gpu_sim.MatExp(S_acc, E);
    gpu_sim.ReleaseMatrix(S_acc);

    // Softmax each row of E and accumulate the weighted value sum.
    Matrix *answer = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *e_r = matrix_memory_allocator.Allocate("e_" + std::to_string(r));
      gpu_sim.GetRow(E, r, e_r, kInSharedMemory);

      Matrix *sum_r = matrix_memory_allocator.Allocate("sum_" + std::to_string(r));
      gpu_sim.Sum(e_r, sum_r);

      Matrix *p_r = matrix_memory_allocator.Allocate("p_" + std::to_string(r));
      gpu_sim.MatDiv(e_r, sum_r, p_r);

      // ans_r = sum_k p_r[k] * V[k] using 1x1 MatMul as scalar multiply.
      Matrix *ans_acc = nullptr;
      for (size_t k = 0; k <= i; ++k) {
        Matrix *v_k = matrix_memory_allocator.Allocate("v_" + std::to_string(k));
        gpu_sim.GetRow(v_concat, k, v_k, kInSharedMemory);

        Matrix *p_rk = matrix_memory_allocator.Allocate("prk_" + std::to_string(k));
        gpu_sim.GetColumn(p_r, k, p_rk, kInSharedMemory);

        Matrix *scaled = matrix_memory_allocator.Allocate("scaled_" + std::to_string(k));
        gpu_sim.MatMul(p_rk, v_k, scaled); // [1,1] * [1,d] -> [1,d]

        if (k == 0) {
          ans_acc = scaled;
          gpu_sim.ReleaseMatrix(v_k);
          gpu_sim.ReleaseMatrix(p_rk);
        } else {
          Matrix *ans_new = matrix_memory_allocator.Allocate("ansacc_" + std::to_string(k));
          gpu_sim.MatAdd(ans_acc, scaled, ans_new);
          gpu_sim.ReleaseMatrix(ans_acc);
          gpu_sim.ReleaseMatrix(v_k);
          gpu_sim.ReleaseMatrix(p_rk);
          gpu_sim.ReleaseMatrix(scaled);
          ans_acc = ans_new;
        }
      }

      Matrix *ans_r = ans_acc;
      if (r == 0) {
        answer = ans_r;
      } else {
        Matrix *answer_new = matrix_memory_allocator.Allocate("answer_new_" + std::to_string(r));
        gpu_sim.Concat(answer, ans_r, answer_new, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(ans_r);
        answer = answer_new;
      }

      gpu_sim.ReleaseMatrix(e_r);
      gpu_sim.ReleaseMatrix(sum_r);
      gpu_sim.ReleaseMatrix(p_r);
    }
    gpu_sim.ReleaseMatrix(E);

    // Move final answer to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(answer);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
