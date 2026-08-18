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

    // Bring inputs to SRAM
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build transposed key block [d, 1] in place
    gpu_sim.Transpose(keys[i], kInSharedMemory);
    Matrix *k_i_T = keys[i];

    // Build new K^T [d, i+1]
    Matrix *new_kt;
    if (kt_concat == nullptr) {
      new_kt = k_i_T; // first round, reuse the transposed key
    } else {
      new_kt = matrix_memory_allocator.Allocate("KT_" + std::to_string(i));
      gpu_sim.Concat(kt_concat, k_i_T, new_kt, 1, kInSharedMemory);
      gpu_sim.ReleaseMatrix(kt_concat);
      gpu_sim.ReleaseMatrix(k_i_T);
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

    // Compute attention row by row
    Matrix *answer = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      Matrix *q_j = matrix_memory_allocator.Allocate("q_" + std::to_string(j));
      gpu_sim.GetRow(current_query, j, q_j, kInSharedMemory);

      Matrix *s_j = matrix_memory_allocator.Allocate("s_" + std::to_string(j));
      gpu_sim.MatMul(q_j, kt_concat, s_j); // [1,d] * [d,n] -> [1,n]

      Matrix *e_j = matrix_memory_allocator.Allocate("e_" + std::to_string(j));
      gpu_sim.MatExp(s_j, e_j);

      Matrix *sum_j = matrix_memory_allocator.Allocate("sum_" + std::to_string(j));
      gpu_sim.Sum(e_j, sum_j);

      Matrix *p_j = matrix_memory_allocator.Allocate("p_" + std::to_string(j));
      gpu_sim.MatDiv(e_j, sum_j, p_j);

      Matrix *ans_j = matrix_memory_allocator.Allocate("ans_" + std::to_string(j));
      gpu_sim.MatMul(p_j, v_concat, ans_j); // [1,n] * [n,d] -> [1,d]

      if (j == 0) {
        answer = ans_j;
      } else {
        Matrix *answer_new = matrix_memory_allocator.Allocate("answer_new_" + std::to_string(j));
        gpu_sim.Concat(answer, ans_j, answer_new, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(ans_j);
        answer = answer_new;
      }

      gpu_sim.ReleaseMatrix(q_j);
      gpu_sim.ReleaseMatrix(s_j);
      gpu_sim.ReleaseMatrix(e_j);
      gpu_sim.ReleaseMatrix(sum_j);
      gpu_sim.ReleaseMatrix(p_j);
    }

    // Move final answer to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(answer);
    gpu_sim.ReleaseMatrix(current_query);

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
