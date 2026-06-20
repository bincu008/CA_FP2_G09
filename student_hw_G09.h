#pragma once

#include <array>

namespace student_hw {

// ============================================================
// Student-tunable hardware configuration
// ------------------------------------------------------------
// This file defines the hardware parameters that students are
// allowed to tune for exploration.
//
// IMPORTANT:
//   You may change ONLY the numeric values on the right-hand side.
//   Do NOT change the format of this file.
//
// Specifically, do NOT:
//   1) rename any field,
//   2) remove any field,
//   3) add any new field,
//   4) change NUM_BUFS,
//   5) change the order of the 12 SRAM bank fields,
//   6) add comments inside the BUF_SIZES initializer list.
//
// Why this restriction exists:
//   The TA-side grading script uses regular expressions to read
//   the numeric values from this file. If you change the format,
//   the script may fail to parse your submission correctly.
//
// Grading flow:
//   - Students submit this header together with their instruction file.
//   - The TA will place the submitted parameters into the official
//     grading environment and re-run the simulator.
//   - Final correctness and score are based on the TA-side run only.
//
// ------------------------------------------------------------
// Official grading baseline values
// ------------------------------------------------------------
// ARRAY_MACS_PER_CYCLE = 64
// DRAM_LANES           = 64
// SRAM_DMA_WIDTH       = 8
// NOC_LINK_WIDTH       = 8
// VECTOR_LANES         = 128
//
// ------------------------------------------------------------
// Fixed local SRAM bank IDs
// ------------------------------------------------------------
// The simulator always uses exactly 12 local SRAM banks.
// Their IDs and functions are FIXED and must NOT be changed.
//
//   Bank  0 : Array Unit input
//   Bank  1 : Array Unit output
//   Bank  2 : Vector Unit input
//   Bank  3 : Vector Unit weight
//   Bank  4 : Vector Unit output
//   Bank  5 : Other input
//   Bank  6 : Other output
//   Bank  7 : K
//   Bank  8 : V
//   Bank  9 : Q_SV
//   Bank 10 : Residual / RMS
//   Bank 11 : Core_Controller
//
// You are allowed to change only the CAPACITY of each bank,
// measured in FP32 elements.
//
// Example:
//   If BUF_ARRAY_IN_SIZE = 2048, then bank 0 can store 2048
//   FP32 elements.
// ============================================================
struct Config {
  static constexpr int ARRAY_MACS_PER_CYCLE = 24;
  static constexpr int DRAM_LANES = 24;
  static constexpr int SRAM_DMA_WIDTH = 14;
  static constexpr int NOC_LINK_WIDTH = 12;
  static constexpr int VECTOR_LANES = 32;

  // Fixed interface constant. Do NOT change this value.
  static constexpr int NUM_BUFS = 12;

  // ----------------------------------------------------------
  // Local SRAM bank capacities (unit: FP32 elements)
  // ----------------------------------------------------------
  static constexpr int BUF_ARRAY_IN_SIZE = 1488;      // Bank 0
  static constexpr int BUF_ARRAY_OUT_SIZE = 4058;    // Bank 1
  static constexpr int BUF_VEC_IN_SIZE = 1488;        // Bank 2
  static constexpr int BUF_VEC_W_SIZE = 1488;         // Bank 3
  static constexpr int BUF_VEC_OUT_SIZE = 1488;       // Bank 4
  static constexpr int BUF_OTHER_IN_SIZE = 0;        // Bank 5
  static constexpr int BUF_OTHER_OUT_SIZE = 0;       // Bank 6
  static constexpr int BUF_K_SIZE = 1;             // Bank 7
  static constexpr int BUF_V_SIZE = 1;               // Bank 8
  static constexpr int BUF_Q_SV_SIZE = 0;            // Bank 9
  static constexpr int BUF_RMS_SIZE = 0;             // Bank 10
  static constexpr int BUF_CORE_CONTROLLER_SIZE = 0; // Bank 11

  inline static constexpr std::array<int, NUM_BUFS> BUF_SIZES = {
      BUF_ARRAY_IN_SIZE,  BUF_ARRAY_OUT_SIZE, BUF_VEC_IN_SIZE,
      BUF_VEC_W_SIZE,     BUF_VEC_OUT_SIZE,   BUF_OTHER_IN_SIZE,
      BUF_OTHER_OUT_SIZE, BUF_K_SIZE,         BUF_V_SIZE,
      BUF_Q_SV_SIZE,      BUF_RMS_SIZE,       BUF_CORE_CONTROLLER_SIZE};
};

} // namespace student_hw