# VFB - GPU-Accelerated LTL Formula Synthesis

A CUDA-based tool for synthesizing Linear Temporal Logic (LTL) formulas from positive and negative trace examples using GPU acceleration.

## Overview

This project implements efficient algorithms for LTL formula synthesis that leverage GPU parallel processing to search through the space of possible formulas. Given a set of positive traces (behaviors that should satisfy the formula) and negative traces (behaviors that should violate it), the tool finds an LTL formula that correctly classifies all examples.

## Features

- **GPU-accelerated enumeration**: Parallel exploration of formula space using CUDA
- **Multiple implementations**:
  - `ltl_nocache.cu`: Basic enumeration without caching
  - `LTL_BS.cu`: Full implementation with hash-based caching using WarpCore hash sets
  - `LTL_BS_light.cu`: Lightweight version with optimized caching
- **Efficient trace representation**: Bitwise encoding of temporal traces (up to 64 time steps)
- **Hash-based deduplication**: Uses WarpCore GPU hash tables to avoid redundant formula checks
- **Flexible input format**: JSON-based trace specification

## Supported LTL Operators

- `~` (Not): Negation
- `&` (And): Conjunction
- `|` (Or): Disjunction
- `X` (Next): Next state
- `F` (Finally): Eventually in the future
- `G` (Globally): Always in the future
- `U` (Until): Until operator

## Requirements

- **NVIDIA GPU** with compute capability 8.0 or higher (configured for sm_80)
- **CUDA Toolkit** (version supporting C++20)
- **nvcc** compiler
- C++20 compatible compiler

## Building

```bash
make
```

This will compile the `ltl_nocache.cu` implementation targeting sm_80 architecture.

For other implementations, compile manually:
```bash
nvcc LTL_BS.cu -o ltl_bs -arch=sm_80 -std=c++20
nvcc LTL_BS_light.cu -o ltl_bs_light -arch=sm_80 -std=c++20
```

## Input Format

The tool reads trace specifications from JSON files. See [ExampleLTLF.json](ExampleLTLF.json) for an example.

```json
{
  "number_atomic_propositions": 2,
  "number_traces": 3,
  "number_positive_traces": 1,
  "max_length_traces": 4,
  "atomic_propositions": ["a0", "a1"],
  "positive_traces": [
    {
      "a0": [0, 1, 1, 0],
      "a1": [0, 1, 0, 1]
    }
  ],
  "negative_traces": [
    {
      "a0": [1, 1, 1, 0],
      "a1": [0, 1, 0, 1]
    },
    {
      "a0": [0, 1, 1, 0],
      "a1": [0, 1, 1, 1]
    }
  ]
}
```

### JSON Fields

- `number_atomic_propositions`: Number of boolean variables
- `number_traces`: Total number of traces (positive + negative)
- `number_positive_traces`: Number of positive examples
- `max_length_traces`: Maximum trace length
- `atomic_propositions`: Names of the boolean variables
- `positive_traces`: Array of traces that should satisfy the formula
- `negative_traces`: Array of traces that should not satisfy the formula

Each trace is an object where keys are atomic proposition names and values are arrays of 0s and 1s representing the truth values over time.

## Usage

```bash
./ltl_nocache <json_file> [max_formula_size]
```

Example:
```bash
./ltl_nocache ExampleLTLF.json 10
```

This will search for LTL formulas up to size 10 that satisfy all positive traces and reject all negative traces.

## Implementation Details

### Trace Encoding

Traces are encoded as 64-bit unsigned integers where each bit represents the truth value of an atomic proposition at a specific time step. This allows for efficient bitwise operations when evaluating temporal operators.

### Formula Enumeration

Formulas are enumerated in increasing size order using a systematic numbering scheme. Each formula is represented in Reverse Polish Notation (RPN) for efficient GPU evaluation.

### Hash-Based Caching (LTL_BS variants)

The BS (Binary Search) variants use GPU hash sets from the WarpCore library to cache already-seen formula semantics, avoiding redundant evaluations:
- Formulas are evaluated to produce characteristic sets (CS) representing their behavior on traces
- Hash values are computed from CS patterns
- Duplicate semantic patterns are filtered out using GPU hash tables

### Relaxed Uniqueness Checking

When trace lengths exceed 126 time steps combined, the implementations use hash-based relaxed uniqueness checking to maintain efficiency while keeping hash values within 128 bits.

## Project Structure

```
.
├── ltl_nocache.cu              # Main implementation without caching
├── LTL_BS.cu                   # Full implementation with caching
├── LTL_BS_light.cu             # Lightweight cached version
├── json.hpp                    # JSON parsing library (nlohmann/json)
├── ExampleLTLF.json            # Example input file
├── Makefile                    # Build configuration
└── modified_libraries/
    ├── helpers/                # CUDA utility functions
    │   ├── cuda_helpers.cuh
    │   ├── hashers.cuh
    │   └── ...
    └── warpcore/               # GPU hash table library
        ├── hash_set.cuh
        ├── single_value_hash_table.cuh
        └── ...
```

## Limitations

Current version constraints (configurable in source):
- Maximum 10 atomic propositions (`ltl_nocache`)
- Maximum 100 traces (`ltl_nocache`)
- Maximum 63 traces (BS variants, due to hash encoding)
- Maximum 64 time steps per trace
- Maximum formula size: 32 operators/variables

## Performance

The GPU-accelerated approach provides significant speedup over CPU-based enumeration, especially for larger formula sizes. The hash-based caching in BS variants further reduces redundant work by eliminating semantically equivalent formulas.

## References

This implementation is based on research in GPU-accelerated LTL synthesis and uses the WarpCore library for efficient GPU hash tables.