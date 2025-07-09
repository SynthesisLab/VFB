#include "json.hpp"
#include <fstream>
#include <iostream>
#include <stack>

using namespace std;
using json = nlohmann::json;

const int maxNumOfVars = 10;
const int maxNumOfTraces = 100;
const int maxFormulaSize = 32;
const int maxTraceLength = 64;
int numVar;
int numOfTraces;
int numOfPosTraces;
vector<string> varNames;
const char *opStr[7] = {"(~)", "(&)", "(|)", "(X)", "(F)", "(G)", "(U)"};

__constant__ int d_numVar;
__constant__ int d_numOfTraces;
__constant__ int d_numOfPosTraces;
__constant__ uint64_t d_inputData[maxNumOfVars * maxNumOfTraces];
__constant__ int d_traceLen[maxNumOfTraces];
__constant__ uint64_t d_numForm[maxFormulaSize + 1][8];
__device__ __constant__ char d_opChar[7] = {'~', '&', '|', 'X', 'F', 'G', 'U'};

inline cudaError_t checkCuda(cudaError_t res) {
  if (res != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
    assert(res == cudaSuccess);
  }
  return res;
}

tuple<uint64_t *, int *, int> readJsonFile(const string &filename) {

  ifstream file(filename);
  if (!file) {
    cerr << "Error: Could not open file " << filename << endl;
    return {nullptr, nullptr, 0};
  }

  json j;
  file >> j;

  numVar = j["number_atomic_propositions"];
  numOfTraces = j["number_traces"];
  numOfPosTraces = j["number_positive_traces"];
  const int traceLength = j["max_length_traces"];

  uint64_t *inputData = new uint64_t[numOfTraces * numVar];
  int *traceLen = new int[numOfTraces];

  varNames = j["atomic_propositions"];

  auto fillTrace = [&](const json &trace, int index) {
    traceLen[index] = trace[varNames[0]].size();
    for (int v = 0; v < numVar; ++v) {
      string varName = varNames[v];
      const vector<int> &values = trace[varName];
      uint64_t encoded = 0;
      for (int t = 0; t < values.size(); ++t)
        if (values[t])
          encoded |= (1ULL << t);
      inputData[index * numVar + v] = encoded;
    }
  };

  int traceIndex = 0;
  for (const auto &trace : j["positive_traces"])
    fillTrace(trace, traceIndex++);
  for (const auto &trace : j["negative_traces"])
    fillTrace(trace, traceIndex++);

  return {inputData, traceLen, traceLength};
}

__device__ bool evaluateRPN(char formula[maxFormulaSize], int size) {

  uint64_t stack[32];
  int stackIdx;

  for (int i = 0; i < d_numOfTraces; ++i) {

    stackIdx = -1;

    for (int j = 0; j < size; ++j) {

      char token = formula[j];

      if (token >= '0' && token <= '9') {
        // Variables
        int var = token - '0';
        stack[++stackIdx] = d_inputData[i * d_numVar + var];
      } else if (token == '~' || token == 'X' || token == 'F' || token == 'G') {
        // Unary operators
        uint64_t left = stack[stackIdx];
        switch (token) {
        case '~':
          stack[stackIdx] = ~left & (1ULL << d_traceLen[i]) - 1;
          break;
        case 'X':
          stack[stackIdx] = left >> 1;
          break;
        case 'F':
          left |= left >> 1;
          left |= left >> 2;
          left |= left >> 4;
          left |= left >> 8;
          left |= left >> 16;
          left |= left >> 32;
          stack[stackIdx] = left;
          break;
        case 'G':
          left = ~left & (1ULL << d_traceLen[i]) - 1;
          left |= left >> 1;
          left |= left >> 2;
          left |= left >> 4;
          left |= left >> 8;
          left |= left >> 16;
          left |= left >> 32;
          stack[stackIdx] &= ~left;
          break;
        }
      } else {
        // Binary operators
        uint64_t right = stack[stackIdx--];
        uint64_t left = stack[stackIdx];
        switch (token) {
        case '&':
          stack[stackIdx] = left & right;
          break;
        case '|':
          stack[stackIdx] = left | right;
          break;
        case 'U':
          right |= left & (right >> 1);
          left &= left >> 1;
          right |= left & (right >> 2);
          left &= left >> 2;
          right |= left & (right >> 4);
          left &= left >> 4;
          right |= left & (right >> 8);
          left &= left >> 8;
          right |= left & (right >> 16);
          left &= left >> 16;
          right |= left & (right >> 32);
          stack[stackIdx] = right;
          break;
        }
      }
    }

    if (!(stack[0] & 1) && i < d_numOfPosTraces)
      return false;
    if ((stack[0] & 1) && i >= d_numOfPosTraces)
      return false;
  }

  return true;
}

struct StackEntry {
  uint64_t n;
  int size;
  int shift;
};

__device__ void printFormula(int size, char formula[maxFormulaSize]) {
  for (int i = 0; i < size; ++i)
    printf("%c ", formula[i]);
  printf("\n");
}

__device__ void numberToFormula(uint64_t n, int size,
                                char formula[maxFormulaSize]) {

  StackEntry stack[maxFormulaSize];
  int shift = 0;
  stack[0] = {n, size, shift};
  int stackIdx = 0;

  uint64_t partSum;
  int opIdx;
  int fIdx;
  uint64_t numRightForm;

  while (stackIdx >= 0) {

    StackEntry entry = stack[stackIdx--];
    n = entry.n;
    size = entry.size;
    shift = entry.shift;

    if (size == 1)
      formula[shift] = '0' + n;

    else {

      // Find the next operator
      partSum = 0;
      opIdx = 0;
      while (n >= partSum + d_numForm[size][opIdx])
        partSum += d_numForm[size][opIdx++];

      n -= partSum;
      size--;
      formula[shift + size] = d_opChar[opIdx];

      // If the operator is unary, push it to the stack
      if (opIdx == 0 || opIdx == 3 || opIdx == 4 || opIdx == 5)
        stack[++stackIdx] = {n, size, shift};

      else {

        // If the operator is binary, find the size of the operands
        partSum = 0;
        fIdx = 1;
        while (n >= partSum + d_numForm[fIdx][7] * d_numForm[size - fIdx][7]) {
          partSum += d_numForm[fIdx][7] * d_numForm[size - fIdx++][7];
        }

        n -= partSum;
        numRightForm = d_numForm[size - fIdx][7];
        stack[++stackIdx] = {n / numRightForm, fIdx, shift};
        stack[++stackIdx] = {n % numRightForm, size - fIdx, shift + fIdx};
      }
    }
  }
}

__global__ void processOperator(int LTLLen, uint64_t maxTid, uint64_t offset,
                                char *d_LTLFormula, int *d_foundFlag) {

  uint64_t tid =
      static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(blockIdx.x) +
      static_cast<uint64_t>(threadIdx.x);

  if (tid < maxTid) {

    tid += offset;

    char formula[maxFormulaSize];
    numberToFormula(tid, LTLLen, formula);
    bool found = evaluateRPN(formula, LTLLen);

    // If the formula verify all samples, copy it
    if (found && atomicCAS(d_foundFlag, 0, 1) == 0)
      for (int i = 0; i < LTLLen; ++i)
        d_LTLFormula[i] = formula[i];
  }
}

void printMatrix(uint64_t *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      printf("%-12lu ", matrix[i * n + j]);
    printf("\n");
  }
}

uint64_t *generateMatrix(int maxLen) {

  // Last column is the number of formulas of the given size
  uint64_t *numForm = new uint64_t[(maxLen + 1) * 8]();

  // Initialisation with variables and unary operators
  numForm[1 * 8 + 7] = numVar;     // Total number of formulas of size 1
  numForm[2 * 8 + 0] = numVar;     // Number of ~ formulas of size 2
  numForm[2 * 8 + 3] = numVar;     // Number of X formulas of size 2
  numForm[2 * 8 + 4] = numVar;     // Number of F formulas of size 2
  numForm[2 * 8 + 5] = numVar;     // Number of G formulas of size 2
  numForm[2 * 8 + 7] = 4 * numVar; // Total number of formulas of size 2

  // Generation
  uint64_t unaryForm;
  uint64_t commutBinaryForm;
  uint64_t notCommutBinaryForm;

  for (int i = 3; i <= maxLen; ++i) {
    unaryForm = numForm[(i - 1) * 8 + 7];
    commutBinaryForm = 0;
    notCommutBinaryForm = 0;
    for (int j = 1; j <= (i - 1) / 2; ++j)
      commutBinaryForm += numForm[j * 8 + 7] * numForm[(i - j - 1) * 8 + 7];
    for (int j = 1; j <= i - 2; ++j)
      notCommutBinaryForm += numForm[j * 8 + 7] * numForm[(i - j - 1) * 8 + 7];
    numForm[i * 8] = unaryForm;               // Number of ~ formulas of size i
    numForm[i * 8 + 1] = commutBinaryForm;    // Number of & formulas of size i
    numForm[i * 8 + 2] = commutBinaryForm;    // Number of | formulas of size i
    numForm[i * 8 + 3] = unaryForm;           // Number of X formulas of size i
    numForm[i * 8 + 4] = unaryForm;           // Number of F formulas of size i
    numForm[i * 8 + 5] = unaryForm;           // Number of G formulas of size i
    numForm[i * 8 + 6] = notCommutBinaryForm; // Number of U formulas of size i
    numForm[i * 8 + 7] =
        4 * unaryForm + 2 * commutBinaryForm +
        notCommutBinaryForm; // Total number of formulas of size i
  }

  // printMatrix(numForm, maxLen + 1, 9);
  return numForm;
}

string LTLToString(const char *LTLFormula) {

  stack<string> stack;

  for (int i = 0; LTLFormula[i] != '\0'; ++i) {

    char token = LTLFormula[i];

    if (token >= '0' && token <= '9') {
      stack.push(varNames[token - '0']);
    } else {
      if (token == '~' || token == 'X' || token == 'F' || token == 'G') {
        string left = stack.top();
        stack.pop();
        stack.push(string(1, token) + "(" + left + ")");
      } else {
        string right = stack.top();
        stack.pop();
        string left = stack.top();
        stack.pop();
        stack.push("(" + left + ") " + string(1, token) + " (" + right + ")");
      }
    }
  }

  if (stack.size() != 1)
    return "Error : Incorrect Formula.\n";
  return stack.top();
}

string LTL(const int maxLen, uint64_t *inputData, int *traceLen,
           int traceLength) {

  // --------------------------------------
  // Memory allocation & Checking variables
  // --------------------------------------

  if (maxLen > maxFormulaSize) {
    printf("This version supports formulas of size at most %d.\n",
           maxFormulaSize);
    return "see_the_error";
  }

  if (numOfTraces > maxNumOfTraces) {
    printf("This version supports at most %d samples.\n", maxNumOfTraces);
    return "see_the_error";
  }

  if (numVar > maxNumOfVars) {
    printf("This version supports at most %d variables.\n", maxNumOfVars);
    return "see_the_error";
  }

  if (traceLength > maxTraceLength) {
    printf("This version supports traces of size at most %d.\n",
           maxTraceLength);
    return "see_the_error";
  }

  // Copying number of vars, inputs and outputs into the constant memory
  checkCuda(cudaMemcpyToSymbol(d_numVar, &numVar, sizeof(int)));
  checkCuda(cudaMemcpyToSymbol(d_numOfTraces, &numOfTraces, sizeof(int)));
  checkCuda(cudaMemcpyToSymbol(d_numOfPosTraces, &numOfPosTraces, sizeof(int)));
  checkCuda(cudaMemcpyToSymbol(d_inputData, inputData,
                               numVar * numOfTraces * sizeof(uint64_t)));
  checkCuda(
      cudaMemcpyToSymbol(d_traceLen, traceLen, numOfTraces * sizeof(int)));

  // Number of generated formulas
  uint64_t allLTLs{};

  // Checking variables as potential solution
  printf("Length %-2d | Vars | CheckedLTLs: %-13lu | ToBeChecked: %-12d \n", 1,
         allLTLs, numVar);
  bool found;
  for (int i = 0; i < numVar; ++i) {
    found = true;
    for (int j = 0; j < numOfPosTraces; j++)
      if (!(inputData[j * numVar + i] & 1))
        found = false;
    for (int j = numOfPosTraces; j < numOfTraces; j++)
      if (inputData[j * numVar + i] & 1)
        found = false;
    allLTLs++;
    if (found)
      return varNames[i];
  }

  // Memory allocation for the potential solution
  char *d_LTLFormula;
  char *LTLFormula = new char[maxFormulaSize];
  LTLFormula[0] = '\0';
  checkCuda(cudaMalloc(&d_LTLFormula, maxFormulaSize * sizeof(char)));
  int *d_foundFlag;
  checkCuda(cudaMalloc(&d_foundFlag, sizeof(int)));
  checkCuda(cudaMemset(d_foundFlag, 0, sizeof(int)));

  // Number of formulas matrix
  uint64_t *numForm = generateMatrix(maxLen);
  checkCuda(cudaMemcpyToSymbol(d_numForm, numForm,
                               (maxLen + 1) * 8 * sizeof(uint64_t)));

  // ----------------------------
  // Enumeration of the next LTLs
  // ----------------------------

  uint64_t offset;
  uint64_t N;
  uint64_t blockSize;
  bool stop = false;

  for (int LTLLen = 2; LTLLen <= maxLen && !stop; ++LTLLen) {

    offset = 0;

    for (int i = 0; i < 7 && !stop; ++i) {

      N = numForm[LTLLen * 8 + i];
      blockSize = (N + 1023) / 1024;
      printf("Length %-2d | %-4s | CheckedLTLs: %-13lu | ToBeChecked: %-12lu\n",
             LTLLen, opStr[i], allLTLs, N);
      if (N > 0)
        processOperator<<<blockSize, 1024>>>(LTLLen, N, offset, d_LTLFormula,
                                             d_foundFlag);
      offset += N;
      allLTLs += N;

      checkCuda(cudaPeekAtLastError());
      checkCuda(cudaMemcpy(LTLFormula, d_LTLFormula,
                           maxFormulaSize * sizeof(char),
                           cudaMemcpyDeviceToHost));
      if (LTLFormula[0] != '\0')
        stop = true;
    }
  }

  // --------------------------------
  // Returning the solution & Cleanup
  // --------------------------------

  string output;

  if (LTLFormula[0] != '\0') {
    output = LTLToString(LTLFormula);
  } else {
    output = "Not found !";
  }

  delete[] numForm;
  cudaFree(d_LTLFormula);
  cudaFree(d_foundFlag);
  return output;
}

int main(int argc, char *argv[]) {

  // -----------------
  // Reading the input
  // -----------------

  if (argc != 3) {
    printf("Arguments should be in the form of\n");
    printf(
        "-----------------------------------------------------------------\n");
    printf("%s <input_file_address> <maxLen>\n", argv[0]);
    printf(
        "-----------------------------------------------------------------\n");
    return 0;
  }

  if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50) {
    printf("Argument maxLen = %s should be between 1 and %d", argv[2],
           maxFormulaSize);
    return 0;
  }

  string fileName = argv[1];
  auto [inputData, traceLen, traceLength] = readJsonFile(fileName);
  int maxLen = atoi(argv[2]);

  // ---------------------------
  // Linear Temporal Logic (LTL)
  // ---------------------------

  string output = LTL(maxLen, inputData, traceLen, traceLength);
  if (output == "see_the_error")
    return 0;

  printf("\nLTL: \"%s\"\n", output.c_str());

  return 0;
}