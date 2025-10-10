#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <warpcore/hash_set.cuh>

using namespace std;

const size_t maxNumOfTraces = 63;

set<char> alphabet; // Set of characters in the traces
int alphabetSize; // Size of the alphabet
int numOfTraces; // Number of traces
int numOfP; // Number of positive traces
int lenSum; // Sum of the lengths of all traces

__constant__ int d_alphabetSize;
__constant__ int d_numOfTraces;
__constant__ int d_numOfP;
__constant__ int d_lenSum;
__constant__ char d_traceLen[maxNumOfTraces]; // Length of each trace

enum class Op { Not, And, Or, Next, Finally, Globally, Until }; // Supported operators

inline
cudaError_t checkCuda(cudaError_t res) {
    if (res != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
        assert(res == cudaSuccess);
    }
    return res;
}

// Making the hash values from the CSs
__device__ void generateCSHashs(
    uint64_t* CS,
    uint64_t& hash)
{

    if (d_lenSum > 64) {

        // We use a hash function on all CSs to create the final hash values
        for (int i = 0; i < d_numOfTraces; ++i) {
            uint64_t x = CS[i];
            x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
            x = x ^ (x >> 31);
            hash ^= x;
        }

    }

    else {

        // We simply concatenate everything together
        hash = CS[0];
        for (int i = 1; i < d_numOfTraces; ++i) {
            hash <<= d_traceLen[i - 1];
            hash |= CS[i];
        }

    }

}

// Initialising the hashSet with the alphabet before starting the enumeration
template<class hash_set_t>
__global__ void hashSetsInit(
    hash_set_t hashSet,
    uint64_t* d_LTLcache)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    uint64_t CS[maxNumOfTraces];
    for (int i = 0; i < d_numOfTraces; ++i)
        CS[i] = d_LTLcache[tid * d_numOfTraces + i];

    uint64_t hash{};
    generateCSHashs(CS, hash);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    hashSet.insert(hash, group);

}

// Initialising the hashSet with the reduced LTL formulas before starting the enumeration
template<class hash_set_t>
__global__ void BSHashSetsInit(
    hash_set_t hashSet,
    uint64_t* d_BSCache)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    hashSet.insert(d_BSCache[tid], group);

}

// Delete LTL formulas reducing to the same formula when switching to BS
void BSInitialisation(
    int& LTLLastIdx,
    uint64_t* d_BSCache)
{

    thrust::device_ptr<uint64_t> new_end_ptr;
    thrust::device_ptr<uint64_t> d_BSCache_ptr(d_BSCache);
    new_end_ptr = thrust::remove(d_BSCache_ptr, d_BSCache_ptr + LTLLastIdx, (uint64_t)-1);
    LTLLastIdx = static_cast<int>(new_end_ptr - d_BSCache_ptr);

}

// Applying op on the formulas indexed by ldx and rdx in the cache
template<Op op>
__device__ void applyOperator(
    uint64_t* CS,
    uint64_t* d_LTLcache,
    int ldx, int rdx)
{

    if constexpr (op == Op::Not) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            uint64_t negationFixer = ((uint64_t)1 << d_traceLen[i]) - 1;
            CS[i] = ~d_LTLcache[ldx * d_numOfTraces + i] & negationFixer;
        }
    } else if constexpr (op == Op::And) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * d_numOfTraces + i] & d_LTLcache[rdx * d_numOfTraces + i];
        }
    } else if constexpr (op == Op::Or) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * d_numOfTraces + i] | d_LTLcache[rdx * d_numOfTraces + i];
        }
    } else if constexpr (op == Op::Next) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * d_numOfTraces + i] >> 1;
        }
    } else if constexpr (op == Op::Finally) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * d_numOfTraces + i];
            CS[i] |= CS[i] >> 1; CS[i] |= CS[i] >> 2; CS[i] |= CS[i] >> 4;
            CS[i] |= CS[i] >> 8; CS[i] |= CS[i] >> 16; CS[i] |= CS[i] >> 32;
        }
    } else if constexpr (op == Op::Globally) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * d_numOfTraces + i];
            uint64_t cs = ~CS[i] & (((uint64_t)1 << d_traceLen[i]) - 1);
            cs |= cs >> 1; cs |= cs >> 2; cs |= cs >> 4;
            cs |= cs >> 8; cs |= cs >> 16; cs |= cs >> 32;
            CS[i] &= ~cs;
        }
    } else if constexpr (op == Op::Until) {
        for (int i = 0; i < d_numOfTraces; ++i) {
            uint64_t l = d_LTLcache[ldx * d_numOfTraces + i];
            uint64_t r = d_LTLcache[rdx * d_numOfTraces + i];
            r |= l & (r >> 1);  l &= l >> 1;
            r |= l & (r >> 2);  l &= l >> 2;
            r |= l & (r >> 4);  l &= l >> 4;
            r |= l & (r >> 8);  l &= l >> 8;
            r |= l & (r >> 16); l &= l >> 16;
            r |= l & (r >> 32);
            CS[i] = r;
        }
    } else {
        [] <bool flag = false>() { static_assert(flag, "Unhandled operator"); }();
    }

}

template<Op op>
__device__ void BSApplyOperator(
    uint64_t& CS,
    uint64_t* d_BSCache,
    int ldx, int rdx)
{

    if constexpr (op == Op::Not) {
        CS = ~d_BSCache[ldx] & (((uint64_t)1 << d_numOfTraces) - 1);
    } else if constexpr (op == Op::And) {
        CS = d_BSCache[ldx] & d_BSCache[rdx];
    } else if constexpr (op == Op::Or) {
        CS = d_BSCache[ldx] | d_BSCache[rdx];
    } else {
        [] <bool flag = false>() { static_assert(flag, "Unhandled operator"); }();
    }

}

// Checking the uniqueness of the CSs
// hashSet.insert returns a negative value if CS is unique, positive if it is a duplicate
template <typename hash_set_t>
__device__ bool checkCSUniqueness(
    uint64_t* CS,
    hash_set_t& hashSet)
{

    uint64_t hash{};
    generateCSHashs(CS, hash);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    return hashSet.insert(hash, group) < 0;

}

template <typename hash_set_t>
__device__ bool BSCheckCSUniqueness(
    uint64_t& CS,
    hash_set_t& hashSet)
{

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    return hashSet.insert(CS, group) < 0;

}

// Inserting the CSs into the temporary cache if they are unique
// Also stores the left and right indices to be able to reconstruct the formula later
// If the CS is not unique, it replaces it with -1 to mark that index as invalid
// That's why the MSB of CSs is not used and should not be used
// We also check if the CSs satisfies the positive and negative traces
__device__ void insertInCache(
    bool isUnqCS,
    uint64_t* CS,
    int tid,
    int ldx, int rdx,
    uint64_t* d_temp_LTLcache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalLTLIdx)
{

    if (isUnqCS) {

        for (int i = 0; i < d_numOfTraces; ++i)
            d_temp_LTLcache[tid * d_numOfTraces + i] = CS[i];
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;

        bool found = true;
        for (int i = 0; found && i < d_numOfP; ++i) if (!(CS[i] & 1)) found = false;
        for (int i = d_numOfP; found && i < d_numOfTraces; ++i) if (CS[i] & 1) found = false;
        if (found) atomicCAS(d_FinalLTLIdx, -1, tid);

    } else {

        for (int i = 0; i < d_numOfTraces; ++i)
            d_temp_LTLcache[tid * d_numOfTraces + i] = (uint64_t)-1;
        d_temp_leftIdx[tid] = -1; d_temp_rightIdx[tid] = -1;

    }

}

__device__ void BSInsertInCache(
    bool isUnqCS,
    uint64_t& CS,
    int tid,
    int ldx, int rdx,
    uint64_t* d_temp_BSCache,
    int* d_temp_BSLeftIdx, int* d_temp_BSRightIdx,
    int* d_FinalBSIdx)
{

    if (isUnqCS) {

        d_temp_BSCache[tid] = CS;
        d_temp_BSLeftIdx[tid] = ldx; d_temp_BSRightIdx[tid] = rdx;

        uint64_t mask = ((uint64_t)1 << d_numOfP) - 1;
        if (CS == mask) atomicCAS(d_FinalBSIdx, -1, tid);

    } else {

        d_temp_BSCache[tid] = (uint64_t)-1;
        d_temp_BSLeftIdx[tid] = -1; d_temp_BSRightIdx[tid] = -1;

    }

}

// Combining all together : create CSs, check uniqueness, insert in cache
// There is a special case for Until since it is not commutative
template<Op op, class hash_set_t>
__global__ void processOperator(
    const int idx1, const int idx2,
    const int idx3, const int idx4,
    uint64_t* d_LTLcache, uint64_t* d_temp_LTLcache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    hash_set_t hashSet,
    int* d_FinalLTLIdx)
{

    const int realTid = (blockDim.x * blockIdx.x + threadIdx.x);
    const int tid = (op == Op::Until) ? (realTid * 2) : realTid;
    constexpr bool isUnary = (op == Op::Not || op == Op::Next || op == Op::Finally || op == Op::Globally);
    int maxTid = isUnary ? (idx2 - idx1 + 1) : ((idx4 - idx3 + 1) * (idx2 - idx1 + 1));

    if (tid < maxTid) {

        int ldx = isUnary ? (idx1 + tid) : (idx1 + tid / (idx4 - idx3 + 1));
        int rdx = isUnary ? 0 : (idx3 + tid % (idx4 - idx3 + 1));
        uint64_t CS[maxNumOfTraces];

        applyOperator<op>(CS, d_LTLcache, ldx, rdx);
        bool isUnqCS = checkCSUniqueness(CS, hashSet);
        insertInCache(isUnqCS, CS, tid, ldx, rdx, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx);

        if (op == Op::Until) {

            applyOperator<Op::Until>(CS, d_LTLcache, rdx, ldx);
            bool isUnqCS = checkCSUniqueness(CS, hashSet);
            insertInCache(isUnqCS, CS, tid + 1, rdx, ldx, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx);

        }

    }

}

template<Op op, class hash_set_t>
__global__ void BSProcessOperator(
    const int idx1, const int idx2,
    const int idx3, const int idx4,
    uint64_t* d_BSCache, uint64_t* d_temp_BSCache,
    int* d_temp_BSLeftIdx, int* d_temp_BSRightIdx,
    hash_set_t hashSet,
    int* d_FinalBSIdx)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr bool isUnary = op == Op::Not;
    int maxTid = isUnary ? (idx2 - idx1 + 1) : ((idx4 - idx3 + 1) * (idx2 - idx1 + 1));

    if (tid < maxTid) {

        int ldx = isUnary ? (idx1 + tid) : (idx1 + tid / (idx4 - idx3 + 1));
        int rdx = isUnary ? 0 : (idx3 + tid % (idx4 - idx3 + 1));
        uint64_t CS;

        BSApplyOperator<op>(CS, d_BSCache, ldx, rdx);
        bool isUnqCS = BSCheckCSUniqueness(CS, hashSet);
        BSInsertInCache(isUnqCS, CS, tid, ldx, rdx, d_temp_BSCache, d_temp_BSLeftIdx, d_temp_BSRightIdx, d_FinalBSIdx);

    }

}

// Transfering the unique CSs from temporary to main cache
// This is where we delete the duplicates marked with -1
// If the main cache is full, we return true to stop the enumeration
bool storeUnqLTLs(
    int N,
    int& LTLLastIdx,
    const int LTLCacheCapacity,
    uint64_t* d_LTLcache, uint64_t* d_temp_LTLcache,
    int* d_leftIdx, int* d_temp_leftIdx,
    int* d_rightIdx, int* d_temp_rightIdx)
{

    thrust::device_ptr<uint64_t> new_end_ptr;
    thrust::device_ptr<uint64_t> d_LTLcache_ptr(d_LTLcache + numOfTraces * LTLLastIdx);
    thrust::device_ptr<uint64_t> d_temp_LTLcache_ptr(d_temp_LTLcache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + LTLLastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + LTLLastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);

    new_end_ptr =
        thrust::remove(d_temp_LTLcache_ptr, d_temp_LTLcache_ptr + numOfTraces * N, (uint64_t)-1);
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

    int newLTLs = static_cast<int>(new_end_ptr - d_temp_LTLcache_ptr) / numOfTraces;
    bool lastRound = false;
    if (LTLLastIdx + newLTLs > LTLCacheCapacity) {
        newLTLs = LTLCacheCapacity - LTLLastIdx;
        lastRound = true;
    }

    thrust::copy_n(d_temp_LTLcache_ptr, numOfTraces * newLTLs, d_LTLcache_ptr);
    thrust::copy_n(d_temp_leftIdx_ptr, newLTLs, d_leftIdx_ptr);
    thrust::copy_n(d_temp_rightIdx_ptr, newLTLs, d_rightIdx_ptr);

    LTLLastIdx += newLTLs;
    return lastRound;

}

bool storeUnqBSs(
    int N,
    int& BSLastIdx,
    const int BSCacheCapacity,
    uint64_t* d_BSCache, uint64_t* d_temp_BSCache,
    int* d_BSLeftIdx, int* d_temp_BSLeftIdx,
    int* d_BSRightIdx, int* d_temp_BSRightIdx)
{

    thrust::device_ptr<uint64_t> new_end_ptr;
    thrust::device_ptr<uint64_t> d_BSCache_ptr(d_BSCache + BSLastIdx);
    thrust::device_ptr<uint64_t> d_temp_BSCache_ptr(d_temp_BSCache);
    thrust::device_ptr<int> d_BSLeftIdx_ptr(d_BSLeftIdx + BSLastIdx);
    thrust::device_ptr<int> d_BSRightIdx_ptr(d_BSRightIdx + BSLastIdx);
    thrust::device_ptr<int> d_temp_BSLeftIdx_ptr(d_temp_BSLeftIdx);
    thrust::device_ptr<int> d_temp_BSRightIdx_ptr(d_temp_BSRightIdx);

    new_end_ptr = thrust::remove(d_temp_BSCache_ptr, d_temp_BSCache_ptr + N, (uint64_t)-1);
    thrust::remove(d_temp_BSLeftIdx_ptr, d_temp_BSLeftIdx_ptr + N, -1);
    thrust::remove(d_temp_BSRightIdx_ptr, d_temp_BSRightIdx_ptr + N, -1);

    int newBSs = static_cast<int>(new_end_ptr - d_temp_BSCache_ptr);
    bool lastRound = false;
    if (BSLastIdx + newBSs > BSCacheCapacity) {
        newBSs = BSCacheCapacity - BSLastIdx;
        lastRound = true;
    }

    thrust::copy_n(d_temp_BSCache_ptr, N, d_BSCache_ptr);
    thrust::copy_n(d_temp_BSLeftIdx_ptr, N, d_BSLeftIdx_ptr);
    thrust::copy_n(d_temp_BSRightIdx_ptr, N, d_BSRightIdx_ptr);
    BSLastIdx += newBSs;
    return lastRound;

}

// Bringing back the left and right indices from device to host
__global__ void generateResIndices(
    const int index,
    const int* d_leftIdx, const int* d_rightIdx,
    int* d_FinalLTLIdx)
{

    int resIdx = 0;
    while (d_FinalLTLIdx[resIdx] != -1) resIdx++;
    int queue[100];
    queue[0] = index;
    int head = 0;
    int tail = 1;

    while (head < tail) {
        int ltl = queue[head];
        int l = d_leftIdx[ltl];
        int r = d_rightIdx[ltl];
        d_FinalLTLIdx[resIdx++] = ltl;
        d_FinalLTLIdx[resIdx++] = l;
        d_FinalLTLIdx[resIdx++] = r;
        if (l >= d_alphabetSize) queue[tail++] = l;
        if (r >= d_alphabetSize) queue[tail++] = r;
        head++;
    }

}

__global__ void BSGenerateResIndices(
    const int index,
    const int LTLLastIdx,
    const int* d_BSLeftIdx, const int* d_BSRightIdx,
    int* d_FinalBSIdx)
{

    int resIdx = 0;
    while (d_FinalBSIdx[resIdx] != -1) resIdx++;
    int queue[100];
    queue[0] = index;
    int head = 0;
    int tail = 1;

    while (head < tail) {
        int ltl = queue[head];
        int l = d_BSLeftIdx[ltl];
        int r = d_BSRightIdx[ltl];
        d_FinalBSIdx[resIdx++] = ltl;
        d_FinalBSIdx[resIdx++] = l;
        d_FinalBSIdx[resIdx++] = r;
        if (l >= LTLLastIdx) queue[tail++] = l;
        if (r >= LTLLastIdx) queue[tail++] = r;
        head++;
    }

}

// Convert indices to string representation of the formula
string LTLToString(
    int index,
    map<int, pair<int, int>>& indicesMap,
    const int* LTLStartPoints)
{

    if (index < alphabetSize) {
        string s(1, *next(alphabet.begin(), index));
        return s;
    }

    int i = 0;
    while (index >= LTLStartPoints[i]) { i++; }
    i--;

    if (i % 7 == 0) {
        string res = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
        return "~(" + res + ")";
    }

    if (i % 7 == 1) {
        string left = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
        string right = LTLToString(indicesMap[index].second, indicesMap, LTLStartPoints);
        return "(" + left + ")" + "&" + "(" + right + ")";
    }

    if (i % 7 == 2) {
        string left = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
        string right = LTLToString(indicesMap[index].second, indicesMap, LTLStartPoints);
        return "(" + left + ")" + "|" + "(" + right + ")";
    }

    if (i % 7 == 3) {
        string res = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
        return "X(" + res + ")";
    }

    if (i % 7 == 4) {
        string res = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
        return "F(" + res + ")";
    }

    if (i % 7 == 5) {
        string res = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
        return "G(" + res + ")";
    }

    string left = LTLToString(indicesMap[index].first, indicesMap, LTLStartPoints);
    string right = LTLToString(indicesMap[index].second, indicesMap, LTLStartPoints);
    return "(" + left + ")" + "U" + "(" + right + ")";

}

// Constructing the string representation of the formula
string LTLString(
    const int LTLLastIdx,
    const int* LTLStartPoints,
    const int* d_leftIdx, const int* d_rightIdx,
    const int* d_temp_leftIdx, const int* d_temp_rightIdx,
    const int FinalLTLIdx)
{

    int* LIdx = new int[1];
    int* RIdx = new int[1];

    checkCuda(cudaMemcpy(LIdx, d_temp_leftIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(RIdx, d_temp_rightIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));

    int* d_resIndices;
    checkCuda(cudaMalloc(&d_resIndices, 100 * sizeof(int)));
    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 100, -1);

    if (*LIdx >= alphabetSize) generateResIndices << <1, 1 >> > (*LIdx, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= alphabetSize) generateResIndices << <1, 1 >> > (*RIdx, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[100];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 100 * sizeof(int), cudaMemcpyDeviceToHost));

    map<int, pair<int, int>> indicesMap;
    indicesMap.insert(make_pair(INT_MAX - 1, make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 100) {
        int ltl = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert(make_pair(ltl, make_pair(l, r)));
        i += 3;
    }

    if (i + 2 >= 100) return "Size of the output is too big";
    cudaFree(d_resIndices);
    return LTLToString(INT_MAX - 1, indicesMap, LTLStartPoints);

}

string BSToString(
    int index,
    map<int, pair<int, int>>& indicesMap,
    const uint64_t LTLLastIdx,
    const int* LTLStartPoints, const int* BSStartPoints,
    const int* d_leftIdx, const int* d_rightIdx)
{

    if (index < LTLLastIdx) {
        LTLString(LTLLastIdx, LTLStartPoints, d_leftIdx, d_rightIdx, d_leftIdx, d_rightIdx, index);
    }

    int i = 0;
    while (index >= BSStartPoints[i]) { i++; }
    i--;

    if (i % 3 == 0) {
        string res = BSToString(indicesMap[index].first, indicesMap, LTLLastIdx, LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        return "~(" + res + ")";
    }

    if (i % 3 == 1) {
        string left = BSToString(indicesMap[index].first, indicesMap, LTLLastIdx, LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        string right = BSToString(indicesMap[index].second, indicesMap, LTLLastIdx, LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        return "(" + left + ")" + "&" + "(" + right + ")";
    }

    else {
        string left = BSToString(indicesMap[index].first, indicesMap, LTLLastIdx, LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        string right = BSToString(indicesMap[index].second, indicesMap, LTLLastIdx, LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        return "(" + left + ")" + "|" + "(" + right + ")";
    }

}

string BSString(
    const int FinalBSIdx,
    const int LTLLastIdx,
    const int* LTLStartPoints, const int* BSStartPoints,
    const int* d_leftIdx, const int* d_rightIdx,
    const int* d_BSLeftIdx, const int* d_BSRightIdx,
    const int* d_temp_BSLeftIdx, const int* d_temp_BSRightIdx)
{

    auto* LIdx = new int[1];
    auto* RIdx = new int[1];

    checkCuda(cudaMemcpy(LIdx, d_temp_BSLeftIdx + FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(RIdx, d_temp_BSRightIdx + FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));

    int* d_resIndices;
    checkCuda(cudaMalloc(&d_resIndices, 100 * sizeof(int)));
    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 100, -1);

    if (*LIdx >= LTLLastIdx) BSGenerateResIndices << <1, 1 >> > (*LIdx, LTLLastIdx, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= LTLLastIdx) BSGenerateResIndices << <1, 1 >> > (*RIdx, LTLLastIdx, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[100];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 100 * sizeof(int), cudaMemcpyDeviceToHost));

    map<int, pair<int, int>> indicesMap;
    indicesMap.insert(make_pair(INT_MAX - 1, make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 100) {
        int ltl = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert(make_pair(ltl, make_pair(l, r)));
        i += 3;
    }

    cudaFree(d_resIndices);

    if (i + 2 >= 100) return "Size of the output is too big";
    else return BSToString(INT_MAX - 1, indicesMap, LTLLastIdx, LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);

}

// Transfer the LTLCache in the BSCache
__global__ void transfer(
    const int LTLLastIdx,
    const uint64_t* d_LTLcache,
    uint64_t* d_BSCache)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < LTLLastIdx) {
        uint64_t CS{};
        for (int i = 0; i < d_numOfTraces; ++i)
            CS |= (d_LTLcache[tid * d_numOfTraces + i] & (uint64_t)1) << i;
        d_BSCache[tid] = CS;
    }

}

// Enumerate all BS formulas
string BS(
    const unsigned short BSMaxCost,
    int LTLLastIdx,
    int* LTLStartPoints,
    int LTLCost,
    uint64_t* d_BSCache,
    const int BSCacheCapacity,
    int* d_leftIdx, int* d_rightIdx) {

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int BSLastIdx = LTLLastIdx;
    cout << LTLLastIdx << endl;
    const int temp_BSCacheCapacity = BSCacheCapacity / 2;

    // 3 for ~, &, |
    int* BSStartPoints = new int[(BSMaxCost + 2) * 3]();
    for (int i = 2; i <= LTLCost; ++i) BSStartPoints[i * 3] = LTLStartPoints[i * 7];

    int* d_FinalBSIdx;
    int* FinalBSIdx = new int[1]; *FinalBSIdx = -1;
    checkCuda(cudaMalloc(&d_FinalBSIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalBSIdx, FinalBSIdx, sizeof(int), cudaMemcpyHostToDevice));

    uint64_t* d_temp_BSCache;
    int* d_BSLeftIdx, * d_BSRightIdx, * d_temp_BSLeftIdx, * d_temp_BSRightIdx;
    checkCuda(cudaMalloc(&d_BSLeftIdx, BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_BSRightIdx, BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_BSLeftIdx, temp_BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_BSRightIdx, temp_BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_BSCache, temp_BSCacheCapacity * sizeof(uint64_t)));

    using hash_set_t = warpcore::HashSet<
        uint64_t,         // Key type
        uint64_t(0) - 1,  // Empty key
        uint64_t(0) - 2,  // Tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <uint64_t>>>;

    hash_set_t hashSet(2 * BSCacheCapacity);
    BSHashSetsInit<hash_set_t> << <1, LTLLastIdx >> > (hashSet, d_BSCache);

    // ---------------------------
    // Enumeration of the next BSs
    // ---------------------------

    int BSCost;
    uint64_t allBSs{};
    for (BSCost = LTLCost; BSCost <= BSMaxCost; ++BSCost) {

        // Negation (~)
        if (BSCost > 1) {

            int idx1 = BSStartPoints[(BSCost - 1) * 3];
            int idx2 = BSStartPoints[BSCost * 3] - 1;
            int N = idx2 - idx1 + 1;

            if (N) {
                int x = idx1, y;
                do {
                    y = x + min(temp_BSCacheCapacity - 1, idx2 - x);
                    N = y - x + 1;
                    printf("Cost %-2d | (~) | AllBSs:  %-11lu | StoredBSs:  %-10d | ToBeChecked: %-10d \n", BSCost, allBSs, BSLastIdx, N);
                    int numBlocks = (N + 1023) / 1024;
                    BSProcessOperator<Op::Not, hash_set_t> << <numBlocks, 1024 >> > (
                        x, y, 0, 0, d_BSCache, d_temp_BSCache,
                        d_temp_BSLeftIdx, d_temp_BSRightIdx, hashSet, d_FinalBSIdx);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalBSIdx, d_FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allBSs += N;
                    if (*FinalBSIdx != -1) { BSStartPoints[BSCost * 3 + 1] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUnqBSs(
                        N, BSLastIdx, BSCacheCapacity, d_BSCache, d_temp_BSCache,
                        d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx2);
            }

        }

        BSStartPoints[BSCost * 3 + 1] = BSLastIdx;

        // Intersection (&)
        for (int i = 1; 2 * i <= BSCost - 1; ++i) {

            int idx1 = BSStartPoints[i * 3];
            int idx2 = BSStartPoints[(i + 1) * 3] - 1;
            int idx3 = BSStartPoints[(BSCost - i - 1) * 3];
            int idx4 = BSStartPoints[(BSCost - i) * 3] - 1;
            int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(temp_BSCacheCapacity / (idx2 - idx1 + 1) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
                    printf("Cost %-2d | (&) | AllBSs:  %-11lu | StoredBSs:  %-10d | ToBeChecked: %-10d \n", BSCost, allBSs, BSLastIdx, N);
                    int numBlocks = (N + 1023) / 1024;
                    BSProcessOperator<Op::And, hash_set_t> << <numBlocks, 1024 >> > (
                        idx1, idx2, x, y, d_BSCache, d_temp_BSCache,
                        d_temp_BSLeftIdx, d_temp_BSRightIdx, hashSet, d_FinalBSIdx);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalBSIdx, d_FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allBSs += N;
                    if (*FinalBSIdx != -1) { BSStartPoints[BSCost * 3 + 2] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUnqBSs(
                        N, BSLastIdx, BSCacheCapacity, d_BSCache, d_temp_BSCache,
                        d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }

        BSStartPoints[BSCost * 3 + 2] = BSLastIdx;

        // Union (|)
        for (int i = 1; 2 * i <= BSCost - 1; ++i) {

            int idx1 = BSStartPoints[i * 3];
            int idx2 = BSStartPoints[(i + 1) * 3] - 1;
            int idx3 = BSStartPoints[(BSCost - i - 1) * 3];
            int idx4 = BSStartPoints[(BSCost - i) * 3] - 1;
            int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(temp_BSCacheCapacity / (idx2 - idx1 + 1) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
                    printf("Cost %-2d | (|) | AllBSs:  %-11lu | StoredBSs:  %-10d | ToBeChecked: %-10d \n", BSCost, allBSs, BSLastIdx, N);
                    int numBlocks = (N + 1023) / 1024;
                    BSProcessOperator<Op::Or, hash_set_t> << <numBlocks, 1024 >> > (
                        idx1, idx2, x, y, d_BSCache, d_temp_BSCache,
                        d_temp_BSLeftIdx, d_temp_BSRightIdx, hashSet, d_FinalBSIdx);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalBSIdx, d_FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allBSs += N;
                    if (*FinalBSIdx != -1) { BSStartPoints[BSCost * 3 + 3] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUnqBSs(
                        N, BSLastIdx, BSCacheCapacity, d_BSCache, d_temp_BSCache,
                        d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }

        BSStartPoints[BSCost * 3 + 3] = BSLastIdx;

    }

exitEnumeration:

    string output;
    if (*FinalBSIdx != -1) {
        output = BSString(*FinalBSIdx, LTLLastIdx, LTLStartPoints, BSStartPoints,
            d_leftIdx, d_rightIdx, d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
    } else {
        output = "not_found";
    }

    cudaFree(d_BSCache); cudaFree(d_temp_BSCache);
    cudaFree(d_BSLeftIdx); cudaFree(d_temp_BSLeftIdx);
    cudaFree(d_BSRightIdx); cudaFree(d_temp_BSRightIdx);
    cudaFree(d_FinalBSIdx);

    return output;

}

string LTLI(
    const unsigned short LTLMaxCost,
    const unsigned short BSMaxCost,
    const vector<vector<string>> pos,
    const vector<vector<string>> neg) {

    // --------------------------------
    // Generating and checking alphabet
    // --------------------------------

    int maxLenOfTraces{};
    auto* traceLen = new char[numOfTraces];
    int TLIdx{};
    for (const auto& trace : pos) {
        lenSum += trace.size();
        traceLen[TLIdx++] = trace.size();
        if (trace.size() > maxLenOfTraces) maxLenOfTraces = trace.size();
    } for (const auto& trace : neg) {
        lenSum += trace.size();
        traceLen[TLIdx++] = trace.size();
        if (trace.size() > maxLenOfTraces) maxLenOfTraces = trace.size();
    }

    if (numOfTraces > maxNumOfTraces || maxLenOfTraces > sizeof(uint64_t) * 8 - 1) {
        printf("In this version, The input can have:\n");
        printf("1) At most %zu traces. It is currently %d.\n", maxNumOfTraces, numOfTraces);
        printf("2) Max(len(trace)) = %d. It is currently %d.\n", static_cast<int>(sizeof(uint64_t) * 8 - 1), maxLenOfTraces);
        return "see_the_error";
    }

    checkCuda(cudaMemcpyToSymbol(d_lenSum, &lenSum, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_traceLen, traceLen, numOfTraces * sizeof(char)));

    uint64_t* LTLcache = new uint64_t[alphabetSize * numOfTraces];
    int LTLLastIdx{}; uint64_t allLTLs{};
    printf("Cost %-2d | (A) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", 1, allLTLs, 0, alphabetSize);

    int index{};
    for (int i = 0; i < alphabetSize; ++i) {
        bool found = true;
        string ch(1, *next(alphabet.begin(), i));
        for (const auto& trace : pos) {
            uint64_t binTrace{};
            uint64_t idx = 1;
            for (const auto& token : trace) {
                for (const auto& c : token)
                    if (c == ch[0]) { binTrace |= idx; break; }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
            if (!(binTrace & 1)) found = false;
        }
        for (const auto& trace : neg) {
            uint64_t binTrace{};
            uint64_t idx = 1;
            for (const auto& token : trace) {
                for (const auto& c : token)
                    if (c == ch[0]) { binTrace |= idx; break; }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
            if (binTrace & 1) found = false;
        }
        allLTLs++; LTLLastIdx++;
        if (found) return ch;
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);
    const int LTLCacheCapacity = maxAllocationSize / (numOfTraces * sizeof(uint64_t));
    const int temp_LTLCacheCapacity = LTLCacheCapacity / 2;

    // 7 for ~, &, |, X, F, G, U
    int* LTLStartPoints = new int[(LTLMaxCost + 2) * 7]();
    LTLStartPoints[1 * 7 + 6] = LTLLastIdx;
    LTLStartPoints[2 * 7] = LTLLastIdx;

    int* d_FinalLTLIdx;
    auto* FinalLTLIdx = new int[1]; *FinalLTLIdx = -1;
    checkCuda(cudaMalloc(&d_FinalLTLIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalLTLIdx, FinalLTLIdx, sizeof(int), cudaMemcpyHostToDevice));

    uint64_t* d_LTLcache, * d_temp_LTLcache;
    int* d_leftIdx, * d_rightIdx, * d_temp_leftIdx, * d_temp_rightIdx;
    checkCuda(cudaMalloc(&d_leftIdx, LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, temp_LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, temp_LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_LTLcache, LTLCacheCapacity * numOfTraces * sizeof(uint64_t)));
    checkCuda(cudaMemcpy(d_LTLcache, LTLcache, alphabetSize * numOfTraces * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&d_temp_LTLcache, temp_LTLCacheCapacity * numOfTraces * sizeof(uint64_t)));

    using hash_set_t = warpcore::HashSet<
        uint64_t,         // key type
        uint64_t(0) - 1,  // empty key
        uint64_t(0) - 2,  // tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <uint64_t>>>;

    hash_set_t hashSet(2 * LTLCacheCapacity);
    hashSetsInit<hash_set_t> << <1, alphabetSize >> > (hashSet, d_LTLcache);

    // ----------------------------
    // Enumeration of the next LTLs
    // ----------------------------

    int LTLCost;
    int idx1; int idx2; int idx3; int idx4;
    int N;

    for (LTLCost = 2; LTLCost <= LTLMaxCost; ++LTLCost) {

        // Negation (~)
        idx1 = LTLStartPoints[(LTLCost - 1) * 7];
        idx2 = LTLStartPoints[LTLCost * 7] - 1;
        N = idx2 - idx1 + 1;

        if (N) {
            int x = idx1, y;
            do {
                y = x + min(temp_LTLCacheCapacity - 1, idx2 - x);
                N = (y - x + 1);
                printf("Cost %-2d | (~) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, N);
                int numBlocks = (N + 1023) / 1024;
                processOperator<Op::Not, hash_set_t> << <numBlocks, 1024 >> > (
                    x, y, 0, 0, d_LTLcache, d_temp_LTLcache,
                    d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                allLTLs += N;
                if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 1] = INT_MAX; goto exitEnumeration; }
                bool lastRound = storeUnqLTLs(
                    N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                    d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                if (lastRound) goto exitEnumeration;
                x = y + 1;
            } while (y < idx2);
        }

        LTLStartPoints[LTLCost * 7 + 1] = LTLLastIdx;

        // Intersection (&)
        for (int i = 1; 2 * i <= LTLCost - 1; ++i) {

            idx1 = LTLStartPoints[i * 7];
            idx2 = LTLStartPoints[(i + 1) * 7] - 1;
            idx3 = LTLStartPoints[(LTLCost - i - 1) * 7];
            idx4 = LTLStartPoints[(LTLCost - i) * 7] - 1;
            N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(temp_LTLCacheCapacity / (idx2 - idx1 + 1) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
                    printf("Cost %-2d | (&) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, N);
                    int numBlocks = (N + 1023) / 1024;
                    processOperator<Op::And, hash_set_t> << <numBlocks, 1024 >> > (
                        idx1, idx2, x, y, d_LTLcache, d_temp_LTLcache,
                        d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 2] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUnqLTLs(
                        N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }

        LTLStartPoints[LTLCost * 7 + 2] = LTLLastIdx;

        // Union (|)
        for (int i = 1; 2 * i <= LTLCost - 1; ++i) {

            idx1 = LTLStartPoints[i * 7];
            idx2 = LTLStartPoints[(i + 1) * 7] - 1;
            idx3 = LTLStartPoints[(LTLCost - i - 1) * 7];
            idx4 = LTLStartPoints[(LTLCost - i) * 7] - 1;
            N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(temp_LTLCacheCapacity / (idx2 - idx1 + 1) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
                    printf("Cost %-2d | (|) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, N);
                    int numBlocks = (N + 1023) / 1024;
                    processOperator<Op::Or, hash_set_t> << <numBlocks, 1024 >> > (
                        idx1, idx2, x, y, d_LTLcache, d_temp_LTLcache,
                        d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 3] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUnqLTLs(
                        N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }

        LTLStartPoints[LTLCost * 7 + 3] = LTLLastIdx;

        // Next (X)
        idx1 = LTLStartPoints[(LTLCost - 1) * 7];
        idx2 = LTLStartPoints[LTLCost * 7] - 1;
        N = idx2 - idx1 + 1;

        if (N) {
            int x = idx1, y;
            do {
                y = x + min(temp_LTLCacheCapacity - 1, idx2 - x);
                N = (y - x + 1);
                printf("Cost %-2d | (X) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, N);
                int numBlocks = (N + 1023) / 1024;
                processOperator<Op::Not, hash_set_t> << <numBlocks, 1024 >> > (
                    x, y, 0, 0, d_LTLcache, d_temp_LTLcache,
                    d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                allLTLs += N;
                if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 4] = INT_MAX; goto exitEnumeration; }
                bool lastRound = storeUnqLTLs(
                    N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                    d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                if (lastRound) goto exitEnumeration;
                x = y + 1;
            } while (y < idx2);
        }

        LTLStartPoints[LTLCost * 7 + 4] = LTLLastIdx;

        // Finally (F)
        idx1 = LTLStartPoints[(LTLCost - 1) * 7];
        idx2 = LTLStartPoints[LTLCost * 7] - 1;
        N = idx2 - idx1 + 1;

        if (N) {
            int x = idx1, y;
            do {
                y = x + min(temp_LTLCacheCapacity - 1, idx2 - x);
                N = (y - x + 1);
                printf("Cost %-2d | (F) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, N);
                int numBlocks = (N + 1023) / 1024;
                processOperator<Op::Finally, hash_set_t> << <numBlocks, 1024 >> > (
                    x, y, 0, 0, d_LTLcache, d_temp_LTLcache,
                    d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                allLTLs += N;
                if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 5] = INT_MAX; goto exitEnumeration; }
                bool lastRound = storeUnqLTLs(
                    N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                    d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                if (lastRound) goto exitEnumeration;
                x = y + 1;
            } while (y < idx2);
        }

        LTLStartPoints[LTLCost * 7 + 5] = LTLLastIdx;

        // Globally (G)
        idx1 = LTLStartPoints[(LTLCost - 1) * 7];
        idx2 = LTLStartPoints[LTLCost * 7] - 1;
        N = idx2 - idx1 + 1;

        if (N) {
            int x = idx1, y;
            do {
                y = x + min(temp_LTLCacheCapacity - 1, idx2 - x);
                N = (y - x + 1);
                printf("Cost %-2d | (G) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, N);
                int numBlocks = (N + 1023) / 1024;
                processOperator<Op::Globally, hash_set_t> << <numBlocks, 1024 >> > (
                    x, y, 0, 0, d_LTLcache, d_temp_LTLcache,
                    d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                allLTLs += N;
                if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 6] = INT_MAX; goto exitEnumeration; }
                bool lastRound = storeUnqLTLs(
                    N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                    d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                if (lastRound) goto exitEnumeration;
                x = y + 1;
            } while (y < idx2);
        }

        LTLStartPoints[LTLCost * 7 + 6] = LTLLastIdx;

        // Until (U)
        // Not commutative
        for (int i = 1; 2 * i <= LTLCost - 1; ++i) {

            idx1 = LTLStartPoints[i * 7];
            idx2 = LTLStartPoints[(i + 1) * 7] - 1;
            idx3 = LTLStartPoints[(LTLCost - i - 1) * 7];
            idx4 = LTLStartPoints[(LTLCost - i) * 7] - 1;
            N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(temp_LTLCacheCapacity / (2 * (idx2 - idx1 + 1)) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
                    printf("Cost %-2d | (U) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n", LTLCost, allLTLs, LTLLastIdx, 2 * N);
                    int numBlocks = (N + 1023) / 1024;
                    processOperator<Op::Until, hash_set_t> << <numBlocks, 1024 >> > (
                        idx1, idx2, x, y, d_LTLcache, d_temp_LTLcache,
                        d_temp_leftIdx, d_temp_rightIdx, hashSet, d_FinalLTLIdx);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += 2 * N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[(LTLCost + 1) * 7] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUnqLTLs(
                        2 * N, LTLLastIdx, LTLCacheCapacity, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }

        LTLStartPoints[(LTLCost + 1) * 7] = LTLLastIdx;

    }

exitEnumeration:

    string output;

    if (*FinalLTLIdx != -1) {
        output = LTLString(LTLLastIdx, LTLStartPoints, d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx, *FinalLTLIdx);
        cudaFree(d_LTLcache); cudaFree(d_temp_LTLcache);
        cudaFree(d_leftIdx); cudaFree(d_rightIdx);
        cudaFree(d_temp_leftIdx); cudaFree(d_temp_rightIdx);
        cudaFree(d_FinalLTLIdx);
    }

    else {
        uint64_t* d_BSCache;
        const int BSCacheCapacity = maxAllocationSize / sizeof(uint64_t) * 1.5;
        checkCuda(cudaMalloc(&d_BSCache, BSCacheCapacity * sizeof(uint64_t)));
        int numBlocks = (LTLLastIdx + 1023) / 1024;
        transfer << <numBlocks, 1024 >> > (LTLLastIdx, d_LTLcache, d_BSCache);

        cudaFree(d_LTLcache); cudaFree(d_temp_LTLcache);
        cudaFree(d_temp_leftIdx); cudaFree(d_temp_rightIdx);
        cudaFree(d_FinalLTLIdx);
        output = BS(BSMaxCost, LTLLastIdx, LTLStartPoints, LTLCost, d_BSCache, BSCacheCapacity, d_leftIdx, d_rightIdx);
        cudaFree(d_leftIdx); cudaFree(d_rightIdx);
    }

    return output;

}

// Reading the input file
bool readFile(
    const string& fileName,
    vector<vector<string>>& pos,
    vector<vector<string>>& neg)
{

    ifstream file(fileName);
    if (file.is_open()) {
        file.seekg(0, ios::end);
        string line;
        char ch;
        bool foundNewline = false;
        while (!foundNewline && file.tellg() > 0) {
            file.seekg(-2, ios::cur);
            file.get(ch);
            if (ch == '\n') foundNewline = true;
        }
        getline(file, line);
        string alpha;
        for (auto& c : line)
            if (c >= 'a' && c <= 'z') {
                alphabet.insert(c);
                alpha += c;
            }
        file.seekg(0, ios::beg);
        while (getline(file, line)) {
            vector<string> trace;
            if (line != "---") {
                string token;
                int j{};
                for (auto& c : line) {
                    if (c == ';') {
                        trace.push_back(token);
                        token = "";
                        j = 0;
                    } else if (c == ',') continue;
                    else {
                        if (c == '1') token += alpha[j];
                        j++;
                    }
                }
                trace.push_back(token);
                pos.push_back(trace);
            } else break;
        }
        while (getline(file, line)) {
            vector<string> trace;
            if (line != "---") {
                string token;
                int j{};
                for (auto& c : line) {
                    if (c == ';') {
                        trace.push_back(token);
                        token = "";
                        j = 0;
                    } else if (c == ',') continue;
                    else {
                        if (c == '1') token += alpha[j];
                        j++;
                    }
                }
                trace.push_back(token);
                neg.push_back(trace);
            } else break;
        }
        file.close();
        return true;
    } else printf("Failed to open the input file.\n");

    return false;

}

int main(int argc, char* argv[]) {

    // -----------------
    // Reading the input
    // -----------------

    if (argc != 4) {
        printf("Arguments should be in the form of\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s <input_file_address> <LTLMaxCost> <BSMaxCost>\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        printf("\nFor example\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s ./input.txt 10 20\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        return 0;
    }

    bool argError = false;
    if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50) {
        printf("Argument number 2, LTLMaxCost = \"%s\", should be between 1 and 50.\n", argv[2]);
        argError = true;
    }
    if (atoi(argv[3]) < 0 || atoi(argv[3]) > 100) {
        printf("Argument number 3, BSMaxCost = \"%s\", should be between 1 and 50.\n", argv[3]);
        argError = true;
    }
    if (argError) return 0;

    string fileName = argv[1];
    vector<vector<string>> pos, neg;
    if (!readFile(fileName, pos, neg)) return 0;
    unsigned short LTLMaxCost = atoi(argv[2]);
    unsigned short BSMaxCost = atoi(argv[3]);

    // -------------------
    // Assigning constants
    // -------------------

    alphabetSize = static_cast<int>(alphabet.size());
    numOfP = pos.size();
    numOfTraces = numOfP + neg.size();

    cudaMemcpyToSymbol(d_alphabetSize, &alphabetSize, sizeof(int));
    cudaMemcpyToSymbol(d_numOfTraces, &numOfTraces, sizeof(int));
    cudaMemcpyToSymbol(d_numOfP, &numOfP, sizeof(int));

    // --------------------------------------
    // Linear Temporal Logic Inference (LTLI)
    // --------------------------------------

    string output = LTLI(LTLMaxCost, BSMaxCost, pos, neg);
    if (output == "see_the_error") return 0;

    // -------------------
    // Printing the output
    // -------------------

    printf("\nPositive: \n");
    for (const auto& trace : pos) {
        printf("\t");
        for (const auto& t : trace) {
            string s;
            for (const auto& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("\n");
    }

    printf("\nNegative: \n");
    for (const auto& trace : neg) {
        printf("\t");
        for (const auto& t : trace) {
            string s;
            for (const auto& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("\n");
    }

    printf("\nNumber of Traces: %d", numOfTraces);
    printf("\n\nLTL: \"%s\"\n", output.c_str());

    return 0;

}