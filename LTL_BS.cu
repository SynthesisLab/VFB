#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <warpcore/hash_set.cuh>

using UINT_64 = std::uint64_t;

const std::size_t maxNumOfTraces = 63;

__constant__ char d_traceLen[maxNumOfTraces];

enum class Op { Not, And, Or, Next, Finally, Globally, Until };

inline
cudaError_t checkCuda(cudaError_t res) {
#ifndef MEASUREMENT_MODE
    if (res != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
        assert(res == cudaSuccess);
    }
#endif
    return res;
}

__device__ void makeRlxUnqChkCSs(
    UINT_64* CS,
    UINT_64& hCS,
    UINT_64& lCS,
    const int numOfTraces,
    const int RlxUnqChkTyp,
    const int lenSum)
{

    if (lenSum > 126) {

        // we need an relaxed uniqueness check

        switch (RlxUnqChkTyp) {

        case 1: {
            const int stride = lenSum / 126;
            int j = 0;
            for (int i = 0; i < numOfTraces; ++i) {
                for (int k = 0; k < d_traceLen[i]; k += stride, ++j) {
                    if (j < 63) {
                        if (CS[i] & ((UINT_64)1 << k)) lCS |= (UINT_64)1 << j;
                    } else if (j < 126) {
                        if (CS[i] & ((UINT_64)1 << k)) hCS |= (UINT_64)1 << (j - 63);
                    } else break;
                }
            }

            break;
        }

        case 2: {
            int j = 0;
            for (int i = 0; i < numOfTraces; ++i) {
                UINT_64 bitPtr = 1;
                int maxbitsForThisTrace = (126 * d_traceLen[i] + lenSum) / lenSum;
                for (int k = 0; k < maxbitsForThisTrace; ++k, ++j, bitPtr <<= 1) {
                    if (j < 63) {
                        if (CS[i] & bitPtr) lCS |= (UINT_64)1 << j;
                    } else if (j < 126) {
                        if (CS[i] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                    } else break;
                }
            }

            break;
        }

        case 3: {
            for (int i = 0; i < numOfTraces; ++i) {
                UINT_64 x = CS[i];
                x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
                x = x ^ (x >> 31);
                if (i < numOfTraces / 2) hCS ^= x; else lCS ^= x;
            }

            break;
        }

        }

    } else {

        // the result will be minimal

        int j = 0;
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 bitPtr = 1;
            for (int k = 0; k < d_traceLen[i]; ++k, ++j, bitPtr <<= 1) {
                if (j < 63) {
                    if (CS[i] & bitPtr) lCS |= (UINT_64)1 << j;
                } else if (j < 126) {
                    if (CS[i] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                } else break;
            }
        }

    }

}

template<class hash_set_t>
__global__ void BSHashSetsInitialisation(
    hash_set_t hashSet,
    UINT_64* d_BSCache) {

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    if (hashSet.insert(d_BSCache[tid], group) > 0) d_BSCache[tid] = (UINT_64)-1;

}

void BSInitialisation(
    int& lastIdx,
    UINT_64* d_BSCache)
{

    thrust::device_ptr<UINT_64> new_end_ptr;
    thrust::device_ptr<UINT_64> d_BSCache_ptr(d_BSCache);
    new_end_ptr = thrust::remove(d_BSCache_ptr, d_BSCache_ptr + lastIdx, (UINT_64)-1);
    lastIdx = static_cast<int>(new_end_ptr - d_BSCache_ptr);

}

// Initialising the hashSets with the alphabet before starting the enumeration
template<class hash_set_t>
__global__ void hashSetsInitialisation(
    const int numOfTraces,
    const int RlxUnqChkTyp,
    const int lenSum,
    hash_set_t cHashSet, hash_set_t iHashSet,
    UINT_64* d_LTLcache)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    UINT_64 CS[maxNumOfTraces];

    for (int i = 0; i < numOfTraces; ++i)
        CS[i] = d_LTLcache[tid * numOfTraces + i];

    UINT_64 hCS{}, lCS{};
    makeRlxUnqChkCSs(CS, hCS, lCS, numOfTraces, RlxUnqChkTyp, lenSum);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    int H = cHashSet.insert(hCS, group); int L = cHashSet.insert(lCS, group);
    H = (H > 0) ? H : -H; L = (L > 0) ? L : -L;
    UINT_64 HL = H; HL <<= 32; HL |= L;
    iHashSet.insert(HL, group);

}

template<Op op>
__device__ void BSApplyOperator(
    UINT_64& CS,
    UINT_64* d_BSCache,
    int ldx, int rdx,
    int numOfTraces)
{

    if constexpr (op == Op::Not) {
        UINT_64 negationFixer = ((UINT_64)1 << numOfTraces) - 1;
        CS = ~d_BSCache[ldx] & negationFixer;
    } else if constexpr (op == Op::And) {
        CS = d_BSCache[ldx] & d_BSCache[rdx];
    } else if constexpr (op == Op::Or) {
        CS = d_BSCache[ldx] | d_BSCache[rdx];
    } else {
        [] <bool flag = false>() { static_assert(flag, "Unhandled operator"); }();
    }

}

template<Op op>
__device__ void applyOperator(
    UINT_64* CS,
    UINT_64* d_LTLcache,
    int ldx, int rdx,
    int numOfTraces)
{

    if constexpr (op == Op::Not) {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 negationFixer = ((UINT_64)1 << d_traceLen[i]) - 1;
            CS[i] = ~d_LTLcache[ldx * numOfTraces + i] & negationFixer;
        }
    } else if constexpr (op == Op::And) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i] & d_LTLcache[rdx * numOfTraces + i];
        }
    } else if constexpr (op == Op::Or) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i] | d_LTLcache[rdx * numOfTraces + i];
        }
    } else if constexpr (op == Op::Next) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i] >> 1;
        }
    } else if constexpr (op == Op::Finally) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i];
            CS[i] |= CS[i] >> 1; CS[i] |= CS[i] >> 2; CS[i] |= CS[i] >> 4;
            CS[i] |= CS[i] >> 8; CS[i] |= CS[i] >> 16; CS[i] |= CS[i] >> 32;
        }
    } else if constexpr (op == Op::Globally) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i];
            UINT_64 cs = ~CS[i] & (((UINT_64)1 << d_traceLen[i]) - 1);
            cs |= cs >> 1; cs |= cs >> 2; cs |= cs >> 4;
            cs |= cs >> 8; cs |= cs >> 16; cs |= cs >> 32;
            CS[i] &= ~cs;
        }
    } else if constexpr (op == Op::Until) {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 l = d_LTLcache[ldx * numOfTraces + i];
            UINT_64 r = d_LTLcache[rdx * numOfTraces + i];
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

template <typename hash_set_t>
__device__ bool BSProcessUniqueCS(
    UINT_64& CS,
    hash_set_t& hashSet)
{

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    return (hashSet.insert(CS, group) > 0) ? false : true;

}

template <typename hash_set_t>
__device__ bool processUniqueCS(
    UINT_64* CS,
    const int numOfTraces,
    const int RlxUnqChkTyp,
    const int lenSum,
    hash_set_t& cHashSet, hash_set_t& iHashSet)
{

    UINT_64 hCS{}, lCS{};
    makeRlxUnqChkCSs(CS, hCS, lCS, numOfTraces, RlxUnqChkTyp, lenSum);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    int H = cHashSet.insert(hCS, group); int L = cHashSet.insert(lCS, group);
    H = (H > 0) ? H : -H; L = (L > 0) ? L : -L;
    UINT_64 HL = H; HL <<= 32; HL |= L;
    return (iHashSet.insert(HL, group) > 0) ? false : true;

}

__device__ void BSInsertInCache(
    bool CS_is_unique,
    UINT_64& CS,
    int tid,
    int numOfP,
    int ldx, int rdx,
    UINT_64* d_temp_BSCache,
    int* d_temp_BSLeftIdx, int* d_temp_BSRightIdx,
    int* d_FinalBSIdx)
{

    if (CS_is_unique) {

        d_temp_BSCache[tid] = CS;
        d_temp_BSLeftIdx[tid] = ldx; d_temp_BSRightIdx[tid] = rdx;

        UINT_64 mask = ((UINT_64)1 << numOfP) - 1;
        if (CS == mask) atomicCAS(d_FinalBSIdx, -1, tid);

    } else {

        d_temp_BSCache[tid] = (UINT_64)-1;
        d_temp_BSLeftIdx[tid] = -1; d_temp_BSRightIdx[tid] = -1;

    }

}

__device__ void insertInCache(
    bool CS_is_unique,
    UINT_64* CS,
    int tid,
    int numOfTraces, int numOfP,
    int ldx, int rdx,
    UINT_64* d_temp_LTLcache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalLTLIdx)
{

    if (CS_is_unique) {

        for (int i = 0; i < numOfTraces; ++i)
            d_temp_LTLcache[tid * numOfTraces + i] = CS[i];
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;

        bool found = true;
        for (int i = 0; found && i < numOfP; ++i) if (!(CS[i] & 1)) found = false;
        for (int i = numOfP; found && i < numOfTraces; ++i) if (CS[i] & 1) found = false;
        if (found) atomicCAS(d_FinalLTLIdx, -1, tid);

    } else {

        for (int i = 0; i < numOfTraces; ++i)
            d_temp_LTLcache[tid * numOfTraces + i] = (UINT_64)-1;
        d_temp_leftIdx[tid] = -1; d_temp_rightIdx[tid] = -1;

    }

}

__device__ void processOnTheFly(
    UINT_64* CS,
    int tid,
    int numOfTraces, int numOfP,
    int ldx, int rdx,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalLTLIdx)
{

    bool found = true;
    for (int i = 0; found && i < numOfP; ++i) if (!(CS[i] & 1)) found = false;
    for (int i = numOfP; found && i < numOfTraces; ++i) if (CS[i] & 1) found = false;

    if (found) {
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;
        atomicCAS(d_FinalLTLIdx, -1, tid);
    }

}

template<Op op, class hash_set_t>
__global__ void BSProcessOperator(
    const int idx1, const int idx2,
    const int idx3, const int idx4,
    const int numOfTraces, const int numOfP,
    UINT_64* d_BSCache, UINT_64* d_temp_BSCache,
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
        UINT_64 CS;
        BSApplyOperator<op>(CS, d_BSCache, ldx, rdx, numOfTraces);

        bool CS_is_unique = BSProcessUniqueCS(CS, hashSet);
        BSInsertInCache(
            CS_is_unique, CS, tid, numOfP, ldx, rdx,
            d_temp_BSCache, d_temp_BSLeftIdx, d_temp_BSRightIdx, d_FinalBSIdx
        );

    }

}

template<Op op, class hash_set_t>
__global__ void processOperator(
    const int idx1, const int idx2,
    const int idx3, const int idx4,
    const int numOfP, const int numOfN,
    const int RlxUnqChkTyp,
    const int lenSum,
    const bool onTheFly,
    UINT_64* d_LTLcache, UINT_64* d_temp_LTLcache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    hash_set_t cHashSet, hash_set_t iHashSet,
    int* d_FinalLTLIdx)
{

    const int realTid = (blockDim.x * blockIdx.x + threadIdx.x);
    const int tid = (op == Op::Until) ? (realTid * 2) : realTid;
    const int numOfTraces = numOfP + numOfN;
    constexpr bool isUnary = (op == Op::Not || op == Op::Next || op == Op::Finally || op == Op::Globally);
    int maxTid = isUnary ? (idx2 - idx1 + 1) : ((idx4 - idx3 + 1) * (idx2 - idx1 + 1));

    if (tid < maxTid) {

        int ldx = isUnary ? (idx1 + tid) : (idx1 + tid / (idx4 - idx3 + 1));
        int rdx = isUnary ? 0 : (idx3 + tid % (idx4 - idx3 + 1));
        UINT_64 CS[maxNumOfTraces];
        applyOperator<op>(CS, d_LTLcache, ldx, rdx, numOfTraces);

        if (onTheFly) {
            processOnTheFly(
                CS, tid, numOfTraces, numOfP, ldx, rdx,
                d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
            );
        } else {
            bool CS_is_unique =
                processUniqueCS(CS, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
            insertInCache(
                CS_is_unique, CS, tid, numOfTraces, numOfP, ldx, rdx,
                d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
            );
        }

        if (op == Op::Until) {

            applyOperator<Op::Until>(CS, d_LTLcache, rdx, ldx, numOfTraces);

            if (onTheFly) {
                processOnTheFly(
                    CS, tid + 1, numOfTraces, numOfP, rdx, ldx,
                    d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            } else {
                bool CS_is_unique =
                    processUniqueCS(CS, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
                insertInCache(
                    CS_is_unique, CS, tid + 1, numOfTraces, numOfP, rdx, ldx,
                    d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            }

        }

    }

}

bool storeUniqueBSs(
    int N,
    int& lastIdx,
    const int BSCacheCapacity,
    UINT_64* d_BSCache,
    UINT_64* d_temp_BSCache,
    int* d_BSLeftIdx,
    int* d_BSRightIdx,
    int* d_temp_BSLeftIdx,
    int* d_temp_BSRightIdx)
{

    thrust::device_ptr<UINT_64> new_end_ptr;
    thrust::device_ptr<UINT_64> d_BSCache_ptr(d_BSCache + lastIdx);
    thrust::device_ptr<UINT_64> d_temp_BSCache_ptr(d_temp_BSCache);
    thrust::device_ptr<int> d_BSLeftIdx_ptr(d_BSLeftIdx + lastIdx);
    thrust::device_ptr<int> d_BSRightIdx_ptr(d_BSRightIdx + lastIdx);
    thrust::device_ptr<int> d_temp_BSLeftIdx_ptr(d_temp_BSLeftIdx);
    thrust::device_ptr<int> d_temp_BSRightIdx_ptr(d_temp_BSRightIdx);

    new_end_ptr = thrust::remove(d_temp_BSCache_ptr, d_temp_BSCache_ptr + N, (UINT_64)-1);
    thrust::remove(d_temp_BSLeftIdx_ptr, d_temp_BSLeftIdx_ptr + N, -1);
    thrust::remove(d_temp_BSRightIdx_ptr, d_temp_BSRightIdx_ptr + N, -1);

    int numberOfNewUniqueBSs = static_cast<int>(new_end_ptr - d_temp_BSCache_ptr);
    if (lastIdx + numberOfNewUniqueBSs > BSCacheCapacity) {
        N = BSCacheCapacity - lastIdx;
        lastIdx += N;
        return true;
    } else {
        N = numberOfNewUniqueBSs;
        thrust::copy_n(d_temp_BSCache_ptr, N, d_BSCache_ptr);
        thrust::copy_n(d_temp_BSLeftIdx_ptr, N, d_BSLeftIdx_ptr);
        thrust::copy_n(d_temp_BSRightIdx_ptr, N, d_BSRightIdx_ptr);
        lastIdx += N;
        return false;
    }

}

// Transfering the unique CSs from temp to main LTLcache
void storeUniqueLTLs(
    int N,
    int& lastIdx,
    const int numOfTraces,
    const int LTLCacheCapacity,
    bool& onTheFly,
    UINT_64* d_LTLcache,
    UINT_64* d_temp_LTLcache,
    int* d_leftIdx,
    int* d_rightIdx,
    int* d_temp_leftIdx,
    int* d_temp_rightIdx)
{

    thrust::device_ptr<UINT_64> new_end_ptr;
    thrust::device_ptr<UINT_64> d_LTLcache_ptr(d_LTLcache + numOfTraces * lastIdx);
    thrust::device_ptr<UINT_64> d_temp_LTLcache_ptr(d_temp_LTLcache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + lastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + lastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);

    new_end_ptr = // end of d_temp_LTLcache
        thrust::remove(d_temp_LTLcache_ptr, d_temp_LTLcache_ptr + numOfTraces * N, (UINT_64)-1);
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

    // It stores all (or a part of) unique CSs until language cache gets full
    // If language cache gets full, it makes onTheFly mode on
    int numberOfNewUniqueLTLs = static_cast<int>(new_end_ptr - d_temp_LTLcache_ptr) / numOfTraces;
    if (lastIdx + numberOfNewUniqueLTLs > LTLCacheCapacity) {
        N = LTLCacheCapacity - lastIdx;
        onTheFly = true;
    } else N = numberOfNewUniqueLTLs;

    thrust::copy_n(d_temp_LTLcache_ptr, numOfTraces * N, d_LTLcache_ptr);
    thrust::copy_n(d_temp_leftIdx_ptr, N, d_leftIdx_ptr);
    thrust::copy_n(d_temp_rightIdx_ptr, N, d_rightIdx_ptr);

    lastIdx += N;

}

__global__ void BSGenerateResIndices(
    const int index,
    const int nLTL,
    const int* d_BSLeftIdx,
    const int* d_BSRightIdx,
    int* d_FinalBSIdx)
{

    int resIdx = 0;
    while (d_FinalBSIdx[resIdx] != -1) resIdx++;
    int queue[600];
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
        if (l >= nLTL) queue[tail++] = l;
        if (r >= nLTL) queue[tail++] = r;
        head++;
    }

}

// Finding the left and right indices that makes the final LTL to bring to the host later
__global__ void generateResIndices(
    const int index,
    const int alphabetSize,
    const int* d_leftIdx,
    const int* d_rightIdx,
    int* d_FinalLTLIdx)
{

    int resIdx = 0;
    while (d_FinalLTLIdx[resIdx] != -1) resIdx++;
    int queue[600];
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
        if (l >= alphabetSize) queue[tail++] = l;
        if (r >= alphabetSize) queue[tail++] = r;
        head++;
    }

}

// Generating the final RE string recursively
// When all the left and right indices are ready in the host
std::string LTLToString(
    int index,
    std::map<int, std::pair<int, int>>& indicesMap,
    const std::set<char>& alphabet,
    const int* LTLStartPoints)
{

    if (index < alphabet.size()) {
        std::string s(1, *next(alphabet.begin(), index));
        return s;
    }
    int i = 0;
    while (index >= LTLStartPoints[i]) { i++; }
    i--;

    if (i % 7 == 0) {
        std::string res = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
        return "~(" + res + ")";
    }

    if (i % 7 == 1) {
        std::string left = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
        std::string right = LTLToString(indicesMap[index].second, indicesMap, alphabet, LTLStartPoints);
        return "(" + left + ")" + "&" + "(" + right + ")";
    }

    if (i % 7 == 2) {
        std::string left = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
        std::string right = LTLToString(indicesMap[index].second, indicesMap, alphabet, LTLStartPoints);
        return "(" + left + ")" + "|" + "(" + right + ")";
    }

    if (i % 7 == 3) {
        std::string res = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
        return "X(" + res + ")";
    }

    if (i % 7 == 4) {
        std::string res = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
        return "F(" + res + ")";
    }

    if (i % 7 == 5) {
        std::string res = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
        return "G(" + res + ")";
    }

    std::string left = LTLToString(indicesMap[index].first, indicesMap, alphabet, LTLStartPoints);
    std::string right = LTLToString(indicesMap[index].second, indicesMap, alphabet, LTLStartPoints);
    return "(" + left + ")" + "U" + "(" + right + ")";

}

// Bringing the left and right indices of the LTL from device to host
// If LTL is found, this index is from the temp memory               (temp = true)
// For printing other LTLs if needed, indices are in the main memory (temp = false)
std::string LTLString(
    bool temp,
    const int FinalLTLIdx,
    const int lastIdx,
    const std::set<char>& alphabet,
    const int* LTLStartPoints,
    const int* d_leftIdx,
    const int* d_rightIdx,
    const int* d_temp_leftIdx,
    const int* d_temp_rightIdx)
{

    auto* LIdx = new int[1];
    auto* RIdx = new int[1];

    if (temp) {
        checkCuda(cudaMemcpy(LIdx, d_temp_leftIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(RIdx, d_temp_rightIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        checkCuda(cudaMemcpy(LIdx, d_leftIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(RIdx, d_rightIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
    }

    auto alphabetSize = static_cast<int> (alphabet.size());

    int* d_resIndices;
    checkCuda(cudaMalloc(&d_resIndices, 600 * sizeof(int)));

    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 600, -1);

    if (*LIdx >= alphabetSize) generateResIndices << <1, 1 >> > (*LIdx, alphabetSize, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= alphabetSize) generateResIndices << <1, 1 >> > (*RIdx, alphabetSize, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[600];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 600 * sizeof(int), cudaMemcpyDeviceToHost));

    std::map<int, std::pair<int, int>> indicesMap;

    if (temp) indicesMap.insert(std::make_pair(INT_MAX - 1, std::make_pair(*LIdx, *RIdx)));
    else      indicesMap.insert(std::make_pair(FinalLTLIdx, std::make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 600) {
        int ltl = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert(std::make_pair(ltl, std::make_pair(l, r)));
        i += 3;
    }

    if (i + 2 >= 600) return "Size of the output is too big";

    cudaFree(d_resIndices);

    if (temp) return LTLToString(INT_MAX - 1, indicesMap, alphabet, LTLStartPoints);
    else      return LTLToString(FinalLTLIdx, indicesMap, alphabet, LTLStartPoints);

}

std::string BSToString(
    int index,
    std::map<int, std::pair<int, int>>& indicesMap,
    const UINT_64 nLTL,
    const std::set<char>& alphabet,
    const int* LTLStartPoints, const int* BSStartPoints,
    const int* d_leftIdx, const int* d_rightIdx)
{

    if (index < nLTL) {
        return LTLString(false, index, nLTL, alphabet, LTLStartPoints, d_leftIdx, d_rightIdx, d_leftIdx, d_rightIdx);
    }

    int i = 0;
    while (index >= BSStartPoints[i]) { i++; }
    i--;

    if (i % 3 == 0) {
        std::string res = BSToString(
            indicesMap[index].first, indicesMap, nLTL, alphabet,
            LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        return "~(" + res + ")";
    }

    if (i % 3 == 1) {
        std::string left = BSToString(
            indicesMap[index].first, indicesMap, nLTL, alphabet,
            LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        std::string right = BSToString(
            indicesMap[index].second, indicesMap, nLTL, alphabet,
            LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        return "(" + left + ")" + "&" + "(" + right + ")";
    }

    else {
        std::string left = BSToString(
            indicesMap[index].first, indicesMap, nLTL, alphabet,
            LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        std::string right = BSToString(
            indicesMap[index].second, indicesMap, nLTL, alphabet,
            LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);
        return "(" + left + ")" + "|" + "(" + right + ")";
    }

}

std::string BSString(
    const int FinalBSIdx,
    const int nLTL,
    const std::set<char>& alphabet,
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
    checkCuda(cudaMalloc(&d_resIndices, 600 * sizeof(int)));

    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 600, -1);

    if (*LIdx >= nLTL) BSGenerateResIndices << <1, 1 >> > (*LIdx, nLTL, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= nLTL) BSGenerateResIndices << <1, 1 >> > (*RIdx, nLTL, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[600];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 600 * sizeof(int), cudaMemcpyDeviceToHost));

    std::map<int, std::pair<int, int>> indicesMap;
    indicesMap.insert(std::make_pair(INT_MAX - 1, std::make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 600) {
        int ltl = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert(std::make_pair(ltl, std::make_pair(l, r)));
        i += 3;
    }

    cudaFree(d_resIndices);

    if (i + 2 >= 600) return "Size of the output is too big";
    else return BSToString(
        INT_MAX - 1, indicesMap, nLTL, alphabet,
        LTLStartPoints, BSStartPoints, d_leftIdx, d_rightIdx);

}

int costOf(const int index, const int* LTLStartPoints) {
    int i = 0;
    while (index >= LTLStartPoints[i]) { i++; }
    return((i - 1) / 4);
}

__global__ void transfer(
    const int numOfTraces,
    const int lastIdx,
    const UINT_64* d_LTLcache,
    UINT_64* d_BSCache) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < lastIdx) {
        UINT_64 CS{};
        for (int i = 0; i < numOfTraces; ++i)
            CS |= (d_LTLcache[tid * numOfTraces + i] & (UINT_64)1) << i;
        d_BSCache[tid] = CS;
    }

}

__global__ void printFullLTLCache(
    const int numOfTraces,
    const int lastIdx,
    const UINT_64* d_LTLCache) {

    printf("LTL Cache:\n");
    for (int i = 0; i < lastIdx; ++i) {
        for (int j = 0; j < numOfTraces; ++j) {
            printf("%lu ", d_LTLCache[i * numOfTraces + j]);
        }
        printf("\n");
    }

}

__global__ void printLTLCache(
    const int numOfTraces,
    const int lastIdx,
    const UINT_64* d_LTLCache) {

    printf("LTL Cache:\n");
    for (int i = 0; i < lastIdx; ++i) {
        for (int j = 0; j < numOfTraces; ++j) {
            printf("%lu", d_LTLCache[i * numOfTraces + j] & 1);
        }
        printf("\n");
    }

}

__global__ void printBSCache(
    const int numOfTraces,
    const int lastIdx,
    const UINT_64* d_BSCache) {

    printf("BS Cache:\n");
    for (int i = 0; i < lastIdx; ++i) {
        printf("Index %3d: ", i);
        for (int j = 0; j < numOfTraces; ++j) {
            printf("%lu", (d_BSCache[i] >> j) & 1);
        }
        printf("\n");
    }

}

std::string BS(
    const unsigned short* costFun,
    const unsigned short BSMaxCost,
    int nLTL,
    const std::set<char>& alphabet,
    int* LTLStartPoints,
    int LTLSize,
    UINT_64* d_BSCache,
    int numOfTraces, int numOfP,
    int* d_leftIdx, int* d_rightIdx) {

    int lastIdx = nLTL;

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    // Cost function
    int c1 = costFun[0]; // Cost of p
    int c2 = costFun[1]; // Cost of ~
    int c3 = costFun[2]; // Cost of &
    int c4 = costFun[3]; // Cost of |

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int BSCacheCapacity = maxAllocationSize / (10 * sizeof(UINT_64));
    const int temp_BSCacheCapacity = BSCacheCapacity / 2;

    // 3 for ~, &, |
    int* BSStartPoints = new int[(BSMaxCost + 2) * 3]();
    for (int i = 2; i <= LTLSize; ++i) {
        BSStartPoints[i * 3] = LTLStartPoints[i * 7];
    }

    int* d_FinalBSIdx;
    int* FinalBSIdx = new int[1]; *FinalBSIdx = -1;
    checkCuda(cudaMalloc(&d_FinalBSIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalBSIdx, FinalBSIdx, sizeof(int), cudaMemcpyHostToDevice));

    UINT_64* d_temp_BSCache;
    int* d_BSLeftIdx, * d_BSRightIdx, * d_temp_BSLeftIdx, * d_temp_BSRightIdx;
    checkCuda(cudaMalloc(&d_BSLeftIdx, BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_BSRightIdx, BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_BSLeftIdx, temp_BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_BSRightIdx, temp_BSCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_BSCache, temp_BSCacheCapacity * sizeof(UINT_64)));

    using hash_set_t = warpcore::HashSet<
        UINT_64,         // Key type
        UINT_64(0) - 1,  // Empty key
        UINT_64(0) - 2,  // Tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <UINT_64>>>;

    hash_set_t hashSet(2 * BSCacheCapacity);
    BSHashSetsInitialisation<hash_set_t> << <1, nLTL >> > (hashSet, d_BSCache);
    BSInitialisation(lastIdx, d_BSCache);

    // ---------------------------
    // Enumeration of the next BSs
    // ---------------------------

    int BSCost;
    UINT_64 allBSs{};
    for (BSCost = LTLSize; BSCost <= BSMaxCost; ++BSCost) {

        // Negation (~)
        if (BSCost - c2 >= c1) {

            int idx1 = BSStartPoints[(BSCost - c2) * 3];
            int idx2 = BSStartPoints[(BSCost - c2 + 1) * 3] - 1;
            int N = idx2 - idx1 + 1;

            if (N) {
                int x = idx1, y;
                do {
                    y = x + std::min(temp_BSCacheCapacity - 1, idx2 - x);
                    N = y - x + 1;
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (~) | AllBSs:  %-11lu | StoredBSs:  %-10d | ToBeChecked: %-10d \n",
                        BSCost, allBSs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    BSProcessOperator<Op::Not, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfTraces, numOfP, d_BSCache, d_temp_BSCache,
                        d_temp_BSLeftIdx, d_temp_BSRightIdx, hashSet, d_FinalBSIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalBSIdx, d_FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allBSs += N;
                    if (*FinalBSIdx != -1) { BSStartPoints[BSCost * 3 + 1] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUniqueBSs(N, lastIdx, BSCacheCapacity, d_BSCache,
                        d_temp_BSCache, d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx2);
            }

        }
        BSStartPoints[BSCost * 3 + 1] = lastIdx;

        // Intersection (&)
        for (int i = c1; 2 * i <= BSCost - c3; ++i) {

            int idx1 = BSStartPoints[i * 3];
            int idx2 = BSStartPoints[(i + 1) * 3] - 1;
            int idx3 = BSStartPoints[(BSCost - i - c3) * 3];
            int idx4 = BSStartPoints[(BSCost - i - c3 + 1) * 3] - 1;
            int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + std::min(temp_BSCacheCapacity / (idx2 - idx1 + 1) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (&) | AllBSs:  %-11lu | StoredBSs:  %-10d | ToBeChecked: %-10d \n",
                        BSCost, allBSs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    BSProcessOperator<Op::And, hash_set_t> << <Blc, 1024 >> > (
                        idx1, idx2, x, y, numOfTraces, numOfP, d_BSCache, d_temp_BSCache,
                        d_temp_BSLeftIdx, d_temp_BSRightIdx, hashSet, d_FinalBSIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalBSIdx, d_FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allBSs += N;
                    if (*FinalBSIdx != -1) { BSStartPoints[BSCost * 3 + 2] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUniqueBSs(N, lastIdx, BSCacheCapacity, d_BSCache,
                        d_temp_BSCache, d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }
        BSStartPoints[BSCost * 3 + 2] = lastIdx;

        // Union (|)
        for (int i = c1; 2 * i <= BSCost - c4; ++i) {

            int idx1 = BSStartPoints[i * 3];
            int idx2 = BSStartPoints[(i + 1) * 3] - 1;
            int idx3 = BSStartPoints[(BSCost - i - c3) * 3];
            int idx4 = BSStartPoints[(BSCost - i - c3 + 1) * 3] - 1;
            int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

            if (N) {
                int x = idx3, y;
                do {
                    y = x + std::min(temp_BSCacheCapacity / (idx2 - idx1 + 1) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (|) | AllBSs:  %-11lu | StoredBSs:  %-10d | ToBeChecked: %-10d \n",
                        BSCost, allBSs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    BSProcessOperator<Op::Or, hash_set_t> << <Blc, 1024 >> > (
                        idx1, idx2, x, y, numOfTraces, numOfP, d_BSCache, d_temp_BSCache,
                        d_temp_BSLeftIdx, d_temp_BSRightIdx, hashSet, d_FinalBSIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalBSIdx, d_FinalBSIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allBSs += N;
                    if (*FinalBSIdx != -1) { BSStartPoints[BSCost * 3 + 3] = INT_MAX; goto exitEnumeration; }
                    bool lastRound = storeUniqueBSs(N, lastIdx, BSCacheCapacity, d_BSCache,
                        d_temp_BSCache, d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
                    if (lastRound) goto exitEnumeration;
                    x = y + 1;
                } while (y < idx4);
            }

        }
        BSStartPoints[BSCost * 3 + 3] = lastIdx;

    }

exitEnumeration:

    std::string output;
    if (*FinalBSIdx != -1) {
        output = BSString(*FinalBSIdx, nLTL, alphabet, LTLStartPoints, BSStartPoints,
            d_leftIdx, d_rightIdx, d_BSLeftIdx, d_BSRightIdx, d_temp_BSLeftIdx, d_temp_BSRightIdx);
    } else {
        output = "not_found";
    }

    cudaFree(d_BSCache);
    cudaFree(d_FinalBSIdx);
    cudaFree(d_temp_BSCache);
    cudaFree(d_BSLeftIdx);
    cudaFree(d_BSRightIdx);
    cudaFree(d_temp_BSLeftIdx);
    cudaFree(d_temp_BSRightIdx);

    return output;

}

std::string LTLI(
    const unsigned short* costFun,
    const unsigned short maxCost,
    const unsigned short BSMaxCost,
    const unsigned int RlxUnqChkTyp,
    const unsigned int NegType,
    const std::set<char> alphabet,
    int& LTLCost,
    std::uint64_t& allLTLs,
    const std::vector<std::vector<std::string>> pos,
    const std::vector<std::vector<std::string>> neg) {

    // --------------------------------
    // Generating and checking alphabet
    // --------------------------------

    const int numOfP = pos.size();
    const int numOfN = neg.size();
    const int numOfTraces = numOfP + numOfN;

    int maxLenOfTraces{};
    auto* traceLen = new char[numOfTraces];

    int TLIdx{};
    int lenSum{};
    for (const auto& trace : pos) {
        lenSum += trace.size();
        traceLen[TLIdx++] = trace.size();
        if (trace.size() > maxLenOfTraces) maxLenOfTraces = trace.size();
    }
    for (const auto& trace : neg) {
        lenSum += trace.size();
        traceLen[TLIdx++] = trace.size();
        if (trace.size() > maxLenOfTraces) maxLenOfTraces = trace.size();
    }

    if (numOfTraces > maxNumOfTraces || maxLenOfTraces > sizeof(UINT_64) * 8 - 1) {
        printf("In this version, The input can have:\n");
        printf("1) At most %zu traces. It is currently %d.\n", maxNumOfTraces, numOfTraces);
        printf("2) Max(len(trace)) = %d. It is currently %d.\n", static_cast<int>(sizeof(UINT_64) * 8 - 1), maxLenOfTraces);
        return "see_the_error";
    }

    // Copying the length of traces into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_traceLen, traceLen, numOfTraces * sizeof(char)));

    const int alphabetSize = static_cast<int>(alphabet.size());

    auto* LTLcache = new UINT_64[alphabetSize * numOfTraces];

    // Index of the last free position in the LTLcache
    int lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Cost %-2d | (A) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
        costFun[0], allLTLs, 0, alphabetSize);
#endif

    int index{};
    for (int i = 0; i < alphabetSize; ++i) {
        bool found = true;
        std::string ch(1, *next(alphabet.begin(), i));
        for (const auto& trace : pos) {
            UINT_64 binTrace{};
            UINT_64 idx = 1;
            for (const auto& token : trace) {
                for (const auto& c : token) {
                    if (c == ch[0]) {
                        binTrace |= idx;
                        break;
                    }
                }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
            if (!(binTrace & 1)) found = false;
        }
        for (const auto& trace : neg) {
            UINT_64 binTrace{};
            UINT_64 idx = 1;
            for (const auto& token : trace) {
                for (const auto& c : token) {
                    if (c == ch[0]) {
                        binTrace |= idx;
                        break;
                    }
                }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
            if (binTrace & 1) found = false;
        }
        allLTLs++; lastIdx++;
        if (found) return ch;
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    // cost function
    int c1 = costFun[0]; // cost of p
    int c2 = costFun[1]; // cost of ~
    int c3 = costFun[2]; // cost of &
    int c4 = costFun[3]; // cost of |
    int c5 = costFun[4]; // cost of X
    int c6 = costFun[5]; // cost of F
    int c7 = costFun[6]; // cost of G
    int c8 = costFun[7]; // cost of U

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int LTLCacheCapacity = maxAllocationSize / (numOfTraces * sizeof(UINT_64)) * 1.5;
    const int temp_LTLCacheCapacity = LTLCacheCapacity / 2;

    // const int LTLCacheCapacity = 2000000;
    // const int temp_LTLCacheCapacity = 100000000;

    // 7 for ~, &, |, X, F, G, U
    int* LTLStartPoints = new int[(maxCost + 2) * 7]();
    LTLStartPoints[c1 * 7 + 6] = lastIdx;
    LTLStartPoints[(c1 + 1) * 7] = lastIdx;

    int* d_FinalLTLIdx;
    auto* FinalLTLIdx = new int[1]; *FinalLTLIdx = -1;
    checkCuda(cudaMalloc(&d_FinalLTLIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalLTLIdx, FinalLTLIdx, sizeof(int), cudaMemcpyHostToDevice));

    UINT_64* d_LTLcache, * d_temp_LTLcache;
    int* d_leftIdx, * d_rightIdx, * d_temp_leftIdx, * d_temp_rightIdx;
    checkCuda(cudaMalloc(&d_leftIdx, LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, temp_LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, temp_LTLCacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_LTLcache, LTLCacheCapacity * numOfTraces * sizeof(UINT_64)));
    checkCuda(cudaMalloc(&d_temp_LTLcache, temp_LTLCacheCapacity * numOfTraces * sizeof(UINT_64)));

    using hash_set_t = warpcore::HashSet<
        UINT_64,         // key type
        UINT_64(0) - 1,  // empty key
        UINT_64(0) - 2,  // tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <UINT_64>>>;

    hash_set_t cHashSet(2 * LTLCacheCapacity);
    hash_set_t iHashSet(2 * LTLCacheCapacity);

    checkCuda(cudaMemcpy(d_LTLcache, LTLcache, alphabetSize * numOfTraces * sizeof(UINT_64), cudaMemcpyHostToDevice));
    hashSetsInitialisation<hash_set_t> << <1, alphabetSize >> > (numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet, d_LTLcache);

    // ----------------------------
    // Enumeration of the next LTLs
    // ----------------------------

    bool onTheFly = false, lastRound = false;
    int shortageCost = -1;

    for (LTLCost = c1 + 1; LTLCost <= maxCost; ++LTLCost) {


        // Once it uses a previous cost that is not fully stored, it should continue as the last round
        if (onTheFly) {
            int dif = LTLCost - shortageCost;
            if (dif == c2 || dif == c1 + c3 || dif == c1 + c4 || dif == c5 || dif == c6 || dif == c7 || dif == c1 + c8) lastRound = true;
        }


        // negation (~)
        // NegType = 1 is for negation of phi
        // NegType = 2 is for negation of char only
        if ((NegType == 1 && LTLCost - c2 >= c1) || (NegType == 2 && LTLCost - c2 == c1))
            if (LTLCost - c2 >= c1) {

                int Idx1 = LTLStartPoints[(LTLCost - c2) * 7];
                int Idx2 = LTLStartPoints[(LTLCost - c2 + 1) * 7] - 1;
                int N = Idx2 - Idx1 + 1;

                if (N) {
                    int x = Idx1, y;
                    do {
                        y = x + std::min(temp_LTLCacheCapacity - 1, Idx2 - x);
                        N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (~) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                            LTLCost, allLTLs, lastIdx, N);
#endif
                        int Blc = (N + 1023) / 1024;
                        processOperator<Op::Not, hash_set_t> << <Blc, 1024 >> > (
                            x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                            d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                            cHashSet, iHashSet, d_FinalLTLIdx
                            );
                        checkCuda(cudaPeekAtLastError());
                        checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                        allLTLs += N;
                        if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 1] = INT_MAX; goto exitEnumeration; }
                        if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                            d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                        x = y + 1;
                    } while (y < Idx2);
                }

            }
        LTLStartPoints[LTLCost * 7 + 1] = lastIdx;

        // intersection (&)
        for (int i = c1; 2 * i <= LTLCost - c3; ++i) {

            int Idx1 = LTLStartPoints[i * 7];
            int Idx2 = LTLStartPoints[(i + 1) * 7] - 1;
            int Idx3 = LTLStartPoints[(LTLCost - i - c3) * 7];
            int Idx4 = LTLStartPoints[(LTLCost - i - c3 + 1) * 7] - 1;
            int N = (Idx4 - Idx3 + 1) * (Idx2 - Idx1 + 1);

            if (N) {
                int x = Idx3, y;
                do {
                    y = x + std::min(temp_LTLCacheCapacity / (Idx2 - Idx1 + 1) - 1, Idx4 - x);
                    N = (y - x + 1) * (Idx2 - Idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (&) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLCost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::And, hash_set_t> << <Blc, 1024 >> > (
                        Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 2] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx4);
            }

        }
        LTLStartPoints[LTLCost * 7 + 2] = lastIdx;

        // union (|)
        for (int i = c1; 2 * i <= LTLCost - c4; ++i) {

            int Idx1 = LTLStartPoints[i * 7];
            int Idx2 = LTLStartPoints[(i + 1) * 7] - 1;
            int Idx3 = LTLStartPoints[(LTLCost - i - c4) * 7];
            int Idx4 = LTLStartPoints[(LTLCost - i - c4 + 1) * 7] - 1;
            int N = (Idx4 - Idx3 + 1) * (Idx2 - Idx1 + 1);

            if (N) {
                int x = Idx3, y;
                do {
                    y = x + std::min(temp_LTLCacheCapacity / (Idx2 - Idx1 + 1) - 1, Idx4 - x);
                    N = (y - x + 1) * (Idx2 - Idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (|) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLCost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Or, hash_set_t> << <Blc, 1024 >> > (
                        Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 3] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx4);
            }

        }
        LTLStartPoints[LTLCost * 7 + 3] = lastIdx;

        // next (X)
        if (LTLCost - c5 >= c1) {

            int Idx1 = LTLStartPoints[(LTLCost - c5) * 7];
            int Idx2 = LTLStartPoints[(LTLCost - c5 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLCacheCapacity - 1, Idx2 - x);
                    N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (X) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLCost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Next, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 4] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        LTLStartPoints[LTLCost * 7 + 4] = lastIdx;

        // finally (F)
        if (LTLCost - c6 >= c1) {

            int Idx1 = LTLStartPoints[(LTLCost - c6) * 7];
            int Idx2 = LTLStartPoints[(LTLCost - c6 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLCacheCapacity - 1, Idx2 - x);
                    N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (F) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLCost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Finally, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 5] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        LTLStartPoints[LTLCost * 7 + 5] = lastIdx;

        // globally (G)
        if (LTLCost - c7 >= c1) {

            int Idx1 = LTLStartPoints[(LTLCost - c7) * 7];
            int Idx2 = LTLStartPoints[(LTLCost - c7 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLCacheCapacity - 1, Idx2 - x);
                    N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (G) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLCost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Globally, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[LTLCost * 7 + 6] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        LTLStartPoints[LTLCost * 7 + 6] = lastIdx;

        // until (U)
        for (int i = c1; 2 * i <= LTLCost - c8; ++i) {

            int Idx1 = LTLStartPoints[i * 7];
            int Idx2 = LTLStartPoints[(i + 1) * 7] - 1;
            int Idx3 = LTLStartPoints[(LTLCost - i - c8) * 7];
            int Idx4 = LTLStartPoints[(LTLCost - i - c8 + 1) * 7] - 1;
            int N = (Idx4 - Idx3 + 1) * (Idx2 - Idx1 + 1);

            if (N) {
                int x = Idx3, y;
                do {
                    y = x + std::min(temp_LTLCacheCapacity / (2 * (Idx2 - Idx1 + 1)) - 1, Idx4 - x); // 2 is for until only (lUr and rUl)
                    N = (y - x + 1) * (Idx2 - Idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (U) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLCost, allLTLs, lastIdx, 2 * N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Until, hash_set_t> << <Blc, 1024 >> > (
                        Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += 2 * N;
                    if (*FinalLTLIdx != -1) { LTLStartPoints[(LTLCost + 1) * 7] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(2 * N, lastIdx, numOfTraces, LTLCacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx4);
            }

        }
        LTLStartPoints[(LTLCost + 1) * 7] = lastIdx;

        if (lastRound) break;
        if (onTheFly && shortageCost == -1) shortageCost = LTLCost;

    }

exitEnumeration:

    std::string output;
    bool isLTLFromTempLTLcache = true;

    if (*FinalLTLIdx != -1) {
        output = LTLString(isLTLFromTempLTLcache, *FinalLTLIdx, lastIdx, alphabet, LTLStartPoints,
            d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
        cudaFree(d_LTLcache);
        cudaFree(d_FinalLTLIdx);
        cudaFree(d_temp_LTLcache);
        cudaFree(d_leftIdx);
        cudaFree(d_rightIdx);
        cudaFree(d_temp_leftIdx);
        cudaFree(d_temp_rightIdx);
    }

    else {
        UINT_64* d_BSCache;
        const int BSCacheCapacity = maxAllocationSize / sizeof(UINT_64) * 1.5;
        checkCuda(cudaMalloc(&d_BSCache, BSCacheCapacity * sizeof(UINT_64)));
        int numBlocks = (lastIdx + 1023) / 1024;
        transfer << <numBlocks, 1024 >> > (numOfTraces, lastIdx, d_LTLcache, d_BSCache);
        cudaFree(d_LTLcache);
        cudaFree(d_FinalLTLIdx);
        cudaFree(d_temp_LTLcache);
        cudaFree(d_temp_leftIdx);
        cudaFree(d_temp_rightIdx);
        output = BS(
            costFun, BSMaxCost, lastIdx, alphabet, LTLStartPoints, LTLCost, d_BSCache,
            numOfTraces, numOfP, d_leftIdx, d_rightIdx);
        cudaFree(d_leftIdx);
        cudaFree(d_rightIdx);
    }

    return output;

}

// Reading the input file
bool readFile(
    const std::string& fileName,
    std::set<char>& alphabet,
    std::vector<std::vector<std::string>>& pos,
    std::vector<std::vector<std::string>>& neg)
{

    std::ifstream file(fileName);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::string line;
        char ch;
        bool foundNewline = false;
        while (!foundNewline && file.tellg() > 0) {
            file.seekg(-2, std::ios::cur);
            file.get(ch);
            if (ch == '\n') foundNewline = true;
        }
        std::getline(file, line);
        std::string alpha;
        for (auto& c : line)
            if (c >= 'a' && c <= 'z') {
                alphabet.insert(c);
                alpha += c;
            }
        file.seekg(0, std::ios::beg);
        while (std::getline(file, line)) {
            std::vector<std::string> trace;
            if (line != "---") {
                std::string token;
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
        while (std::getline(file, line)) {
            std::vector<std::string> trace;
            if (line != "---") {
                std::string token;
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

    if (argc != 14) {
        printf("Arguments should be in the form of\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s <input_file_address> <c1> <c2> <c3> <c4> <c5> <c6> <c7> <c8> <maxCost> <BSMaxCost> <RlxUnqChkTyp> <NegType>\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        printf("\nFor example\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s ./input.txt 1 1 1 1 1 1 1 1 500 3 2\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        return 0;
    }

    bool argError = false;
    for (int i = 2; i < argc - 2; ++i) {
        if (std::atoi(argv[i]) <= 0 || std::atoi(argv[i]) > SHRT_MAX) {
            printf("Argument number %d, \"%s\", should be a positive short integer.\n", i, argv[i]);
            argError = true;
        }
    }
    if (std::atoi(argv[12]) < 1 || std::atoi(argv[12]) > 3) {
        printf("Argument number 12, RlxUnqChkTyp = \"%s\", should be 1, 2, or 3.\n", argv[11]);
        argError = true;
    }
    if (std::atoi(argv[13]) < 1 || std::atoi(argv[13]) > 2) {
        printf("Argument number 13, NegType = \"%s\", should be 1, or 2.\n", argv[12]);
        argError = true;
    }

    if (argError) return 0;

    std::string fileName = argv[1];
    std::set<char> alphabet;
    std::vector<std::vector<std::string>> pos, neg;
    if (!readFile(fileName, alphabet, pos, neg)) return 0;
    unsigned short costFun[8];
    for (int i = 0; i < 8; i++)
        costFun[i] = std::atoi(argv[i + 2]);
    unsigned short maxCost = std::atoi(argv[10]);
    unsigned short BSMaxCost = std::atoi(argv[11]);
    unsigned int RlxUnqChkTyp = std::atoi(argv[12]);
    unsigned int NegType = std::atoi(argv[13]);

    // --------------------------------------
    // Linear Temporal Logic Inference (LTLI)
    // --------------------------------------

#ifdef MEASUREMENT_MODE
    auto start = std::chrono::high_resolution_clock::now();
#endif

    std::uint64_t allLTLs{}; int LTLCost = costFun[0];
    std::string output = LTLI(costFun, maxCost, BSMaxCost, RlxUnqChkTyp, NegType, alphabet, LTLCost, allLTLs, pos, neg);
    if (output == "see_the_error") return 0;

#ifdef MEASUREMENT_MODE
    auto stop = std::chrono::high_resolution_clock::now();
#endif

    // -------------------
    // Printing the output
    // -------------------

    printf("\nPositive: \n");
    for (const auto& trace : pos) {
        printf("\t");
        for (const auto& t : trace) {
            std::string s;
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
            std::string s;
            for (const auto& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("\n");
    }

    printf("\nCost Function: p:%u, ~:%u, &:%u, |:%u, X:%u, F:%u, G:%u, U:%u",
        costFun[0], costFun[1], costFun[2], costFun[3], costFun[4], costFun[5], costFun[6], costFun[7]);
    printf("\nNumber of Traces: %d", static_cast<int>(pos.size() + neg.size()));
#ifdef MEASUREMENT_MODE
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("\nNumber of All LTLs: %lu", allLTLs);
    printf("\nCost of Final LTL: %d", LTLCost);
    printf("\nRunning Time: %f s", (double)duration * 0.000001);
#endif
    printf("\n\nLTL: \"%s\"\n", output.c_str());

    return 0;

}