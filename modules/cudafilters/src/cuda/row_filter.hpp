/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace row_filter
{
    #define MAX_KERNEL_SIZE 128

    __constant__ float c_kernel[MAX_KERNEL_SIZE];

    template <int KSIZE, typename T, typename D, typename B>
    __global__ void linearRowFilter(const PtrStepSz<T> src, PtrStep<D> dst, const int anchor, const B brd)
    {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
            const int BLOCK_DIM_X = 32;
            const int BLOCK_DIM_Y = 8;
            const int PATCH_PER_BLOCK = 4;
            const int HALO_SIZE = KSIZE <= 32 ? 1 : 2;
        #else
            const int BLOCK_DIM_X = 32;
            const int BLOCK_DIM_Y = 4;
            const int PATCH_PER_BLOCK = 4;
            const int HALO_SIZE = 1;
        #endif

        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

        __shared__ sum_t smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];

        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (y >= src.rows)
            return;

        const T* src_row = src.ptr(y);

        const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

        if (blockIdx.x > 0)
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = saturate_cast<sum_t>(src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X]);
        }
        else
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_low(xStart - (HALO_SIZE - j) * BLOCK_DIM_X, src_row));
        }

        if (blockIdx.x + 2 < gridDim.x)
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(src_row[xStart + j * BLOCK_DIM_X]);

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X]);
        }
        else
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_high(xStart + j * BLOCK_DIM_X, src_row));

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_high(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src_row));
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
        {
            const int x = xStart + j * BLOCK_DIM_X;

            if (x < src.cols)
            {
                sum_t sum = VecTraits<sum_t>::all(0);

                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum = sum + smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X - anchor + k] * c_kernel[k];

                dst(y, x) = saturate_cast<D>(sum);
            }
        }
    }

    template <int KSIZE, typename T, typename D, template<typename> class B>
    void caller(PtrStepSz<T> src, PtrStepSz<D> dst, int anchor, int cc, cudaStream_t stream)
    {
        int BLOCK_DIM_X;
        int BLOCK_DIM_Y;
        int PATCH_PER_BLOCK;

        if (cc >= 20)
        {
            BLOCK_DIM_X = 32;
            BLOCK_DIM_Y = 8;
            PATCH_PER_BLOCK = 4;
        }
        else
        {
            BLOCK_DIM_X = 32;
            BLOCK_DIM_Y = 4;
            PATCH_PER_BLOCK = 4;
        }

        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid(divUp(src.cols, BLOCK_DIM_X * PATCH_PER_BLOCK), divUp(src.rows, BLOCK_DIM_Y));

        B<T> brd(src.cols);

        linearRowFilter<KSIZE, T, D><<<grid, block, 0, stream>>>(src, dst, anchor, brd);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

namespace filter
{
    template <typename T, typename D>
    void linearRow(PtrStepSzb src, PtrStepSzb dst, const float* kernel, int ksize, int anchor, int brd_type, int cc, cudaStream_t stream)
    {
        typedef void (*caller_t)(PtrStepSz<T> src, PtrStepSz<D> dst, int anchor, int cc, cudaStream_t stream);

        static const caller_t callers[5][129] =
        {
            {
                0,
                row_filter::caller< 1, T, D, BrdRowConstant>,
                row_filter::caller< 2, T, D, BrdRowConstant>,
                row_filter::caller< 3, T, D, BrdRowConstant>,
                row_filter::caller< 4, T, D, BrdRowConstant>,
                row_filter::caller< 5, T, D, BrdRowConstant>,
                row_filter::caller< 6, T, D, BrdRowConstant>,
                row_filter::caller< 7, T, D, BrdRowConstant>,
                row_filter::caller< 8, T, D, BrdRowConstant>,
                row_filter::caller< 9, T, D, BrdRowConstant>,
                row_filter::caller<10, T, D, BrdRowConstant>,
                row_filter::caller<11, T, D, BrdRowConstant>,
                row_filter::caller<12, T, D, BrdRowConstant>,
                row_filter::caller<13, T, D, BrdRowConstant>,
                row_filter::caller<14, T, D, BrdRowConstant>,
                row_filter::caller<15, T, D, BrdRowConstant>,
                row_filter::caller<16, T, D, BrdRowConstant>,
                row_filter::caller<17, T, D, BrdRowConstant>,
                row_filter::caller<18, T, D, BrdRowConstant>,
                row_filter::caller<19, T, D, BrdRowConstant>,
                row_filter::caller<20, T, D, BrdRowConstant>,
                row_filter::caller<21, T, D, BrdRowConstant>,
                row_filter::caller<22, T, D, BrdRowConstant>,
                row_filter::caller<23, T, D, BrdRowConstant>,
                row_filter::caller<24, T, D, BrdRowConstant>,
                row_filter::caller<25, T, D, BrdRowConstant>,
                row_filter::caller<26, T, D, BrdRowConstant>,
                row_filter::caller<27, T, D, BrdRowConstant>,
                row_filter::caller<28, T, D, BrdRowConstant>,
                row_filter::caller<29, T, D, BrdRowConstant>,
                row_filter::caller<30, T, D, BrdRowConstant>,
                row_filter::caller<31, T, D, BrdRowConstant>,
                row_filter::caller<32, T, D, BrdRowConstant>,
                row_filter::caller<33, T, D, BrdRowConstant>,
                row_filter::caller<34, T, D, BrdRowConstant>,
                row_filter::caller<35, T, D, BrdRowConstant>,
                row_filter::caller<36, T, D, BrdRowConstant>,
                row_filter::caller<37, T, D, BrdRowConstant>,
                row_filter::caller<38, T, D, BrdRowConstant>,
                row_filter::caller<39, T, D, BrdRowConstant>,
                row_filter::caller<40, T, D, BrdRowConstant>,
                row_filter::caller<41, T, D, BrdRowConstant>,
                row_filter::caller<42, T, D, BrdRowConstant>,
                row_filter::caller<43, T, D, BrdRowConstant>,
                row_filter::caller<44, T, D, BrdRowConstant>,
                row_filter::caller<45, T, D, BrdRowConstant>,
                row_filter::caller<46, T, D, BrdRowConstant>,
                row_filter::caller<47, T, D, BrdRowConstant>,
                row_filter::caller<48, T, D, BrdRowConstant>,
                row_filter::caller<49, T, D, BrdRowConstant>,
                row_filter::caller<50, T, D, BrdRowConstant>,
                row_filter::caller<51, T, D, BrdRowConstant>,
                row_filter::caller<52, T, D, BrdRowConstant>,
                row_filter::caller<53, T, D, BrdRowConstant>,
                row_filter::caller<54, T, D, BrdRowConstant>,
                row_filter::caller<55, T, D, BrdRowConstant>,
                row_filter::caller<56, T, D, BrdRowConstant>,
                row_filter::caller<57, T, D, BrdRowConstant>,
                row_filter::caller<58, T, D, BrdRowConstant>,
                row_filter::caller<59, T, D, BrdRowConstant>,
                row_filter::caller<60, T, D, BrdRowConstant>,
                row_filter::caller<61, T, D, BrdRowConstant>,
                row_filter::caller<62, T, D, BrdRowConstant>,
                row_filter::caller<63, T, D, BrdRowConstant>,
                row_filter::caller<64, T, D, BrdRowConstant>,
                row_filter::caller<65, T, D, BrdRowConstant>,
                row_filter::caller<66, T, D, BrdRowConstant>,
                row_filter::caller<67, T, D, BrdRowConstant>,
                row_filter::caller<68, T, D, BrdRowConstant>,
                row_filter::caller<69, T, D, BrdRowConstant>,
                row_filter::caller<70, T, D, BrdRowConstant>,
                row_filter::caller<71, T, D, BrdRowConstant>,
                row_filter::caller<72, T, D, BrdRowConstant>,
                row_filter::caller<73, T, D, BrdRowConstant>,
                row_filter::caller<74, T, D, BrdRowConstant>,
                row_filter::caller<75, T, D, BrdRowConstant>,
                row_filter::caller<76, T, D, BrdRowConstant>,
                row_filter::caller<77, T, D, BrdRowConstant>,
                row_filter::caller<78, T, D, BrdRowConstant>,
                row_filter::caller<79, T, D, BrdRowConstant>,
                row_filter::caller<80, T, D, BrdRowConstant>,
                row_filter::caller<81, T, D, BrdRowConstant>,
                row_filter::caller<82, T, D, BrdRowConstant>,
                row_filter::caller<83, T, D, BrdRowConstant>,
                row_filter::caller<84, T, D, BrdRowConstant>,
                row_filter::caller<85, T, D, BrdRowConstant>,
                row_filter::caller<86, T, D, BrdRowConstant>,
                row_filter::caller<87, T, D, BrdRowConstant>,
                row_filter::caller<88, T, D, BrdRowConstant>,
                row_filter::caller<89, T, D, BrdRowConstant>,
                row_filter::caller<90, T, D, BrdRowConstant>,
                row_filter::caller<91, T, D, BrdRowConstant>,
                row_filter::caller<92, T, D, BrdRowConstant>,
                row_filter::caller<93, T, D, BrdRowConstant>,
                row_filter::caller<94, T, D, BrdRowConstant>,
                row_filter::caller<95, T, D, BrdRowConstant>,
                row_filter::caller<96, T, D, BrdRowConstant>,
                row_filter::caller<97, T, D, BrdRowConstant>,
                row_filter::caller<98, T, D, BrdRowConstant>,
                row_filter::caller<99, T, D, BrdRowConstant>,
                row_filter::caller<100, T, D, BrdRowConstant>,
                row_filter::caller<101, T, D, BrdRowConstant>,
                row_filter::caller<102, T, D, BrdRowConstant>,
                row_filter::caller<103, T, D, BrdRowConstant>,
                row_filter::caller<104, T, D, BrdRowConstant>,
                row_filter::caller<105, T, D, BrdRowConstant>,
                row_filter::caller<106, T, D, BrdRowConstant>,
                row_filter::caller<107, T, D, BrdRowConstant>,
                row_filter::caller<108, T, D, BrdRowConstant>,
                row_filter::caller<109, T, D, BrdRowConstant>,
                row_filter::caller<110, T, D, BrdRowConstant>,
                row_filter::caller<111, T, D, BrdRowConstant>,
                row_filter::caller<112, T, D, BrdRowConstant>,
                row_filter::caller<113, T, D, BrdRowConstant>,
                row_filter::caller<114, T, D, BrdRowConstant>,
                row_filter::caller<115, T, D, BrdRowConstant>,
                row_filter::caller<116, T, D, BrdRowConstant>,
                row_filter::caller<117, T, D, BrdRowConstant>,
                row_filter::caller<118, T, D, BrdRowConstant>,
                row_filter::caller<119, T, D, BrdRowConstant>,
                row_filter::caller<120, T, D, BrdRowConstant>,
                row_filter::caller<121, T, D, BrdRowConstant>,
                row_filter::caller<122, T, D, BrdRowConstant>,
                row_filter::caller<123, T, D, BrdRowConstant>,
                row_filter::caller<124, T, D, BrdRowConstant>,
                row_filter::caller<125, T, D, BrdRowConstant>,
                row_filter::caller<126, T, D, BrdRowConstant>,
                row_filter::caller<127, T, D, BrdRowConstant>,
                row_filter::caller<128, T, D, BrdRowConstant>
            },
            {
                0,
                row_filter::caller< 1, T, D, BrdRowReplicate>,
                row_filter::caller< 2, T, D, BrdRowReplicate>,
                row_filter::caller< 3, T, D, BrdRowReplicate>,
                row_filter::caller< 4, T, D, BrdRowReplicate>,
                row_filter::caller< 5, T, D, BrdRowReplicate>,
                row_filter::caller< 6, T, D, BrdRowReplicate>,
                row_filter::caller< 7, T, D, BrdRowReplicate>,
                row_filter::caller< 8, T, D, BrdRowReplicate>,
                row_filter::caller< 9, T, D, BrdRowReplicate>,
                row_filter::caller<10, T, D, BrdRowReplicate>,
                row_filter::caller<11, T, D, BrdRowReplicate>,
                row_filter::caller<12, T, D, BrdRowReplicate>,
                row_filter::caller<13, T, D, BrdRowReplicate>,
                row_filter::caller<14, T, D, BrdRowReplicate>,
                row_filter::caller<15, T, D, BrdRowReplicate>,
                row_filter::caller<16, T, D, BrdRowReplicate>,
                row_filter::caller<17, T, D, BrdRowReplicate>,
                row_filter::caller<18, T, D, BrdRowReplicate>,
                row_filter::caller<19, T, D, BrdRowReplicate>,
                row_filter::caller<20, T, D, BrdRowReplicate>,
                row_filter::caller<21, T, D, BrdRowReplicate>,
                row_filter::caller<22, T, D, BrdRowReplicate>,
                row_filter::caller<23, T, D, BrdRowReplicate>,
                row_filter::caller<24, T, D, BrdRowReplicate>,
                row_filter::caller<25, T, D, BrdRowReplicate>,
                row_filter::caller<26, T, D, BrdRowReplicate>,
                row_filter::caller<27, T, D, BrdRowReplicate>,
                row_filter::caller<28, T, D, BrdRowReplicate>,
                row_filter::caller<29, T, D, BrdRowReplicate>,
                row_filter::caller<30, T, D, BrdRowReplicate>,
                row_filter::caller<31, T, D, BrdRowReplicate>,
                row_filter::caller<32, T, D, BrdRowReplicate>,
                row_filter::caller<33, T, D, BrdRowReplicate>,
                row_filter::caller<34, T, D, BrdRowReplicate>,
                row_filter::caller<35, T, D, BrdRowReplicate>,
                row_filter::caller<36, T, D, BrdRowReplicate>,
                row_filter::caller<37, T, D, BrdRowReplicate>,
                row_filter::caller<38, T, D, BrdRowReplicate>,
                row_filter::caller<39, T, D, BrdRowReplicate>,
                row_filter::caller<40, T, D, BrdRowReplicate>,
                row_filter::caller<41, T, D, BrdRowReplicate>,
                row_filter::caller<42, T, D, BrdRowReplicate>,
                row_filter::caller<43, T, D, BrdRowReplicate>,
                row_filter::caller<44, T, D, BrdRowReplicate>,
                row_filter::caller<45, T, D, BrdRowReplicate>,
                row_filter::caller<46, T, D, BrdRowReplicate>,
                row_filter::caller<47, T, D, BrdRowReplicate>,
                row_filter::caller<48, T, D, BrdRowReplicate>,
                row_filter::caller<49, T, D, BrdRowReplicate>,
                row_filter::caller<50, T, D, BrdRowReplicate>,
                row_filter::caller<51, T, D, BrdRowReplicate>,
                row_filter::caller<52, T, D, BrdRowReplicate>,
                row_filter::caller<53, T, D, BrdRowReplicate>,
                row_filter::caller<54, T, D, BrdRowReplicate>,
                row_filter::caller<55, T, D, BrdRowReplicate>,
                row_filter::caller<56, T, D, BrdRowReplicate>,
                row_filter::caller<57, T, D, BrdRowReplicate>,
                row_filter::caller<58, T, D, BrdRowReplicate>,
                row_filter::caller<59, T, D, BrdRowReplicate>,
                row_filter::caller<60, T, D, BrdRowReplicate>,
                row_filter::caller<61, T, D, BrdRowReplicate>,
                row_filter::caller<62, T, D, BrdRowReplicate>,
                row_filter::caller<63, T, D, BrdRowReplicate>,
                row_filter::caller<64, T, D, BrdRowReplicate>,
                row_filter::caller<65, T, D, BrdRowReplicate>,
                row_filter::caller<66, T, D, BrdRowReplicate>,
                row_filter::caller<67, T, D, BrdRowReplicate>,
                row_filter::caller<68, T, D, BrdRowReplicate>,
                row_filter::caller<69, T, D, BrdRowReplicate>,
                row_filter::caller<70, T, D, BrdRowReplicate>,
                row_filter::caller<71, T, D, BrdRowReplicate>,
                row_filter::caller<72, T, D, BrdRowReplicate>,
                row_filter::caller<73, T, D, BrdRowReplicate>,
                row_filter::caller<74, T, D, BrdRowReplicate>,
                row_filter::caller<75, T, D, BrdRowReplicate>,
                row_filter::caller<76, T, D, BrdRowReplicate>,
                row_filter::caller<77, T, D, BrdRowReplicate>,
                row_filter::caller<78, T, D, BrdRowReplicate>,
                row_filter::caller<79, T, D, BrdRowReplicate>,
                row_filter::caller<80, T, D, BrdRowReplicate>,
                row_filter::caller<81, T, D, BrdRowReplicate>,
                row_filter::caller<82, T, D, BrdRowReplicate>,
                row_filter::caller<83, T, D, BrdRowReplicate>,
                row_filter::caller<84, T, D, BrdRowReplicate>,
                row_filter::caller<85, T, D, BrdRowReplicate>,
                row_filter::caller<86, T, D, BrdRowReplicate>,
                row_filter::caller<87, T, D, BrdRowReplicate>,
                row_filter::caller<88, T, D, BrdRowReplicate>,
                row_filter::caller<89, T, D, BrdRowReplicate>,
                row_filter::caller<90, T, D, BrdRowReplicate>,
                row_filter::caller<91, T, D, BrdRowReplicate>,
                row_filter::caller<92, T, D, BrdRowReplicate>,
                row_filter::caller<93, T, D, BrdRowReplicate>,
                row_filter::caller<94, T, D, BrdRowReplicate>,
                row_filter::caller<95, T, D, BrdRowReplicate>,
                row_filter::caller<96, T, D, BrdRowReplicate>,
                row_filter::caller<97, T, D, BrdRowReplicate>,
                row_filter::caller<98, T, D, BrdRowReplicate>,
                row_filter::caller<99, T, D, BrdRowReplicate>,
                row_filter::caller<100, T, D, BrdRowReplicate>,
                row_filter::caller<101, T, D, BrdRowReplicate>,
                row_filter::caller<102, T, D, BrdRowReplicate>,
                row_filter::caller<103, T, D, BrdRowReplicate>,
                row_filter::caller<104, T, D, BrdRowReplicate>,
                row_filter::caller<105, T, D, BrdRowReplicate>,
                row_filter::caller<106, T, D, BrdRowReplicate>,
                row_filter::caller<107, T, D, BrdRowReplicate>,
                row_filter::caller<108, T, D, BrdRowReplicate>,
                row_filter::caller<109, T, D, BrdRowReplicate>,
                row_filter::caller<110, T, D, BrdRowReplicate>,
                row_filter::caller<111, T, D, BrdRowReplicate>,
                row_filter::caller<112, T, D, BrdRowReplicate>,
                row_filter::caller<113, T, D, BrdRowReplicate>,
                row_filter::caller<114, T, D, BrdRowReplicate>,
                row_filter::caller<115, T, D, BrdRowReplicate>,
                row_filter::caller<116, T, D, BrdRowReplicate>,
                row_filter::caller<117, T, D, BrdRowReplicate>,
                row_filter::caller<118, T, D, BrdRowReplicate>,
                row_filter::caller<119, T, D, BrdRowReplicate>,
                row_filter::caller<120, T, D, BrdRowReplicate>,
                row_filter::caller<121, T, D, BrdRowReplicate>,
                row_filter::caller<122, T, D, BrdRowReplicate>,
                row_filter::caller<123, T, D, BrdRowReplicate>,
                row_filter::caller<124, T, D, BrdRowReplicate>,
                row_filter::caller<125, T, D, BrdRowReplicate>,
                row_filter::caller<126, T, D, BrdRowReplicate>,
                row_filter::caller<127, T, D, BrdRowReplicate>,
                row_filter::caller<128, T, D, BrdRowReplicate>
            },
            {
                0,
                row_filter::caller< 1, T, D, BrdRowReflect>,
                row_filter::caller< 2, T, D, BrdRowReflect>,
                row_filter::caller< 3, T, D, BrdRowReflect>,
                row_filter::caller< 4, T, D, BrdRowReflect>,
                row_filter::caller< 5, T, D, BrdRowReflect>,
                row_filter::caller< 6, T, D, BrdRowReflect>,
                row_filter::caller< 7, T, D, BrdRowReflect>,
                row_filter::caller< 8, T, D, BrdRowReflect>,
                row_filter::caller< 9, T, D, BrdRowReflect>,
                row_filter::caller<10, T, D, BrdRowReflect>,
                row_filter::caller<11, T, D, BrdRowReflect>,
                row_filter::caller<12, T, D, BrdRowReflect>,
                row_filter::caller<13, T, D, BrdRowReflect>,
                row_filter::caller<14, T, D, BrdRowReflect>,
                row_filter::caller<15, T, D, BrdRowReflect>,
                row_filter::caller<16, T, D, BrdRowReflect>,
                row_filter::caller<17, T, D, BrdRowReflect>,
                row_filter::caller<18, T, D, BrdRowReflect>,
                row_filter::caller<19, T, D, BrdRowReflect>,
                row_filter::caller<20, T, D, BrdRowReflect>,
                row_filter::caller<21, T, D, BrdRowReflect>,
                row_filter::caller<22, T, D, BrdRowReflect>,
                row_filter::caller<23, T, D, BrdRowReflect>,
                row_filter::caller<24, T, D, BrdRowReflect>,
                row_filter::caller<25, T, D, BrdRowReflect>,
                row_filter::caller<26, T, D, BrdRowReflect>,
                row_filter::caller<27, T, D, BrdRowReflect>,
                row_filter::caller<28, T, D, BrdRowReflect>,
                row_filter::caller<29, T, D, BrdRowReflect>,
                row_filter::caller<30, T, D, BrdRowReflect>,
                row_filter::caller<31, T, D, BrdRowReflect>,
                row_filter::caller<32, T, D, BrdRowReflect>,
                row_filter::caller<33, T, D, BrdRowReflect>,
                row_filter::caller<34, T, D, BrdRowReflect>,
                row_filter::caller<35, T, D, BrdRowReflect>,
                row_filter::caller<36, T, D, BrdRowReflect>,
                row_filter::caller<37, T, D, BrdRowReflect>,
                row_filter::caller<38, T, D, BrdRowReflect>,
                row_filter::caller<39, T, D, BrdRowReflect>,
                row_filter::caller<40, T, D, BrdRowReflect>,
                row_filter::caller<41, T, D, BrdRowReflect>,
                row_filter::caller<42, T, D, BrdRowReflect>,
                row_filter::caller<43, T, D, BrdRowReflect>,
                row_filter::caller<44, T, D, BrdRowReflect>,
                row_filter::caller<45, T, D, BrdRowReflect>,
                row_filter::caller<46, T, D, BrdRowReflect>,
                row_filter::caller<47, T, D, BrdRowReflect>,
                row_filter::caller<48, T, D, BrdRowReflect>,
                row_filter::caller<49, T, D, BrdRowReflect>,
                row_filter::caller<50, T, D, BrdRowReflect>,
                row_filter::caller<51, T, D, BrdRowReflect>,
                row_filter::caller<52, T, D, BrdRowReflect>,
                row_filter::caller<53, T, D, BrdRowReflect>,
                row_filter::caller<54, T, D, BrdRowReflect>,
                row_filter::caller<55, T, D, BrdRowReflect>,
                row_filter::caller<56, T, D, BrdRowReflect>,
                row_filter::caller<57, T, D, BrdRowReflect>,
                row_filter::caller<58, T, D, BrdRowReflect>,
                row_filter::caller<59, T, D, BrdRowReflect>,
                row_filter::caller<60, T, D, BrdRowReflect>,
                row_filter::caller<61, T, D, BrdRowReflect>,
                row_filter::caller<62, T, D, BrdRowReflect>,
                row_filter::caller<63, T, D, BrdRowReflect>,
                row_filter::caller<64, T, D, BrdRowReflect>,
                row_filter::caller<65, T, D, BrdRowReflect>,
                row_filter::caller<66, T, D, BrdRowReflect>,
                row_filter::caller<67, T, D, BrdRowReflect>,
                row_filter::caller<68, T, D, BrdRowReflect>,
                row_filter::caller<69, T, D, BrdRowReflect>,
                row_filter::caller<70, T, D, BrdRowReflect>,
                row_filter::caller<71, T, D, BrdRowReflect>,
                row_filter::caller<72, T, D, BrdRowReflect>,
                row_filter::caller<73, T, D, BrdRowReflect>,
                row_filter::caller<74, T, D, BrdRowReflect>,
                row_filter::caller<75, T, D, BrdRowReflect>,
                row_filter::caller<76, T, D, BrdRowReflect>,
                row_filter::caller<77, T, D, BrdRowReflect>,
                row_filter::caller<78, T, D, BrdRowReflect>,
                row_filter::caller<79, T, D, BrdRowReflect>,
                row_filter::caller<80, T, D, BrdRowReflect>,
                row_filter::caller<81, T, D, BrdRowReflect>,
                row_filter::caller<82, T, D, BrdRowReflect>,
                row_filter::caller<83, T, D, BrdRowReflect>,
                row_filter::caller<84, T, D, BrdRowReflect>,
                row_filter::caller<85, T, D, BrdRowReflect>,
                row_filter::caller<86, T, D, BrdRowReflect>,
                row_filter::caller<87, T, D, BrdRowReflect>,
                row_filter::caller<88, T, D, BrdRowReflect>,
                row_filter::caller<89, T, D, BrdRowReflect>,
                row_filter::caller<90, T, D, BrdRowReflect>,
                row_filter::caller<91, T, D, BrdRowReflect>,
                row_filter::caller<92, T, D, BrdRowReflect>,
                row_filter::caller<93, T, D, BrdRowReflect>,
                row_filter::caller<94, T, D, BrdRowReflect>,
                row_filter::caller<95, T, D, BrdRowReflect>,
                row_filter::caller<96, T, D, BrdRowReflect>,
                row_filter::caller<97, T, D, BrdRowReflect>,
                row_filter::caller<98, T, D, BrdRowReflect>,
                row_filter::caller<99, T, D, BrdRowReflect>,
                row_filter::caller<100, T, D, BrdRowReflect>,
                row_filter::caller<101, T, D, BrdRowReflect>,
                row_filter::caller<102, T, D, BrdRowReflect>,
                row_filter::caller<103, T, D, BrdRowReflect>,
                row_filter::caller<104, T, D, BrdRowReflect>,
                row_filter::caller<105, T, D, BrdRowReflect>,
                row_filter::caller<106, T, D, BrdRowReflect>,
                row_filter::caller<107, T, D, BrdRowReflect>,
                row_filter::caller<108, T, D, BrdRowReflect>,
                row_filter::caller<109, T, D, BrdRowReflect>,
                row_filter::caller<110, T, D, BrdRowReflect>,
                row_filter::caller<111, T, D, BrdRowReflect>,
                row_filter::caller<112, T, D, BrdRowReflect>,
                row_filter::caller<113, T, D, BrdRowReflect>,
                row_filter::caller<114, T, D, BrdRowReflect>,
                row_filter::caller<115, T, D, BrdRowReflect>,
                row_filter::caller<116, T, D, BrdRowReflect>,
                row_filter::caller<117, T, D, BrdRowReflect>,
                row_filter::caller<118, T, D, BrdRowReflect>,
                row_filter::caller<119, T, D, BrdRowReflect>,
                row_filter::caller<120, T, D, BrdRowReflect>,
                row_filter::caller<121, T, D, BrdRowReflect>,
                row_filter::caller<122, T, D, BrdRowReflect>,
                row_filter::caller<123, T, D, BrdRowReflect>,
                row_filter::caller<124, T, D, BrdRowReflect>,
                row_filter::caller<125, T, D, BrdRowReflect>,
                row_filter::caller<126, T, D, BrdRowReflect>,
                row_filter::caller<127, T, D, BrdRowReflect>,
                row_filter::caller<128, T, D, BrdRowReflect>
            },
            {
                0,
                row_filter::caller< 1, T, D, BrdRowWrap>,
                row_filter::caller< 2, T, D, BrdRowWrap>,
                row_filter::caller< 3, T, D, BrdRowWrap>,
                row_filter::caller< 4, T, D, BrdRowWrap>,
                row_filter::caller< 5, T, D, BrdRowWrap>,
                row_filter::caller< 6, T, D, BrdRowWrap>,
                row_filter::caller< 7, T, D, BrdRowWrap>,
                row_filter::caller< 8, T, D, BrdRowWrap>,
                row_filter::caller< 9, T, D, BrdRowWrap>,
                row_filter::caller<10, T, D, BrdRowWrap>,
                row_filter::caller<11, T, D, BrdRowWrap>,
                row_filter::caller<12, T, D, BrdRowWrap>,
                row_filter::caller<13, T, D, BrdRowWrap>,
                row_filter::caller<14, T, D, BrdRowWrap>,
                row_filter::caller<15, T, D, BrdRowWrap>,
                row_filter::caller<16, T, D, BrdRowWrap>,
                row_filter::caller<17, T, D, BrdRowWrap>,
                row_filter::caller<18, T, D, BrdRowWrap>,
                row_filter::caller<19, T, D, BrdRowWrap>,
                row_filter::caller<20, T, D, BrdRowWrap>,
                row_filter::caller<21, T, D, BrdRowWrap>,
                row_filter::caller<22, T, D, BrdRowWrap>,
                row_filter::caller<23, T, D, BrdRowWrap>,
                row_filter::caller<24, T, D, BrdRowWrap>,
                row_filter::caller<25, T, D, BrdRowWrap>,
                row_filter::caller<26, T, D, BrdRowWrap>,
                row_filter::caller<27, T, D, BrdRowWrap>,
                row_filter::caller<28, T, D, BrdRowWrap>,
                row_filter::caller<29, T, D, BrdRowWrap>,
                row_filter::caller<30, T, D, BrdRowWrap>,
                row_filter::caller<31, T, D, BrdRowWrap>,
                row_filter::caller<32, T, D, BrdRowWrap>,
                row_filter::caller<33, T, D, BrdRowWrap>,
                row_filter::caller<34, T, D, BrdRowWrap>,
                row_filter::caller<35, T, D, BrdRowWrap>,
                row_filter::caller<36, T, D, BrdRowWrap>,
                row_filter::caller<37, T, D, BrdRowWrap>,
                row_filter::caller<38, T, D, BrdRowWrap>,
                row_filter::caller<39, T, D, BrdRowWrap>,
                row_filter::caller<40, T, D, BrdRowWrap>,
                row_filter::caller<41, T, D, BrdRowWrap>,
                row_filter::caller<42, T, D, BrdRowWrap>,
                row_filter::caller<43, T, D, BrdRowWrap>,
                row_filter::caller<44, T, D, BrdRowWrap>,
                row_filter::caller<45, T, D, BrdRowWrap>,
                row_filter::caller<46, T, D, BrdRowWrap>,
                row_filter::caller<47, T, D, BrdRowWrap>,
                row_filter::caller<48, T, D, BrdRowWrap>,
                row_filter::caller<49, T, D, BrdRowWrap>,
                row_filter::caller<50, T, D, BrdRowWrap>,
                row_filter::caller<51, T, D, BrdRowWrap>,
                row_filter::caller<52, T, D, BrdRowWrap>,
                row_filter::caller<53, T, D, BrdRowWrap>,
                row_filter::caller<54, T, D, BrdRowWrap>,
                row_filter::caller<55, T, D, BrdRowWrap>,
                row_filter::caller<56, T, D, BrdRowWrap>,
                row_filter::caller<57, T, D, BrdRowWrap>,
                row_filter::caller<58, T, D, BrdRowWrap>,
                row_filter::caller<59, T, D, BrdRowWrap>,
                row_filter::caller<60, T, D, BrdRowWrap>,
                row_filter::caller<61, T, D, BrdRowWrap>,
                row_filter::caller<62, T, D, BrdRowWrap>,
                row_filter::caller<63, T, D, BrdRowWrap>,
                row_filter::caller<64, T, D, BrdRowWrap>,
                row_filter::caller<65, T, D, BrdRowWrap>,
                row_filter::caller<66, T, D, BrdRowWrap>,
                row_filter::caller<67, T, D, BrdRowWrap>,
                row_filter::caller<68, T, D, BrdRowWrap>,
                row_filter::caller<69, T, D, BrdRowWrap>,
                row_filter::caller<70, T, D, BrdRowWrap>,
                row_filter::caller<71, T, D, BrdRowWrap>,
                row_filter::caller<72, T, D, BrdRowWrap>,
                row_filter::caller<73, T, D, BrdRowWrap>,
                row_filter::caller<74, T, D, BrdRowWrap>,
                row_filter::caller<75, T, D, BrdRowWrap>,
                row_filter::caller<76, T, D, BrdRowWrap>,
                row_filter::caller<77, T, D, BrdRowWrap>,
                row_filter::caller<78, T, D, BrdRowWrap>,
                row_filter::caller<79, T, D, BrdRowWrap>,
                row_filter::caller<80, T, D, BrdRowWrap>,
                row_filter::caller<81, T, D, BrdRowWrap>,
                row_filter::caller<82, T, D, BrdRowWrap>,
                row_filter::caller<83, T, D, BrdRowWrap>,
                row_filter::caller<84, T, D, BrdRowWrap>,
                row_filter::caller<85, T, D, BrdRowWrap>,
                row_filter::caller<86, T, D, BrdRowWrap>,
                row_filter::caller<87, T, D, BrdRowWrap>,
                row_filter::caller<88, T, D, BrdRowWrap>,
                row_filter::caller<89, T, D, BrdRowWrap>,
                row_filter::caller<90, T, D, BrdRowWrap>,
                row_filter::caller<91, T, D, BrdRowWrap>,
                row_filter::caller<92, T, D, BrdRowWrap>,
                row_filter::caller<93, T, D, BrdRowWrap>,
                row_filter::caller<94, T, D, BrdRowWrap>,
                row_filter::caller<95, T, D, BrdRowWrap>,
                row_filter::caller<96, T, D, BrdRowWrap>,
                row_filter::caller<97, T, D, BrdRowWrap>,
                row_filter::caller<98, T, D, BrdRowWrap>,
                row_filter::caller<99, T, D, BrdRowWrap>,
                row_filter::caller<100, T, D, BrdRowWrap>,
                row_filter::caller<101, T, D, BrdRowWrap>,
                row_filter::caller<102, T, D, BrdRowWrap>,
                row_filter::caller<103, T, D, BrdRowWrap>,
                row_filter::caller<104, T, D, BrdRowWrap>,
                row_filter::caller<105, T, D, BrdRowWrap>,
                row_filter::caller<106, T, D, BrdRowWrap>,
                row_filter::caller<107, T, D, BrdRowWrap>,
                row_filter::caller<108, T, D, BrdRowWrap>,
                row_filter::caller<109, T, D, BrdRowWrap>,
                row_filter::caller<110, T, D, BrdRowWrap>,
                row_filter::caller<111, T, D, BrdRowWrap>,
                row_filter::caller<112, T, D, BrdRowWrap>,
                row_filter::caller<113, T, D, BrdRowWrap>,
                row_filter::caller<114, T, D, BrdRowWrap>,
                row_filter::caller<115, T, D, BrdRowWrap>,
                row_filter::caller<116, T, D, BrdRowWrap>,
                row_filter::caller<117, T, D, BrdRowWrap>,
                row_filter::caller<118, T, D, BrdRowWrap>,
                row_filter::caller<119, T, D, BrdRowWrap>,
                row_filter::caller<120, T, D, BrdRowWrap>,
                row_filter::caller<121, T, D, BrdRowWrap>,
                row_filter::caller<122, T, D, BrdRowWrap>,
                row_filter::caller<123, T, D, BrdRowWrap>,
                row_filter::caller<124, T, D, BrdRowWrap>,
                row_filter::caller<125, T, D, BrdRowWrap>,
                row_filter::caller<126, T, D, BrdRowWrap>,
                row_filter::caller<127, T, D, BrdRowWrap>,
                row_filter::caller<128, T, D, BrdRowWrap>
            },
            {
                0,
                row_filter::caller< 1, T, D, BrdRowReflect101>,
                row_filter::caller< 2, T, D, BrdRowReflect101>,
                row_filter::caller< 3, T, D, BrdRowReflect101>,
                row_filter::caller< 4, T, D, BrdRowReflect101>,
                row_filter::caller< 5, T, D, BrdRowReflect101>,
                row_filter::caller< 6, T, D, BrdRowReflect101>,
                row_filter::caller< 7, T, D, BrdRowReflect101>,
                row_filter::caller< 8, T, D, BrdRowReflect101>,
                row_filter::caller< 9, T, D, BrdRowReflect101>,
                row_filter::caller<10, T, D, BrdRowReflect101>,
                row_filter::caller<11, T, D, BrdRowReflect101>,
                row_filter::caller<12, T, D, BrdRowReflect101>,
                row_filter::caller<13, T, D, BrdRowReflect101>,
                row_filter::caller<14, T, D, BrdRowReflect101>,
                row_filter::caller<15, T, D, BrdRowReflect101>,
                row_filter::caller<16, T, D, BrdRowReflect101>,
                row_filter::caller<17, T, D, BrdRowReflect101>,
                row_filter::caller<18, T, D, BrdRowReflect101>,
                row_filter::caller<19, T, D, BrdRowReflect101>,
                row_filter::caller<20, T, D, BrdRowReflect101>,
                row_filter::caller<21, T, D, BrdRowReflect101>,
                row_filter::caller<22, T, D, BrdRowReflect101>,
                row_filter::caller<23, T, D, BrdRowReflect101>,
                row_filter::caller<24, T, D, BrdRowReflect101>,
                row_filter::caller<25, T, D, BrdRowReflect101>,
                row_filter::caller<26, T, D, BrdRowReflect101>,
                row_filter::caller<27, T, D, BrdRowReflect101>,
                row_filter::caller<28, T, D, BrdRowReflect101>,
                row_filter::caller<29, T, D, BrdRowReflect101>,
                row_filter::caller<30, T, D, BrdRowReflect101>,
                row_filter::caller<31, T, D, BrdRowReflect101>,
                row_filter::caller<32, T, D, BrdRowReflect101>,
                row_filter::caller<33, T, D, BrdRowReflect101>,
                row_filter::caller<34, T, D, BrdRowReflect101>,
                row_filter::caller<35, T, D, BrdRowReflect101>,
                row_filter::caller<36, T, D, BrdRowReflect101>,
                row_filter::caller<37, T, D, BrdRowReflect101>,
                row_filter::caller<38, T, D, BrdRowReflect101>,
                row_filter::caller<39, T, D, BrdRowReflect101>,
                row_filter::caller<40, T, D, BrdRowReflect101>,
                row_filter::caller<41, T, D, BrdRowReflect101>,
                row_filter::caller<42, T, D, BrdRowReflect101>,
                row_filter::caller<43, T, D, BrdRowReflect101>,
                row_filter::caller<44, T, D, BrdRowReflect101>,
                row_filter::caller<45, T, D, BrdRowReflect101>,
                row_filter::caller<46, T, D, BrdRowReflect101>,
                row_filter::caller<47, T, D, BrdRowReflect101>,
                row_filter::caller<48, T, D, BrdRowReflect101>,
                row_filter::caller<49, T, D, BrdRowReflect101>,
                row_filter::caller<50, T, D, BrdRowReflect101>,
                row_filter::caller<51, T, D, BrdRowReflect101>,
                row_filter::caller<52, T, D, BrdRowReflect101>,
                row_filter::caller<53, T, D, BrdRowReflect101>,
                row_filter::caller<54, T, D, BrdRowReflect101>,
                row_filter::caller<55, T, D, BrdRowReflect101>,
                row_filter::caller<56, T, D, BrdRowReflect101>,
                row_filter::caller<57, T, D, BrdRowReflect101>,
                row_filter::caller<58, T, D, BrdRowReflect101>,
                row_filter::caller<59, T, D, BrdRowReflect101>,
                row_filter::caller<60, T, D, BrdRowReflect101>,
                row_filter::caller<61, T, D, BrdRowReflect101>,
                row_filter::caller<62, T, D, BrdRowReflect101>,
                row_filter::caller<63, T, D, BrdRowReflect101>,
                row_filter::caller<64, T, D, BrdRowReflect101>,
                row_filter::caller<65, T, D, BrdRowReflect101>,
                row_filter::caller<66, T, D, BrdRowReflect101>,
                row_filter::caller<67, T, D, BrdRowReflect101>,
                row_filter::caller<68, T, D, BrdRowReflect101>,
                row_filter::caller<69, T, D, BrdRowReflect101>,
                row_filter::caller<70, T, D, BrdRowReflect101>,
                row_filter::caller<71, T, D, BrdRowReflect101>,
                row_filter::caller<72, T, D, BrdRowReflect101>,
                row_filter::caller<73, T, D, BrdRowReflect101>,
                row_filter::caller<74, T, D, BrdRowReflect101>,
                row_filter::caller<75, T, D, BrdRowReflect101>,
                row_filter::caller<76, T, D, BrdRowReflect101>,
                row_filter::caller<77, T, D, BrdRowReflect101>,
                row_filter::caller<78, T, D, BrdRowReflect101>,
                row_filter::caller<79, T, D, BrdRowReflect101>,
                row_filter::caller<80, T, D, BrdRowReflect101>,
                row_filter::caller<81, T, D, BrdRowReflect101>,
                row_filter::caller<82, T, D, BrdRowReflect101>,
                row_filter::caller<83, T, D, BrdRowReflect101>,
                row_filter::caller<84, T, D, BrdRowReflect101>,
                row_filter::caller<85, T, D, BrdRowReflect101>,
                row_filter::caller<86, T, D, BrdRowReflect101>,
                row_filter::caller<87, T, D, BrdRowReflect101>,
                row_filter::caller<88, T, D, BrdRowReflect101>,
                row_filter::caller<89, T, D, BrdRowReflect101>,
                row_filter::caller<90, T, D, BrdRowReflect101>,
                row_filter::caller<91, T, D, BrdRowReflect101>,
                row_filter::caller<92, T, D, BrdRowReflect101>,
                row_filter::caller<93, T, D, BrdRowReflect101>,
                row_filter::caller<94, T, D, BrdRowReflect101>,
                row_filter::caller<95, T, D, BrdRowReflect101>,
                row_filter::caller<96, T, D, BrdRowReflect101>,
                row_filter::caller<97, T, D, BrdRowReflect101>,
                row_filter::caller<98, T, D, BrdRowReflect101>,
                row_filter::caller<99, T, D, BrdRowReflect101>,
                row_filter::caller<100, T, D, BrdRowReflect101>,
                row_filter::caller<101, T, D, BrdRowReflect101>,
                row_filter::caller<102, T, D, BrdRowReflect101>,
                row_filter::caller<103, T, D, BrdRowReflect101>,
                row_filter::caller<104, T, D, BrdRowReflect101>,
                row_filter::caller<105, T, D, BrdRowReflect101>,
                row_filter::caller<106, T, D, BrdRowReflect101>,
                row_filter::caller<107, T, D, BrdRowReflect101>,
                row_filter::caller<108, T, D, BrdRowReflect101>,
                row_filter::caller<109, T, D, BrdRowReflect101>,
                row_filter::caller<110, T, D, BrdRowReflect101>,
                row_filter::caller<111, T, D, BrdRowReflect101>,
                row_filter::caller<112, T, D, BrdRowReflect101>,
                row_filter::caller<113, T, D, BrdRowReflect101>,
                row_filter::caller<114, T, D, BrdRowReflect101>,
                row_filter::caller<115, T, D, BrdRowReflect101>,
                row_filter::caller<116, T, D, BrdRowReflect101>,
                row_filter::caller<117, T, D, BrdRowReflect101>,
                row_filter::caller<118, T, D, BrdRowReflect101>,
                row_filter::caller<119, T, D, BrdRowReflect101>,
                row_filter::caller<120, T, D, BrdRowReflect101>,
                row_filter::caller<121, T, D, BrdRowReflect101>,
                row_filter::caller<122, T, D, BrdRowReflect101>,
                row_filter::caller<123, T, D, BrdRowReflect101>,
                row_filter::caller<124, T, D, BrdRowReflect101>,
                row_filter::caller<125, T, D, BrdRowReflect101>,
                row_filter::caller<126, T, D, BrdRowReflect101>,
                row_filter::caller<127, T, D, BrdRowReflect101>,
                row_filter::caller<128, T, D, BrdRowReflect101>
            }
        };

        if (stream == 0)
            cudaSafeCall( cudaMemcpyToSymbol(row_filter::c_kernel, kernel, ksize * sizeof(float), 0, cudaMemcpyDeviceToDevice) );
        else
            cudaSafeCall( cudaMemcpyToSymbolAsync(row_filter::c_kernel, kernel, ksize * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream) );

        callers[brd_type][ksize]((PtrStepSz<T>)src, (PtrStepSz<D>)dst, anchor, cc, stream);
    }
}
