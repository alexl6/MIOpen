/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "batchnorm_functions.hpp"

extern "C" __global__ void
    MIOpenBatchNormBwdPerActivationSavedHIP(const FP_TYPE* in,
                                            const FP_TYPE* dy_in,
                                            unsigned int N,
                                            unsigned int in_nstride,
                                            unsigned int in_cstride,
                                            FP_TYPE* dx_out,
                                            const FP_TYPE_PREC* scale,
                                            FP_TYPE_PREC* delta_scale,
                                            FP_TYPE_PREC* delta_bias,
                                            const FP_TYPE_PREC* savedMean,
                                            const FP_TYPE_PREC*savedInvVariance)
{
    int xgid    = blockIdx.x * blockDim.x + threadIdx.x;
    int ygid    = blockIdx.y * blockDim.y + threadIdx.y;
    int yglb_sz = gridDim.y * blockDim.y;
    int cidx    = in_cstride * xgid;

    unsigned int index, adjIndex;
    FP_TYPE_PREC mean, invVar;
    FP_TYPE_PREC xhat, dyelem;
    FP_TYPE_PREC pvt_scale, pvt_dscale;
    FP_TYPE_PREC pvt_dbias;
    FP_TYPE_PREC tmp1, tmp2, tmp3;
    FP_TYPE_PREC dxhat    = static_cast<FP_TYPE_PREC>(0.);
    FP_TYPE_PREC dxhathat = static_cast<FP_TYPE_PREC>(0.);

    // move across the sections of an image in the mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        adjIndex   = cidx + idx;
        mean       = savedMean[adjIndex];
        invVar     = savedInvVariance[adjIndex];
        pvt_scale  = scale[adjIndex];
        pvt_dscale = static_cast<FP_TYPE_PREC>(0.);
        pvt_dbias  = static_cast<FP_TYPE_PREC>(0.);
        dxhat      = static_cast<FP_TYPE_PREC>(0.);
        dxhathat   = static_cast<FP_TYPE_PREC>(0.);

        for(int n = 0; n < N; n++)
        {
            // per (x-dims) channel load a block of data into LDS
            index  = in_nstride * n + adjIndex;
            xhat   = (static_cast<FP_TYPE_PREC>(in[index]) - mean) * invVar;
            dyelem = static_cast<FP_TYPE_PREC>(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = fma(xhat, dyelem, pvt_dscale);
            tmp1       = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = fma(tmp1, xhat, dxhathat);
        } // end for(n)

        for(int n = 0; n < N; n++)
        {
            index         = in_nstride * n + adjIndex;
            xhat          = (static_cast<FP_TYPE_PREC>(in[index]) - mean) * invVar;
            tmp1          = fma(xhat, dxhathat, dxhat);
            tmp2          = fma(static_cast<FP_TYPE_PREC>(N), static_cast<FP_TYPE_PREC>(dy_in[index]) * pvt_scale, -tmp1);
            tmp3          = invVar / static_cast<FP_TYPE_PREC>(N);
            dx_out[index] = static_cast<FP_TYPE>(tmp3 * tmp2);
        }
        // Write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    }
}

extern "C" __global__ void
    MIOpenBatchNormBwdPerActivationHIP(const FP_TYPE* in,
                                       const FP_TYPE* dy_in,
                                       unsigned int N,
                                       unsigned int in_nstride,
                                       unsigned int in_cstride,
                                       FP_TYPE* dx_out,
                                       const FP_TYPE_PREC* scale,
                                       FP_TYPE_PREC* delta_scale,
                                       FP_TYPE_PREC* delta_bias,
                                       double epsilon)
{

    int xgid    = blockIdx.x * blockDim.x + threadIdx.x;
    int ygid    = blockIdx.y * blockDim.y + threadIdx.y;
    int yglb_sz = gridDim.y * blockDim.y;
    int cidx    = in_cstride * xgid;

    unsigned int index, adjIndex;
    FP_TYPE_PREC mean, invVar;
    FP_TYPE_PREC xhat, dyelem;
    FP_TYPE_PREC pvt_scale, pvt_dscale;
    FP_TYPE_PREC pvt_dbias;
    _FLOAT_ACCUM tmp1, tmp2, tmp3;
    FP_TYPE_PREC dxhat     = static_cast<FP_TYPE_PREC>(0.);
    FP_TYPE_PREC dxhathat  = static_cast<FP_TYPE_PREC>(0.);
    _FLOAT_ACCUM variance = 0.;

    // move across the sections of the image mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        mean     = static_cast<FP_TYPE_PREC>(0.);
        adjIndex = cidx + idx; // gamma and beta tensor index
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = in_nstride * n + adjIndex;
            mean += static_cast<FP_TYPE_PREC>(in[index]);
        } // end for(n)
        mean /= static_cast<FP_TYPE_PREC>(N);
        variance = 0.;

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index             = in_nstride * n + adjIndex;
            FP_TYPE_PREC xdiff = static_cast<FP_TYPE_PREC>(in[index]) - mean;
            variance += (xdiff * xdiff);
        } // end for(n)
        variance /= static_cast<FP_TYPE_PREC>(N);
        invVar = rsqrt(variance + epsilon);

        pvt_scale  = *(scale + adjIndex);
        pvt_dscale = static_cast<FP_TYPE_PREC>(0.);
        pvt_dbias  = static_cast<FP_TYPE_PREC>(0.);
        dxhat      = static_cast<FP_TYPE_PREC>(0.);
        dxhathat   = static_cast<FP_TYPE_PREC>(0.);

        for(int n = 0; n < MIO_BN_N; n++)
        {
            // per (x-dims) channel load a block of data into LDS
            index  = in_nstride * n + adjIndex;
            xhat   = (static_cast<FP_TYPE_PREC>(in[index]) - mean) * invVar;
            dyelem = static_cast<FP_TYPE_PREC>(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = fma(xhat, dyelem, pvt_dscale);
            tmp1       = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = fma(tmp1, xhat, dxhathat);
        } // end for(n)

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index         = in_nstride * n + adjIndex;
            xhat          = (static_cast<FP_TYPE_PREC>(in[index]) - mean) * invVar;
            tmp1          = fma(xhat, dxhathat, dxhat);
            tmp2          = fma(static_cast<FP_TYPE_PREC>(N), static_cast<FP_TYPE_PREC>(dy_in[index]) * pvt_scale, -tmp1);
            tmp3          = invVar / (static_cast<FP_TYPE_PREC>(N));
            dx_out[index] = static_cast<FP_TYPE>(tmp3 * tmp2);
        }
        // Write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    } // end for(idx) //image mini_batch is processed
}

//================== END PER ACTIVATION ====================
