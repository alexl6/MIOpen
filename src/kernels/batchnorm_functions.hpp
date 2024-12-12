
#include "bfloat16_dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#define _FLOAT_ACCUM float
#if MIOPEN_USE_FP16 == 1
#define MIO_BN_NODPP 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FP_TYPE half
#define FP_TYPE_PREC half
#define EPSILON static_cast<FP_TYPE>(0.0001)
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define FP_TYPE float
#define FP_TYPE_PREC float
#define EPSILON static_cast<FP_TYPE>(0.000001)
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif
#if MIOPEN_USE_FPMIX == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FP_TYPE half

#ifdef MIO_BN_NODPP
#undef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#endif

#ifdef FP_TYPE_PREC
#undef FP_TYPE_PREC
#endif
#define FP_TYPE_PREC float

#ifdef EPSILON
#undef EPSILON
#endif
#define EPSILON static_cast<FP_TYPE>(0.000001)

#endif

#if MIOPEN_USE_BFPMIX == 1
#define FP_TYPE ushort

#ifdef MIO_BN_NODPP
#undef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#endif

#ifdef FP_TYPE_PREC
#undef FP_TYPE_PREC
#endif
#define FP_TYPE_PREC float

#ifdef EPSILON
#undef EPSILON
#endif
#define EPSILON static_cast<FP_TYPE_PREC>(0.000001)

#define FLOAT2FLOATPREC(x) (bfloat16_to_float(x))
#define FLOATPREC2FLOAT(x) (float_to_bfloat16(x))
#define FLOAT2ACCUM(x) (FLOAT2FLOATPREC(x))
#define ACCUM2FLOAT(x) (FLOATPREC2FLOAT(x))

#else

#define FLOAT2FLOATPREC(x) ((FP_TYPE_PREC)(x))
#define FLOATPREC2FLOAT(x) ((FP_TYPE)(x))
#define FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define ACCUM2FLOAT(x) ((FP_TYPE)(x))
#endif

#define _FLOAT2 PPCAT(FP_TYPE, TWO)
#define _FLOAT4 PPCAT(FP_TYPE, FOUR)
#define _FLOAT8 PPCAT(FP_TYPE, EIGHT)
#define _AS_FLOAT PPCAT(as_, FP_TYPE)

#define _FLOAT_PREC4 PPCAT(FP_TYPE_PREC, FOUR)

#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXN
#define MIO_BN_LOOP_UNROLL_MAXN 768
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXHW
#define MIO_BN_LOOP_UNROLL_MAXHW 2500
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 255
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif

// TODO: Spaghetti code!!!
// MIOPEN_USE_AMDGCN may be defined before this header.
#ifndef MIOPEN_USE_AMDGCN
#if defined(__AMDGCN__) &&                           \
    !((defined(MIO_BN_GFX103X) && MIO_BN_GFX103X) || \
      (defined(MIO_BN_GFX110X) && MIO_BN_GFX110X) || (defined(MIO_BN_GFX120X) && MIO_BN_GFX120X))
#define MIOPEN_USE_AMDGCN 1
#else
#define MIOPEN_USE_AMDGCN 0
#endif
#endif

// MIOPEN_USE_AMDGCN is guaranteed to be defined at this point.

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif(MIO_BN_NODPP == 1 && MIO_BN_VARIANT != 0)
#undef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 0
#endif

#ifndef MIO_SAVE_MEAN_VARIANCE
#define MIO_SAVE_MEAN_VARIANCE 0
#endif

#ifndef MIO_RUNNING_RESULT
#define MIO_RUNNING_RESULT 0
#endif

#ifndef MIO_BN_GFX103X
#define MIO_BN_GFX103X 0
#endif

#ifndef MIO_BN_GFX110X
#define MIO_BN_GFX110X 0
#endif

#ifndef MIO_BN_GFX120X
#define MIO_BN_GFX120X 0
#endif

#define UNUSED __attribute__((__unused__))

#if(MIO_BN_VARIANT != 4)
static inline void running_stash(global FP_TYPE_PREC* resultRunningMean,
                                 global FP_TYPE_PREC* resultRunningVariance,
                                 double expAvgFactor,
                                 _FLOAT_ACCUM mean,
                                 _FLOAT_ACCUM variance,
                                 uint channel)
{
    _FLOAT_ACCUM pvt_runMean = (_FLOAT_ACCUM)(*(resultRunningMean + channel));
    _FLOAT_ACCUM pvt_newRunMean =
        mad((_FLOAT_ACCUM)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
    resultRunningMean[channel] =
        (FP_TYPE_PREC)mad(mean, (_FLOAT_ACCUM)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
    const _FLOAT_ACCUM adjust =
        (_FLOAT_ACCUM)((MIO_BN_NHW == 1)
                           ? variance
                           : variance * ((_FLOAT_ACCUM)MIO_BN_NHW /
                                         ((_FLOAT_ACCUM)MIO_BN_NHW - (_FLOAT_ACCUM)1.0)));
    resultRunningVariance[channel] =
        (FP_TYPE_PREC)((1 - (_FLOAT_ACCUM)expAvgFactor) *
                          (_FLOAT_ACCUM)(*(resultRunningVariance + channel)) +
                      (_FLOAT_ACCUM)expAvgFactor * adjust);
}

static inline void running_stash_pa(global FP_TYPE_PREC* resultRunningMean,
                                    global FP_TYPE_PREC* resultRunningVariance,
                                    double expAvgFactor,
                                    _FLOAT_ACCUM mean,
                                    _FLOAT_ACCUM variance,
                                    uint index)
{
    FP_TYPE_PREC N              = (FP_TYPE_PREC)MIO_BN_N;
    FP_TYPE_PREC pvt_runMean    = *(resultRunningMean + index); // previous: oldRunMean
    FP_TYPE_PREC pvt_newRunMean = mad((FP_TYPE_PREC)-expAvgFactor,
                                     pvt_runMean,
                                     pvt_runMean); // tmp = oldRunMean*(1-factor)

    resultRunningMean[index] = mad((FP_TYPE_PREC)mean,
                                   (FP_TYPE_PREC)expAvgFactor,
                                   pvt_newRunMean); // newMean*factor + tmp

    const FP_TYPE_PREC adjust = (MIO_BN_N == 1) ? variance : variance * (N / (N - 1.0));
    resultRunningVariance[index] =
        (1 - (FP_TYPE_PREC)expAvgFactor) * *(resultRunningVariance + index) +
        (FP_TYPE_PREC)expAvgFactor * adjust;
}

#else

static inline void running_stash_dyn(global FP_TYPE_PREC* resultRunningMean,
                                     global FP_TYPE_PREC* resultRunningVariance,
                                     double expAvgFactor,
                                     _FLOAT_ACCUM mean,
                                     _FLOAT_ACCUM variance,
                                     uint channel,
                                     _FLOAT_ACCUM inhw)
{
    _FLOAT_ACCUM pvt_runMean = (_FLOAT_ACCUM)(*(resultRunningMean + channel));
    _FLOAT_ACCUM pvt_newRunMean =
        mad((_FLOAT_ACCUM)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
    resultRunningMean[channel] =
        (FP_TYPE_PREC)mad(mean, (_FLOAT_ACCUM)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
    const _FLOAT_ACCUM adjust =
        (_FLOAT_ACCUM)((inhw == 1) ? variance : variance * (1. / (1. - inhw)));
    resultRunningVariance[channel] =
        (FP_TYPE_PREC)((1 - (_FLOAT_ACCUM)expAvgFactor) *
                          (_FLOAT_ACCUM)(*(resultRunningVariance + channel)) +
                      (_FLOAT_ACCUM)expAvgFactor * adjust);
}
#endif

static inline void saved_stash(global FP_TYPE_PREC* resultSaveMean,
                               global FP_TYPE_PREC* resultSaveInvVariance,
                               _FLOAT_ACCUM mean,
                               _FLOAT_ACCUM invVariance,
                               uint channel)
{
    *(resultSaveMean + channel)        = (FP_TYPE_PREC)mean;
    *(resultSaveInvVariance + channel) = (FP_TYPE_PREC)invVariance;
}
