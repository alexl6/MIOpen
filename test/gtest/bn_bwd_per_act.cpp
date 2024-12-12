#include "get_handle.hpp"
#include "random.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>

// #include <src/kernels/batchnorm_functions.hpp>
#include <iostream>
#include <string>
#include <vector>

// Quick hack to keep vscode happy, should probably get rid of this
#ifndef FP_TYPE
#define FP_TYPE float
#define FP_TYPE_PREC float
#endif


namespace {

struct BnBWDPATestCase
{
    int N;
    int C;
    int H;
    int W;
};

std::vector<BnBWDPATestCase> GetBnBwdPA_TestConfig()
{
    return {
        {8, 16, 32, 32},
        {4, 32, 28, 28},
        {2, 64, 8, 8}
    };
}

struct BnBwdPATest : public ::testing::TestWithParam<BnBWDPATestCase>
{
protected:
    // Test Functions
    void SetUp() override
    {
        bn_config = GetParam();
        
        N = bn_config.N;
        C = bn_config.C;
        H = bn_config.H;
        W = bn_config.W;

        size_t in_size = N * C * H * W;
        size_t per_act_size = C * H * W;

        // Adjust input vector size
        // in_host = tesnro<FP_TYPE>{N, C, H, W};
        // dy_host.resize(in_size);
        // scale_host.resize(per_act_size);
        // savedMean_host.resize(per_act_size);
        // savedInvVariance_host.resize(per_act_size);

        in_host = tensor<FP_TYPE>{N, C, H, W};
        dy_host = tensor<FP_TYPE>{N, C, H, W};
        scale_host = tensor<FP_TYPE_PREC>{1, C, H, W};
        savedMean_host = tensor<FP_TYPE_PREC>{1, C, H, W};
        savedInvVariance_host = tensor<FP_TYPE_PREC>{1, C, H, W};

        // Adjust output vector size
        // dx_host_ocl.resize(in_size);
        // dscale_host_ocl.resize(per_act_size);
        // dbias_host_ocl.resize(per_act_size);

        // dx_host_hip.resize(in_size);
        // dscale_host_hip.resize(per_act_size);
        // dbias_host_hip.resize(per_act_size);

        dx_host_ocl = tensor<FP_TYPE>{N, C, H, W};
        dscale_host_ocl = tensor<FP_TYPE_PREC>{1, C, H, W};
        dbias_host_ocl = tensor<FP_TYPE_PREC>{1, C, H, W};

        dx_host_hip = tensor<FP_TYPE>{N, C, H, W};
        dscale_host_hip = tensor<FP_TYPE_PREC>{1, C, H, W};
        dbias_host_hip = tensor<FP_TYPE_PREC>{1, C, H, W};

        // Generate some test data (not sure if this is any good) ⚠️
        for (size_t i = 0; i < in_size; ++i)
        {
            in_host[i] = (i % 255) / 255.0f;
            dy_host[i] = (i % 127) / 127.0f;
        }

        for (size_t i = 0; i < per_act_size; ++i)
        {
            scale_host[i] = (1.0f + (i % 10) * 0.1f);
            savedMean_host[i] = (1.0f + (i % 10) * 0.1f);
            savedInvVariance_host[i] = (1.0f + (i % 10) * 0.1f);
        }

    }

    void RunOCLKernel()
    {
        auto&& handle = get_handle();

        // Write generated data to input tensors
        // might need to pre-allocate with something like the line below first????? ⚠️
        // in_dev = tensor<FP_TYPE>{in_size};
        in_dev = handle.Write(in_host.data);
        dy_dev = handle.Write(dy_host.data);
        scale_dev = handle.Write(scale_host.data);

        // Allocate and initialize additional input tensors
        // Might need to reset in_dev, dy_dev, scale_dev here ⚠️
        savedMean_dev = handle.Write(savedMean_host.data);
        savedInvVar_dev = handle.Write(savedInvVariance_host.data);

        // Probably need to pre-alclearlocate output tensors
        dx_dev = handle.Write(dx_host_ocl.data);
        dscale_dev = handle.Write(dscale_host_ocl.data);
        dbias_dev = handle.Write(dbias_host_ocl.data);

        int in_nstride = C * H * W;
        int in_cstride = H * W;

        // Adjust global/local sizes as needed for your kernel
        // The original OCL kernel likely launched with these dims:
        // global: (C, H*W), local: (some factors)
        // For simplicity, let's assume a workgroup of (1, 1) and global (C, H*W).
        // Adjust as per the original kernel invocation if known.
        std::vector<size_t> vgd = {static_cast<size_t>(C), static_cast<size_t>(H*W), 1};
        std::vector<size_t> vld = {1, 1, 1};

        std::string program_name = "MIOpenBatchNormBwdPerAct.cl";
        std::string kernel_name  = "MIOpenBatchNormBwdPerActivationSaved";
        std::string network_config = "bn_bwd_pa_ocl_test";

        miopen::KernelBuildParameters options{};
        std::string params = options.GenerateFor(miopen::kbp::OpenCL{});


        // Call handle.AddKernel()
        auto k = handle.AddKernel("bn_bwd_pa_ocl",
                                  network_config,
                                  program_name,
                                  kernel_name,
                                  vld, vgd, params);

        k(in_dev.get(),
          dy_dev.get(),
          static_cast<unsigned int>(N),
          static_cast<unsigned int>(in_nstride),
          static_cast<unsigned int>(in_cstride),
          dx_dev.get(),
          scale_dev.get(),
          dscale_dev.get(),
          dbias_dev.get(),
          savedMean_dev.get(),
          savedInvVar_dev.get());


        // Read results back
        dx_host_ocl.data     = handle.Read<FP_TYPE>(dx_dev, dx_host_ocl.data.size());
        dscale_host_ocl.data = handle.Read<FP_TYPE_PREC>(dscale_dev, dscale_host_ocl.data.size());
        dbias_host_ocl.data  = handle.Read<FP_TYPE_PREC>(dbias_dev, dbias_host_ocl.data.size());
    }

    void RunHIPKernel()
    {
        // Todo
        // auto&& handle = get_handle();
        return;
    }

    void VerifyResults()
    {
        // Todo
        std::cout << "dx_host_ocl: ";
        for(const auto& v : dx_host_ocl) {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        std::cout << "dscale_host_ocl: ";
        for(const auto& v : dscale_host_ocl) {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        std::cout << "dbias_host_ocl: ";
        for(const auto& v : dbias_host_ocl) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    // Variables
    int N, C, H, W;
    BnBWDPATestCase bn_config;
    
    // Input tensors
    tensor<FP_TYPE> in_host;
    tensor<FP_TYPE> dy_host;
    tensor<FP_TYPE_PREC> scale_host;
    // Input vector for saved versions
    tensor<FP_TYPE_PREC> savedMean_host;
    tensor<FP_TYPE_PREC> savedInvVariance_host;

    // Output vectors (dx_out, delta_scale, delta_bias)
    tensor<FP_TYPE> dx_host_ocl;
    tensor<FP_TYPE_PREC> dscale_host_ocl;
    tensor<FP_TYPE_PREC> dbias_host_ocl;

    tensor<FP_TYPE> dx_host_hip;
    tensor<FP_TYPE_PREC> dscale_host_hip;
    tensor<FP_TYPE_PREC> dbias_host_hip;

    // Tensors for corresnpoding vectors
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr savedMean_dev;
    miopen::Allocator::ManageDataPtr savedInvVar_dev;

    miopen::Allocator::ManageDataPtr dx_dev;
    miopen::Allocator::ManageDataPtr dscale_dev;
    miopen::Allocator::ManageDataPtr dbias_dev;
};

TEST_P(BnBwdPATest, BackwardBatchNormPeractTest)
{
    RunOCLKernel();
    // RunHIPKernel();
    VerifyResults();
}

INSTANTIATE_TEST_SUITE_P(Smoke, BnBwdPATest, testing::ValuesIn(GetBnBwdPA_TestConfig()));

}   // anonymous namespace
