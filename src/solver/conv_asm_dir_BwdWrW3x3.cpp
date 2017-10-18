/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <sstream>
#include <unordered_map>
#include <limits>
#include <iterator>
#include <chrono>

#include "miopen/gcn_asm_utils.hpp"
#include "miopen/env.hpp"
#include "miopen/logger.hpp"
#include "miopen/handle.hpp"
#include "miopen/solver.hpp"

#define MIOPEN_GCNASM3X3WRW_SEARCH_LWC_FIXED 1
#define MIOPEN_GCNASM3X3WRW_INIT_OUTPUT_BUFFER 0

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS)

#define MIOPEN_LOG_E(...) MIOPEN_LOG(miopen::LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_W(...) MIOPEN_LOG(miopen::LoggingLevel::Warning, __VA_ARGS__)
#define MIOPEN_LOG_I(...) MIOPEN_LOG(miopen::LoggingLevel::Info, __VA_ARGS__)
#define MIOPEN_LOG_I2(...) MIOPEN_LOG(miopen::LoggingLevel::Info2, __VA_ARGS__)

namespace miopen {

class Timer
{
    public:
    Timer(){};
    void start() { st = std::chrono::steady_clock::now(); }
    float elapsed_ms()
    {
        capture();
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
            .count();
    }

    private:
    void capture() { et = std::chrono::steady_clock::now(); }
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

namespace solver {

static bool IsReverseInOutAllowed(const ConvolutionContext& config)
{
    return config.kernel_stride0 == 1 && config.kernel_stride1 == 1;
}

class PerformanceConfigAsmDirect3x3WrW : public PerformanceConfig
{
    int limit_wave_cnt;   // [0..9]
    int reverse_inout;    // [0..1], 1 is allowed for stride=1x1 only.
    int chunk_size;       // {16,8}, Smaller values increase register pressure.
    int k_per_wave;       // {1,2,4,8} && ((chunk_size * k_per_wave) <= 64).
                          // Higher values increase register pressure.
    int pipe_lines_depth; // [1..16] && (pipe_lines_depth <= img_h).
                          // Higher values increase register pressure.
    int n_per_group;      // [1..8] && (n_per_group <= batch_size).

    bool IsValidRange() const;
    void EuristicInit(const ConvolutionContext& config);

    std::string ToString() const;

    int GetCPerWave() const
    {
        assert(chunk_size != 0);
        return 64 / chunk_size;
    }

    PerformanceConfigAsmDirect3x3WrW(int limit_wave_cnt_,
                                     int reverse_inout_,
                                     int chunk_size_,
                                     int k_per_wave_,
                                     int pipe_lines_depth_,
                                     int n_per_group_)
        : limit_wave_cnt(limit_wave_cnt_),
          reverse_inout(reverse_inout_),
          chunk_size(chunk_size_),
          k_per_wave(k_per_wave_),
          pipe_lines_depth(pipe_lines_depth_),
          n_per_group(n_per_group_)
    {
    }

    bool IsValid(const ConvolutionContext& config) const;

    bool IsEqual(const PerformanceConfigAsmDirect3x3WrW& other) const
    {
        return limit_wave_cnt == other.limit_wave_cnt && reverse_inout == other.reverse_inout &&
               chunk_size == other.chunk_size && k_per_wave == other.k_per_wave &&
               pipe_lines_depth == other.pipe_lines_depth && n_per_group == other.n_per_group;
    }

    public:
    PerformanceConfigAsmDirect3x3WrW() : PerformanceConfigAsmDirect3x3WrW(-1, -1, -1, -1, -1, -1) {}
    void Serialize(std::ostream&) const override;
    bool Deserialize(const std::string& str) override;

    friend class ConvAsmBwdWrW3x3;
    friend class VirtualIterator;
};

class VirtualIterator;

/// This container (together with corresponding iterator) provies access
/// to a set of performance configs which, by definition, must be
/// suitable for the given problem config.
///
/// It does not hold values themselves as these would take too much memory
/// but can be easily computed (and that is what iterator actually does).
///
/// The container holds problem config information instead. This info
/// is required for advancing the iterator to the next valid configuration.
/// Also it provides const_iterator, begin() and end().
class VirtualContainer
{
    // Valid iterator shall denote an element of a container.
    const ConvolutionContext& config;
    friend class VirtualIterator;

    public:
    typedef VirtualIterator const_iterator;
    VirtualContainer(const ConvolutionContext& config_) : config(config_) {}
    VirtualIterator begin() const;
    VirtualIterator end() const;
};

// Iterator shall advance to the next valid config, i.e. the one which
// satisfies PerformanceConfig.IsValid(ProblemConfig)
class VirtualIterator
    : public std::iterator<std::input_iterator_tag, PerformanceConfigAsmDirect3x3WrW>
{
    value_type v; // PerformanceConfigAsmDirect3x3WrW
    const VirtualContainer* container;

    static const value_type minValue;
    static const value_type outOfRangeValue;

    /// Implements begin()
    VirtualIterator(const VirtualContainer* container_) : v(minValue), container(container_)
    {
        if(!IsValid())
            Next();
    }
    friend class VirtualContainer; // Passes itself to private ctor in order to construct begin().
    void Next();
    bool IsValid();

    public:
    /// Implementes end() and also serves as a default ctor.
    VirtualIterator() : v(outOfRangeValue), container(nullptr) {}
    VirtualIterator(const VirtualIterator& it) : v(it.v), container(it.container) {}

    bool operator!=(VirtualIterator const& other) const;
    const value_type& operator*() const { return v; }
    const value_type* operator->() const { return &v; }
    VirtualIterator& operator++()
    {
        Next();
        return *this;
    }
};

inline VirtualIterator VirtualContainer::begin() const { return VirtualIterator(this); }

inline VirtualIterator VirtualContainer::end() const { return VirtualIterator(); }

const VirtualIterator::value_type VirtualIterator::minValue =
    PerformanceConfigAsmDirect3x3WrW(0, 0, 8, 1, 1, 1);
const VirtualIterator::value_type VirtualIterator::outOfRangeValue =
    PerformanceConfigAsmDirect3x3WrW(-1, -1, -1, -1, -1, -1);

inline bool VirtualIterator::IsValid()
{
    if(!container)
        return false;
    return v.IsValid(container->config);
}

inline bool VirtualIterator::operator!=(VirtualIterator const& other) const
{
    return !(v.IsEqual(other.v) && container == other.container);
}

void VirtualIterator::Next()
{
    if(container == nullptr)
    {
        v = outOfRangeValue;
        return;
    }
    do
    {
        // Increment with wrap-around:
        do
        {
#if MIOPEN_GCNASM3X3WRW_SEARCH_LWC_FIXED == 0
            // (0 <= limit_wave_cnt && limit_wave_cnt <= 9)
            if(++v.limit_wave_cnt <= 9)
                break;
#endif
            v.limit_wave_cnt = 0;
            // (0 <= reverse_inout && reverse_inout <= 1)
            if(++v.reverse_inout <= 1)
                break;
            v.reverse_inout = 0;
            // (8 == chunk_size || 16 == chunk_size)
            if((v.chunk_size += 8) <= 16)
                break;
            v.chunk_size = 8;
            // (1 == k_per_wave || 2 == k_per_wave || 4 == k_per_wave || 8 == k_per_wave)
            if(1 == v.k_per_wave)
            {
                v.k_per_wave = 2;
                break;
            }
            if(2 == v.k_per_wave)
            {
                v.k_per_wave = 4;
                break;
            }
            if(4 == v.k_per_wave)
            {
                v.k_per_wave = 8;
                break;
            }
            v.k_per_wave = 1;
            // (1 <= pipe_lines_depth && pipe_lines_depth <= 16)
            if(++v.pipe_lines_depth <= 16)
                break;
            v.pipe_lines_depth = 1;
            // (1 <= n_per_group && n_per_group <= 8);
            if(++v.n_per_group <= 8)
                break;
            v.n_per_group = 1;
            // All the fields (components) of performance confic have wrapped around.
            // The next one is not the min (in the allowed range) but a one beyond the end.
            // Iterator is useless from now.
            v         = outOfRangeValue;
            container = nullptr;
            return;
        } while(false);
    } while(!IsValid());
}

bool PerformanceConfigAsmDirect3x3WrW::IsValidRange() const
{
    return (0 <= limit_wave_cnt && limit_wave_cnt <= 9) &&
           (0 <= reverse_inout && reverse_inout <= 1) && (8 == chunk_size || 16 == chunk_size) &&
           (1 == k_per_wave || 2 == k_per_wave || 4 == k_per_wave || 8 == k_per_wave) &&
           (1 <= pipe_lines_depth && pipe_lines_depth <= 16) &&
           (1 <= n_per_group && n_per_group <= 8);
}

bool PerformanceConfigAsmDirect3x3WrW::IsValid(const ConvolutionContext& config) const
{
    if(!IsValidRange())
        return false;
    assert(chunk_size != 0);
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        return false;
    if((reverse_inout ? config.n_inputs : config.n_outputs) % GetCPerWave() != 0)
        return false;
    if(!(chunk_size * k_per_wave <= 64))
        return false;
    if((reverse_inout ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        return false;
    if(!(n_per_group <= config.batch_sz))
        return false;
    if(!(1 <= pipe_lines_depth && pipe_lines_depth <= std::min(config.out_height, 16)))
        return false;
    if(reverse_inout && !IsReverseInOutAllowed(config))
        return false;

    {
        const int accums_cnt =
            (config.kernel_size0 * config.kernel_size1 * GetCPerWave() * k_per_wave * chunk_size) /
            64;
        assert(chunk_size);
        int gprs_per_line_in = (config.out_width + chunk_size - 1) / chunk_size;
        if(chunk_size != 16)
        {
            assert(chunk_size - config.pad0);
            gprs_per_line_in =
                (config.out_width + chunk_size - config.pad0 - 1) / (chunk_size - config.pad0);
        }
        assert(config.kernel_stride0);
        gprs_per_line_in += gprs_per_line_in % config.kernel_stride0;
        const int gprs_per_line_out =
            (gprs_per_line_in > 1) ? gprs_per_line_in / config.kernel_stride0 : 1;

        const int lines_in = pipe_lines_depth + config.kernel_size1 - 1;
        assert(config.kernel_stride1);
        const int lines_out =
            (pipe_lines_depth + config.kernel_stride1 - 1) / config.kernel_stride1;
        const int vgprs =
            accums_cnt + lines_in * gprs_per_line_in + lines_out * gprs_per_line_out + 6;
        if(!(vgprs <= 256))
            return false;
        if(n_per_group > 4)
            if(!(vgprs <= 128))
                return false;
        if(limit_wave_cnt != 0 && limit_wave_cnt * 4 < n_per_group)
            return false;
        const int lds_size = (n_per_group - 1) * 64 /*wavesize*/ * sizeof(float) * accums_cnt;
        if(!(lds_size <= 65536))
            return false;

        const int unroll_factor = pipe_lines_depth * (pipe_lines_depth + 2);
        const int steps         = std::max(0, config.out_height - 1 - pipe_lines_depth);
        assert(unroll_factor);
        const int loops   = pipe_lines_depth + unroll_factor + steps % unroll_factor + 1;
        const int m_instr = 3 + (gprs_per_line_in + 3) / 4;
        const int v_instr =
            (k_per_wave * config.kernel_size1 * gprs_per_line_out * config.kernel_size0 * 4) / 3;
        const int total = loops * (m_instr + v_instr); // instructions
        if(total >= 32000)                             // Estimation, a bit smaller than 32K.
            return false;
    }
    return true;
}

void PerformanceConfigAsmDirect3x3WrW::EuristicInit(const ConvolutionContext& config)
{
    limit_wave_cnt = 0;

    chunk_size = (config.out_width < 48) ? 8 : 16;
    if((config.n_outputs % (64 / chunk_size) != 0) && (config.n_inputs % (64 / chunk_size) != 0))
        chunk_size = 16; // Fixup for correctness

    reverse_inout = 0;
    if(IsReverseInOutAllowed(config) && ((config.n_outputs % 4 != 0) || (config.out_width < 8)))
        reverse_inout = 1;

    const auto c_k = config.n_outputs * config.n_inputs; // C*K
    if(c_k < 256)
        k_per_wave = 1;
    else if(c_k < 16384)
        k_per_wave = 2;
    else // C*K >= 16k
        k_per_wave = ((chunk_size == 8) ? 2 : 4);
    while((reverse_inout ? config.n_outputs : config.n_inputs) % k_per_wave != 0)
        k_per_wave /= 2; // Fixup for correctness

    if(c_k <= 512)
        n_per_group = 8;
    else if(c_k <= 4096)
        n_per_group = 4;
    else if(c_k <= 8192)
        n_per_group = 2;
    else
        n_per_group = 1;
    if(n_per_group > config.batch_sz)
        n_per_group = config.batch_sz; // n_per_group should never be > batch size.
    if(config.out_width >= 256 &&
       n_per_group > 4) // when width >= 256, n_per_group should not be > 4.
        n_per_group = 4;

    pipe_lines_depth = (config.out_height <= 1) ? 1 : 2;
    if((config.out_height < 8) && (config.out_width < 64))
    {
        pipe_lines_depth = config.out_height; // Special case.
    }

    if(!IsValid(config))
    {
        MIOPEN_LOG_I("!IsValid(): " << ToString() << ". Conservative re-init...");
        limit_wave_cnt   = 0;
        reverse_inout    = 0;
        chunk_size       = 16; // CPerWave() = 4;
        k_per_wave       = 1;
        pipe_lines_depth = 2;
        n_per_group      = 1;
        if(config.n_outputs % 4 != 0)
        {
            /// (1) If reverse is Off, then both (C % c_per_wave) and (K % k_per_wave) must be 0.
            /// Toggling reverse swaps C and K in the condition above.
            /// (2) From the other hand, IsApplicable() ensures that either C or K is evenly
            /// divisable by 4.
            /// (3) We just set k_per_wave=1, c_per_wave=4. Therefore, (1) always can be satisfied
            /// here. If (C % c_per_wave) is not zero, just push reverse button so K and C will
            /// swap.
            ///
            /// \note C (input channels) resides in n_outputs, K (output channels) - in n_inputs,
            /// because that's how reverse convolutions are handled in MIOpen.
            reverse_inout = 1;
        }
        assert(IsValid(config));
    }
    MIOPEN_LOG_I(ToString());
}

static bool DeserializeField(const char separator, std::istream& from, int& to)
{
    std::string part;

    if(!std::getline(from, part, separator))
        return false;

    const auto start = part.c_str();
    char* end;
    to = std::strtol(start, &end, 10);
    return start != end;
}

void PerformanceConfigAsmDirect3x3WrW::Serialize(std::ostream& stream) const
{
    static const auto sep = ','; // clang-format off
    stream << limit_wave_cnt
        << sep << reverse_inout
        << sep << chunk_size
        << sep << k_per_wave
        << sep << pipe_lines_depth
        << sep << n_per_group; // clang-format on
}

bool PerformanceConfigAsmDirect3x3WrW::Deserialize(const std::string& str)
{
    PerformanceConfigAsmDirect3x3WrW out;
    {
        std::istringstream tmp(str);
        const auto ok = // clang-format off
            DeserializeField(',', tmp, out.limit_wave_cnt) &&
            DeserializeField(',', tmp, out.reverse_inout) &&
            DeserializeField(',', tmp, out.chunk_size) &&
            DeserializeField(',', tmp, out.k_per_wave) &&
            DeserializeField(',', tmp, out.pipe_lines_depth) &&
            DeserializeField(',', tmp, out.n_per_group); // clang-format on

        if(!ok)
            return false;
    }
    *this = out;
    return true;
}

std::string PerformanceConfigAsmDirect3x3WrW::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

std::unique_ptr<PerformanceConfig> ConvAsmBwdWrW3x3::PerformanceConfigImpl() const
{
    return make_unique<PerformanceConfigAsmDirect3x3WrW>();
}

void ConvAsmBwdWrW3x3::InitPerformanceConfigImpl(const ConvolutionContext& params,
                                                 PerformanceConfig& result) const
{
    std::string s;
    PerformanceConfigAsmDirect3x3WrW pp;
    const auto p_asciz = miopen::GetStringEnv(MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS{});
    if(p_asciz)
    {
        s = std::string(p_asciz);
    }
    if(!s.empty()) // Otherwise, nothing is set in env -> nothing to parse.
    {
        static const std::string h("MIOPEN_DEBUG_GCN_ASM_DIRECT_3X3WRW_PERF_VALS: ");
        if(!pp.Deserialize(s))
        {
            MIOPEN_THROW(h + "Bad format:" + s);
        }
        if(!pp.IsValid(params))
        {
            MIOPEN_THROW(h + "Out of range of invalid for the problem config:" + s);
        }
        MIOPEN_LOG_I("From env: " << pp.ToString());
    }
    else
    {
        pp.EuristicInit(params);
    }
    MIOPEN_LOG_I(pp.ToString());
    dynamic_cast<PerformanceConfigAsmDirect3x3WrW&>(result) = pp;
}

bool ConvAsmBwdWrW3x3::IsValidPerformanceConfigImpl(const ConvolutionContext& problem,
                                                    const PerformanceConfig& config) const
{
    const auto& c = dynamic_cast<const PerformanceConfigAsmDirect3x3WrW&>(config);
    return c.IsValidRange() && c.IsValid(problem);
}

bool ConvAsmBwdWrW3x3::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.assembler_available)
    {
        return false;
    }

    if(params.n_passes)
    {
        return false;
    }

    const std::string name = params.GetStream().GetDeviceName();
    if(name.find("gfx8") == std::string::npos && name.find("gfx9") == std::string::npos)
    {
        return false;
    }
    assert(params.weights_layout.length() == 0); // _weights_layout is not supported yet
    bool ok = params.pad0 == 1                   // -q  pad_w
              && params.pad1 == 1                // -p  pad_h
              && (params.kernel_stride0 <= 2)    // -u  stride_w
              && (params.kernel_stride1 <= 2)    // -v  stride_h
              && params.kernel_size0 == 3        // -x  S wei_w
              && params.kernel_size1 == 3        // -y  R wei_h
              && params.kernal_dilation0 == 1 && params.kernal_dilation1 == 1 && params.bias == 0 &&
              params.in_layout == "NCHW";
    // && _weights_layout == "KCHW"
    if(!ok)
    {
        return false; // Early exit to speed up the check.
    }
    // Check limits:
    const auto h_w     = static_cast<long>(params.out_height) * params.out_width;
    const auto r_s     = static_cast<long>(params.kernel_size1) * params.kernel_size0;
    const auto c_h_w   = static_cast<long>(params.n_outputs) * h_w;   // C*H*W
    const auto k_h_w   = static_cast<long>(params.n_inputs) * h_w;    // K*H*W
    const auto c_r_s   = static_cast<long>(params.n_outputs) * r_s;   // C*R*S
    const auto k_r_s   = static_cast<long>(params.n_inputs) * r_s;    // K*R*S
    const auto n_c_h_w = static_cast<long>(params.batch_sz) * c_h_w;  // N*C*H*W
    const auto n_k_h_w = static_cast<long>(params.batch_sz) * k_h_w;  // N*K*H*W
    const auto c_k_r_s = static_cast<long>(params.n_outputs) * k_r_s; // C*K*R*S
    // clang-format off
    ok = params.out_width > 0
         && params.out_width <= 512
         && (IsReverseInOutAllowed(params)
                ? ((params.n_outputs % 4 == 0) || (params.n_inputs % 4 == 0))
                : (params.n_outputs % 4 == 0))
         && params.out_height < std::pow(2, 16) // -H   H img_h
         && params.batch_sz < std::pow(2, 16)   // -n   N batch_size
         && params.n_outputs < std::pow(2, 16)  // -c   C input_channels
         && params.n_inputs < std::pow(2, 16)   // -k   K output_channels
         && c_h_w < std::pow(2, 22)
         && k_h_w < std::pow(2, 22)
         && c_r_s < std::pow(2, 22)
         && k_r_s < std::pow(2, 22)
         && n_c_h_w < std::pow(2, 29)
         && n_k_h_w < std::pow(2, 29)
         && c_k_r_s < std::pow(2, 29);                                    // clang-format on
    return ok;
}

bool ConvAsmBwdWrW3x3::IsFast(const ConvolutionContext&) const { return true; }

ConvSolution ConvAsmBwdWrW3x3::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfig& config) const
{
    ConvSolution result;
    std::ostringstream options;
    GenerateClangDefsym(options, "batch_size", params.batch_sz); // N
    GenerateClangDefsym(options, "img_h", params.out_height);    // H
    GenerateClangDefsym(options, "img_w", params.out_width);     // W
    // Note that params.n_outputs and params.n_inputs are swapped for backward convolutions.
    GenerateClangDefsym(options, "input_channels", params.n_outputs); // C
    GenerateClangDefsym(options, "output_channels", params.n_inputs); // K
    GenerateClangDefsym(options, "wei_h", params.kernel_size1);       // R
    GenerateClangDefsym(options, "wei_w", params.kernel_size0);       // S
    GenerateClangDefsym(options, "pad_h", params.pad1);
    GenerateClangDefsym(options, "pad_w", params.pad0);
    GenerateClangDefsym(options, "stride_h", params.kernel_stride1);
    GenerateClangDefsym(options, "stride_w", params.kernel_stride0);
    GenerateClangDefsym(options, "weights_layout", 0);
    GenerateClangDefsym(options, "reverse_weights", 0);
    // Perf tune:
    const auto& pp = dynamic_cast<const PerformanceConfigAsmDirect3x3WrW&>(config);
    GenerateClangDefsym(options, "limit_wave_cnt", pp.limit_wave_cnt);
    GenerateClangDefsym(options, "chunk_size", pp.chunk_size);
    GenerateClangDefsym(options, "c_per_wave", pp.GetCPerWave());
    GenerateClangDefsym(options, "k_per_wave", pp.k_per_wave);
    GenerateClangDefsym(options, "n_per_group", pp.n_per_group);
    GenerateClangDefsym(options, "pipe_lines_depth", pp.pipe_lines_depth);
    GenerateClangDefsym(options, "reverse_inout", pp.reverse_inout);
    // Debugging:
    GenerateClangDefsym(options, "enable_debug_output", 0);

    KernelInfo kernel;

    kernel.comp_options = options.str();

    kernel.l_wk.clear(); // workgroupsize
    kernel.l_wk.push_back(64 * pp.n_per_group);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.clear(); // gridsize
    kernel.g_wk.push_back(64 * pp.n_per_group);

    if(pp.reverse_inout == 0)
    {
        kernel.g_wk.push_back(params.n_outputs / pp.GetCPerWave());
        kernel.g_wk.push_back(params.n_inputs / pp.k_per_wave);
    }
    else
    {
        kernel.g_wk.push_back(params.n_outputs / pp.k_per_wave);
        kernel.g_wk.push_back(params.n_inputs / pp.GetCPerWave());
    }

    kernel.kernel_file = "conv3x3wrw.s";
    kernel.kernel_name = "gcnAsmConv3x3WrW";

    result.construction_params.push_back(kernel);
    result.workspce_sz = 0;
    return result;
}

int ConvAsmBwdWrW3x3::Measure(miopen::Handle& profile_h,
                              Data_t bot_ocl_buf,
                              Data_t top_ocl_buf,
                              Data_t wei_ocl_buf,
                              double& processing_time,
                              const ConvolutionContext& params,
                              const PerformanceConfig& config) const
{
    ConvSolution solution = GetSolution(params, config);
    if(!solution.Succeeded())
    {
        return 1;
    }
    const KernelInfo k_info = solution.construction_params[0];
#ifdef NDEBUG
    try
#endif
    {
        processing_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.GetKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);
        int unused       = 0;
        int* return_addr = nullptr;
        int n_groups =
            static_cast<int>(params.GetStream().GetMaxComputeUnits()); // kernel needs int32

        kernel(params.batch_sz,   // N
               params.n_outputs,  // C
               params.out_height, // H
               params.out_width,  // W
               params.n_inputs,   // K
               n_groups,          // n_groups
               unused,
               unused,
               top_ocl_buf,
               wei_ocl_buf,
               bot_ocl_buf,
               return_addr);
        processing_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception&)
    {
        return -1;
    }
#endif
    return 0;
}

static void
InitVectorRandomly(std::vector<float>& vec, const double offset = 0.0, const double factor = 1.0)
{
    float* p = vec.data();
    for(int i = 0; i < vec.size(); ++i)
        *p++ = static_cast<float>((rand() * (1.0 / RAND_MAX) + offset) * factor);
}

class HeartBeat
{
    size_t n_within_beat;
    size_t n_best;
    double best_time; // within beat
    double elapsed_cumulative;
    miopen::Timer timer;
    PerformanceConfigAsmDirect3x3WrW best_config;

    void Continue()
    {
        best_time     = std::numeric_limits<float>::max();
        n_within_beat = 0;
        timer.start();
    }

    public:
    HeartBeat() : n_within_beat(0), n_best(0), best_time(0.0) {}

    void Start()
    {
        elapsed_cumulative = 0;
        best_config        = PerformanceConfigAsmDirect3x3WrW();
        Continue();
    }

    void Monitor(const double recent_time,
                 const size_t n_recent,
                 const double total_best,
                 size_t n_failed,
                 size_t n_total,
                 const PerformanceConfigAsmDirect3x3WrW& recent_config)
    {
        ++n_within_beat;
        if(recent_time < best_time)
        {
            best_time   = recent_time;
            n_best      = n_recent;
            best_config = recent_config;
        }
        const float elapsed = timer.elapsed_ms();
        if(elapsed > 3000)
        {
            elapsed_cumulative += elapsed;
            const float eta_sec =
                n_recent ? ((n_total - n_recent) * (elapsed_cumulative / n_recent) / 1000)
                         : 0; // paraniod
            MIOPEN_LOG_W(n_recent << '/' << n_failed << '/' << n_total << " " << total_best
                                  << ",  best within recent "
                                  << n_within_beat
                                  << ": "
                                  << best_time
                                  << " #"
                                  << n_best
                                  << " "
                                  << best_config
                                  << ", ETA:"
                                  << eta_sec
                                  << " sec.");
            Continue();
        }
    }
};

bool ConvAsmBwdWrW3x3::Search(const ConvolutionContext& params, PerformanceConfig& config) const
{
    miopen::Handle profile_h;
    profile_h.EnableProfiling(true);

    // Allocate & init input buffers.
    std::vector<float> bot(params.bot_sz / sizeof(float));
    InitVectorRandomly(bot);
    auto bot_ocl_buf = profile_h.Write(bot);

    std::vector<float> top(params.top_sz / sizeof(float));
    InitVectorRandomly(top);
    auto top_ocl_buf = profile_h.Write(top);

    // Allocate output buffer & prepare random initializer for it.
    std::vector<float> wei(params.weights_sz / sizeof(float));
    auto wei_ocl_buf = profile_h.Write(wei);
#if MIOPEN_GCNASM3X3WRW_INIT_OUTPUT_BUFFER // FIXME delete?
    std::vector<float> init_wei(wei.size());
    InitVectorRandomly(init_wei, -0.5, 0.001);
#endif

    // const
    int n_runs_total = 0;
    VirtualContainer configs(params);
    {
        for(const auto c : configs)
        {
            ++n_runs_total;
            (void)c;
        }
    }
    MIOPEN_LOG_W("Searching the best solution among " << n_runs_total << "...");
    auto& best = dynamic_cast<PerformanceConfigAsmDirect3x3WrW&>(config);
    best.EuristicInit(params);
    bool is_passed   = false;
    double best_time = std::numeric_limits<float>::max();
    size_t n_failed  = 0;
    size_t n_run     = 0;
    size_t n_best    = 0;
    HeartBeat heartBeat;
    heartBeat.Start();
    for(const auto c : configs)
    {
        double processing_time;
#if MIOPEN_GCNASM3X3WRW_INIT_OUTPUT_BUFFER // FIXME delete?
        profile_h.WriteTo(reinterpret_cast<const void*>(init_wei.data()),
                          wei_ocl_buf,
                          init_wei.size() * sizeof(decltype(init_wei)::value_type));
#endif

        MIOPEN_LOG_I2("#" << n_run << " (" << n_runs_total << ") " << c);
        const auto ret = Measure(profile_h,
                                 bot_ocl_buf.get(),
                                 top_ocl_buf.get(),
                                 wei_ocl_buf.get(),
                                 processing_time,
                                 params,
                                 c);
        if(ret == 0)
        {
            is_passed = true;
            if(processing_time < best_time)
            {
                MIOPEN_LOG_I("#" << n_run << "/" << n_failed << "/" << n_runs_total << " "
                                 << processing_time
                                 << " < "
                                 << best_time
                                 << ", new candidate: "
                                 << c);
                best      = c;
                best_time = processing_time;
                n_best    = n_run;
            }
        }
        else
        {
            MIOPEN_LOG_E("#" << n_run << " (" << n_runs_total << ") "
                             << " Failed rc="
                             << ret);
            ++n_failed;
        }
        heartBeat.Monitor(processing_time, n_run, best_time, n_failed, n_runs_total, c);
        ++n_run;
    }

    profile_h.EnableProfiling(false);
    MIOPEN_LOG_W("Done: " << n_runs_total << "/" << n_failed << "/" << n_runs_total << ", best #"
                          << n_best
                          << " "
                          << best_time
                          << " "
                          << best);
    return is_passed;
}

} // namespace solver
} // namespace miopen
