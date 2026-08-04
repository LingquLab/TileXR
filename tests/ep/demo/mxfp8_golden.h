#ifndef TILEXR_TESTS_EP_DEMO_MXFP8_GOLDEN_H
#define TILEXR_TESTS_EP_DEMO_MXFP8_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace TileXREpDemo {

constexpr std::size_t kMxfp8BlockSize = 32;

enum class Mxfp8Format : uint64_t {
    E4M3,
    E5M2,
};

struct Mxfp8Tensor {
    std::vector<uint8_t> elements;
    std::vector<uint8_t> scales;
    std::size_t scaleCountPerRow = 0;
};

inline std::size_t Mxfp8ScaleCountPerRow(std::size_t h)
{
    const std::size_t blockCount = (h + kMxfp8BlockSize - 1) / kMxfp8BlockSize;
    return blockCount + (blockCount & 1U);
}

inline float RoundToNearestEven(float value)
{
    const float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction < 0.5f) {
        return lower;
    }
    if (fraction > 0.5f) {
        return lower + 1.0f;
    }
    return std::fmod(lower, 2.0f) == 0.0f ? lower : lower + 1.0f;
}

inline uint8_t EncodeFp8(float value, Mxfp8Format format)
{
    const int exponentBits = format == Mxfp8Format::E4M3 ? 4 : 5;
    const int mantissaBits = format == Mxfp8Format::E4M3 ? 3 : 2;
    const int exponentBias = format == Mxfp8Format::E4M3 ? 7 : 15;
    const float maxFinite = format == Mxfp8Format::E4M3 ? 448.0f : 57344.0f;
    const uint8_t sign = std::signbit(value) ? 0x80U : 0U;
    float magnitude = std::min(std::fabs(value), maxFinite);
    if (magnitude == 0.0f) {
        return sign;
    }

    const int minNormalExponent = 1 - exponentBias;
    const float minNormal = std::ldexp(1.0f, minNormalExponent);
    int encodedExponent = 0;
    int encodedMantissa = 0;
    if (magnitude < minNormal) {
        const float subnormalStep = std::ldexp(1.0f, minNormalExponent - mantissaBits);
        encodedMantissa = static_cast<int>(RoundToNearestEven(magnitude / subnormalStep));
        if (encodedMantissa >= (1 << mantissaBits)) {
            encodedExponent = 1;
            encodedMantissa = 0;
        }
    } else {
        int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
        const float normalized = std::ldexp(magnitude, -exponent);
        encodedMantissa = static_cast<int>(RoundToNearestEven(
            (normalized - 1.0f) * static_cast<float>(1 << mantissaBits)));
        if (encodedMantissa == (1 << mantissaBits)) {
            ++exponent;
            encodedMantissa = 0;
        }
        encodedExponent = exponent + exponentBias;
    }
    const int exponentMask = (1 << exponentBits) - 1;
    return static_cast<uint8_t>(sign |
        (static_cast<uint8_t>(encodedExponent & exponentMask) << mantissaBits) |
        static_cast<uint8_t>(encodedMantissa));
}

inline float DecodeFp8(uint8_t value, Mxfp8Format format)
{
    const int mantissaBits = format == Mxfp8Format::E4M3 ? 3 : 2;
    const int exponentBits = format == Mxfp8Format::E4M3 ? 4 : 5;
    const int exponentBias = format == Mxfp8Format::E4M3 ? 7 : 15;
    const int exponentMask = (1 << exponentBits) - 1;
    const int mantissaMask = (1 << mantissaBits) - 1;
    const int exponent = (value >> mantissaBits) & exponentMask;
    const int mantissa = value & mantissaMask;
    const float sign = (value & 0x80U) == 0 ? 1.0f : -1.0f;

    if (exponent == 0) {
        if (mantissa == 0) {
            return std::copysign(0.0f, sign);
        }
        return sign * std::ldexp(static_cast<float>(mantissa),
            1 - exponentBias - mantissaBits);
    }
    if (format == Mxfp8Format::E5M2 && exponent == exponentMask) {
        return mantissa == 0 ? sign * std::numeric_limits<float>::infinity() :
            std::numeric_limits<float>::quiet_NaN();
    }
    if (format == Mxfp8Format::E4M3 && exponent == exponentMask && mantissa == mantissaMask) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) /
        static_cast<float>(1 << mantissaBits), exponent - exponentBias);
}

inline void QuantizeMxfp8Row(const float *input, std::size_t h, Mxfp8Format format,
    uint8_t *elements, uint8_t *scales)
{
    const std::size_t scaleCount = Mxfp8ScaleCountPerRow(h);
    std::fill(scales, scales + scaleCount, 0U);
    const int elementMaxExponent = format == Mxfp8Format::E4M3 ? 8 : 15;
    const int exponentBits = format == Mxfp8Format::E4M3 ? 4 : 5;
    const int mantissaBits = format == Mxfp8Format::E4M3 ? 3 : 2;
    const int minPrivateExponent = -(1 << (exponentBits - 1)) + 2;
    const float maxFinite = format == Mxfp8Format::E4M3 ? 448.0f : 57344.0f;

    const std::size_t blockCount = (h + kMxfp8BlockSize - 1) / kMxfp8BlockSize;
    for (std::size_t block = 0; block < blockCount; ++block) {
        const std::size_t begin = block * kMxfp8BlockSize;
        const std::size_t end = std::min(begin + kMxfp8BlockSize, h);
        float maxAbs = 0.0f;
        for (std::size_t index = begin; index < end; ++index) {
            maxAbs = std::max(maxAbs, std::fabs(input[index]));
        }
        int sharedExponent = maxAbs == 0.0f ? -127 :
            static_cast<int>(std::floor(std::log2(maxAbs))) - elementMaxExponent;
        sharedExponent = std::max(-127, std::min(127, sharedExponent));
        scales[block] = static_cast<uint8_t>(sharedExponent + 127);
        const float sharedScale = std::ldexp(1.0f, sharedExponent);

        for (std::size_t index = begin; index < end; ++index) {
            const float scaled = input[index] / sharedScale;
            int privateExponent = scaled == 0.0f ? 0 :
                static_cast<int>(std::floor(std::log2(std::fabs(scaled))));
            privateExponent = std::max(privateExponent, minPrivateExponent);
            const float privateScale = std::ldexp(1.0f, privateExponent);
            float quantized = RoundToNearestEven(
                scaled / privateScale * static_cast<float>(1 << mantissaBits));
            quantized = quantized / static_cast<float>(1 << mantissaBits) * privateScale;
            quantized = std::max(-maxFinite, std::min(maxFinite, quantized));
            elements[index] = EncodeFp8(quantized, format);
        }
    }
}

inline Mxfp8Tensor QuantizeMxfp8(const std::vector<float> &input, std::size_t rows,
    std::size_t h, Mxfp8Format format)
{
    Mxfp8Tensor result;
    if (rows == 0 || h == 0 || input.size() != rows * h) {
        return result;
    }
    result.scaleCountPerRow = Mxfp8ScaleCountPerRow(h);
    result.elements.resize(rows * h);
    result.scales.resize(rows * result.scaleCountPerRow);
    for (std::size_t row = 0; row < rows; ++row) {
        QuantizeMxfp8Row(input.data() + row * h, h, format,
            result.elements.data() + row * h,
            result.scales.data() + row * result.scaleCountPerRow);
    }
    return result;
}

inline std::vector<float> DequantizeMxfp8(const Mxfp8Tensor &input, std::size_t rows,
    std::size_t h, Mxfp8Format format)
{
    const std::size_t scaleCount = Mxfp8ScaleCountPerRow(h);
    if (rows == 0 || h == 0 || input.scaleCountPerRow != scaleCount ||
        input.elements.size() != rows * h || input.scales.size() != rows * scaleCount) {
        return {};
    }

    std::vector<float> result(rows * h, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t index = 0; index < h; ++index) {
            const std::size_t block = index / kMxfp8BlockSize;
            const uint8_t scale = input.scales[row * scaleCount + block];
            result[row * h + index] = DecodeFp8(input.elements[row * h + index], format) *
                std::ldexp(1.0f, static_cast<int>(scale) - 127);
        }
    }
    return result;
}

inline std::vector<float> RoundTripMxfp8(const std::vector<float> &input, std::size_t rows,
    std::size_t h, Mxfp8Format format)
{
    return DequantizeMxfp8(QuantizeMxfp8(input, rows, h, format), rows, h, format);
}

} // namespace TileXREpDemo

#endif // TILEXR_TESTS_EP_DEMO_MXFP8_GOLDEN_H
