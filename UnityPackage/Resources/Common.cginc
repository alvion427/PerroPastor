#define kThreadsPerGroup 256

#define kFixedPointScale (256.0f * 256.0f * 256.0f)

#if QUANT_WEIGHT_32
    #define WEIGHT_TYPE_VEC float4

    float4 StoreWeights(float4 value)
    {
        return value;
    }

    float4 LoadWeights(float4 weights)
    {
        return weights;
    }

#elif QUANT_WEIGHT_16

    #define WEIGHT_TYPE_VEC uint2

    uint2 StoreWeights(float4 v)
    {
        return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
    }

    float4 LoadWeights(uint2 weights)
    {
        return float4(f16tof32(weights.x & 0xFFFF), f16tof32(weights.x >> 16), f16tof32(weights.y & 0xFFFF), f16tof32(weights.y >> 16));
    }

#endif
