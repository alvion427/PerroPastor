#define kThreadsPerGroup 256

#define kFixedPointScale (256.0f * 256.0f * 256.0f)

float4 StoreVec32f(float4 value)
{
    return value;
}

float4 LoadVec32f(float4 weights)
{
    return weights;
}

uint2 StoreVec16f(float4 v)
{
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}

float4 LoadVec16f(uint2 weights)
{
    return float4(f16tof32(weights.x & 0xFFFF), f16tof32(weights.x >> 16), f16tof32(weights.y & 0xFFFF), f16tof32(weights.y >> 16));
}

#if QUANT_WEIGHT_16
    #define WEIGHT_TYPE_VEC uint2
    #define StoreWeights StoreVec16f
    #define LoadWeights LoadVec16f
#else /* QUANT_WEIGHT_32 */
    #define WEIGHT_TYPE_VEC float4
    #define StoreWeights StoreVec32f
    #define LoadWeights LoadVec32f
#endif

#if QUANT_RUNTIME_16
    #define RUNTIME_TYPE_VEC uint2
    #define StoreRuntime StoreVec16f
    #define LoadRuntime LoadVec16f
#else  /* QUANT_RUNTIME_32 */
    #define RUNTIME_TYPE_VEC float4
    #define StoreRuntime StoreVec32f
    #define LoadRuntime LoadVec32f
#endif
