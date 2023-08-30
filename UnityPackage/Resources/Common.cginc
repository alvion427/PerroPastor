#define kThreadsPerGroup 256

#define kFixedPointScale (256.0f * 256.0f)

struct Q8_0_Block
{
    float scale;
    uint values[8];
};

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

uint Encode_Q8_0(float4 v, float scale)
{
    float4 scaled = v / scale;
    
    uint x = (uint)((int)scaled.x) & 0xff;
    uint y = (uint)((int)scaled.y) & 0xff;
    uint z = (uint)((int)scaled.z) & 0xff;
    uint w = (uint)((int)scaled.w) & 0xff;

    uint blockValue = x | (y << 8) | (z << 16) | (w << 24);
    return blockValue;
}

float4 Decode_Q8_0(uint blockValue, float scale)
{
    int x = (int)(blockValue << 24) >> 24;
    int y = (int)(blockValue << 16) >> 24;
    int z = (int)(blockValue << 8) >> 24;
    int w = (int)blockValue >> 24;

    return float4(x, y, z, w) * scale;
}