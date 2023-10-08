#define kThreadsPerGroup 256

// Our standard fixed point format uses 16 bits for whole numbers and 16 bits for fraction
#define kFixedPointScale (256.0f * 256.0f)

// When we are accumulating fractions, we can use a little more precision for the decimal, although we may be
// accumulating a LOT of them, so we still need a decent amount of room for whole numbers
#define kFixedPointFractionScale (128.0f * 256.0f * 256.0f)

uint FloatToFixedUint(float val) {
    int fixedInt = (int)(val * kFixedPointScale);
    return (uint)(fixedInt) + 0x80000000;
}

float FixedUintToFloat(uint fixedUint) {
    return ((int)(fixedUint - 0x80000000)) / (float)kFixedPointScale;
}

int2 FloatToFixedInt64(float f)
{
    int whole = (int)f;
    int fraction = (int)((f - (float)whole) * kFixedPointFractionScale);
    return int2(whole, fraction);
}

float FixedInt64ToFloat(int2 i)
{
    return (float)i.x + (float)i.y / kFixedPointFractionScale;
}

#define InterlockedAddFixedInt64(loc, val) \
    InterlockedAdd(loc.x, val.x); \
    InterlockedAdd(loc.y, val.y);

#define GetVector(b, idx) float4( \
    b[idx * 4 + 0], \
    b[idx * 4 + 1], \
    b[idx * 4 + 2], \
    b[idx * 4 + 3]);

#define SetVector(b, idx, v) \
    b[idx * 4 + 0] = v.x; \
    b[idx * 4 + 1] = v.y; \
    b[idx * 4 + 2] = v.z; \
    b[idx * 4 + 3] = v.w;


struct Q8_0_Block
{
    float scale;
    uint values[8];
};

struct Q5_1_Block
{
    uint scaleAndMin;
    uint highBits;
    uint values[4];
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

//void Encode_Q8_0(float4 v, out Q8_0_Block block, int valueIndex, float scale)
//{
#define Encode_Q8_0(v, block, valueIndex)                               \
{                                                                       \
    float4 scaled = v / block.scale;                                    \
    uint x = (uint)((int)scaled.x) & 0xff;                              \
    uint y = (uint)((int)scaled.y) & 0xff;                              \
    uint z = (uint)((int)scaled.z) & 0xff;                              \
    uint w = (uint)((int)scaled.w) & 0xff;                              \
    block.values[valueIndex] = x | (y << 8) | (z << 16) | (w << 24);    \
}

float4 Decode_Q8_0(const Q8_0_Block block, int valueIndex)
{
    uint blockValue = block.values[valueIndex];
    int x = (int)(blockValue << 24) >> 24;
    int y = (int)(blockValue << 16) >> 24;
    int z = (int)(blockValue << 8) >> 24;
    int w = (int)blockValue >> 24;

    return float4(x, y, z, w) * block.scale;
}

void Encode_Q5_1(float4 v, out Q5_1_Block block, int valueIndex)
{
    block.values[0] = 0;
}

float4 Decode_Q5_1(Q5_1_Block block, uint valueIndex)
{
    const float scale = f16tof32(block.scaleAndMin & 0xFFFF);
    const float min = f16tof32(block.scaleAndMin >> 16);

    uint v = block.values[valueIndex % 4];
    uint hb = (block.highBits >> (valueIndex * 4)) << 4;

    // The first pass through uses the low bits, the second pass uses the high bits
    if (valueIndex >= 4) {
      v >>= 4;
    }

    float4 result;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
      const float f = (hb & 0x10) | (v & 0xf);
      result[i] = f * scale + min;
      v >>= 8;
      hb >>= 1;
    }

    return result;
}
