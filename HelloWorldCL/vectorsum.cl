kernel void vectorsum(global float* in1, global float* in2, global float* out)
{
    size_t i = get_global_id(0);
    out[i] = in1[i] + in2[i];
}