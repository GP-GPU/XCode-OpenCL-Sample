kernel void matrixmult(global float* in1, global float* in2, global float* out,
                       int cols1, int cols2)
{
    int ix = get_global_id(1);
    int iy = get_global_id(0);
    int idx1 = iy * cols1;
    int idx2 = ix;

    int i;
    float res = 0.f;
    for (i = 0; i < cols1; i++) {
        res += in1[idx1] * in2[idx2];
        idx1 += 1;
        idx2 += cols2;
    }
    out[iy*cols2 + ix] = res;
}
