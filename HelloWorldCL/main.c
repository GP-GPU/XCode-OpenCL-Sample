//
//  main.c
//  HelloWorldCL
//
//  Created by Гурьянов Антон on 27.04.13.
//  Copyright (c) 2013 Gurrrik. All rights reserved.
//

#include <stdio.h>
#include <math.h>

#include <OpenCL/OpenCL.h>

#include "helloworld.cl.h"
#include "vectorsum.cl.h"
#include "matrixmult.cl.h"

#define NUM_VALUES 1024
#define EPS 1e-3

static int validate_square(cl_float* input, cl_float* output)
{
    int i;
    for (i = 0; i < NUM_VALUES; i++) {
        if (fabs(output[i] - input[i] * input[i]) > EPS) {
            fprintf(stderr, "Error: Element %d did not match expected output.\n", i);
            fprintf(stderr, "Saw: %1.4f, expected %1.4f\n", output[i], input[i]*input[i]);
            fflush(stderr);
            return 0;
        }
    }
    return 1;
}

static int validate_vectorsum(cl_float* in1, cl_float* in2, cl_float* output)
{
    int i;
    for (i = 0; i < NUM_VALUES; i++) {
        if (fabs(in1[i] + in2[i] - output[i]) > EPS) {
            fprintf(stderr, "Error: Element %d did not match expected output.\n", i);
            fprintf(stderr, "Saw: %1.4f, expected %1.4f\n", output[i], in1[i] + in2[i]);
            fflush(stderr);
            return 0;
        }
    }
    return 1;
}

static int validate_matrixmult(cl_float* output)
{
    int i;
    float correct[12] = {10.f, 6.f, 3.f, 3.f, 10.f, 9.f, 7.f, 2.f, 0.f, -3.f, -4.f, 1.f};
    for (i = 0; i < 12; i++) {
        if (fabs(output[i] - correct[i]) > EPS) {
            fprintf(stderr, "Error: Element %d did not match expected output.\n", i);
            fprintf(stderr, "Saw: %1.4f, expected %1.4f\n", output[i], correct[i]);
            fflush(stderr);
            return 0;
        }
    }
    return 1;
}

static void print_device_info(cl_device_id device)
{
    char name[128];
    char vendor[128];
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    
    fprintf(stdout, "%s : %s\n", vendor, name);
}

#pragma mark -
#pragma mark Hello World - Sample 1

int main(int argc, const char** argv)
{
    int i;
    const size_t byte_size = sizeof(cl_float) * NUM_VALUES;

    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    if (queue == NULL)
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    print_device_info(gpu);

    // ======== HelloWorld =========
    float *test_in = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    for (i = 0; i < NUM_VALUES; i++)
        test_in[i] = (cl_float)i;
    
    float *test_out = (float*)malloc(sizeof(cl_float) * NUM_VALUES);
    
    void *mem_in = gcl_malloc(sizeof(cl_float) * NUM_VALUES, test_in, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_out = gcl_malloc(sizeof(cl_float) * NUM_VALUES, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(square_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        cl_ndrange range = { 1                           //number of dimensions
                           , {0, 0, 0}                   //offsets in dimensions
                           , {NUM_VALUES, 0, 0}          //global range
                           , {wgs, 0, 0}};               //local size of workgroup
        square_kernel(&range, (cl_float*)mem_in, (cl_float*)mem_out);
        gcl_memcpy(test_out, mem_out, sizeof(cl_float) * NUM_VALUES);
    });
    
    if (validate_square(test_in, test_out))
        fprintf(stdout, "All values were squared.\n");
    
    gcl_free(mem_in);
    gcl_free(mem_out);
    
    free(test_in);
    free(test_out);

    // ========= VectorSum =========
    float *test_in1 = (float*)malloc(byte_size);
    float *test_in2 = (float*)malloc(byte_size);
    for (i = 0; i < NUM_VALUES; i++) {
        test_in1[i] = (cl_float)i;
        test_in2[i] = (cl_float)i * 2;
    }

    test_out = (float*)malloc(byte_size);

    void *mem_in1 = gcl_malloc(byte_size, test_in1, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_in2 = gcl_malloc(byte_size, test_in2, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    mem_out = gcl_malloc(byte_size, NULL, CL_MEM_WRITE_ONLY);

    dispatch_sync(queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(vectorsum_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        cl_ndrange range = { 1
                           , {0, 0, 0}
                           , {NUM_VALUES, 0, 0}
                           , {wgs, 0, 0}};
        vectorsum_kernel(&range, (cl_float*)mem_in1, (cl_float*)mem_in2, (cl_float*)mem_out);
        gcl_memcpy(test_out, mem_out, byte_size);
    });

    if (validate_vectorsum(test_in1, test_in2, test_out))
        fprintf(stdout, "All values were summed.\n");

    gcl_free(mem_in1);
    gcl_free(mem_in2);
    gcl_free(mem_out);

    free(test_in1);
    free(test_in2);
    free(test_out);

    // ========= MatrixProduct ========
    float test_mat1[6] = {1.f, 2.f, -1.f, 3.f, 2.f, -1.f};
    int test_mat1_rows = 3;
    int test_mat1_cols = 2;
    float test_mat2[8] = {2.f, 0.f, -1.f, 1.f, 4.f, 3.f, 2.f, 1.f};
    int test_mat2_rows = 2;
    int test_mat2_cols = 4;
    test_out = (float*)malloc(sizeof(cl_float) * 12);
    int test_mat3_rows = 3;
    int test_mat3_cols = 4;

    mem_in1 = gcl_malloc(sizeof(cl_float) * 6, test_mat1, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    mem_in2 = gcl_malloc(sizeof(cl_float) * 8, test_mat2, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    mem_out = gcl_malloc(sizeof(cl_float) * 12, NULL, CL_MEM_WRITE_ONLY);

    dispatch_sync(queue, ^{
        cl_ndrange range = { 2
            , {0, 0, 0}
            , {test_mat3_rows, test_mat3_cols, 0}
            , {1, 1, 0}};
        matrixmult_kernel(&range, (cl_float*)mem_in1, (cl_float*)mem_in2, (cl_float*)mem_out, (cl_int)test_mat1_cols, (cl_int)test_mat2_cols);
        gcl_memcpy(test_out, mem_out, sizeof(cl_float) * 12);
    });

    if (validate_matrixmult(test_out))
        fprintf(stdout, "Matrices were multiplied.\n");

    gcl_free(mem_in1);
    gcl_free(mem_in2);
    gcl_free(mem_out);

    free(test_out);

    dispatch_release(queue);
    
    return 0;
}