TESTMACROS_TEMPLATE = '''#pragma once 
#include <chrono>
#include "data_utils.h"
#include "acl/acl.h"


// Define some macros for easy life 
#define SYNC_STREAM() \\
    CHECK_ACL(aclrtSynchronizeStream(stream));

#define DEFINE_VARIABLE(x, size) \\
    size_t x##_datasize = size; \\
    uint8_t *x##_host; \\
    uint8_t *x; \\
    CHECK_ACL(aclrtMallocHost((void **)(&x##_host), x##_datasize)); \\
    CHECK_ACL(aclrtMalloc((void **)&x, x##_datasize, ACL_MEM_MALLOC_HUGE_FIRST)); \\
    printf(#x" size: %ld MB\\n", x##_datasize/1024/1024);

#define COPY_VARIABLE(x) \\
    ReadFile("./input/input_"#x".bin", x##_datasize, x##_host, x##_datasize); \\
    CHECK_ACL(aclrtMemcpy(x, x##_datasize, x##_host, x##_datasize, ACL_MEMCPY_HOST_TO_DEVICE));\\
    CHECK_ACL(aclrtSynchronizeStream(stream));

#define WRITE_VARIABLE(x) \\
    CHECK_ACL(aclrtMemcpy(x##_host, x##_datasize, x, x##_datasize, ACL_MEMCPY_DEVICE_TO_HOST)); \\
    CHECK_ACL(aclrtSynchronizeStream(stream));\\
    WriteFile("./output/output_"#x".bin", x##_host, x##_datasize);

#define INIT_ACL() \\
    CHECK_ACL(aclInit(nullptr)); \\
    int32_t deviceId = 0; \\
    CHECK_ACL(aclrtSetDevice(deviceId)); \\
    aclrtStream stream = nullptr; \\
    CHECK_ACL(aclrtCreateStream(&stream));

#define END_ACL() \\
    CHECK_ACL(aclrtDestroyStream(stream)); \\
    CHECK_ACL(aclrtResetDevice(deviceId)); \\
    CHECK_ACL(aclFinalize());

#define RUN_KERNEL(x, ...) ACLRT_LAUNCH_KERNEL(x)(blockDim, stream, __VA_ARGS__)

#define SPEED_TEST(x, ...) \\
    RUN_KERNEL(x, __VA_ARGS__); \\
    RUN_KERNEL(x, __VA_ARGS__); \\
    RUN_KERNEL(x, __VA_ARGS__); \\
    CHECK_ACL(aclrtSynchronizeStream(stream)); \\
    auto startTime = std::chrono::steady_clock::now(); \\
    for (int i=0; i<100; ++i){ RUN_KERNEL(x, __VA_ARGS__); } \\
    CHECK_ACL(aclrtSynchronizeStream(stream)); \\
    auto endTime = std::chrono::steady_clock::now(); \\
    double duration_second = std::chrono::duration<double>(endTime - startTime).count(); \\
    duration_second = duration_second * 1000.0 / 100.0; \\
    printf("Time cost: %fms\\n", duration_second);

'''