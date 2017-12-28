//
// Created by byd on 17-12-1.
//

#include <cstring>
#include "densecrf_util.h"

float* allocate(size_t N) {
    float* r = NULL;
    if (N > 0) {
        r = new float[N];
    }
    memset(r, 0, sizeof(float)*N);
    return r;
}

void deallocate(float* &ptr) {
    if (ptr)
        delete[] ptr;
    ptr = NULL;
}