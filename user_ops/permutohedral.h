//
// Created by byd on 17-11-30.
//

#ifndef _PERMUTOHEDRAL_H
#define _PERMUTOHEDRAL_H

#include <cstdlib>

#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

class Permutohedral {
protected:
    int* offset_;
    float* barycentric_;

    struct Neighbors {
        int n1, n2;
        Neighbors(int n1=0, int n2=0) : n1(n1), n2(n2) {}
    };

    Neighbors* blur_neighbors_;
    // Number of elements, size of sparse discretized space, dimension of features
    int N_, M_, d_;

public:
    Permutohedral();

    virtual ~Permutohedral();

    void init(const float* feature, int feature_size, int N);

    void compute(float* out, const float* in, int value_size, int in_offset=0, \
		int out_offset = 0, int in_size = -1, int out_size = -1) const;
};

#endif //_PERMUTOHEDRAL_H
