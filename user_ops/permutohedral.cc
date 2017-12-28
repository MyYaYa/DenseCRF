//
// Created by byd on 17-11-30.
//

#include "permutohedral.h"

/************************************************/
/***                Hash Table                ***/
/************************************************/

class HashTable {
private:
    size_t key_size_, filled_, capacity_;
    short* keys_;
    int* table_;
    void grow() {
        // temporarily save old memory
        short* old_keys_ = keys_;
        int* old_table_ = table_;
        size_t old_capacity_ = capacity_;
        capacity_ *= 2;
        // Allocate new memory
        keys_ = new short[(old_capacity_ + 10)*key_size_];
        table_ = new int[capacity_];
        memset(table_, -1, capacity_ * sizeof(int));
        memcpy(keys_, old_keys_, filled_ * key_size_ * sizeof(short));
        // Recovery table
        for (size_t i = 0; i < old_capacity_; i++) {
            if (old_table_[i] > 0) {
                int e = old_table_[i];
                size_t h = hash(old_keys_ + (getKey(e) - keys_)) % capacity_;
                for (; table_[h] >= 0; h = h < capacity_ - 1 ? h + 1 : 0);
                table_[h] = e;
            }
        }
        // clean
        delete [] old_keys_;
        delete [] old_table_;
    }

    // hash function
    size_t hash(const short* k) {
        size_t r = 0;
        for (size_t i = 0; i < key_size_; i++) {
            r += k[i];
            r *= 1664525;
        }
        return r;
    }

public:
    explicit HashTable(int key_size, int n_elements) : key_size_(key_size), filled_(0), capacity_(2 * n_elements) {
        table_ = new int[capacity_];
        keys_ = new short[(capacity_ / 2 + 10) * key_size_];
        memset(table_, -1, capacity_ * sizeof(int));
    }

    ~HashTable() {
        delete [] table_;
        delete [] keys_;
    }

    int size() const {
        return (int)filled_;
    }

    void reset() {
        filled_ = 0;
        memset(table_, -1, capacity_ * sizeof(int));
    }

    int find(const short* k, bool create = false) {
        if (2 * filled_ >= capacity_)	grow();
        // Compute hash value
        size_t h = hash(k) % capacity_;
        // Find the element with the right key, using linear probing
        while (1) {
            int e = table_[h];
            if (e == -1) {
                if (create) {
                    // Insert the new key, and return the new id;
                    for (size_t i = 0; i < key_size_; i++) {
                        keys_[filled_ * key_size_ + i] = k[i];
                    }
                    return table_[h] = (int)filled_++;
                } else {
                    return -1;
                }
            }
            // Check the value is correct
            bool good = true;
            for (size_t i = 0; i < key_size_ && good; i++) {
                if (keys_[e * key_size_ + i] != k[i])
                    good = false;
            }
            if (good)
                return e;
            // Searching false, continue searching in next position.
            h++;
            if (h == capacity_)
                h = 0;
        }
    }

    const short * getKey(int i) const {
        return keys_ + i * key_size_;
    }
};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

Permutohedral::Permutohedral() {
    offset_ = NULL;
    barycentric_ = NULL;
    blur_neighbors_ = NULL;
    N_ = 0;
    M_ = 0;
    d_ = 0;
}

Permutohedral::~Permutohedral() {
    if (offset_)			delete[] offset_;
    if (barycentric_)		delete[] barycentric_;
    if (blur_neighbors_)	delete[] blur_neighbors_;
}

void Permutohedral::init(const float* feature, int feature_size, int N) {
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here]
    N_ = N;
    d_ = feature_size;
    HashTable hash_table(d_, N_ * (d_ + 1));

    // Allocate the class memory
    if (offset_)	delete[] offset_;
    offset_ = new int[(d_ + 1)*N_];
    if (barycentric_)	delete[] barycentric_;
    barycentric_ = new float[(d_ + 1)*N_];

    // Allocate the local memory
    float * scale_factor = new float[d_];
    float * elevated = new float[d_ + 1];
    float * rem0 = new float[d_ + 1];
    float * barycentric = new float[d_ + 2];
    short * rank = new short[d_ + 1];
    short * canonical = new short[(d_ + 1) * (d_ + 1)];
    short * key = new short[d_ + 1];

    // Compute the canonical simplex
    for (int i = 0; i <= d_; i++) {
        for (int j = 0; j <= d_ - i; j++)
            canonical[i * (d_ + 1) + j] = i;
        for (int j = d_ - i + 1; j <= d_; j++)
            canonical[i * (d_ + 1) + j] = i - (d_ + 1);
    }

    // Expected standard deviation of our filter
    float inv_std_dev = sqrtf(2.f / 3.f) * (d_ + 1);
    // Compute the diagonal part of E
    for (int i = 0; i < d_; i++)
        scale_factor[i] = 1.f / sqrtf((i + 2.f) * (i + 1.f)) * inv_std_dev;

    // Compute the simplex each feature lies in
    for (int k = 0; k < N_; k++) {
        // Elevate the feature
        const float* f = feature + k * feature_size;
        // sm contains the sum of 1 .. n of our feature vector
        float sm = 0;
        for (int j = d_; j > 0; j--) {
            float cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - j * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.0f / (d_ + 1);
        float up_factor = (d_ + 1);
        int sum = 0;
        for (int i = 0; i <= d_; i++) {
            int rd = (int)round(down_factor * elevated[i]);
            rem0[i] = rd * up_factor;
            sum += rd;
        }

        // Find the simplex we are in and store it in rank (where rank describes
        // what position coordinate i has in the sorted order of the features values)
        for (int i = 0; i <= d_; i++)
            rank[i] = 0;
        for (int i = 0; i < d_; i++) {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= d_; j++)
                if (di < elevated[j] - rem0[j])
                    rank[i]++;
                else
                    rank[j]++;
        }

        // If the point doesn't lie on the plane (sum! = 0) bring it back
        for (int i = 0; i <= d_; i++) {
            rank[i] += sum;
            if (rank[i] < 0) {
                rank[i] += d_ + 1;
                rem0[i] += d_ + 1;
            } else if (rank[i] > d_) {
                rank[i] -= d_ + 1;
                rem0[i] -= d_ + 1;
            }
        }

        // Compute the barycentric coordinates
        for (int i = 0; i <= d_ + 1; i++)
            barycentric[i] = 0;
        for (int i = 0; i <= d_; i++) {
            float v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[d_ - rank[i]] += v;
            barycentric[d_ - rank[i] + 1] -= v;
        }
        // Warp around
        barycentric[0] += 1.0f + barycentric[d_ + 1];

        // Compute all vertices and their offset, (all but the 
        // last coordinate - it's redundant because they sum to zero)
        for (int remainder = 0; remainder <= d_; remainder++) {
            for (int i = 0; i < d_; i++)
                key[i] = rem0[i] + canonical[remainder * (d_ + 1) + rank[i]];
            offset_[k * (d_ + 1) + remainder] = hash_table.find(key, true);
            barycentric_[k * (d_ + 1) + remainder] = barycentric[remainder];
        }
    }

    delete [] scale_factor;
    delete [] elevated;
    delete [] rem0;
    delete [] barycentric;
    delete [] rank;
    delete [] canonical;
    delete [] key;

    // Find the Neighbors of each lattice point

    // Get the number if vertices in the lattice
    M_ = hash_table.size();

    // Create the neighborhood structure
    if (blur_neighbors_)	delete [] blur_neighbors_;
    blur_neighbors_ = new Neighbors[(d_ + 1)*M_];

    short* n1 = new short[d_ + 1];
    short* n2 = new short[d_ + 1];

    // For each of d+1 axes
    for (int j = 0; j <= d_; j++) {
        for (int i = 0; i < M_; i++) {
            const short* key = hash_table.getKey(i);
            for (int k = 0; k < d_; k++) {
                n1[k] = key[k] - 1;
                n2[k] = key[k] + 1;
            }
            n1[j] = key[j] + d_;
            n2[j] = key[j] - d_;

            blur_neighbors_[j * M_ + i].n1 = hash_table.find(n1);
            blur_neighbors_[j * M_ + i].n2 = hash_table.find(n2);
        }
    }
    delete[] n1;
    delete[] n2;
}

void Permutohedral::compute(float* out, const float* in, int value_size, int in_offset,
                            int out_offset, int in_size, int out_size) const {

    if (in_size == -1)	in_size = N_ - in_offset;
    if (out_size == -1)	out_size = N_ - out_offset;

    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    float* values = new float[(M_+2)*value_size];
    float* new_values = new float[(M_+2)*value_size];

    for (int i = 0; i < (M_+2)*value_size; i++)
        values[i] = new_values[i] = 0;

    // Splatting
    for (int i = 0; i < in_size; i++) {
        for (int j = 0; j <= d_; j++) {
            int o = offset_[(in_offset+i) * (d_+1) + j] + 1;
            float w = barycentric_[(in_offset+i) * (d_+1) + j];
            for (int k = 0; k < value_size; k++)
                values[o*value_size+k] += w * in[i*value_size + k];
        }
    }

    for (int j = 0; j <= d_; j++) {
        for (int i = 0; i < M_; i++) {
            float* old_val = values + (i+1) * value_size;
            float* new_val = new_values + (i+1) * value_size;

            int n1 = blur_neighbors_[j*M_+i].n1 + 1;
            int n2 = blur_neighbors_[j*M_+i].n2 + 1;
            float* n1_val = values + n1*value_size;
            float* n2_val = values + n2*value_size;
            for (int k = 0; k < value_size; k++)
                new_val[k] = old_val[k] + 0.5f*(n1_val[k] + n2_val[k]);
        }
        float* tmp = values;
        values = new_values;
        new_values = tmp;
    }

    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha =  1.0f / (1.f + powf(2.f, -(float)d_));

    // Slicing
    for (int i = 0; i < out_size; i++) {
        for (int k = 0; k < value_size; k++)
            out[i*value_size + k] = 0;
        for (int j = 0; j <= d_; j++) {
            int o = offset_[(out_offset + i) * (d_+1) + j] + 1;
            float w = barycentric_[(out_offset + i) * (d_+1) + j];
            for (int k = 0; k < value_size; k++)
                out[i*value_size + k] += w * values[o*value_size + k] * alpha;
        }
    }

    delete[] values;
    delete[] new_values;
}