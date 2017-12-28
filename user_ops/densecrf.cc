#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <vector>

#include "densecrf_pairwise.h"
#include "densecrf_util.h"

using namespace tensorflow;

REGISTER_OP("DenseCRF")
	.Attr("max_iter : int")
	.Attr("pos_w : float")
	.Attr("pos_xy_std : float")
	.Attr("bi_w : float")
	.Attr("bi_xy_std : float")
	.Attr("bi_rgb_std : float")
	.Input("feature_map : float32")
	.Input("data_dim : float32")
	.Input("data : float32")
	.Output("crf_inf : float32");


class DenseCRFOp : public OpKernel {
public:
	explicit DenseCRFOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context,
			context->GetAttr("max_iter", &max_iter_));
		OP_REQUIRES_OK(context,
			context->GetAttr("pos_w", &pos_w_));
		OP_REQUIRES_OK(context,
			context->GetAttr("pos_xy_std", &pos_xy_std_));
		OP_REQUIRES_OK(context,
			context->GetAttr("bi_w", &bi_w_));
		OP_REQUIRES_OK(context,
			context->GetAttr("bi_xy_std", &bi_xy_std_));
		OP_REQUIRES_OK(context,
			context->GetAttr("bi_rgb_std", &bi_rgb_std_));

		unary_element_ = 0;
		map_element_ = 0;
		unary_ = NULL;
		current_ = NULL;
		next_ = NULL;
		tmp_ = NULL;
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& input_feat = context->input(0);
		// data_dim shape is (batch_size, 1, 1, 2)
		const Tensor& input_data_dims = context->input(1);
		const Tensor& input_data = context->input(2);

		/* Begin ----- Reshape function */
		const TensorShape& input_feat_shape = input_feat.shape();
		num_ = input_feat_shape.dim_size(0);
		pad_height_ = input_feat_shape.dim_size(1);
		pad_width_ = input_feat_shape.dim_size(2);
		M_ = input_feat_shape.dim_size(3);

		const TensorShape& input_data_shape = input_data.shape();
		int img_h = input_feat_shape.dim_size(1);
		int img_w = input_feat_shape.dim_size(2);

		// Square size
		int num_pixel = pad_height_ * pad_width_;
		// Cube size
		int cur_unary_element = num_pixel * M_;

		if (unary_element_ < cur_unary_element) {
			unary_element_ = cur_unary_element;
			map_element_ = num_pixel;
			DeAllocateAllData();
			AllocateAllData();
		}

		// allocate largest possible size for output
		TensorShape output_shape = TensorShape({num_, pad_height_, pad_width_, M_});
    	Tensor* output = NULL;
    	OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    	// TensorShape sum_multiplier_shape = TensorShape({1, 1, 1, M_});
    	// OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, sum_multiplier_shape, &sum_multiplier_));
    	// auto multiplier_data = sum_multiplier_->flat<DT_FLOAT>().data;
    	// std::fill_n(multiplier_data, M_, 1);

    	// TensorShape norm_data_shape = TensorShape({1, pad_height_, pad_width_, M_});
    	// OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, norm_data_shape, &norm_data_));

		/* End -------- Reshape function*/

		// Begin -------- Forward cpu function
		const float* bottom_data = input_feat.flat<float>().data();	// format NHWC
		const float* data_dims = input_data_dims.flat<float>().data(); // [1,1,1,2]
		const float* im = input_data.flat<float>().data();			// format NHWC
		float* top_data = output->flat<float>().data();

		int bottom_data_offset;
		int data_dim_offset;
		int im_offset;
		int top_data_offset;

		for (int n = 0; n < num_; ++n) {
			// TODO : tensorflow no offset() function
			bottom_data_offset = n * cur_unary_element;
			data_dim_offset = n * 2;
			im_offset = n * (img_h * img_w * 3);
			top_data_offset = n * cur_unary_element;

			// check dimension of data arrays, if too small, reallocate memory
			// int real_img_height = input_data_dims(data_dim_offset);
			// int real_img_width = input_data_dims(data_dim_offset+1);
			int real_img_height = *(data_dims + data_dim_offset);
			int real_img_width = *(data_dims + data_dim_offset + 1);
			if (pad_height_ <= real_img_height && pad_width_ <= real_img_width) {
				H_ = pad_height_;
				W_ = pad_width_;
			} else {
				H_ = real_img_height;
				W_ = real_img_width;
			}
			N_ = W_ * H_;

			SetupUnaryEnergy(bottom_data + bottom_data_offset);
			SetupPairwiseFunctions(im + im_offset);
			ComputeMap(top_data + top_data_offset);
			ClearPairwiseFunctions();
		}
	}


private:
	int64 num_;		// batch_size
	int64 pad_height_;	// may have padded rows
	int64 pad_width_;	// may have padded cols

	int64 M_;		// number of input feature (channel)
	int64 W_;		// effective width
	int64 H_;		// effective height
	int64 N_;		// = W_ * H_


	int max_iter_;
	// Gaussian pairwise potential with weight and positional standard deviation
	float pos_w_;
	float pos_xy_std_;
	// Bilateral pairwise potential with weight, positional std, and color std
	float bi_w_;
	float bi_xy_std_;
	float bi_rgb_std_;

	std::vector<PairwisePotential*> pairwise_;

	int unary_element_;
	int map_element_;

	float* unary_;		// unary energy
	float* current_;	// current inference values, will copy to output
	float* next_;		// next inference values
	float* tmp_;		// buffer

	// sum_multiplier is used to carry out sum using BLAS
	Tensor* sum_multiplier_;
	// scale is an intermediate Tensor to hold temporary results
	Tensor* scale_;
	// norm_data is an intermediate Tensor to hold temporary results;
	Tensor* norm_data_;

	// the output format is probability or score
	bool output_prob_;

	void DeAllocateAllData() {
		deallocate(unary_);
		deallocate(current_);
		deallocate(next_);
		deallocate(tmp_);
	}

	void AllocateAllData() {
		unary_ 	= allocate(unary_element_);
		current_= allocate(unary_element_);
		next_	= allocate(unary_element_);
		tmp_	= allocate(unary_element_);
	}

	void ClearPairwiseFunctions() {
		for (size_t i = 0; i < pairwise_.size(); ++i) {
			delete pairwise_[i];
		}
		pairwise_.clear();
	}

	void SetupUnaryEnergy(const float* bottom_data) {
		for (int h = 0 ; h < H_; ++h) {
			for (int w = 0; w < W_; ++w) {
				for (int c = 0; c < M_; ++c) {
					int in_index = (h * pad_width_ + w) * M_ + c;
					int out_index = (h * W_ + w) * M_ + c;
					// unary format HWC
					unary_[out_index] = -bottom_data[in_index];
				}
			}
		}
	}

	void SetupPairwiseFunctions(const float* im) {
		// im NHWC
		ClearPairwiseFunctions();
		// add pairwise Gaussian
		float* features = new float[N_*2];
		for (int j = 0; j < H_; ++j) {
			for (int i = 0; i < W_; ++i) {
				features[(j*W_+i)*2 + 0] = i / pos_xy_std_;
				features[(j*W_+i)*2 + 1] = j / pos_xy_std_;
			}
		}
		pairwise_.push_back(new PottsPotential(features, 2, N_, pos_w_));
		delete[] features;

		// bilateral pairwise
		// !! Warning !! ---------- Assume im is CHW format. but tensorflow use HWC
		features = new float[N_*5];
		// Note H_ and W_ are the effective dimension of image (not padded dimensions)
		for (int j = 0; j < H_; ++j) {
			for (int i = 0; i < W_; ++i) {
				features[(j*W_+i)*5+0] = i / bi_xy_std_;
				features[(j*W_+i)*5+1] = j / bi_xy_std_;
				int img_index = (j * pad_width_ + i) * 3;
				// im is BGR, Assume im is mean-centered (not affect gaussian blur)
				// and assume im is preprocessing by scale = 1 (may cause problem if not 1)
				features[(j*W_+i)*5+2] = im[img_index] / bi_rgb_std_;
				features[(j*W_+i)*5+3] = im[img_index + 1] / bi_rgb_std_;
				features[(j*W_+i)*5+4] = im[img_index + 2] / bi_rgb_std_;
			}
		}
		pairwise_.push_back(new PottsPotential(features, 5, N_, bi_w_));
		delete[] features;
	}

	void ComputeMap(float* top_inf) {
		memset(top_inf, 0, sizeof(float)*pad_height_*pad_width_*M_);
		// result are saved to current_ after call RunInference()
		RunInference();

		int in_index;
		int out_index;
		// copy current_ to top
		for (int h = 0; h < H_; ++h) {
			for (int w = 0; w < W_; ++w) {
				for (int c = 0; c < M_; ++c) {
					in_index = (h * W_ + w) * M_ + c;	// format HWC
					out_index = (h * pad_height_ + w) * M_ + c;	// format HWC
					top_inf[out_index] = static_cast<float>(current_[in_index]);
				}
			}
		}
	}

	void RunInference() {
		StartInference();
		for (int i = 0; i < max_iter_; ++i) {
			StepInference();
		}
	}

	void StartInference() {
		ExpAndNormalize(current_, unary_, -1.0);
	}

	void ExpAndNormalize(float* out, const float* in, float scale) {
		// in's format is HWC, out's format is HWC
		float* V = new float[M_];
		for (int i = 0; i < N_; ++i) {
			const float* b = in + i * M_;
			float mx = scale * b[0];
			for (int j = 1; j < M_; ++j) {
				if (mx < scale * b[j])
					mx = scale * b[j];
			}
			float tt = 0;
			for (int j = 0; j < M_; ++j) {
				V[j] = fast_exp(scale*b[j] - mx);
				tt += V[j];
			}
			// Make it a probability
			for (int j = 0; j < M_; ++j)
				V[j] /= tt;
			float* a = out + i * M_;
			for (int j = 0; j < M_; ++j)
				a[j] = V[j];
		}
		delete[] V;
	}

	void StepInference() {
		for (int i = 0; i < N_*M_; ++i)
			next_[i] = -unary_[i];
		// Add up all pairwise potentials
		for (size_t i = 0; i < pairwise_.size(); ++i)
			pairwise_[i]->apply(next_, current_, tmp_, M_);

		ExpAndNormalize(current_, next_, 1.0);
	}

};


REGISTER_KERNEL_BUILDER(Name("DenseCRF").Device(DEVICE_CPU), DenseCRFOp);