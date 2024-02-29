//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
#include "cuda.h"
namespace nvinfer1 {


template<typename scalar_t>
void im2col_radiance(const int n, const scalar_t *data_im,
                     const int height, const int width, const int kernel_h, const int kernel_w,
                     const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     const int height_col, const int width_col,
                     scalar_t *data_col);

}// namespace nvinfer1