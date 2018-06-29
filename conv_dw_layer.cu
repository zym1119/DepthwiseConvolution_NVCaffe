#include <vector>
#include "caffe/layers/conv_dw_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Ftype>
__global__ void ConvolutionDepthwiseWeightForward(const int nthreads,
    const Ftype* const bottom_data, const Ftype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Ftype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const Ftype* weight = weight_data + c * kernel_h * kernel_w;
    Ftype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh)
    {
      for (int kw = 0; kw < kernel_w; ++kw)
      {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
        {
          const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}

template <typename Ftype>
__global__ void ConvolutionDepthwiseBiasForward(const int nthreads,
    const Ftype* const bias_data, const int num, const int channels,
    const int top_height, const int top_width, Ftype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / top_height / top_width) % channels;
    top_data[index] += bias_data[c];
  }
}

template <typename Ftype, typename Btype>
void ConvolutionDepthwiseLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const Ftype* weight_data = this->blobs_[0]->gpu_data<Ftype>();
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  ConvolutionDepthwiseWeightForward<Ftype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight_data, num, channels,
      top_height, top_width, bottom_height, bottom_width,
      kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_h_, pad_w_, dilation_h_, dilation_w_, top_data);
  if (this->layer_param_.convolution_param().bias_term())
  {
    const Ftype* bias_data = this->blobs_[1]->gpu_data<Ftype>();
    ConvolutionDepthwiseBiasForward<Ftype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bias_data, num, channels,
        top_height, top_width, top_data);
  }
}

template <typename Ftype, typename Btype>
__global__ void ConvolutionDepthwiseWeightBackward(const int nthreads,
    const Btype* const top_diff, const Btype* const bottom_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Btype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int kh = (index / kernel_w / num / top_height / top_width) % kernel_h;
    const int kw = (index / num / top_height / top_width) % kernel_w;
    const int h_in = -pad_h + h * stride_h + kh * dilation_h;
    const int w_in = -pad_w + w * stride_w + kw * dilation_w;
    if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
    {
      const int c = index / kernel_h / kernel_w / num / top_height / top_width;
      const int n = (index / top_height / top_width) % num;
      const int top_offset = ((n * channels + c) * top_height + h) * top_width + w;
      const int bottom_offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
      buffer_data[index] = top_diff[top_offset] * bottom_data[bottom_offset];
    }
    else
    {
      buffer_data[index] = 0;
    }
  }
}

template <typename Btype>
__global__ void ConvolutionDepthwiseBottomBackward(const int nthreads,
    const Btype* const top_diff, const Btype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Btype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    const Btype* weight = weight_data + c * kernel_h * kernel_w;
    Btype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh)
    {
      for (int kw = 0; kw < kernel_w; ++kw)
      {
        const int h_out_s = h + pad_h - kh * dilation_h;
        const int w_out_s = w + pad_w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0))
        {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width))
          {
            const int offset = ((n * channels + c) * top_height + h_out) * top_width + w_out;
            value += (*weight) * top_diff[offset];
          }
        }
        ++weight;
      }
    }
    bottom_diff[index] += value;
  }
}

template <typename Btype>
__global__ void ConvolutionDepthwiseBiasBackward(const int nthreads,
    const Btype* const top_diff, const int num, const int channels,
    const int top_height, const int top_width, Btype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index / num / top_height / top_width;
    const int n = (index / top_height / top_width) % num;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int offset = ((n * channels + c) * top_height + h) * top_width + w;
    buffer_data[index] = top_diff[offset];
  }
}

template <typename Ftype, typename Btype>
void ConvolutionDepthwiseLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  const int bottom_count = bottom[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  const int length = num * top_height * top_width;
  caffe_gpu_set(bottom_count, Btype(0), bottom[0]->mutable_gpu_diff<Btype>());
  if (this->layer_param_.convolution_param().bias_term() && this->param_propagate_down_[1])
  {
    const int bias_buffer_count = bias_buffer_.count();
    Btype* bias_buffer_mutable_data = bias_buffer_.mutable_gpu_data<Btype>();
    ConvolutionDepthwiseBiasBackward<Btype><<<CAFFE_GET_BLOCKS(bias_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
        bias_buffer_count, top_diff, num, channels,
        top_height, top_width, bias_buffer_mutable_data);
    const int bias_count = this->blobs_[1]->count();
    const Btype* bias_buffer_data = bias_buffer_.gpu_data<Btype>();
    Btype* bias_diff = this->blobs_[1]->mutable_gpu_diff<Btype>();
    const Btype* bias_multiplier_data = bias_multiplier_.gpu_data<Btype>();
    caffe_gpu_gemv(CblasNoTrans, bias_count, length, Btype(1), bias_buffer_data, bias_multiplier_data, Btype(1), bias_diff);
  }
  if (this->param_propagate_down_[0])
  {
    const int weight_buffer_count = weight_buffer_.count();
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    Btype* weight_buffer_mutable_data = weight_buffer_.mutable_gpu_data<Btype>();
    ConvolutionDepthwiseWeightBackward<Btype><<<CAFFE_GET_BLOCKS(weight_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
        weight_buffer_count, top_diff, bottom_data, num, channels,
        top_height, top_width, bottom_height, bottom_width,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, dilation_h_, dilation_w_, weight_buffer_mutable_data);
    const int weight_count = this->blobs_[0]->count();
    const Btype* weight_buffer_data = weight_buffer_.gpu_data<Btype>();
    Btype* weight_diff = this->blobs_[0]->mutable_gpu_diff<Btype>();
    const Btype* weight_multiplier_data = weight_multiplier_.gpu_data<Btype>();
    caffe_gpu_gemv(CblasNoTrans, weight_count, length, Btype(1), weight_buffer_data, weight_multiplier_data, Btype(1), weight_diff);
  }
  if (propagate_down[0])
  {
    const Btype* weight_data = this->blobs_[0]->gpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    ConvolutionDepthwiseBottomBackward<Btype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, top_diff, weight_data, num, channels,
        top_height, top_width, bottom_height, bottom_width,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, dilation_h_, dilation_w_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionDepthwiseLayer);

}  // namespace caffe
