// Copyright (c) 2025 Dmitry Rogozhkin.
// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#pragma once

#include "DeviceInterface.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class XpuDeviceInterface : public DeviceInterface {
 public:
  XpuDeviceInterface(const StableDevice& device);

  virtual ~XpuDeviceInterface();

  std::optional<const AVCodec*> find_codec(
      const AVCodecID& codec_id,
      bool is_decoder = true) override;

  void initialize(const SharedAVCodecContext& codec_context) override;

  void initialize_video(
      const AVStream* av_stream,
      const UniqueDecodingAVFormatContext& av_format_ctx,
      const VideoStreamOptions& video_stream_options,
      const std::vector<std::unique_ptr<Transform>>&
          transforms,
      const std::optional<FrameDims>& resized_output_dims)
      override;

  void register_hardware_device_with_codec(AVCodecContext* codec_context) override;

  void convert_av_frame_to_frame_output(
      UniqueAVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt) override;

 private:
  // We sometimes encounter frames that cannot be decoded on the XPU device.
  // Rather than erroring out, we decode them on the CPU.
  std::unique_ptr<DeviceInterface> cpu_interface_;

  VideoStreamOptions video_stream_options_;
  AVRational time_base_;
  bool has_fp64_;

  UniqueAVBufferRef ctx_;

  std::unique_ptr<FilterGraph> filter_graph_;

  // Used to know whether a new FilterGraphContext should
  // be created before decoding a new frame.
  FiltersConfig prev_filters_config_;

  // Optimized conversion. Return value indicates if conversion was
  // successfull.
  bool convert_av_frame_to_frame_output_with_sycl(
      UniqueAVFrame& av_frame,
      torch::stable::Tensor& dst);
  // Fallback conversion if optimized path is not available.
  void convert_av_frame_to_frame_output_with_filter_graph(
      UniqueAVFrame& av_frame,
      torch::stable::Tensor& dst);
};

} // namespace facebook::torchcodec
