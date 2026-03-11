// Copyright (c) 2025 Dmitry Rogozhkin.
// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#pragma once

#include "DeviceInterface.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class XpuDeviceInterface : public DeviceInterface {
 public:
  XpuDeviceInterface(const torch::Device& device);

  virtual ~XpuDeviceInterface();

  std::optional<const AVCodec*> findCodec(
      const AVCodecID& codecId,
      bool isDecoder = true) override;

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void initializeVideo(
      const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims)
      override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;
  bool has_fp64_;

  UniqueAVBufferRef ctx_;

  std::unique_ptr<FilterGraph> filterGraphContext_;

  // Used to know whether a new FilterGraphContext should
  // be created before decoding a new frame.
  FiltersContext prevFiltersContext_;

  // Optimized conversion. Return value indicates if conversion was
  // successfull.
  bool convertAVFrameToFrameOutput_SYCL(
      UniqueAVFrame& avFrame,
      torch::Tensor& dst);
  // Fallback conversion if optimized path is not available.
  void convertAVFrameToFrameOutput_FilterGraph(
      UniqueAVFrame& avFrame,
      torch::Tensor& dst);
};

} // namespace facebook::torchcodec
