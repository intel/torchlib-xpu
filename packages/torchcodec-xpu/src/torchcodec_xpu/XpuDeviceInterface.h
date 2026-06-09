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

  std::optional<const AVCodec*> findCodec(
      const AVCodecID& codecId,
      bool isDecoder = true) override;

  void initialize(const SharedAVCodecContext& codecContext) override;

  void initializeVideo(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const VideoStreamOptions& videoStreamOptions,
      const std::vector<std::unique_ptr<Transform>>&
          transforms,
      const std::optional<FrameDims>& resizedOutputDims)
      override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::stable::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  // We sometimes encounter frames that cannot be decoded on the XPU device.
  // Rather than erroring out, we decode them on the CPU.
  std::unique_ptr<DeviceInterface> cpuInterface_;

  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;
  bool has_fp64_;

  UniqueAVBufferRef ctx_;

  std::unique_ptr<FilterGraph> filterGraph_;

  // Used to know whether a new FilterGraphContext should
  // be created before decoding a new frame.
  FiltersConfig prevFiltersConfig_;

  // Optimized conversion. Return value indicates if conversion was
  // successfull.
  bool convertAVFrameToFrameOutput_SYCL(
      UniqueAVFrame& avFrame,
      torch::stable::Tensor& dst);
  // Fallback conversion if optimized path is not available.
  void convertAVFrameToFrameOutput_FilterGraph(
      UniqueAVFrame& avFrame,
      torch::stable::Tensor& dst);
};

} // namespace facebook::torchcodec
