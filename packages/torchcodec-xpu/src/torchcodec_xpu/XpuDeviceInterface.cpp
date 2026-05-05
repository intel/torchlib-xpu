// Copyright (c) 2025 Dmitry Rogozhkin.
// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

#include <level_zero/ze_api.h>
#include <va/va_drmcommon.h>

#include <ATen/DLConvertor.h>
#include <c10/xpu/XPUStream.h>

#include "ColorConversionKernel.h"
#include "Cache.h"
#include "FFMPEGCommon.h"
#include "XpuDeviceInterface.h"

extern "C" {
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace facebook::torchcodec {

namespace xpu {

const char* USE_SYCL_KERNELS = std::getenv("USE_SYCL_KERNELS");

static bool g_xpu = registerDeviceInterface(
    DeviceInterfaceKey(StableDeviceType::XPU),
    [](const StableDevice& device) { return new XpuDeviceInterface(device); });

const int MAX_XPU_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
PerGpuCache<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>
    g_cached_hw_device_ctxs(MAX_XPU_GPUS, MAX_CONTEXTS_PER_GPU_IN_CACHE);

inline bool to_bool(std::string str) {
    static const std::unordered_map<std::string, bool> bool_map = {
        {"1", true},  {"0", false},
        {"on", true}, {"off", false},
        {"true", true}, {"false", false}
    };

    auto it = bool_map.find(str);
    if (it != bool_map.end()) {
        return it->second;
    }
    return false;
}

inline bool use_sycl_color_conversion_kernel() {
#ifndef WITH_SYCL_KERNELS
  return false;
#else
  if (!USE_SYCL_KERNELS) {
    return true;
  }
  return to_bool(USE_SYCL_KERNELS);
#endif
}

bool has_fp64(const StableDevice& device) {
  int deviceIndex = getDeviceIndex(device);
  sycl::device syclDevice = c10::xpu::get_raw_device(deviceIndex);
  return syclDevice.has(sycl::aspect::fp64);
}

sycl::ext::oneapi::experimental::architecture getArchitecture(
    const StableDevice& device) {
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device.index());
  return queue.get_device().get_info<
      sycl::ext::oneapi::experimental::info::device::architecture>();
}

// Resolves the VAAPI render-node path this XPU device should open.
std::string resolveRenderD(const StableDevice& device) {
  std::string renderD = "/dev/dri/renderD128";
  int deviceIndex = getDeviceIndex(device);
  sycl::device syclDevice = c10::xpu::get_raw_device(deviceIndex);
  if (syclDevice.has(sycl::aspect::ext_intel_pci_address)) {
    auto BDF =
        syclDevice.get_info<sycl::ext::intel::info::device::pci_address>();
    renderD = "/dev/dri/by-path/pci-" + BDF + "-render";
  }
  return renderD;
}

UniqueAVBufferRef getVaapiContext(const StableDevice& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("vaapi");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find vaapi device");

  UniqueAVBufferRef hw_device_ctx = g_cached_hw_device_ctxs.get(device);
  if (hw_device_ctx) {
    return hw_device_ctx;
  }

  std::string renderD = resolveRenderD(device);

  AVBufferRef* ctx = nullptr;
  int err = av_hwdevice_ctx_create(&ctx, type, renderD.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device: ",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return UniqueAVBufferRef(ctx);
}

torch::stable::Tensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const StableDevice& device) {
  STD_TORCH_CHECK(
      frameDims.height > 0, "height must be > 0, got: ", frameDims.height);
  STD_TORCH_CHECK(
      frameDims.width > 0, "width must be > 0, got: ", frameDims.width);
  return torch::stable::empty(
      {frameDims.height, frameDims.width, 3},
      kStableUInt8,
      std::nullopt,
      device);
}

// Self-contained SW NV12/YUV->RGB24 conversion for the CPU fallback path.
// Uses libswscale directly rather than delegating to CpuDeviceInterface: the
// relevant torchcodec symbols (CpuDeviceInterface ctor, createDeviceInterface)
// are not exported from the installed libtorchcodec_core6.so, so delegation is
// not linkable against the shipped wheel.
void convertSWFrameToRGB_sws(
    AVFrame* avFrame,
    torch::stable::Tensor& dstRGB_CPU) {
  const int width = avFrame->width;
  const int height = avFrame->height;
  auto srcFormat = static_cast<AVPixelFormat>(avFrame->format);

  SwsContext* sws = sws_getContext(width, height, srcFormat, width, height,
      AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  TORCH_CHECK(
      sws != nullptr, "sws_getContext failed for ", av_get_pix_fmt_name(srcFormat),
      " -> RGB24 at ", width, "x",
      height);

  uint8_t* dstData[4] = {static_cast<uint8_t*>(dstRGB_CPU.mutable_data_ptr()),
      nullptr, nullptr, nullptr};
  int dstLinesize[4] = {width * 3, 0, 0, 0};

  int scaled = sws_scale(sws, avFrame->data, avFrame->linesize, 0,
      height, dstData, dstLinesize);
  sws_freeContext(sws);
  TORCH_CHECK(
      scaled == height, "sws_scale produced ", scaled, " lines, expected ", height);
}

} // namespace xpu

int getDeviceIndex(const StableDevice& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and XPU
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < xpu::MAX_XPU_GPUS,
      "Invalid device index = ",
      deviceIndex);

  return (deviceIndex == -1)? 0: deviceIndex;
}

XpuDeviceInterface::XpuDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  TORCH_CHECK(xpu::g_xpu, "XpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == kStableXPU, "Unsupported device: must be XPU");

  // It is important for pytorch itself to create the xpu context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the xpu context.
  torch::stable::Tensor dummyTensorForXpuInitialization = torch::stable::empty(
      {1}, kStableUInt8, std::nullopt, StableDevice(device));

  auto arch = xpu::getArchitecture(device);
  // Checking for devices which don't have HW media engines so we can skip
  // initialization of VAAPI context.
  if (arch != sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc &&
      arch != sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg) {
      ctx_ = xpu::getVaapiContext(device_);
  }

  if (xpu::use_sycl_color_conversion_kernel()) {
    VLOG(1) << "XpuDeviceInterface initialized with SYCL kernel backend";
    VLOG(1) << "Backend: SYCL_KERNEL (Direct NV12→RGB)";
  } else {
    VLOG(1) << "XpuDeviceInterface initialized with VAAPI filter graph backend";
    VLOG(1) << "Backend: VAAPI_FILTER (Flexible, with scaling)";
  }

  has_fp64_ = xpu::has_fp64(device);
  VLOG(1) << "Device supports FP64: " << has_fp64_;
}

XpuDeviceInterface::~XpuDeviceInterface() {
  if (ctx_) {
    xpu::g_cached_hw_device_ctxs.addIfCacheHasCapacity(device_, std::move(ctx_));
  }
}

void XpuDeviceInterface::initialize(
    const AVStream* avStream,
    [[maybe_unused]] const UniqueDecodingAVFormatContext& avFormatCtx,
    const SharedAVCodecContext& codecContext) {
  TORCH_CHECK(avStream != nullptr, "avStream is null");
  codecContext_ = codecContext;
  timeBase_ = avStream->time_base;
}

void XpuDeviceInterface::initializeVideo(
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {
  videoStreamOptions_ = videoStreamOptions;
}

void XpuDeviceInterface::registerHardwareDeviceWithCodec(
    AVCodecContext* codecContext) {
  if (!ctx_) {
    VLOG(1) << "HW context not initialized, falling back to CPU";
    return;
  }
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  codecContext->hw_device_ctx = av_buffer_ref(ctx_.get());
}

VADisplay getVaDisplayFromAV(AVFrame* avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

struct xpuManagerCtx {
  UniqueAVFrame avFrame;
  ze_context_handle_t zeCtx = nullptr;
};

void deleter(DLManagedTensor* self) {
  std::unique_ptr<DLManagedTensor> tensor(self);
  std::unique_ptr<xpuManagerCtx> context((xpuManagerCtx*)self->manager_ctx);
  zeMemFree(context->zeCtx, self->dl_tensor.data);
  free(self->dl_tensor.shape);
  free(self->dl_tensor.strides);
}

torch::stable::Tensor AVFrameToTensor(
    const StableDevice& device,
    const UniqueAVFrame& frame) {
  TORCH_CHECK_EQ(frame->format, AV_PIX_FMT_VAAPI);

  VADRMPRIMESurfaceDescriptor desc{};

  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(frame.get()),
      (VASurfaceID)(uintptr_t)frame->data[3],
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle failed: ",
      vaErrorStr(sts));

  TORCH_CHECK(desc.num_objects == 1, "Expected 1 fd, got ", desc.num_objects);
  TORCH_CHECK(desc.num_layers == 1, "Expected 1 layer, got ", desc.num_layers);
  TORCH_CHECK(
      desc.layers[0].num_planes == 1,
      "Expected 1 plane, got ",
      desc.layers[0].num_planes);

  std::unique_ptr<xpuManagerCtx> context = std::make_unique<xpuManagerCtx>();
  ze_device_handle_t ze_device{};
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device.index());

  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          context->zeCtx =
              ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          ze_device =
              ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;

  ze_result_t res = zeMemAllocDevice(
      context->zeCtx, &alloc_desc, desc.objects[0].size, 0, ze_device, &usm_ptr);
  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS, "Failed to import fd=", desc.objects[0].fd);

  close(desc.objects[0].fd);

  std::unique_ptr<DLManagedTensor> dl_dst = std::make_unique<DLManagedTensor>();
  int64_t* shape = (int64_t*)malloc(3*sizeof(int64_t));

  shape[0] = frame->height;
  shape[1] = frame->width;
  shape[2] = 4;

  context->avFrame.reset(av_frame_alloc());
  TORCH_CHECK(context->avFrame.get(), "Failed to allocate AVFrame");

  int status = av_frame_ref(context->avFrame.get(), frame.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to reference AVFrame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  dl_dst->manager_ctx = context.release();
  dl_dst->deleter = deleter;
  dl_dst->dl_tensor.data = usm_ptr;
  dl_dst->dl_tensor.device.device_type = kDLOneAPI;
  dl_dst->dl_tensor.device.device_id = device.index();
  dl_dst->dl_tensor.ndim = 3;
  dl_dst->dl_tensor.dtype.code = kDLUInt;
  dl_dst->dl_tensor.dtype.bits = 8;
  dl_dst->dl_tensor.dtype.lanes = 1;
  dl_dst->dl_tensor.shape = shape;
  dl_dst->dl_tensor.strides = nullptr;
  dl_dst->dl_tensor.byte_offset = desc.layers[0].offset[0];

  // torch::stable::Tensor(AtenTensorHandle) constructor steals the ownership, so
  // we need to release at::Tensor after getting its handle.
  auto dst = std::make_unique<at::Tensor>(at::fromDLPack(dl_dst.release()));
  // From: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/inductor/aoti_torch/utils.h#L32
  // NOTE: this conversion migth get broken in the new PyTorch release which will
  // likely result in the segmentation fault. If this will happen - update the conversion.
  AtenTensorHandle dst_handle = reinterpret_cast<AtenTensorHandle>(dst.release());
  return torch::stable::Tensor(dst_handle);
}

VADisplay getVaDisplayFromAV(UniqueAVFrame& avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

void XpuDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor) {
  if (avFrame->format != AV_PIX_FMT_VAAPI) {
    // The frame's format is AV_PIX_FMT_VAAPI if and only if its content is on
    // the GPU. In this branch, the frame is on the CPU. This is what FFmpeg VAAPI
    // decoder gives us if it wasn't able to decode a frame, for whatever reason.
    // Typically that happens if the video's decoder isn't supported by VAAPI in
    // general or on this particular device. In this case we have a frame on the
    // CPU. We send the frame back to the XPU device when we're done.

    // Self-contained SW->RGB conversion via libswscale. We do not delegate to
    // CpuDeviceInterface because its constructor and the createDeviceInterface
    // factory are not exported from the installed libtorchcodec_core wheel.
    auto frameDims = FrameDims(avFrame->height, avFrame->width);
    torch::stable::Tensor cpuRGB = torch::stable::empty(
        {frameDims.height, frameDims.width, 3},
        kStableUInt8,
        std::nullopt,
        StableDevice(kStableCPU));
    xpu::convertSWFrameToRGB_sws(avFrame.get(), cpuRGB);

    // Finally, we need to send the frame back to the GPU. Note that the
    // pre-allocated tensor is on the GPU, so we can't send that to the CPU
    // device interface. We copy it over here.
    if (preAllocatedOutputTensor.has_value()) {
      torch::stable::copy_(preAllocatedOutputTensor.value(), cpuRGB);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = torch::stable::to(cpuRGB, device_);
    }
    return;
  }

  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_VAAPI,
      "Expected format to be AV_PIX_FMT_VAAPI, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)avFrame->format)));
  auto frameDims = FrameDims(avFrame->height, avFrame->width);
  torch::stable::Tensor& dst = frameOutput.data;
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
	    (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        intArrayRefToString(shape));
    dst = preAllocatedOutputTensor.value();
  } else {
    // Explicitly load the version defined in facebook::torchcodec::xpu
    // namespace as facebook::torchcodec defines the same but with the linkage
    // type which we can't use.
    dst = xpu::allocateEmptyHWCTensor(frameDims, device_);
  }

  auto start = std::chrono::high_resolution_clock::now();
  if (!convertAVFrameToFrameOutput_SYCL(avFrame, dst)) {
    convertAVFrameToFrameOutput_FilterGraph(avFrame, dst);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "Conversion of frame height=" << frameDims.height << " width=" << frameDims.width
          << " took: " << duration.count() << "us" << std::endl;
}

void XpuDeviceInterface::convertAVFrameToFrameOutput_FilterGraph(
    UniqueAVFrame& avFrame,
    torch::stable::Tensor& dst) {
  VLOG(1) << "Using VAAPI filter graph backend for conversion";
  auto frameDims = FrameDims(avFrame->height, avFrame->width);

  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);
  FiltersConfig filtersConfig;

  filtersConfig.inputWidth = avFrame->width;
  filtersConfig.inputHeight = avFrame->height;
  filtersConfig.inputFormat = frameFormat;
  filtersConfig.inputAspectRatio = avFrame->sample_aspect_ratio;
  // Actual output color format will be set via filter options
  filtersConfig.outputFormat = AV_PIX_FMT_VAAPI;
  filtersConfig.timeBase = timeBase_;
  filtersConfig.hwFramesCtx.reset(av_buffer_ref(avFrame->hw_frames_ctx));

  std::stringstream filters;
  filters << "scale_vaapi=" << frameDims.width << ":" << frameDims.height;
  // CPU device interface outputs RGB in full (pc) color range.
  // We are doing the same to match.
  filters << ":format=rgba:out_range=pc";

  filtersConfig.filtergraphStr = filters.str();

  if (!filterGraph_ || prevFiltersConfig_ != filtersConfig) {
    filterGraph_ =
        std::make_unique<FilterGraph>(filtersConfig, videoStreamOptions_);
    prevFiltersConfig_ = std::move(filtersConfig);
  }

  // We convert input to the RGBX color format with VAAPI getting WxHx4
  // tensor on the output.
  UniqueAVFrame filteredAVFrame = filterGraph_->convert(avFrame);

  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_VAAPI);

  torch::stable::Tensor dst_rgb4 = AVFrameToTensor(device_, filteredAVFrame);
  torch::stable::copy_(dst, torch::stable::narrow(dst_rgb4, 2, 0, 3));
}

bool XpuDeviceInterface::convertAVFrameToFrameOutput_SYCL(
    [[maybe_unused]] UniqueAVFrame& frame,
    [[maybe_unused]] torch::stable::Tensor& dst) {
  bool converted = false;
  if (!xpu::use_sycl_color_conversion_kernel()) {
    return converted;
  }
  if (!has_fp64_) {
    return converted;
  }

#ifdef WITH_SYCL_KERNELS
  VLOG(1) << "Using SYCL kernel backend for conversion";
  TORCH_CHECK_EQ(frame->format, AV_PIX_FMT_VAAPI);
  VADRMPRIMESurfaceDescriptor desc{};
  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(frame.get()),
      (VASurfaceID)(uintptr_t)frame->data[3],
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle failed: ",
      vaErrorStr(sts));

  TORCH_CHECK(desc.num_objects == 1, "Expected 1 fd, got ", desc.num_objects);
  std::unique_ptr<xpuManagerCtx> context = std::make_unique<xpuManagerCtx>();
  ze_device_handle_t ze_device{};
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device_.index());
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          context->zeCtx =
              ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          ze_device =
              ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;

  ze_result_t res = zeMemAllocDevice(
      context->zeCtx,
      &alloc_desc,
      desc.objects[0].size,
      0,
      ze_device,
      &usm_ptr);
  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS, "Failed to import fd=", desc.objects[0].fd);

  close(desc.objects[0].fd);

  convertNV12ToRGB(
      queue,
      (uint8_t*)usm_ptr + desc.layers[0].offset[0],
      (uint8_t*)usm_ptr + desc.layers[1].offset[0],
      (uint8_t*)dst.data_ptr(),
      frame->width,
      frame->height,
      desc.layers[0].pitch[0],
      frame->color_range,
      frame->colorspace);

  zeMemFree(context->zeCtx, usm_ptr);
  converted = true;
#endif
  return converted;
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the XPU
// device
std::optional<const AVCodec*> XpuDeviceInterface::findCodec(
    const AVCodecID& codecId,
    bool isDecoder) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (isDecoder) {
      if (codec->id != codecId || !av_codec_is_decoder(codec)) {
        continue;
      }
    } else {
      if (codec->id != codecId || !av_codec_is_encoder(codec)) {
        continue;
      }
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

} // namespace facebook::torchcodec
