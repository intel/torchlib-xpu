// Copyright (c) 2025 Dmitry Rogozhkin.
// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#include <unistd.h>
#include <stdlib.h>
#include <cstdio>
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

const char* LOG_LEVEL = std::getenv("TORCHCODEC_XPU_LOG_LEVEL");
const char* USE_SYCL_KERNELS = std::getenv("USE_SYCL_KERNELS");
const char* CPU_FALLBACK = std::getenv("CPU_FALLBACK");
const char* FORCE_CPU_FALLBACK = std::getenv("FORCE_CPU_FALLBACK");

static bool g_xpu = register_device_interface(
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

inline bool cpu_fallback() {
  if (!CPU_FALLBACK) {
    return true;
  }
  return to_bool(CPU_FALLBACK);
}

inline bool force_cpu_fallback() {
  if (!FORCE_CPU_FALLBACK) {
    return false;
  }
  return to_bool(FORCE_CPU_FALLBACK);
}

enum {
    NONE = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    VERBOSE = 4
};

inline int debug_level() {
  if (!LOG_LEVEL) {
    return NONE;
  }

  std::string level(LOG_LEVEL);

  if (level == "0") return NONE;
  else if (level == "1") return ERROR;
  else if (level == "2") return WARNING;
  else if (level == "3") return INFO;
  else if (level == "4") return VERBOSE;

  return NONE;
}

inline int get_debug_level() {
  static int level = debug_level();
  return level;
}

#define DEBUG_LOG(L, M) \
  do { \
    if ((L) <= xpu::get_debug_level()) { \
      std::cout << "[torchcodec-xpu] " << M << "\n"; \
    } \
  } while (0)

bool has_fp64(const StableDevice& device) {
  int deviceIndex = get_device_index(device);
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
  int deviceIndex = get_device_index(device);
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
        get_ffmpeg_error_string_from_error_code(err));
  }
  return UniqueAVBufferRef(ctx);
}

torch::stable::Tensor allocate_empty_hwc_tensor(
    const FrameDims& frame_dims,
    const StableDevice& device) {
  STD_TORCH_CHECK(
      frame_dims.height > 0, "height must be > 0, got: ", frame_dims.height);
  STD_TORCH_CHECK(
      frame_dims.width > 0, "width must be > 0, got: ", frame_dims.width);
  return torch::stable::empty(
      {frame_dims.height, frame_dims.width, 3},
      kStableUInt8,
      std::nullopt,
      device);
}

// Allocates a VAAPI-backed NV12 AVFrame for encoding by pulling a buffer from
// the codec's hw_frames_ctx pool created in setupHardwareFrameContextForEncoding.
UniqueAVFrame allocNV12Frame(
    int width,
    int height,
    int frame_index,
    AVCodecContext* codec_context) {
  TORCH_CHECK(codec_context != nullptr, "codec_context is null");
  TORCH_CHECK(
      codec_context->hw_frames_ctx != nullptr,
      "hw_frames_ctx is null: call setupHardwareFrameContextForEncoding first");

  UniqueAVFrame vaFrame(av_frame_alloc());
  TORCH_CHECK(vaFrame != nullptr, "Failed to allocate AVFrame for encoding");
  vaFrame->format = AV_PIX_FMT_VAAPI;
  vaFrame->width  = width;
  vaFrame->height = height;
  vaFrame->pts    = frame_index;

  int ret = av_hwframe_get_buffer(codec_context->hw_frames_ctx, vaFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "av_hwframe_get_buffer failed: ",
      get_ffmpeg_error_string_from_error_code(ret));
  return vaFrame;
}

} // namespace xpu

int get_device_index(const StableDevice& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and XPU
  // libraries use int. So we use int, too.
  int device_index = static_cast<int>(device.index());
  TORCH_CHECK(
      device_index >= -1 && device_index < xpu::MAX_XPU_GPUS,
      "Invalid device index = ",
      device_index);

  return (device_index == -1)? 0: device_index;
}

XpuDeviceInterface::XpuDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  TORCH_CHECK(xpu::g_xpu, "XpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == kStableXPU, "Unsupported device: must be XPU");

  TORCH_CHECK(!xpu::force_cpu_fallback() || xpu::cpu_fallback(),
      "CPU fallback forced but not allowed");

  // It is important for pytorch itself to create the xpu context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the xpu context.
  torch::stable::Tensor dummyTensorForXpuInitialization = torch::stable::empty(
      {1}, kStableUInt8, std::nullopt, StableDevice(device));

  auto arch = xpu::getArchitecture(device);
  if (!xpu::force_cpu_fallback()) {
    // Checking for devices which don't have HW media engines so we can skip
    // initialization of VAAPI context.
    if (arch != sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc &&
      arch != sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg) {
      ctx_ = xpu::getVaapiContext(device_);
    }
  }

  if (xpu::use_sycl_color_conversion_kernel()) {
    DEBUG_LOG(xpu::INFO, "XpuDeviceInterface initialized with SYCL kernel backend");
    DEBUG_LOG(xpu::INFO, "Backend: SYCL_KERNEL (Direct NV12→RGB)");
  } else {
    DEBUG_LOG(xpu::INFO, "XpuDeviceInterface initialized with VAAPI filter graph backend");
    DEBUG_LOG(xpu::INFO, "Backend: VAAPI_FILTER (Flexible, with scaling)");
  }

  has_fp64_ = xpu::has_fp64(device);
  DEBUG_LOG(xpu::INFO, "Device supports FP64: " << has_fp64_);
}

XpuDeviceInterface::~XpuDeviceInterface() {
  if (ctx_) {
    xpu::g_cached_hw_device_ctxs.add_if_cache_has_capacity(device_, std::move(ctx_));
  }
}

void XpuDeviceInterface::ensure_cpu_interface() {
  if (!cpu_interface_) {
    cpu_interface_ = create_device_interface(kStableCPU);
    STD_TORCH_CHECK(
        cpu_interface_ != nullptr, "Failed to create CPU device interface");
  }
}

void XpuDeviceInterface::initialize(const SharedAVCodecContext& codec_context) {
  codec_context_ = codec_context;

  if (xpu::cpu_fallback()) {
    ensure_cpu_interface();
    cpu_interface_->initialize(codec_context_);
  }
}

void XpuDeviceInterface::initialize_video_decoding(
    const AVStream* av_stream,
    const UniqueDecodingAVFormatContext& av_format_ctx,
    const VideoStreamOptions& video_stream_options) {
  TORCH_CHECK(av_stream != nullptr, "av_stream is null");
  time_base_ = av_stream->time_base;
  video_stream_options_ = video_stream_options;

  if (xpu::cpu_fallback()) {
    ensure_cpu_interface();
    cpu_interface_->initialize_video(
        av_stream,
        av_format_ctx,
        VideoStreamOptions(),
        /*transforms=*/{}, 
        /*resized_output_dims=*/std::nullopt);
  }
}

void XpuDeviceInterface::register_hardware_device_with_codec(
    AVCodecContext* codec_context) {
  if (ctx_) {
    TORCH_CHECK(codec_context != nullptr, "codec_context is null");
    codec_context->hw_device_ctx = av_buffer_ref(ctx_.get());
  }
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
      get_ffmpeg_error_string_from_error_code(status));

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

void XpuDeviceInterface::convert_av_frame_to_frame_output(
    const AVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  if (av_frame.format != AV_PIX_FMT_VAAPI) {
    // The frame's format is AV_PIX_FMT_VAAPI if and only if its content is on
    // the GPU. In this branch, the frame is on the CPU. This is what FFmpeg VAAPI
    // decoder gives us if it wasn't able to decode a frame, for whatever reason.
    // Typically that happens if the video's decoder isn't supported by VAAPI in
    // general or on this particular device. In this case we have a frame on the
    // CPU. We send the frame back to the XPU device when we're done.

    DEBUG_LOG(xpu::VERBOSE, "Incoming frame is on CPU, forwarding to CPU conversion");
    TORCH_CHECK(xpu::cpu_fallback(), "CPU fallback not allowed");
    ensure_cpu_interface();

    FrameOutput cpuFrameOutput;
    cpu_interface_->convert_av_frame_to_frame_output(av_frame, cpuFrameOutput);

    // Finally, we need to send the frame back to the GPU. Note that the
    // pre-allocated tensor is on the GPU, so we can't send that to the CPU
    // device interface. We copy it over here.
    if (pre_allocated_output_tensor.has_value()) {
      torch::stable::copy_(pre_allocated_output_tensor.value(), cpuFrameOutput.data);
      frame_output.data = pre_allocated_output_tensor.value();
    } else {
      frame_output.data = torch::stable::to(cpuFrameOutput.data, device_);
    }
    return;
  }

  TORCH_CHECK(
      av_frame.format == AV_PIX_FMT_VAAPI,
      "Expected format to be AV_PIX_FMT_VAAPI, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)av_frame.format)));
  auto frameDims = FrameDims(av_frame.height, av_frame.width);
  torch::stable::Tensor& dst = frame_output.data;
  if (pre_allocated_output_tensor.has_value()) {
    auto shape = pre_allocated_output_tensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
	    (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        int_array_ref_to_string(shape));
    dst = pre_allocated_output_tensor.value();
  } else {
    // Explicitly load the version defined in facebook::torchcodec::xpu
    // namespace as facebook::torchcodec defines the same but with the linkage
    // type which we can't use.
    dst = xpu::allocate_empty_hwc_tensor(frameDims, device_);
  }

  auto start = std::chrono::high_resolution_clock::now();
  if (!convert_av_frame_to_frame_output_with_sycl(av_frame, dst)) {
    convert_av_frame_to_frame_output_with_filter_graph(av_frame, dst);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  DEBUG_LOG(xpu::VERBOSE, "Conversion of frame height=" << frameDims.height << " width=" << frameDims.width
      << " took: " << duration.count() << "us" << std::endl);
}

void XpuDeviceInterface::convert_av_frame_to_frame_output_with_filter_graph(
    const AVFrame& av_frame,
    torch::stable::Tensor& dst) {
  DEBUG_LOG(xpu::VERBOSE, "Using VAAPI filter graph backend for conversion");
  auto frameDims = FrameDims(av_frame.height, av_frame.width);

  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(av_frame.format);
  FiltersConfig filtersConfig;

  filtersConfig.input_width = av_frame.width;
  filtersConfig.input_height = av_frame.height;
  filtersConfig.input_format = frameFormat;
  filtersConfig.input_aspect_ratio = av_frame.sample_aspect_ratio;
  // Actual output color format will be set via filter options
  filtersConfig.output_format = AV_PIX_FMT_VAAPI;
  filtersConfig.time_base = time_base_;
  filtersConfig.hw_frames_ctx.reset(av_buffer_ref(av_frame.hw_frames_ctx));

  std::stringstream filters;
  filters << "scale_vaapi=" << frameDims.width << ":" << frameDims.height;
  // CPU device interface outputs RGB in full (pc) color range.
  // We are doing the same to match.
  filters << ":format=rgba:out_range=pc";

  filtersConfig.filtergraph_str = filters.str();

  if (!filter_graph_ || prev_filters_config_ != filtersConfig) {
    filter_graph_ =
        std::make_unique<FilterGraph>(filtersConfig, video_stream_options_);
    prev_filters_config_ = std::move(filtersConfig);
  }

  // We convert input to the RGBX color format with VAAPI getting WxHx4
  // tensor on the output.
  UniqueAVFrame filteredAVFrame = filter_graph_->convert(av_frame);

  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_VAAPI);

  torch::stable::Tensor dst_rgb4 = AVFrameToTensor(device_, filteredAVFrame);
  torch::stable::copy_(dst, torch::stable::narrow(dst_rgb4, 2, 0, 3));
}

bool XpuDeviceInterface::convert_av_frame_to_frame_output_with_sycl(
    [[maybe_unused]] const AVFrame& frame,
    [[maybe_unused]] torch::stable::Tensor& dst) {
  bool converted = false;
  if (!xpu::use_sycl_color_conversion_kernel()) {
    return converted;
  }
  if (!has_fp64_) {
    return converted;
  }

#ifdef WITH_SYCL_KERNELS
  DEBUG_LOG(xpu::VERBOSE, "Using SYCL kernel backend for conversion");
  TORCH_CHECK_EQ(frame.format, AV_PIX_FMT_VAAPI);
  VADRMPRIMESurfaceDescriptor desc{};
  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(const_cast<AVFrame*>(&frame)),
      (VASurfaceID)(uintptr_t)frame.data[3],
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
      frame.width,
      frame.height,
      desc.layers[0].pitch[0],
      frame.color_range,
      frame.colorspace);

  zeMemFree(context->zeCtx, usm_ptr);
  converted = true;
#endif
  return converted;
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the XPU
// device
std::optional<const AVCodec*> XpuDeviceInterface::find_codec(
    const AVCodecID& codec_id,
    bool is_decoder) {
  DEBUG_LOG(xpu::INFO, "Looking for " << avcodec_get_name(codec_id)
    << (is_decoder ? " VAAPI decoder" : " VAAPI encoder"));
  if (!ctx_) {
    DEBUG_LOG(xpu::INFO, "No VAAPI context: deferring to software codec");
    return std::nullopt;
  }

  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (is_decoder) {
      if (codec->id != codec_id || !av_codec_is_decoder(codec)) {
        continue;
      }
    } else {
      if (codec->id != codec_id || !av_codec_is_encoder(codec)) {
        continue;
      }
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
        DEBUG_LOG(xpu::INFO, "Found VAAPI codec");
        return codec;
      }
    }
  }

  DEBUG_LOG(xpu::INFO, "VAAPI codec not found");
  return std::nullopt;
}

// ============================================================
// Encoding: getEncodingPixelFormat
// ============================================================
// On the VAAPI path, XPU encoders consume NV12; user-supplied pixel_format
// is rejected. On the CPU fallback path we can't defer to the CPU device
// interface as get_encoding_pixel_format() might be called before it is
// initialized (we can't initialize in constructor due to mutex deadlock).
// So we for all cases consume NV12.
AVPixelFormat XpuDeviceInterface::get_encoding_pixel_format(
    [[maybe_unused]] const AVCodec& av_codec,
    const std::optional<std::string>& user_pixel_format) const {
  STD_TORCH_CHECK(
      !user_pixel_format.has_value(),
      "Video encoding on XPU currently only supports the nv12 pixel format. "
      "Do not set pixel_format to use nv12 by default.");
  return AV_PIX_FMT_NV12;
}

// ============================================================
// Encoding: setupHardwareFrameContextForEncoding
// ============================================================
// On the VAAPI path, the encoder can write directly into VAAPI surfaces (NV12 layout).
// On the CPU fallback path, the SW encoder consumes plain CPU AVFrames.
void XpuDeviceInterface::setup_hardware_frame_context_for_encoding(
    AVCodecContext* codec_context) {
  if (!ctx_) {
    DEBUG_LOG(xpu::INFO, "No VAAPI context: skipping hw_frames_ctx setup (will fall to CPU encoding)");
    // On the CPU fallback path force real-time SW encoding: without tune=zerolatency
    // libx264 buffers frames, breaking mid-encode fragmented-MP4 reads (mov seek EPERM).
    if (codec_context != nullptr){
      if (codec_context->priv_data != nullptr){
        av_opt_set(codec_context->priv_data, "tune", "zerolatency",
                  AV_OPT_SEARCH_CHILDREN);
      }
      codec_context->max_b_frames = 0;
    }
    return;
  }
  TORCH_CHECK(codec_context != nullptr, "codec_context is null");

  AVBufferRef* hwFramesCtxRef = av_hwframe_ctx_alloc(ctx_.get());
  TORCH_CHECK(
      hwFramesCtxRef != nullptr,
      "Failed to allocate VAAPI hw frames context for codec");

  // sw_pix_fmt: the software (CPU-accessible) format the encoder consumes inside the surface
  // pix_fmt:    the hardware wrapper format the codec sees (must match hw_frames_ctx->format)
  codec_context->sw_pix_fmt = AV_PIX_FMT_NV12;
  codec_context->pix_fmt    = AV_PIX_FMT_VAAPI;

  auto* hwFramesCtx = reinterpret_cast<AVHWFramesContext*>(hwFramesCtxRef->data);
  hwFramesCtx->format    = AV_PIX_FMT_VAAPI;
  hwFramesCtx->sw_format = AV_PIX_FMT_NV12;
  hwFramesCtx->width     = codec_context->width;
  hwFramesCtx->height    = codec_context->height;

  // XPU quality is materially better when using BT.709 color space and full range
  // (JPEG) for encoding.
  if (codec_context->color_range == AVCOL_RANGE_UNSPECIFIED) {
      codec_context->color_range = AVCOL_RANGE_JPEG;
  }
  if (codec_context->colorspace == AVCOL_SPC_UNSPECIFIED) {
      codec_context->colorspace = AVCOL_SPC_BT709;
  }
  if (codec_context->color_primaries == AVCOL_PRI_UNSPECIFIED) {
      codec_context->color_primaries = AVCOL_PRI_BT709;
  }
  if (codec_context->color_trc == AVCOL_TRC_UNSPECIFIED) {
      codec_context->color_trc = AVCOL_TRC_BT709;
  }

  // Force non-LP VAAPI engine for better quality under async_depth=1;
  // caller-supplied low_power via extra_options overrides this later.
  if (codec_context->priv_data != nullptr) {
      av_opt_set(codec_context->priv_data, "low_power", "0",
                 AV_OPT_SEARCH_CHILDREN);
      // `quality=1` selects the highest-quality preset on VAAPI codecs that
      // expose it. av_opt_set is silent on codecs that don't, so it's safe.
      av_opt_set(codec_context->priv_data, "quality", "1",
                 AV_OPT_SEARCH_CHILDREN);
  }

  // Disable B-frames (mirrors CUDA/NVENC delay=0) to avoid frames reordering.
  // Callers can restore via extra_options={"bf": "2"}.
  codec_context->max_b_frames = 0;

  int ret = av_hwframe_ctx_init(hwFramesCtxRef);
  if (ret < 0) {
    av_buffer_unref(&hwFramesCtxRef);
    TORCH_CHECK(
        false,
        "Failed to initialize VAAPI hw frames context: ",
        get_ffmpeg_error_string_from_error_code(ret));
  }
  codec_context->hw_frames_ctx = hwFramesCtxRef;
}

// ============================================================
// Encoding: convertTensorToAVFrameForEncoding
// ============================================================
UniqueAVFrame XpuDeviceInterface::convert_tensor_to_av_frame_for_encoding(
    const torch::stable::Tensor& tensor,
    int frame_index,
    AVCodecContext* codec_context) {
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.sizes()[0] == 3,
      "Expected 3D RGB tensor (CHW format), got ",
      tensor.dim(),
      "D tensor");
  TORCH_CHECK(
      tensor.device().type() == kStableXPU,
      "Expected tensor on XPU device, got: ",
      device_type_name(tensor.device().type()));
  const int width = static_cast<int>(tensor.sizes()[2]);
  const int height = static_cast<int>(tensor.sizes()[1]);
  TORCH_CHECK(
      (width % 2) == 0 && (height % 2) == 0,
      "NV12 encoding requires even width/height, got ",
      width,
      "x",
      height);
  TORCH_CHECK(codec_context != nullptr, "codec_context is null");

  // No VAAPI context: copy tensor to CPU and delegate to CpuDeviceInterface.
  if (!ctx_) {
    TORCH_CHECK(xpu::cpu_fallback(), "CPU fallback not allowed");
    ensure_cpu_interface();
    auto cpu_tensor = torch::stable::empty(
        {tensor.sizes()[0], tensor.sizes()[1], tensor.sizes()[2]},
        kStableUInt8, std::nullopt, StableDevice(kStableCPU));
    torch::stable::copy_(cpu_tensor, tensor);
    return cpu_interface_->convert_tensor_to_av_frame_for_encoding(
        cpu_tensor, frame_index, codec_context);
  }

  TORCH_CHECK(
      codec_context->hw_frames_ctx != nullptr,
      "hw_frames_ctx is null: call setupHardwareFrameContextForEncoding first");

  // Try the optimized SYCL path first. It returns a null UniqueAVFrame when
  // SYCL is unavailable (USE_SYCL_KERNELS disabled, no FP64 support, or built
  // without WITH_SYCL_KERNELS). In that case, fall back to the CPU path.
  UniqueAVFrame avFrame =
      convert_tensor_to_av_frame_for_encoding_with_sycl(tensor, frame_index, codec_context);
  if (avFrame) {
    return avFrame;
  }

  return convert_tensor_to_av_frame_for_encoding_with_cpu(tensor, frame_index, codec_context);
}

// ============================================================
// Encoding: convertTensorToAVFrameForEncoding_SYCL
// ============================================================
UniqueAVFrame XpuDeviceInterface::convert_tensor_to_av_frame_for_encoding_with_sycl(
    [[maybe_unused]] const torch::stable::Tensor& tensor,
    [[maybe_unused]] int frame_index,
    [[maybe_unused]] AVCodecContext* codec_context) {
  if (!xpu::use_sycl_color_conversion_kernel()) {
    return UniqueAVFrame();
  }
  if (!has_fp64_) {
    return UniqueAVFrame();
  }

  UniqueAVFrame vaFrame;
#ifdef WITH_SYCL_KERNELS
  DEBUG_LOG(xpu::VERBOSE, "Using SYCL kernel backend for conversion");

  const int width  = static_cast<int>(tensor.sizes()[2]);
  const int height = static_cast<int>(tensor.sizes()[1]);
  vaFrame = xpu::allocNV12Frame(width, height, frame_index, codec_context);

  VADisplay display = getVaDisplayFromAV(vaFrame.get());
  VASurfaceID surfaceId = (VASurfaceID)(uintptr_t)vaFrame->data[3];

  VADRMPRIMESurfaceDescriptor desc{};
  VAStatus sts = vaExportSurfaceHandle(
      display,
      surfaceId,
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_WRITE_ONLY,  // write for encoding (vs. READ_ONLY for decoding)
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle (WRITE_ONLY) failed: ",
      vaErrorStr(sts));
  TORCH_CHECK(desc.num_objects == 1, "Expected 1 DMA-BUF object, got ", desc.num_objects);
  // NV12 export layouts seen on Intel iHD/i915:
  //   A: 1 layer × 2 planes (Y, UV).   B: 2 layers × 1 plane (Y; UV) — used by BMG.
  // Both describe the same DMA-BUF; only the plane offsets/pitches differ.
  const bool layoutA = (desc.num_layers == 1 && desc.layers[0].num_planes == 2);
  const bool layoutB = (desc.num_layers == 2 && desc.layers[0].num_planes == 1
                        && desc.layers[1].num_planes == 1);
  TORCH_CHECK(
      layoutA || layoutB,
      "Unsupported NV12 export layout: num_layers=", desc.num_layers,
      " layers[0].num_planes=", desc.layers[0].num_planes);

  // Get Level Zero context and device handles via SYCL interop.
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device_.index());
  ze_context_handle_t zeCtx  = nullptr;
  ze_device_handle_t  zeDevice = nullptr;
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          zeCtx    = ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          zeDevice = ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd    = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;
  ze_result_t res = zeMemAllocDevice(
      zeCtx, &alloc_desc, desc.objects[0].size, 0, zeDevice, &usm_ptr);

  if (res != ZE_RESULT_SUCCESS) {
    close(desc.objects[0].fd);
    TORCH_CHECK(
        false,
        "zeMemAllocDevice failed importing encode surface fd=",
        desc.objects[0].fd);
   }

  // Extract Y and UV plane pointers and pitches for both layouts
  uint8_t* y_ptr;
  uint8_t* uv_ptr;
  int y_pitch, uv_pitch;
  if (layoutA) {
    y_ptr    = static_cast<uint8_t*>(usm_ptr) + desc.layers[0].offset[0];
    uv_ptr   = static_cast<uint8_t*>(usm_ptr) + desc.layers[0].offset[1];
    y_pitch  = static_cast<int>(desc.layers[0].pitch[0]);
    uv_pitch = static_cast<int>(desc.layers[0].pitch[1]);
  } else {
    y_ptr    = static_cast<uint8_t*>(usm_ptr) + desc.layers[0].offset[0];
    uv_ptr   = static_cast<uint8_t*>(usm_ptr) + desc.layers[1].offset[0];
    y_pitch  = static_cast<int>(desc.layers[0].pitch[0]);
    uv_pitch = static_cast<int>(desc.layers[1].pitch[0]);
  }

  // drm_format_modifier != 0 means tiled (e.g. Intel Tile-Y on BMG/Gen12+).
  const bool is_tiled = (desc.objects[0].drm_format_modifier != 0);
  convertRGBToNV12(
      queue,
      static_cast<const uint8_t*>(tensor.data_ptr()),
      tensor.strides()[0],   // ch_stride
      tensor.strides()[1],   // row_stride
      tensor.strides()[2],   // pixel_stride
      y_ptr,
      uv_ptr,
      vaFrame->width,
      vaFrame->height,
      y_pitch,
      uv_pitch,
      is_tiled,
      codec_context->color_range,
      codec_context->colorspace);

  zeMemFree(zeCtx, usm_ptr);
  close(desc.objects[0].fd);

  vaFrame->colorspace  = codec_context->colorspace;
  vaFrame->color_range = codec_context->color_range;
#endif
  return vaFrame;
}

// ============================================================
// Encoding: convertTensorToAVFrameForEncoding_CPU  (CPU fallback)
// ============================================================
UniqueAVFrame XpuDeviceInterface::convert_tensor_to_av_frame_for_encoding_with_cpu(
    const torch::stable::Tensor& tensor,
    int frame_index,
    AVCodecContext* codec_context) {
  DEBUG_LOG(xpu::VERBOSE, "Using CPU for conversion");

  const int width  = static_cast<int>(tensor.sizes()[2]);
  const int height = static_cast<int>(tensor.sizes()[1]);
  UniqueAVFrame vaFrame =
      xpu::allocNV12Frame(width, height, frame_index, codec_context);

  // Move XPU tensor to CPU (blocking)
  torch::stable::Tensor cpuTensor =
      torch::stable::to(tensor, StableDevice(kStableCPU, 0));

  const uint8_t* data = static_cast<const uint8_t*>(cpuTensor.data_ptr());
  // strides() are in elements (uint8), so they equal byte strides here.
  int64_t ch_stride  = cpuTensor.strides()[0];
  int64_t row_stride = cpuTensor.strides()[1];

  // Allocate an intermediate CPU NV12 frame for sws_scale output
  UniqueAVFrame cpuFrame(av_frame_alloc());
  TORCH_CHECK(cpuFrame != nullptr, "Failed to allocate CPU NV12 AVFrame");
  cpuFrame->format = AV_PIX_FMT_NV12;
  cpuFrame->width  = vaFrame->width;
  cpuFrame->height = vaFrame->height;
  int ret = av_frame_get_buffer(cpuFrame.get(), 0);
  TORCH_CHECK(ret >= 0, "av_frame_get_buffer (NV12) failed: ",
              get_ffmpeg_error_string_from_error_code(ret));

  // Zero-copy GBRP view of the NCHW tensor (GBRP plane order: G=ch1, B=ch2, R=ch0).
  UniqueAVFrame gbrpFrame(av_frame_alloc());
  TORCH_CHECK(gbrpFrame != nullptr, "Failed to allocate GBRP AVFrame");
  gbrpFrame->format = AV_PIX_FMT_GBRP;
  gbrpFrame->width  = vaFrame->width;
  gbrpFrame->height = vaFrame->height;
  gbrpFrame->data[0] = const_cast<uint8_t*>(data + 1 * ch_stride);  // G
  gbrpFrame->data[1] = const_cast<uint8_t*>(data + 2 * ch_stride);  // B
  gbrpFrame->data[2] = const_cast<uint8_t*>(data + 0 * ch_stride);  // R
  gbrpFrame->linesize[0] = static_cast<int>(row_stride);
  gbrpFrame->linesize[1] = static_cast<int>(row_stride);
  gbrpFrame->linesize[2] = static_cast<int>(row_stride);

  // Higher-quality chroma downsampling (GBRP is 4:4:4, NV12 is 4:2:0):
  //   SWS_BICUBIC        - better than bilinear on edges
  //   SWS_ACCURATE_RND   - full-precision rounding in the fixed-point math
  //   SWS_FULL_CHR_H_INT - full horizontal chroma interpolation on output
  const int swsFlags = SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT;

  // GBRP -> NV12 via libswscale
  SwsContext* swsCtx = sws_getContext(
      vaFrame->width, vaFrame->height, AV_PIX_FMT_GBRP,
      vaFrame->width, vaFrame->height, AV_PIX_FMT_NV12,
      swsFlags, nullptr, nullptr, nullptr);
  TORCH_CHECK(swsCtx != nullptr, "sws_getContext(GBRP->NV12) failed");

  const int dstRange = (codec_context->color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
  const int cs = (codec_context->colorspace == AVCOL_SPC_BT709)
                   ? SWS_CS_ITU709
                   : SWS_CS_ITU601;
  sws_setColorspaceDetails(
      swsCtx,
      sws_getCoefficients(cs), /*srcRange=*/1, 
      sws_getCoefficients(cs), dstRange, 
      /*brightness=*/0, /*contrast=*/1 << 16, /*saturation=*/1 << 16);

  int sws_ret = sws_scale(
      swsCtx,
      gbrpFrame->data,
      gbrpFrame->linesize,
      0,
      vaFrame->height,
      cpuFrame->data,
      cpuFrame->linesize);
  sws_freeContext(swsCtx);
  TORCH_CHECK(
      sws_ret >= 0,
      "sws_scale(GBRP->NV12) failed: expected ",
      get_ffmpeg_error_string_from_error_code(sws_ret))
  TORCH_CHECK(
      sws_ret == vaFrame->height,
      "sws_scale(GBRP->NV12) failed: expected ", sws_ret,
      "output lines, expected", vaFrame->height);

  // Upload CPU NV12 -> VAAPI surface
  ret = av_hwframe_transfer_data(vaFrame.get(), cpuFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "av_hwframe_transfer_data (NV12->VAAPI) failed: ",
      get_ffmpeg_error_string_from_error_code(ret));

  vaFrame->colorspace  = codec_context->colorspace;
  vaFrame->color_range = codec_context->color_range;
  return vaFrame;
}

} // namespace facebook::torchcodec
