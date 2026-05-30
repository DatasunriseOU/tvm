/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file metal_module.mm
 * \brief MetalModuleNode — runtime-side, plugin-only.  Reachable from C++
 *        only through the FFI registry keys "ffi.Module.create.metal" and
 *        "ffi.Module.load_from_bytes.metal".  No exported header — codegen-
 *        side construction goes through src/target/metal/metal_fallback_module.h.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/support/io.h>
#include <array>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include "../../support/bytes_io.h"
#include "../file_utils.h"
#include "../metadata.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "metal_common.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of GPU supported in MetalModule. */
static constexpr const int kMetalMaxNumDevice = 32;

// Module to support thread-safe multi-GPU execution.
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class MetalModuleNode final : public ffi::ModuleObj {
 public:
  // Unified factory signature shared with the codegen-side fallback in
  // src/target/metal/metal_fallback_module.h.  The per-kernel `smap`
  // payload is Map<String, Bytes> regardless of whether the format is
  // text MSL ("metal") or compiled metallib ("metallib") — text vs binary
  // distinction lives in `fmt`.
  MetalModuleNode(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                  ffi::Map<ffi::String, FunctionInfo> fmap,
                  ffi::Map<ffi::String, ffi::String> source)
      : smap_(std::move(smap)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  const char* kind() const final { return "metal"; }

  /*! \brief Get the property of the runtime module. */
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    // 3 fields [fmt][fmap][smap].  Source map is in-memory inspection only
    // and is NEVER serialized — matches the cross-backend rule.
    // MetalFallbackModuleNode::SaveToBytes (in
    // src/target/metal/metal_fallback_module.cc) MUST mirror this format
    // byte-for-byte; see one-way comment there.
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(smap_);
    return ffi::Bytes(std::move(result));
  }
  ffi::String InspectSource(const ffi::String& format) const final {
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      if (auto it = source_.find("metal"); it != source_.end()) {
        return (*it).second;
      }
    }
    return ffi::String();
  }

  // Expose the per-function metadata so the direct-launch C API
  // (TVMMetalCreateDirectLaunchHandle) can build a launch handle without going
  // through the packed-func calling convention.
  ffi::Optional<FunctionInfo> GetFunctionInfo(const std::string& name) const {
    return fmap_.Get(name);
  }

  // get a from primary context in device_id
  id<MTLComputePipelineState> GetPipelineState(size_t device_id, const std::string& func_name) {
    metal::MetalWorkspace* w = metal::MetalWorkspace::Global();
    TVM_FFI_ICHECK_LT(device_id, w->devices.size());
    // start lock scope.
    std::lock_guard<std::mutex> lock(mutex_);
    if (finfo_.size() <= device_id) {
      finfo_.resize(device_id + 1, DeviceEntry());
    }
    DeviceEntry& e = finfo_[device_id];
    auto it = e.smap.find(func_name);
    if (it != e.smap.end()) return it->second;
    // compile
    NSError* err_msg = nil;
    id<MTLLibrary> lib = nil;
    auto kernel = smap_.find(func_name);
    // Directly lookup kernels
    TVM_FFI_ICHECK(kernel != smap_.end());
    const ffi::Bytes& source = (*kernel).second;

    if (fmt_ == "metal") {
      MTLCompileOptions* opts = [MTLCompileOptions alloc];
      opts.languageVersion = MTLLanguageVersion2_3;
      opts.fastMathEnabled = YES;
      // opts = nil;
      // Per-kernel payload is bytes; treat as UTF-8 MSL source.
      std::string source_str(source.data(), source.size());
      // Metal 4 cooperative-tensor (mpp::tensor_ops) kernels require MSL 4.0.
      // The runtime JIT (newLibraryWithSource) -- used by the tvm_ffi backend --
      // otherwise caps at MSL 2.3, which rejects the MPP header with
      // "undeclared identifier 'mpp'".  This is the *only* language gate: the
      // mpp ops run on Apple M4 (and presumably M3), not just M5.  Raise to
      // MSL 4.0 when the kernel includes the MPP header and the OS/SDK support
      // it.  (NB: register-heavy kernels can still fail pipeline-state creation
      // with "exceeds available stack space" -- that is a separate occupancy
      // limit, not a language-version or mpp-capability gate.)
      if (source_str.find("MetalPerformancePrimitives") != std::string::npos) {
        if (@available(macOS 26.0, iOS 26.0, *)) {
          opts.languageVersion = MTLLanguageVersion4_0;
        }
      }
      lib = [w->devices[device_id]
          newLibraryWithSource:[NSString stringWithUTF8String:source_str.c_str()]
                       options:opts
                         error:&err_msg];
      [opts dealloc];
      if (lib == nil) {
        LOG(FATAL) << "Fail to compile metal source:"
                   << [[err_msg localizedDescription] UTF8String];
      }
      if (err_msg != nil) {
        LOG(INFO) << "Warning: " << [[err_msg localizedDescription] UTF8String];
      }
    } else {
      // Build from library.
      auto q = dispatch_queue_create("q", DISPATCH_QUEUE_SERIAL);
      auto data = dispatch_data_create(source.data(), source.size(), q,
                                       ^{
                                       });
      lib = [w->devices[device_id] newLibraryWithData:data error:&err_msg];
      if (err_msg != nil || lib == nil) {
        LOG(FATAL) << "Fail to compile metal lib:" << [[err_msg localizedDescription] UTF8String];
      }
    }
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:func_name.c_str()]];
    TVM_FFI_ICHECK(f != nil) << "cannot find function " << func_name;
    id<MTLComputePipelineState> state =
        [w->devices[device_id] newComputePipelineStateWithFunction:f error:&err_msg];
    TVM_FFI_ICHECK(state != nil) << "cannot get state:"
                                 << " for function " << func_name
                                 << [[err_msg localizedDescription] UTF8String];
    [f release];
    [lib release];
    // The state.threadExecutionWidth can change dynamically according
    // to the resource constraint in kernel, so it is not strictly hold
    // Turn of warp aware optimziation for now.
    // TVM_FFI_ICHECK_EQ(state.threadExecutionWidth, w->warp_size[device_id]);
    if (e.smap[func_name] != nil) [e.smap[func_name] release];
    e.smap[func_name] = state;
    return state;
  }

 private:
  // device specific entry
  struct DeviceEntry {
    // state cache;
    std::unordered_map<std::string, id<MTLComputePipelineState>> smap;

    ~DeviceEntry() {
      for (auto&& kv : smap) {
        [kv.second release];
      }
    }
  };
  // Per-kernel payload: kernel-name -> bytes (MSL source for fmt="metal" /
  // metallib blob for fmt="metallib").
  ffi::Map<ffi::String, ffi::Bytes> smap_;
  // The format ("metal" source / "metallib" compiled).
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
  // function information.
  std::vector<DeviceEntry> finfo_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MetalWrappedFunc {
 public:
  // initialize the METAL function.
  void Init(MetalModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            size_t num_buffer_args, size_t num_pack_args,
            const ffi::Array<ffi::String>& launch_param_tags) {
    w_ = metal::MetalWorkspace::Global();
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_buffer_args_ = num_buffer_args;
    num_pack_args_ = num_pack_args;
    std::fill(scache_.begin(), scache_.end(), (id<MTLComputePipelineState>)nil);
    launch_param_config_.Init(num_buffer_args + num_pack_args, launch_param_tags);
    metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
    int dev_id = t->device.device_id;
    scache_[dev_id] = m->GetPipelineState(dev_id, func_name);
  }
  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, const ArgUnion64* pack_args) const {
    AUTORELEASEPOOL {
      metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
      int device_id = t->device.device_id;
      metal::MetalThreadEntry::ExternalCommandBufferState external_state =
          t->GetExternalCommandBufferState(device_id);
      id<MTLCommandBuffer> external_command_buffer = external_state.command_buffer;
      metal::Stream* stream = nullptr;

      if (external_command_buffer == nil) {
        // obtain the stream
        stream =
            metal::MetalWorkspace::Global()->CastStreamOrGetDefault(t->stream[device_id], device_id);

        // skip launching so the error can be printed during sync
        if (stream->HasErrorHappened()) return;
      }

      if (scache_[device_id] == nil) {
        scache_[device_id] = m_->GetPipelineState(device_id, func_name_);
      }
      ThreadWorkLoad wl = launch_param_config_.Extract(args);
      int blockSize = wl.block_dim(0) * wl.block_dim(1) * wl.block_dim(2);
      auto maxTotalThreadsPerThreadgroup = scache_[device_id].maxTotalThreadsPerThreadgroup;
      TVM_FFI_ICHECK_LE(blockSize, maxTotalThreadsPerThreadgroup);
      id<MTLComputeCommandEncoder> encoder;
      if (external_command_buffer != nil) {
        if (external_state.wait_event != nil) {
          [external_command_buffer encodeWaitForEvent:external_state.wait_event
                                                value:external_state.wait_value];
        }
        encoder = [external_command_buffer computeCommandEncoder];
        TVM_FFI_ICHECK(encoder != nil)
            << "Failed to create Metal compute encoder on external command buffer";
      } else {
        // Reuse the pending compute encoder to batch dispatches.
        // The encoder is flushed on sync, copy, or buffer deallocation.
        encoder = stream->GetPendingComputeEncoder(func_name_);
      }
      [encoder setComputePipelineState:scache_[device_id]];
      for (size_t i = 0; i < num_buffer_args_; ++i) {
        void* buf = args[static_cast<int>(i)].cast<void*>();
        [encoder setBuffer:(id<MTLBuffer>)(buf) offset:0 atIndex:i];
      }
      if (num_pack_args_ != 0) {
        [encoder setBytes:pack_args
                   length:num_pack_args_ * sizeof(ArgUnion64)
                  atIndex:num_buffer_args_];
      }
      // launch
      MTLSize dimGrid = MTLSizeMake(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
      MTLSize dimBlock = MTLSizeMake(wl.block_dim(0), wl.block_dim(1), wl.block_dim(2));
      [encoder dispatchThreadgroups:dimGrid threadsPerThreadgroup:dimBlock];
      if (external_command_buffer != nil) {
        [encoder endEncoding];
        if (external_state.signal_event != nil) {
          [external_command_buffer encodeSignalEvent:external_state.signal_event
                                               value:external_state.signal_value];
        }
      }
    };
  }

 private:
  // Reference to global workspace.
  metal::MetalWorkspace* w_;
  // internal module
  MetalModuleNode* m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Number of buffer arguments
  size_t num_buffer_args_;
  // number of packed arguments.
  size_t num_pack_args_;
  // Device state cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<id<MTLComputePipelineState>, kMetalMaxNumDevice> scache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

// Direct-launch handle: lets the MLX/TVM-FFI bridge dispatch a buffer-only
// Metal kernel onto MLX's borrowed command buffer / compute encoder using raw
// MTLBuffer pointers + an int64 launch-arg array, bypassing the packed-func
// DLTensor unpacking path. Used by the TVMMetalDirectLaunch* C API below.
class MetalDirectLaunchHandle {
 public:
  void Init(MetalModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            FunctionInfo info) {
    w_ = metal::MetalWorkspace::Global();
    m_ = m;
    sptr_ = std::move(sptr);
    func_name_ = func_name;
    num_buffer_args_ = NumBufferArgs(info->arg_types);
    TVM_FFI_ICHECK_EQ(num_buffer_args_, info->arg_types.size())
        << "TVM Metal direct launch supports buffer-only kernels; "
        << func_name << " has " << (info->arg_types.size() - num_buffer_args_)
        << " non-buffer arguments";
    std::fill(scache_.begin(), scache_.end(), (id<MTLComputePipelineState>)nil);
    InitLaunchParamMap(info->launch_param_tags);
    metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
    int dev_id = t->device.device_id;
    scache_[dev_id] = m_->GetPipelineState(dev_id, func_name_);
  }

  void Launch(void** buffers, int32_t num_buffers, const int64_t* launch_args,
              int32_t num_launch_args) const {
    AUTORELEASEPOOL {
      TVM_FFI_ICHECK_EQ(static_cast<size_t>(num_buffers), num_buffer_args_);
      TVM_FFI_ICHECK(launch_args != nullptr || num_launch_args == 0);
      metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
      int device_id = t->device.device_id;
      metal::MetalThreadEntry::ExternalCommandBufferState external_state =
          t->GetExternalCommandBufferState(device_id);
      id<MTLCommandBuffer> external_command_buffer = external_state.command_buffer;
      id<MTLComputeCommandEncoder> external_compute_encoder = external_state.compute_encoder;
      metal::Stream* stream = nullptr;

      if (external_command_buffer == nil && external_compute_encoder == nil) {
        stream =
            metal::MetalWorkspace::Global()->CastStreamOrGetDefault(t->stream[device_id], device_id);
        if (stream->HasErrorHappened()) return;
      }

      if (scache_[device_id] == nil) {
        scache_[device_id] = m_->GetPipelineState(device_id, func_name_);
      }
      ThreadWorkLoad wl = ExtractWorkLoad(launch_args, num_launch_args);
      int blockSize = wl.block_dim(0) * wl.block_dim(1) * wl.block_dim(2);
      auto maxTotalThreadsPerThreadgroup = scache_[device_id].maxTotalThreadsPerThreadgroup;
      TVM_FFI_ICHECK_LE(blockSize, maxTotalThreadsPerThreadgroup);
      id<MTLComputeCommandEncoder> encoder;
      if (external_compute_encoder != nil) {
        encoder = external_compute_encoder;
      } else if (external_command_buffer != nil) {
        if (external_state.wait_event != nil) {
          [external_command_buffer encodeWaitForEvent:external_state.wait_event
                                                value:external_state.wait_value];
        }
        encoder = [external_command_buffer computeCommandEncoder];
        TVM_FFI_ICHECK(encoder != nil)
            << "Failed to create Metal compute encoder on external command buffer";
      } else {
        encoder = stream->GetPendingComputeEncoder(func_name_);
      }
      [encoder setComputePipelineState:scache_[device_id]];
      for (size_t i = 0; i < num_buffer_args_; ++i) {
        [encoder setBuffer:(id<MTLBuffer>)(buffers[i]) offset:0 atIndex:i];
      }
      MTLSize dimGrid = MTLSizeMake(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
      MTLSize dimBlock = MTLSizeMake(wl.block_dim(0), wl.block_dim(1), wl.block_dim(2));
      [encoder dispatchThreadgroups:dimGrid threadsPerThreadgroup:dimBlock];
      if (external_command_buffer != nil) {
        [encoder endEncoding];
        if (external_state.signal_event != nil) {
          [external_command_buffer encodeSignalEvent:external_state.signal_event
                                               value:external_state.signal_value];
        }
      }
    };
  }

 private:
  void InitLaunchParamMap(const ffi::Array<ffi::String>& launch_param_tags) {
    expected_launch_args_ = 0;
    for (size_t i = 0; i < launch_param_tags.size(); ++i) {
      std::string tag(launch_param_tags[i]);
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        TVM_FFI_ICHECK_EQ(i, launch_param_tags.size() - 1)
            << "kUseDynamicSharedMemoryTag should be the last tag in launch_param_tags.";
        use_dyn_shared_memory_ = true;
        ++expected_launch_args_;
      } else if (tag == launch_param::kUseProgramaticDependentLaunch) {
        TVM_FFI_THROW(InternalError)
            << "TVM Metal direct launch does not support programmatic dependent launch";
      } else if (tag == launch_param::kUseCooperativeLaunch) {
        TVM_FFI_THROW(InternalError)
            << "TVM Metal direct launch does not support cooperative launch";
      } else {
        ThreadScope ts = ThreadScope::Create(tag);
        launch_arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
        ++expected_launch_args_;
      }
    }
  }

  ThreadWorkLoad ExtractWorkLoad(const int64_t* launch_args, int32_t num_launch_args) const {
    TVM_FFI_ICHECK_EQ(static_cast<size_t>(num_launch_args), expected_launch_args_);
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    size_t arg_pos = 0;
    for (uint32_t index : launch_arg_index_map_) {
      size_t size = static_cast<size_t>(launch_args[arg_pos++]);
      if (size > 0) {
        w.work_size[index] = size;
      }
    }
    if (use_dyn_shared_memory_) {
      w.dyn_shmem_size = static_cast<size_t>(launch_args[arg_pos++]);
    }
    return w;
  }

  metal::MetalWorkspace* w_{nullptr};
  MetalModuleNode* m_{nullptr};
  ffi::ObjectPtr<ffi::Object> sptr_;
  std::string func_name_;
  size_t num_buffer_args_{0};
  mutable std::array<id<MTLComputePipelineState>, kMetalMaxNumDevice> scache_;
  std::vector<uint32_t> launch_arg_index_map_;
  size_t expected_launch_args_{0};
  bool use_dyn_shared_memory_{false};
};

static thread_local std::string metal_direct_launch_last_error;

static void SetMetalDirectLaunchLastError(std::string error) {
  metal_direct_launch_last_error = std::move(error);
}

static std::string CurrentExceptionMessage() {
  try {
    throw;
  } catch (const tvm::ffi::Error& exc) {
    return exc.FullMessage();
  } catch (const std::exception& exc) {
    return exc.what();
  } catch (...) {
    return "unknown exception";
  }
}

ffi::Optional<ffi::Function> MetalModuleNode::GetFunction(const ffi::String& name) {
  ffi::Function ret;
  AUTORELEASEPOOL {
    ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
    TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
    auto opt_info = fmap_.Get(name);
    if (!opt_info.has_value()) {
      return;
    }
    FunctionInfo info = opt_info.value();
    MetalWrappedFunc f;
    size_t num_buffer_args = NumBufferArgs(info->arg_types);
    f.Init(this, sptr_to_self, name, num_buffer_args, info->arg_types.size() - num_buffer_args,
           info->launch_param_tags);
    ret = PackFuncNonBufferArg(f, info->arg_types);
  };
  return ret;
}

extern "C" TVM_RUNTIME_DLL_EXPORT void* TVMMetalCreateDirectLaunchHandle(
    TVMFFIObjectHandle module_handle, const char* func_name) {
  SetMetalDirectLaunchLastError("");
  try {
    TVM_FFI_ICHECK(module_handle != nullptr);
    TVM_FFI_ICHECK(func_name != nullptr);
    auto* module_obj = ffi::details::ObjectUnsafe::RawObjectPtrFromUnowned<ffi::ModuleObj>(
        static_cast<TVMFFIObject*>(module_handle));
    TVM_FFI_ICHECK(module_obj != nullptr);
    TVM_FFI_ICHECK(std::string(module_obj->kind()) == "metal")
        << "TVM Metal direct launch requires a metal module, got " << module_obj->kind();
    auto* metal_module = static_cast<MetalModuleNode*>(module_obj);
    auto opt_info = metal_module->GetFunctionInfo(func_name);
    TVM_FFI_ICHECK(opt_info.has_value())
        << "Cannot create TVM Metal direct launch handle for missing function " << func_name;
    auto sptr = ffi::details::ObjectUnsafe::ObjectPtrFromUnowned<ffi::Object>(
        static_cast<TVMFFIObject*>(module_handle));
    auto* handle = new MetalDirectLaunchHandle();
    handle->Init(metal_module, std::move(sptr), func_name, opt_info.value());
    return handle;
  } catch (...) {
    SetMetalDirectLaunchLastError(CurrentExceptionMessage());
    return nullptr;
  }
}

extern "C" TVM_RUNTIME_DLL_EXPORT void TVMMetalReleaseDirectLaunchHandle(void* handle) {
  delete static_cast<MetalDirectLaunchHandle*>(handle);
}

extern "C" TVM_RUNTIME_DLL_EXPORT int TVMMetalDirectLaunch(
    void* handle, void** buffers, int32_t num_buffers, const int64_t* launch_args,
    int32_t num_launch_args) {
  SetMetalDirectLaunchLastError("");
  try {
    TVM_FFI_ICHECK(handle != nullptr);
    static_cast<MetalDirectLaunchHandle*>(handle)->Launch(
        buffers, num_buffers, launch_args, num_launch_args);
    return 0;
  } catch (...) {
    SetMetalDirectLaunchLastError(CurrentExceptionMessage());
    return -1;
  }
}

extern "C" TVM_RUNTIME_DLL_EXPORT const char* TVMMetalDirectLaunchLastError() {
  return metal_direct_launch_last_error.c_str();
}

static ffi::Module MetalModuleCreateImpl(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                         ffi::Map<ffi::String, FunctionInfo> fmap,
                                         ffi::Map<ffi::String, ffi::String> source) {
  ffi::ObjectPtr<MetalModuleNode> n;
  AUTORELEASEPOOL {
    n = ffi::make_object<MetalModuleNode>(std::move(smap), std::move(fmt), std::move(fmap),
                                          std::move(source));
  };
  return ffi::Module(n);
}

static ffi::Module MetalModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Map<ffi::String, ffi::Bytes> smap;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&smap);
  // Source map is not serialized — reconstructed empty on load.
  return MetalModuleCreateImpl(std::move(smap), std::move(fmt), std::move(fmap),
                               ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.metal" — codegen-time Metal module factory.
  // Used by src/target/metal/metal_fallback_module.h:MetalModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.metal" — disk loader.  Only this
  // (real) module registers a loader; the fallback is codegen-only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.metal", MetalModuleLoadFromBytes)
      .def("ffi.Module.create.metal",
           [](ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
              ffi::Map<ffi::String, FunctionInfo> fmap, ffi::Map<ffi::String, ffi::String> source) {
             return MetalModuleCreateImpl(std::move(smap), std::move(fmt), std::move(fmap),
                                          std::move(source));
           });
}
}  // namespace runtime
}  // namespace tvm
