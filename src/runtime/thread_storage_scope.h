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
 * \file thread_storage_scope.h
 * \brief Extract launch parameters configuration from ffi::PackedArgs.
 */
#ifndef TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
#define TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_

#include <tvm/ffi/function.h>

#include <string>
#include <vector>

#include "metadata.h"

namespace tvm {
namespace runtime {

enum class StorageRank {
  kGlobal = 0,
  kShared = 1,
  kWarp = 2,
  kLocal = 3,
  kWMMAMatrixA = 4,
  kWMMAMatrixB = 5,
  kWMMAAccumulator = 6,
  kTexture = 7,
  kAMXTMM = 8,
  kMMAMatrixA = 9,
  kMMAMatrixB = 10,
  kMMAMatrixC = 11,
  kMetalSimdGroup = 12,
};

inline StorageRank DefaultStorageRank(int thread_scope_rank) {
  switch (thread_scope_rank) {
    case -1:
      return StorageRank::kGlobal;
    case 0:
      return StorageRank::kShared;
    case 1:
      return StorageRank::kLocal;
    default: {
      TVM_FFI_THROW(InternalError) << "unknown rank";
    }
  }
}

struct StorageScope {
  StorageRank rank{StorageRank::kGlobal};
  std::string tag;
  inline bool operator==(const StorageScope& other) const {
    return rank == other.rank && tag == other.tag;
  }
  inline bool operator!=(const StorageScope& other) const { return !(*this == other); }
  inline std::string to_string() const {
    switch (rank) {
      case StorageRank::kGlobal:
        return "global" + tag;
      case StorageRank::kShared:
        return "shared" + tag;
      case StorageRank::kWarp:
        return "warp" + tag;
      case StorageRank::kLocal:
        return "local" + tag;
      case StorageRank::kWMMAMatrixA:
        return "wmma.matrix_a" + tag;
      case StorageRank::kWMMAMatrixB:
        return "wmma.matrix_b" + tag;
      case StorageRank::kWMMAAccumulator:
        return "wmma.accumulator" + tag;
      case StorageRank::kTexture:
        return "texture" + tag;
      case StorageRank::kMMAMatrixA:
        return "m16n8k8.matrixA" + tag;
      case StorageRank::kMMAMatrixB:
        return "m16n8k8.matrixB" + tag;
      case StorageRank::kMMAMatrixC:
        return "m16n8k8.matrixC" + tag;
      case StorageRank::kMetalSimdGroup:
        return "metal.simdgroup" + tag;
      default:
        TVM_FFI_THROW(InternalError) << "unknown storage scope";
        return "";
    }
  }
  static StorageScope Create(const std::string& s) {
    StorageScope r;
    if (s.empty()) {
      r.rank = StorageRank::kGlobal;
    } else if (s.compare(0, 6, "global") == 0) {
      r.rank = StorageRank::kGlobal;
      r.tag = s.substr(6, std::string::npos);
    } else if (s.compare(0, 6, "shared") == 0) {
      r.rank = StorageRank::kShared;
      r.tag = s.substr(6, std::string::npos);
    } else if (s.compare(0, 4, "warp") == 0) {
      r.rank = StorageRank::kWarp;
      r.tag = s.substr(4, std::string::npos);
    } else if (s.compare(0, 5, "local") == 0) {
      r.rank = StorageRank::kLocal;
      r.tag = s.substr(5, std::string::npos);
    } else if (s.compare(0, 13, "wmma.matrix_a") == 0) {
      r.rank = StorageRank::kWMMAMatrixA;
      r.tag = s.substr(13, std::string::npos);
    } else if (s.compare(0, 13, "wmma.matrix_b") == 0) {
      r.rank = StorageRank::kWMMAMatrixB;
      r.tag = s.substr(13, std::string::npos);
    } else if (s.compare(0, 16, "wmma.accumulator") == 0) {
      r.rank = StorageRank::kWMMAAccumulator;
      r.tag = s.substr(16, std::string::npos);
    } else if (s.compare(0, 7, "texture") == 0) {
      r.rank = StorageRank::kTexture;
      r.tag = s.substr(7, std::string::npos);
    } else if (s.compare(0, 7, "amx.tmm") == 0) {
      r.rank = StorageRank::kAMXTMM;
      r.tag = s.substr(7, std::string::npos);
    } else if (s.compare(0, 15, "m16n8k8.matrixA") == 0) {
      r.rank = StorageRank::kMMAMatrixA;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 15, "m16n8k8.matrixB") == 0) {
      r.rank = StorageRank::kMMAMatrixB;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 15, "m16n8k8.matrixC") == 0) {
      r.rank = StorageRank::kMMAMatrixC;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 15, "metal.simdgroup") == 0) {
      r.rank = StorageRank::kMetalSimdGroup;
      r.tag = s.substr(15, std::string::npos);
    } else {
      TVM_FFI_THROW(InternalError) << "unknown storage scope " << s;
    }
    return r;
  }
};

struct ThreadScope {
  int rank{0};
  int dim_index{0};
  static ThreadScope Create(const std::string& s) {
    ThreadScope r;
    if (s.compare(0, 7, "vthread") == 0 || s == "cthread") {
      r.rank = 1;
      r.dim_index = -1;
    } else if (s.compare(0, 9, "blockIdx.") == 0) {
      r.rank = 0;
      r.dim_index = static_cast<int>(s[9] - 'x');
    } else if (s.compare(0, 10, "threadIdx.") == 0) {
      r.rank = 1;
      r.dim_index = static_cast<int>(s[10] - 'x');
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown threadscope " << s;
    }
    return r;
  }
};

struct ThreadWorkLoad {
  size_t work_size[6];
  size_t dyn_shmem_size{0};
  inline size_t block_dim(size_t i) const { return work_size[i + 3]; }
  inline size_t grid_dim(size_t i) const { return work_size[i]; }
};

class LaunchParamConfig {
 public:
  void Init(size_t base, const ffi::Array<ffi::String>& launch_param_tags) {
    base_ = base;
    std::vector<bool> filled(6, false);
    for (size_t i = 0; i < launch_param_tags.size(); ++i) {
      std::string tag(launch_param_tags[i]);
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        TVM_FFI_ICHECK_EQ(i, launch_param_tags.size() - 1)
            << "kUseDynamicSharedMemoryTag should be the last tag in launch_param_tags.";
        use_dyn_shared_memory_ = true;
      } else if (tag == launch_param::kUseProgramaticDependentLaunch) {
        use_programmatic_dependent_launch_ = true;
      } else if (tag == launch_param::kUseCooperativeLaunch) {
        use_cooperative_launch_ = true;
      } else {
        ThreadScope ts = ThreadScope::Create(tag);
        arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
        filled[ts.rank * 3 + ts.dim_index] = true;
      }
    }
    work_dim_ = 1;
    for (int i = 0; i < 3; ++i) {
      if (filled[i] || filled[i + 3]) {
        work_dim_ = i + 1;
      }
    }
  }
  ThreadWorkLoad Extract(ffi::PackedArgs args) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    const TVMFFIAny* raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());

    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      size_t size = static_cast<size_t>(raw_args[base_ + i].v_int64);
      if (size > 0) {
        w.work_size[arg_index_map_[i]] = size;
      }
    }
    if (use_dyn_shared_memory_) {
      w.dyn_shmem_size = static_cast<size_t>(raw_args[base_ + arg_index_map_.size()].v_int64);
    }
    return w;
  }
  size_t work_dim() const { return work_dim_; }

  bool use_programtic_dependent_launch() const { return use_programmatic_dependent_launch_; }

  bool use_cooperative_launch() const { return use_cooperative_launch_; }

 private:
  size_t base_;
  size_t work_dim_;
  std::vector<uint32_t> arg_index_map_;
  bool use_dyn_shared_memory_{false};
  bool use_programmatic_dependent_launch_{false};
  bool use_cooperative_launch_{false};
};

}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::runtime::StorageScope> {
  std::size_t operator()(const ::tvm::runtime::StorageScope& k) const {
    return static_cast<size_t>(k.rank);
  }
};
}  // namespace std
#endif  // TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
