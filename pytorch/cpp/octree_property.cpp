#ifndef KEY64
#define KEY64
#endif
#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

namespace {

Tensor octree_property_gpu(Tensor octree_in, string property, int depth) {
  OctreeParser octree_;
  octree_.set_gpu(octree_in.data_ptr<uint8_t>());

  int node_num = octree_.info().node_num(depth);
  torch::TensorOptions options = octree_in.options();
  Tensor data_out = torch::zeros({1}, options);

  if (property == "key") {
    const uintk* ptr = octree_.key_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    memcpy_gpu(total_num, ptr, (uintk*)data_out.data_ptr<int64_t>());
  }

  if (property == "xyz") {
    const uintk* ptr = octree_.key_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    if (!octree_.info().is_key2xyz()) {
      key2xyz_gpu((uintk*)data_out.data_ptr<int64_t>(), ptr, total_num, depth);
    } else {
      memcpy_gpu(total_num, ptr, (uintk*)data_out.data_ptr<int64_t>());
    }
  }

  if (property == "index") {
    const uintk* key_ptr = octree_.key_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    key2idx_gpu(data_out.data_ptr<int>(), key_ptr, total_num);
  }

  if (property == "child") {
    const int* child_ptr = octree_.children_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kChild);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    memcpy_gpu(total_num, child_ptr, data_out.data_ptr<int>());
  }

  if (property == "neigh") {
    const int* neigh_ptr = octree_.neighbor_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kNeigh);
    int total_num = channel * node_num;
    data_out = torch::zeros({node_num, channel}, options.dtype(torch::kInt32));
    memcpy_gpu(total_num, neigh_ptr, data_out.data_ptr<int>());
  }

  if (property == "feature") {
    const float* feature_ptr = octree_.feature_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kFeature);
    int total_num = channel * node_num;
    data_out = torch::zeros({1, channel, node_num, 1}, options.dtype(torch::kFloat32));
    memcpy_gpu(total_num, feature_ptr, data_out.data_ptr<float>());
  }

  if (property == "label") {
    const float* label_ptr = octree_.label_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kLabel);
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_gpu(total_num, label_ptr, data_out.data_ptr<float>());
  }

  if (property == "split") {
    const float* split_ptr = octree_.split_gpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kSplit);
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_gpu(total_num, split_ptr, data_out.data_ptr<float>());
  }

  if (property == "node_num") {
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_gpu(1, &node_num, data_out.data_ptr<int>());
  }

  if (property == "node_num_ne" || property == "node_num_nempty") {
    int num = octree_.info().node_num_nempty(depth);
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_gpu(1, &num, data_out.data_ptr<int>());
  }

  return data_out;
}

Tensor octree_property_cpu(Tensor octree_in, string property, int depth) {
  OctreeParser octree_;
  octree_.set_cpu(octree_in.data_ptr<uint8_t>());

  int node_num = octree_.info().node_num(depth);
  torch::TensorOptions options = octree_in.options();
  Tensor data_out = torch::zeros({1}, options);

  if (property == "key") {
    const uintk* ptr = octree_.key_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    memcpy_cpu(total_num, ptr, (uintk*)data_out.data_ptr<int64_t>());
  }

  if (property == "xyz") {
    const uintk* ptr = octree_.key_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt64));
    if (!octree_.info().is_key2xyz()) {
      key2xyz_cpu((uintk*)data_out.data_ptr<int64_t>(), ptr, total_num, depth);
    } else {
      memcpy_cpu(total_num, ptr, (uintk*)data_out.data_ptr<int64_t>());
    }
  }

  if (property == "index") {
    const uintk* key_ptr = octree_.key_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kKey);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    key2idx_cpu(data_out.data_ptr<int>(), key_ptr, total_num);
  }

  if (property == "child") {
    const int* child_ptr = octree_.children_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kChild);  // = 1
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kInt32));
    memcpy_cpu(total_num, child_ptr, data_out.data_ptr<int>());
  }

  if (property == "neigh") {
    const int* neigh_ptr = octree_.neighbor_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kNeigh);
    int total_num = channel * node_num;
    data_out = torch::zeros({node_num, channel}, options.dtype(torch::kInt32));
    memcpy_cpu(total_num, neigh_ptr, data_out.data_ptr<int>());
  }

  if (property == "feature") {
    const float* feature_ptr = octree_.feature_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kFeature);
    int total_num = channel * node_num;
    data_out = torch::zeros({1, channel, node_num, 1}, options.dtype(torch::kFloat32));
    memcpy_cpu(total_num, feature_ptr, data_out.data_ptr<float>());
  }

  if (property == "label") {
    const float* label_ptr = octree_.label_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kLabel);
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_cpu(total_num, label_ptr, data_out.data_ptr<float>());
  }

  if (property == "split") {
    const float* split_ptr = octree_.split_cpu(depth);
    int channel = octree_.info().channel(OctreeInfo::kSplit);
    int total_num = channel * node_num;
    data_out = torch::zeros({total_num}, options.dtype(torch::kFloat32));
    memcpy_cpu(total_num, split_ptr, data_out.data_ptr<float>());
  }

  if (property == "node_num") {
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_cpu(1, &node_num, data_out.data_ptr<int>());
  }

  if (property == "node_num_ne" || property == "node_num_nempty") {
    int num = octree_.info().node_num_nempty(depth);
    data_out = torch::zeros({1}, options.dtype(torch::kInt32));
    memcpy_cpu(1, &num, data_out.data_ptr<int>());
  }

  return data_out;
}

}  // anonymous namespace

// API implementation
Tensor octree_property(Tensor octree_in, string property, int depth) {
  if (octree_in.is_cuda()) {
    return octree_property_gpu(octree_in, property, depth);
  } else {
    return octree_property_cpu(octree_in, property, depth);
  }
}