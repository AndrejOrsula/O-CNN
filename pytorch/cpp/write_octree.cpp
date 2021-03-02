#include <octree/octree.h>

#include "ocnn.h"

void write_octree(Tensor input, const std::string &filename)
{
  // Make sure the memory is aligned
  input = input.contiguous();

  // Set octree from Tensor data
  Octree octree;
  octree.set_octree((const char *)input.data_ptr<uint8_t>(), input.numel());

  // Write to file
  octree.write_octree(filename);
}
