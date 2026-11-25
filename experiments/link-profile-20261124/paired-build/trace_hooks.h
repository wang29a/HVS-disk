#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace phase0 {

struct Pages {
  const char *data = nullptr;
  size_t bytes = 0;

  template <typename Request>
  void copy(const std::vector<Request> &requests) const {
    for (const auto &request : requests) {
      if (request.offset > bytes || request.len > bytes - request.offset)
        throw std::runtime_error("native page request outside resident image");
      std::memcpy(request.buf, data + request.offset, request.len);
    }
  }
};

extern Pages topology;
extern Pages coordinates;
extern thread_local std::vector<uint32_t> *expanded;
extern thread_local bool coarse_only;

} // namespace phase0
