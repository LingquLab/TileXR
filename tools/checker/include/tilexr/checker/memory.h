#ifndef TILEXR_CHECKER_MEMORY_H
#define TILEXR_CHECKER_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tilexr/checker/status.h"

namespace tilexr {
namespace checker {

class ByteBuffer {
public:
    ByteBuffer();
    explicit ByteBuffer(size_t size);

    size_t size() const;
    const std::vector<uint8_t> &data() const;
    std::vector<uint8_t> &mutable_data();

    CheckerStatus WriteBytes(size_t offset, const void *src, size_t bytes);
    CheckerStatus ReadBytes(size_t offset, void *dst, size_t bytes) const;
    CheckerStatus WriteInt32(size_t index, int32_t value);
    CheckerStatus ReadInt32(size_t index, int32_t *value) const;

private:
    std::vector<uint8_t> data_;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_MEMORY_H
