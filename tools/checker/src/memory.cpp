#include "tilexr/checker/memory.h"

#include <cstring>
#include <limits>

namespace tilexr {
namespace checker {

namespace {

CheckerStatus CheckRange(size_t total_size, size_t offset, size_t bytes) {
    if (offset > total_size || bytes > (total_size - offset)) {
        return CheckerStatus::Fail("byte buffer access out of bounds");
    }
    return CheckerStatus::Ok();
}

CheckerStatus CheckInt32Index(size_t index) {
    const size_t int32_size = sizeof(int32_t);
    if (index > std::numeric_limits<size_t>::max() / int32_size) {
        return CheckerStatus::Fail("int32 buffer access index overflow");
    }
    return CheckerStatus::Ok();
}

}  // namespace

ByteBuffer::ByteBuffer() = default;

ByteBuffer::ByteBuffer(size_t size) : data_(size, 0) {}

size_t ByteBuffer::size() const {
    return data_.size();
}

uint8_t *ByteBuffer::data_ptr() {
    return data_.empty() ? nullptr : data_.data();
}

const uint8_t *ByteBuffer::data_ptr() const {
    return data_.empty() ? nullptr : data_.data();
}

const std::vector<uint8_t> &ByteBuffer::data() const {
    return data_;
}

std::vector<uint8_t> &ByteBuffer::mutable_data() {
    return data_;
}

CheckerStatus ByteBuffer::WriteBytes(size_t offset, const void *src, size_t bytes) {
    CheckerStatus status = CheckRange(data_.size(), offset, bytes);
    if (!status.ok()) {
        return status;
    }
    std::memcpy(&data_[offset], src, bytes);
    return CheckerStatus::Ok();
}

CheckerStatus ByteBuffer::ReadBytes(size_t offset, void *dst, size_t bytes) const {
    CheckerStatus status = CheckRange(data_.size(), offset, bytes);
    if (!status.ok()) {
        return status;
    }
    std::memcpy(dst, &data_[offset], bytes);
    return CheckerStatus::Ok();
}

CheckerStatus ByteBuffer::WriteInt32(size_t index, int32_t value) {
    CheckerStatus status = CheckInt32Index(index);
    if (!status.ok()) {
        return status;
    }
    return WriteBytes(index * sizeof(int32_t), &value, sizeof(int32_t));
}

CheckerStatus ByteBuffer::ReadInt32(size_t index, int32_t *value) const {
    if (value == nullptr) {
        return CheckerStatus::Fail("int32 destination is null");
    }
    CheckerStatus status = CheckInt32Index(index);
    if (!status.ok()) {
        return status;
    }
    return ReadBytes(index * sizeof(int32_t), value, sizeof(int32_t));
}

}  // namespace checker
}  // namespace tilexr
