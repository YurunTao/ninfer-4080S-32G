#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#if defined(_WIN32)
using ninfer_native_file_t = void*;
#define NINFER_INVALID_FILE nullptr
#else
using ninfer_native_file_t = int;
#define NINFER_INVALID_FILE (-1)
#endif

namespace ninfer::artifact {

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    ninfer_native_file_t fd_                = NINFER_INVALID_FILE;
    mutable ninfer_native_file_t direct_fd_ = NINFER_INVALID_FILE;
    std::uint64_t bytes_                    = 0;
};

} // namespace ninfer::artifact
