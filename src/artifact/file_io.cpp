#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
#if defined(_WIN32)
    throw ArtifactError(path.string() + ": " + operation + ": " +
                        std::system_category().message(static_cast<int>(::GetLastError())));
#else
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
#endif
}

std::uint64_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::uint64_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return offset;
}

#if defined(_WIN32)

// Positional reads are issued through an explicit OVERLAPPED offset; the handle stays
// synchronous so no event object is required.
void seek_to(HANDLE handle, std::uint64_t offset) {
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(offset);
    if (!::SetFilePointerEx(handle, position, nullptr, FILE_BEGIN)) {
        throw ArtifactError("failed to seek artifact file");
    }
}

bool read_block(HANDLE handle, void* destination, std::size_t count, std::size_t& read) {
    const auto request = static_cast<DWORD>(std::min<std::size_t>(count, 0xFFFFFFFFu));
    DWORD transferred  = 0;
    if (!::ReadFile(handle, destination, request, &transferred, nullptr)) { return false; }
    read = transferred;
    return true;
}

std::uint64_t file_size(HANDLE handle) {
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(handle, &size)) { return 0; }
    return static_cast<std::uint64_t>(size.QuadPart);
}

HANDLE open_read_only(const std::filesystem::path& path) {
    return ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
}

// O_DIRECT has no Win32 equivalent. FILE_FLAG_NO_BUFFERING requires sector-aligned
// lengths, which the generic direct path cannot guarantee, so reads stay buffered and
// the caller's alignment contract is still honoured.
HANDLE open_direct(const std::filesystem::path& path) { return open_read_only(path); }

#else

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#if defined(_WIN32)
    const HANDLE handle = open_read_only(path_);
    if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open"); }
    fd_    = handle;
    bytes_ = file_size(handle);
    if (bytes_ == 0) {
        const auto error = ::GetLastError();
        ::CloseHandle(handle);
        fd_ = NINFER_INVALID_FILE;
        ::SetLastError(error);
        fail(path_, "GetFileSizeEx");
    }
#else
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
#if defined(_WIN32)
    if (direct_fd_ != NINFER_INVALID_FILE) { ::CloseHandle(static_cast<HANDLE>(direct_fd_)); }
    if (fd_ != NINFER_INVALID_FILE) { ::CloseHandle(static_cast<HANDLE>(fd_)); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
#if defined(_WIN32)
        seek_to(static_cast<HANDLE>(fd_), offset);
        std::size_t read = 0;
        if (!read_block(static_cast<HANDLE>(fd_), destination.data(), count, read)) {
            fail(path_, "ReadFile");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        const auto consumed = read;
#else
        const auto read = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        const auto consumed = static_cast<std::size_t>(read);
#endif
        offset += static_cast<std::uint64_t>(consumed);
        destination = destination.subspan(consumed);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
#if defined(_WIN32)
    if (direct_fd_ == NINFER_INVALID_FILE) {
        const HANDLE handle = open_direct(path_);
        if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open direct"); }
        direct_fd_ = handle;
    }
    seek_to(static_cast<HANDLE>(direct_fd_), offset);
    std::size_t read = 0;
    if (!read_block(static_cast<HANDLE>(direct_fd_), destination.data(), destination.size(), read)) {
        fail(path_, "direct ReadFile");
    }
    return read;
#else
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
#endif
}

} // namespace ninfer::artifact
