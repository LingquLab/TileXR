#pragma once

namespace spdlog {
namespace level {
enum level_enum {
    trace,
    debug,
    info,
    warn,
    err,
};
} // namespace level

inline void set_level(level::level_enum)
{
}

template <typename... Args>
inline void trace(const char*, Args...)
{
}

template <typename... Args>
inline void debug(const char*, Args...)
{
}

template <typename... Args>
inline void info(const char*, Args...)
{
}

template <typename... Args>
inline void warn(const char*, Args...)
{
}

template <typename... Args>
inline void error(const char*, Args...)
{
}
} // namespace spdlog
