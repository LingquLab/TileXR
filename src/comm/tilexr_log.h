/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_COMM_TILEXR_LOG_H
#define TILEXR_COMM_TILEXR_LOG_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#ifndef TILEXR_LOG_DISABLE_SPDLOG
#if defined(__has_include)
#if __has_include(<spdlog/spdlog.h>)
#define TILEXR_LOG_HAS_SPDLOG 1
#include <spdlog/spdlog.h>
#endif
#endif
#endif

#ifndef TILEXR_LOG_HAS_SPDLOG
#define TILEXR_LOG_HAS_SPDLOG 0
#endif

namespace TileXR {
namespace Log {

enum class Level : int {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
};

inline Level ConfiguredLevel();

class Message {
public:
    Message(Level level, const char* file, int line)
        : level_(level), file_(file), line_(line)
    {
    }

    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;

    Message(Message&& other)
        : level_(other.level_), file_(other.file_), line_(other.line_),
          stream_(std::move(other.stream_)), active_(other.active_)
    {
        other.active_ = false;
    }

    ~Message()
    {
        if (active_) {
            Write(level_, file_, line_, stream_.str());
        }
    }

    template <typename T>
    Message& operator<<(const T& value)
    {
        stream_ << value;
        return *this;
    }

private:
    static const char* BaseName(const char* path)
    {
        const char* slash = std::strrchr(path, '/');
        const char* backslash = std::strrchr(path, '\\');
        const char* last = slash;
        if (backslash != nullptr && (last == nullptr || backslash > last)) {
            last = backslash;
        }
        return last == nullptr ? path : last + 1;
    }

    static const char* LevelName(Level level)
    {
        switch (level) {
            case Level::TRACE:
                return "TRACE";
            case Level::DEBUG:
                return "DEBUG";
            case Level::INFO:
                return "INFO";
            case Level::WARN:
                return "WARN";
            case Level::ERROR:
                return "ERROR";
            default:
                return "INFO";
        }
    }

    static void Write(Level level, const char* file, int line, const std::string& text)
    {
#if TILEXR_LOG_HAS_SPDLOG
        spdlog::set_level(ToSpdlogLevel(ConfiguredLevel()));
        const std::string message = std::string(BaseName(file)) + ":" + std::to_string(line) + " " + text;
        switch (level) {
            case Level::TRACE:
                spdlog::trace("{}", message);
                break;
            case Level::DEBUG:
                spdlog::debug("{}", message);
                break;
            case Level::INFO:
                spdlog::info("{}", message);
                break;
            case Level::WARN:
                spdlog::warn("{}", message);
                break;
            case Level::ERROR:
                spdlog::error("{}", message);
                break;
            default:
                spdlog::info("{}", message);
                break;
        }
#else
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        std::ostream& out = level >= Level::WARN ? std::cerr : std::cout;
        out << LevelName(level) << " " << BaseName(file) << ":" << line << " " << text << std::endl;
#endif
    }

#if TILEXR_LOG_HAS_SPDLOG
    static spdlog::level::level_enum ToSpdlogLevel(Level level)
    {
        switch (level) {
            case Level::TRACE:
                return spdlog::level::trace;
            case Level::DEBUG:
                return spdlog::level::debug;
            case Level::INFO:
                return spdlog::level::info;
            case Level::WARN:
                return spdlog::level::warn;
            case Level::ERROR:
                return spdlog::level::err;
            default:
                return spdlog::level::info;
        }
    }
#endif

    Level level_;
    const char* file_;
    int line_;
    std::ostringstream stream_;
    bool active_ = true;
};

inline Level DefaultLevel()
{
    return Level::INFO;
}

inline bool ParseLevel(const char* value, Level& level)
{
    if (value == nullptr) {
        return false;
    }
    if (std::strcmp(value, "TRACE") == 0) {
        level = Level::TRACE;
        return true;
    }
    if (std::strcmp(value, "DEBUG") == 0) {
        level = Level::DEBUG;
        return true;
    }
    if (std::strcmp(value, "INFO") == 0) {
        level = Level::INFO;
        return true;
    }
    if (std::strcmp(value, "WARN") == 0 || std::strcmp(value, "WARNING") == 0) {
        level = Level::WARN;
        return true;
    }
    if (std::strcmp(value, "ERROR") == 0) {
        level = Level::ERROR;
        return true;
    }
    return false;
}

inline Level ConfiguredLevel()
{
    Level level = DefaultLevel();
    if (ParseLevel(std::getenv("TILEXR_LOG_LEVEL"), level)) {
        return level;
    }
    if (ParseLevel(std::getenv("ASDOPS_LOG_LEVEL"), level)) {
        return level;
    }
    return DefaultLevel();
}

inline bool Enabled(Level level)
{
    return level >= ConfiguredLevel();
}

} // namespace Log
} // namespace TileXR

#define TILEXR_LOG_LEVEL_TRACE ::TileXR::Log::Level::TRACE
#define TILEXR_LOG_LEVEL_DEBUG ::TileXR::Log::Level::DEBUG
#define TILEXR_LOG_LEVEL_INFO ::TileXR::Log::Level::INFO
#define TILEXR_LOG_LEVEL_WARN ::TileXR::Log::Level::WARN
#define TILEXR_LOG_LEVEL_ERROR ::TileXR::Log::Level::ERROR

#define TILEXR_LOG_LEVEL(level) TILEXR_LOG_LEVEL_##level

#define TILEXR_LOG(level) \
    if (!::TileXR::Log::Enabled(TILEXR_LOG_LEVEL(level))) { \
    } else \
        ::TileXR::Log::Message(TILEXR_LOG_LEVEL(level), __FILE__, __LINE__)

#define TILEXR_LOG_IF(condition, level) \
    if (!(condition) || !::TileXR::Log::Enabled(TILEXR_LOG_LEVEL(level))) { \
    } else \
        ::TileXR::Log::Message(TILEXR_LOG_LEVEL(level), __FILE__, __LINE__)

#endif // TILEXR_COMM_TILEXR_LOG_H
