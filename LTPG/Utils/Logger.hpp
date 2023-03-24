#ifndef LOGGER_H
#define LOGGER_H

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <iomanip>
#include <ctime>
#include <mutex>
#include <iostream>
#include <sstream>
#include "LTPG/Utils/DataTypes.hpp"

namespace libtorchPG
{
    class Logger
    {
    private:
        Logger(LogType level, const bool outputFile, const bool overwrite)
        {
            std::vector<spdlog::sink_ptr> sinks;

            auto console_logger = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            if (outputFile)
            {
                std::time_t t = std::time(nullptr);
                std::tm tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << std::put_time(&tm, "%Y-%m-%d");
                std::string filename = "logs/" + oss.str() + ".log";
                auto file_logger = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, overwrite);
                sinks.push_back(file_logger);
            }
            sinks.push_back(console_logger);
            auto logger_ = std::make_shared<spdlog::logger>("libTorchPG", begin(sinks), end(sinks));

            spdlog::register_logger(logger_);
            spdlog::set_default_logger(logger_);
            spdlog::set_level(convertSPDLOGType(level));
            spdlog::set_pattern("[%H:%M:%S %z] %n: %^[%l] %v%$");
        }

        ~Logger() {}

        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        spdlog::level::level_enum convertSPDLOGType(LogType level)
        {
            switch (level)
            {
            case LogType::InfoLog:
                /* code */
                return spdlog::level::level_enum::info;
            case LogType::WarnLog:
                /* code */
                return spdlog::level::level_enum::warn;
            case LogType::CriticalLog:
                /* code */
                return spdlog::level::level_enum::critical;
            case LogType::ErrorLog:
                /* code */
                return spdlog::level::level_enum::err;
            }
        }

    public:
        static Logger &
        getInstance(LogType level, const bool outputFile = true, const bool overwrite = true)
        {
            static Logger instance(level, outputFile, overwrite);
            return instance;
        }

        void logInfo(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE())
        {
            spdlog::info("[{}:{}] {}", file, line, message);
        }

        void logWarning(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE())
        {
            spdlog::warn("[{}:{}] {}", file, line, message);
        }

        void logError(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE())
        {
            spdlog::error("[{}:{}] {}", file, line, message);
        }

        void logCritical(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE())
        {
            spdlog::critical("[{}:{}] {}", file, line, message);
        }
    };

} // namespace libtorchPG

#endif