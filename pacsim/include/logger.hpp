#ifndef PACSCIMLOGGER_HPP
#define PACSCIMLOGGER_HPP

#ifndef PACSIM_STANDALONE
#include "rclcpp/rclcpp.hpp"
#else
#include <iostream>
#endif
#include <string>

class Logger
{
public:
#ifndef PACSIM_STANDALONE
    void logInfo(std::string in) { RCLCPP_INFO_STREAM(rclcpp::get_logger("pacsim_logger"), in); }

    void logWarning(std::string in) { RCLCPP_WARN_STREAM(rclcpp::get_logger("pacsim_logger"), in); }

    void logError(std::string in) { RCLCPP_ERROR_STREAM(rclcpp::get_logger("pacsim_logger"), in); }

    void logFatal(std::string in) { RCLCPP_FATAL_STREAM(rclcpp::get_logger("pacsim_logger"), in); }
#else
    void logInfo(std::string in) { std::cerr << "[pacsim][info] " << in << std::endl; }

    void logWarning(std::string in) { std::cerr << "[pacsim][warning] " << in << std::endl; }

    void logError(std::string in) { std::cerr << "[pacsim][error] " << in << std::endl; }

    void logFatal(std::string in) { std::cerr << "[pacsim][fatal] " << in << std::endl; }
#endif
};

#endif /* PACSCIMLOGGER_HPP */
