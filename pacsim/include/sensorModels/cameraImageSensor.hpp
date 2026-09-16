#ifndef PACSIMCAMERAIMAGESENSOR_HPP
#define PACSIMCAMERAIMAGESENSOR_HPP

#include "sensorBase.hpp"
#include "types.hpp"

#include <string>

class CameraImageSensor : public SensorBase<CameraImageFrame>
{
public:
    CameraImageSensor();

    void configure(const std::string& name, const std::string& frame, double rateHz, double deadTimeSec,
        bool enabled, const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d& orientation = Eigen::Vector3d::Zero());

    bool RunTick(const CameraImageFrame& in, double time);

    std::string getName() const;
    std::string getFrameId() const;
    bool isEnabled() const;

private:
    std::string name;
    std::string frame;
    bool enabled;
};

#endif /* PACSIMCAMERAIMAGESENSOR_HPP */