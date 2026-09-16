#include "sensorModels/cameraImageSensor.hpp"

CameraImageSensor::CameraImageSensor()
{
    this->name = "";
    this->frame = "";
    this->rate = 1.0;
    this->lastSampleTime = 0.0;
    this->deadTime = 0.0;
    this->enabled = true;
}

void CameraImageSensor::configure(
    const std::string& nameIn, const std::string& frameIn, double rateHz, double deadTimeSec,
    bool enabledIn, const Eigen::Vector3d& positionIn, const Eigen::Vector3d& orientationIn)
{
    this->name = nameIn;
    this->frame = frameIn;
    this->rate = (rateHz > 0.0) ? rateHz : 1.0;
    this->deadTime = (deadTimeSec >= 0.0) ? deadTimeSec : 0.0;
    this->enabled = enabledIn;
    this->position = positionIn;
    this->orientation = orientationIn;
    this->lastSampleTime = 0.0;
}

bool CameraImageSensor::RunTick(const CameraImageFrame& in, double time)
{
    if (!enabled)
    {
        return false;
    }

    if (this->sampleReady(time))
    {
        CameraImageFrame value = in;
        value.timestamp = time;
        value.frame = frame;
        this->deadTimeQueue.push(value);
        this->registerSampling();
    }
    return availableDeadTime(time);
}

std::string CameraImageSensor::getName() const
{
    return this->name;
}

std::string CameraImageSensor::getFrameId() const
{
    return this->frame;
}

bool CameraImageSensor::isEnabled() const
{
    return this->enabled;
}