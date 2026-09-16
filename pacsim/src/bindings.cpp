#include "sensorModels/perceptionSensor.hpp"
#include "configParser.hpp"
#include "VehicleModel/VehicleModelBicycle.cpp"
#include "VehicleModel/VehicleModel4Wheel.cpp"
#include "transform.hpp"
#include "sensorModels/imuSensor.hpp"
#include "track/trackLoader.hpp"

#include "logger.hpp"
#include "competitionLogic.hpp"

#include "VehicleModel/deadTime.hpp"
#include "sensorModels/vulkanCameraSim.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class CubicSpline {
    private:
        std::vector<double> x, y;       // data points
        std::vector<double> a, b, c, d; // spline coefficients

    public:
        // Construct spline with given data
        CubicSpline(const std::vector<double>& X, const std::vector<double>& Y) : x(X), y(Y) {
            int n = x.size() - 1;
            a = y;
            b.resize(n);
            c.resize(n + 1);
            d.resize(n);

            std::vector<double> h(n), alpha(n), l(n + 1), mu(n + 1), z(n + 1);

            for (int i = 0; i < n; i++)
                h[i] = x[i + 1] - x[i];

            for (int i = 1; i < n; i++)
                alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);

            l[0] = 1.0; mu[0] = 0.0; z[0] = 0.0;
            for (int i = 1; i < n; i++) {
                l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
                mu[i] = h[i] / l[i];
                z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
            }
            l[n] = 1.0; z[n] = 0.0; c[n] = 0.0;

            for (int j = n - 1; j >= 0; j--) {
                c[j] = z[j] - mu[j] * c[j + 1];
                b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
                d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
            }
        }

        // Evaluate spline at point X
        double operator()(double X) const {
            int i = findInterval(X);
            double dx = X - x[i];
            return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
        }

        // Evaluate derivative at point X
        double derivative(double X) const {
            int i = findInterval(X);
            double dx = X - x[i];
            return b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx * dx;
        }

    private:
        // Find interval [x[i], x[i+1]] containing X
        int findInterval(double X) const {
            int n = x.size() - 1;
            int i = n - 1;
            for (int j = 0; j < n; j++) {
                if (X >= x[j] && X <= x[j + 1]) {
                    i = j;
                    break;
                }
            }
            return i;
        }
};

pybind11::array_t<uint8_t> imageToNumpy(const std::vector<uint8_t>& image, int width, int height)
{
    pybind11::array_t<uint8_t> out({ height, width, 3 });
    auto outBuf = out.request();
    std::memcpy(outBuf.ptr, image.data(), image.size() * sizeof(uint8_t));
    return out;
}

pybind11::array_t<uint8_t> imageToNumpy(std::vector<uint8_t>&& image, int width, int height)
{
    auto* storage = new std::vector<uint8_t>(std::move(image));
    pybind11::capsule owner(storage, [](void* ptr) {
        delete reinterpret_cast<std::vector<uint8_t>*>(ptr);
    });
    return pybind11::array_t<uint8_t>(
        { height, width, 3 },
        { width * 3, 3, 1 },
        storage->data(),
        owner);
}

std::pair<double, double> findCurvlinearCoords(
    const CubicSpline& spline_x, const CubicSpline& spline_y, double arc_length, double x, double y)
{

    // sample at equal step size to get rough guess
    double stepsize = 0.2;
    int steps = (int)(arc_length / stepsize);
    double sampleStepSize = (arc_length / static_cast<double>(steps)) * 0.99999999;

    double bestDistSquared = 999.9 * 999.9;
    double bestS = 999.9;

    for (int i = 0; i < steps; ++i)
    {
        double s = sampleStepSize * i;
        if (i == (steps - 1))
        {
            s = arc_length - 0.00001;
        }
        double dx = spline_x(s) - x;
        double dy = spline_y(s) - y;
        double distSquared = dx * dx + dy * dy;

        if (distSquared < bestDistSquared)
        {
            bestDistSquared = distSquared;
            bestS = s;
        }
    }

    // do binary search to refine
    double refinementStep = stepsize;
    for (int i = 1; i < 10; ++i)
    {
        refinementStep *= 0.5;
        double arc = bestS;
        for (int j = -1; j <= 1; ++j)
        {
            double candidateS = arc + j * refinementStep;
            double dx = spline_x(candidateS) - x;
            double dy = spline_y(candidateS) - y;
            double distSquared = dx * dx + dy * dy;
            if (distSquared < bestDistSquared)
            {
                bestDistSquared = distSquared;
                bestS = candidateS;
            }
        }
    }

    double s = bestS;
    double n = std::sqrt(bestDistSquared);

    std::pair<double, double> ret(s,n);
    
    return ret;
}

class Rangefinder
{
private:
    struct Segment
    {
        double x;
        double y;
        double dx;
        double dy;
    };

    std::vector<double> angleCos;
    std::vector<double> angleSin;
    std::vector<Segment> segments;

    static void appendLaneSegments(
        std::vector<Segment>& out,
        const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& lane)
    {
        auto info = lane.request();
        if (info.ndim != 2 || info.shape[1] < 2)
        {
            throw pybind11::value_error("Rangefinder lane arrays must have shape (N, >=2)");
        }

        const auto rows = static_cast<std::size_t>(info.shape[0]);
        const auto cols = static_cast<std::size_t>(info.shape[1]);
        if (rows == 0)
        {
            return;
        }

        const double* data = static_cast<const double*>(info.ptr);
        out.reserve(out.size() + rows);
        for (std::size_t i = 0; i < rows; ++i)
        {
            const std::size_t previous = (i == 0) ? (rows - 1) : (i - 1);
            const double* p1 = data + previous * cols;
            const double* p2 = data + i * cols;
            out.push_back({
                p1[0],
                p1[1],
                p2[0] - p1[0],
                p2[1] - p1[1],
            });
        }
    }

    double evalRayMinDistance(double rayStartX, double rayStartY, double rayDirX, double rayDirY) const
    {
        double minDistance = std::numeric_limits<double>::infinity();
        for (const auto& segment : segments)
        {
            const double v1x = rayStartX - segment.x;
            const double v1y = rayStartY - segment.y;
            const double denominator = (-segment.dx * rayDirY) + (segment.dy * rayDirX);
            if (std::abs(denominator) < 1e-12)
            {
                continue;
            }

            const double cross21 = (segment.dx * v1y) - (segment.dy * v1x);
            const double t1 = cross21 / denominator;
            const double t2 = ((-v1x * rayDirY) + (v1y * rayDirX)) / denominator;
            if ((t1 >= 0.0) && (t2 >= 0.0) && (t2 <= 1.0) && (t1 < minDistance))
            {
                minDistance = t1;
            }
        }

        if (!std::isfinite(minDistance))
        {
            return 0.0;
        }
        return minDistance;
    }

public:
    Rangefinder(
        const std::vector<double>& angles,
        const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& leftLane,
        const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& rightLane)
    {
        angleCos.reserve(angles.size());
        angleSin.reserve(angles.size());
        for (const auto& angle : angles)
        {
            angleCos.push_back(std::cos(angle));
            angleSin.push_back(std::sin(angle));
        }
        appendLaneSegments(segments, leftLane);
        appendLaneSegments(segments, rightLane);
    }

    std::pair<std::vector<double>, std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d, double>>> rays(
        const Eigen::Vector3d& position,
        const Eigen::Vector3d& orientation) const
    {
        std::pair<std::vector<double>, std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d, double>>> ret;
        ret.first.reserve(angleCos.size());
        ret.second.reserve(angleCos.size());

        const double heading = orientation[2];
        const double headingCos = std::cos(heading);
        const double headingSin = std::sin(heading);
        const double rayStartX = position.x();
        const double rayStartY = position.y();
        Eigen::Vector2d rayStart(rayStartX, rayStartY);

        for (std::size_t i = 0; i < angleCos.size(); ++i)
        {
            const double rayDirX = (headingCos * angleCos[i]) - (headingSin * angleSin[i]);
            const double rayDirY = (headingSin * angleCos[i]) + (headingCos * angleSin[i]);
            double distance = evalRayMinDistance(rayStartX, rayStartY, rayDirX, rayDirY);
            Eigen::Vector2d rayDir(rayDirX, rayDirY);
            ret.first.push_back(distance);
            ret.second.push_back(std::make_tuple(rayStart, rayDir, distance));
        }
        return ret;
    }

    pybind11::array_t<double> normalizedDistances(const Eigen::Vector3d& position, const Eigen::Vector3d& orientation, double maxRange) const
    {
        auto ret_distances = pybind11::array_t<double>(angleCos.size());
        auto ret = ret_distances.mutable_unchecked<1>();

        const double heading = orientation[2];
        const double headingCos = std::cos(heading);
        const double headingSin = std::sin(heading);
        const double rayStartX = position.x();
        const double rayStartY = position.y();

        for (std::size_t i = 0; i < angleCos.size(); ++i)
        {
            const double rayDirX = (headingCos * angleCos[i]) - (headingSin * angleSin[i]);
            const double rayDirY = (headingSin * angleCos[i]) + (headingCos * angleSin[i]);
            double normalizedDistance = evalRayMinDistance(rayStartX, rayStartY, rayDirX, rayDirY) / maxRange;
            if (normalizedDistance < 0.0)
            {
                normalizedDistance = 0.0;
            }
            else if (normalizedDistance > 1.0)
            {
                normalizedDistance = 1.0;
            }
            ret(i) = normalizedDistance;
        }
        return ret_distances;
    }
};

template <typename VehicleModelT>
double forwardIntegrateControlFrame(
    VehicleModelT& model,
    double currentTime,
    double futureTime,
    double dt,
    DeadTime<double>& deadTimeSteering,
    DeadTime<Wheels>& deadTimeRPMSetpoints,
    DeadTime<Wheels>& deadTimeMaxTorques,
    DeadTime<Wheels>& deadTimeMinTorques,
    const Wheels& frictionCoefficients)
{
    while (currentTime < futureTime)
    {
        if (deadTimeSteering.availableDeadTime(currentTime))
        {
            double steer = deadTimeSteering.getOldest();
            model.setSteeringSetpointFront(steer);
        }
        if (deadTimeRPMSetpoints.availableDeadTime(currentTime))
        {
            Wheels rpmSetpoints = deadTimeRPMSetpoints.getOldest();
            model.setRpmSetpoints(rpmSetpoints);
        }
        if (deadTimeMaxTorques.availableDeadTime(currentTime))
        {
            Wheels maxTorques = deadTimeMaxTorques.getOldest();
            model.setMaxTorques(maxTorques);
        }
        if (deadTimeMinTorques.availableDeadTime(currentTime))
        {
            Wheels minTorques = deadTimeMinTorques.getOldest();
            model.setMinTorques(minTorques);
        }

        model.forwardIntegrate(dt, frictionCoefficients);
        currentTime += dt;
    }
    return currentTime;
}

PYBIND11_MODULE(pacsim_pybind, m) {
    m.doc() = "pacSim python bindings";

    pybind11::class_<CubicSpline>(m, "CubicSpline")
    .def(pybind11::init<const std::vector<double>, const std::vector<double>>())
    .def("__call__", &CubicSpline::operator())
    .def("derivative", &CubicSpline::derivative);

    m.def("findCurvlinearCoords", &findCurvlinearCoords);

    pybind11::class_<Rangefinder>(m, "Rangefinder")
    .def(pybind11::init<
        const std::vector<double>&,
        const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>&,
        const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>&>())
    .def("rays", &Rangefinder::rays)
    .def("normalizedDistances", &Rangefinder::normalizedDistances);

    m.def("loadMap", &loadMap);

    pybind11::class_<Config>(m, "Config")
    .def(pybind11::init<std::string>())
    .def("getElement", &ConfigElement::getConfigElement, "Set the pet's age");

    pybind11::class_<ConfigElement>(m, "ConfigElement")
    .def("getElement", &ConfigElement::getConfigElement, "Set the pet's age")
    .def("getElements", pybind11::overload_cast<>(&ConfigElement::getElements), "Set the pet's age");

    pybind11::class_<Wheels>(m, "Wheels")
    .def(pybind11::init())
    .def_readwrite("FL", &Wheels::FL)
    .def_readwrite("FR", &Wheels::FR)
    .def_readwrite("RL", &Wheels::RL)
    .def_readwrite("RR", &Wheels::RR)
    .def_readwrite("timestamp", &Wheels::timestamp);

    pybind11::class_<ImuData>(m, "ImuData")
    .def(pybind11::init())
    .def_readwrite("acceleration", &ImuData::acc)
    .def_readwrite("rot", &ImuData::rot)
    .def_readwrite("acc_cov", &ImuData::acc_cov)
    .def_readwrite("rot_cov", &ImuData::rot_cov)
    .def_readwrite("timestamp", &ImuData::timestamp);

    pybind11::class_<Landmark>(m, "Landmark")
    .def(pybind11::init())
    .def_readwrite("id", &Landmark::id)
    .def_readwrite("position", &Landmark::position)
    .def_readwrite("cov", &Landmark::cov)
    .def_readwrite("beenHit", &Landmark::beenHit);


    pybind11::class_<Track>(m, "Track")
    .def(pybind11::init())
    .def_readwrite("left_lane", &Track::left_lane)
    .def_readwrite("right_lane", &Track::right_lane)
    .def_readwrite("unknown", &Track::unknown)
    .def_readwrite("path_left_point_indices", &Track::path_left_point_indices)
    .def_readwrite("path_right_point_indices", &Track::path_right_point_indices)
    .def_readwrite("time_keeping_gates", &Track::time_keeping_gates);

    pybind11::class_<MainConfig>(m, "MainConfig")
    .def(pybind11::init());

    pybind11::class_<VehicleModel4WheelParameters>(m, "VehicleModel4WheelParameters")
    .def(pybind11::init())
    .def("validate", &VehicleModel4WheelParameters::validate)
    .def_readwrite("lf", &VehicleModel4WheelParameters::lf)
    .def_readwrite("lr", &VehicleModel4WheelParameters::lr)
    .def_readwrite("sf", &VehicleModel4WheelParameters::sf)
    .def_readwrite("sr", &VehicleModel4WheelParameters::sr)
    .def_readwrite("Blat", &VehicleModel4WheelParameters::Blat)
    .def_readwrite("Clat", &VehicleModel4WheelParameters::Clat)
    .def_readwrite("Dlat", &VehicleModel4WheelParameters::Dlat)
    .def_readwrite("Elat", &VehicleModel4WheelParameters::Elat)
    .def_readwrite("relaxationLengthLat", &VehicleModel4WheelParameters::relaxationLengthLat)
    .def_readwrite("Blon", &VehicleModel4WheelParameters::Blon)
    .def_readwrite("Clon", &VehicleModel4WheelParameters::Clon)
    .def_readwrite("Dlon", &VehicleModel4WheelParameters::Dlon)
    .def_readwrite("Elon", &VehicleModel4WheelParameters::Elon)
    .def_readwrite("relaxationLengthLon", &VehicleModel4WheelParameters::relaxationLengthLon)
    .def_readwrite("cla", &VehicleModel4WheelParameters::cla)
    .def_readwrite("cda", &VehicleModel4WheelParameters::cda)
    .def_readwrite("aeroArea", &VehicleModel4WheelParameters::aeroArea)
    .def_readwrite("m", &VehicleModel4WheelParameters::m)
    .def_readwrite("hc", &VehicleModel4WheelParameters::hc)
    .def_readwrite("Izz", &VehicleModel4WheelParameters::Izz)
    .def_readwrite("wheelRadius", &VehicleModel4WheelParameters::wheelRadius)
    .def_readwrite("gearRatio", &VehicleModel4WheelParameters::gearRatio)
    .def_readwrite("innerSteeringRatio", &VehicleModel4WheelParameters::innerSteeringRatio)
    .def_readwrite("outerSteeringRatio", &VehicleModel4WheelParameters::outerSteeringRatio)
    .def_readwrite("nominalVoltageTS", &VehicleModel4WheelParameters::nominalVoltageTS)
    .def_readwrite("powerGroundForce", &VehicleModel4WheelParameters::powerGroundForce)
    .def_readwrite("powertrainEfficiency", &VehicleModel4WheelParameters::powertrainEfficiency)
    .def_readwrite("wspdControlKp", &VehicleModel4WheelParameters::wspdControlKp)
    .def_readwrite("wspdControlKi", &VehicleModel4WheelParameters::wspdControlKi)
    .def_readwrite("steeringW0", &VehicleModel4WheelParameters::steeringW0)
    .def_readwrite("steeringD", &VehicleModel4WheelParameters::steeringD)
    .def_readwrite("steeringMax", &VehicleModel4WheelParameters::steeringMax)
    .def_readwrite("steeringMaxRate", &VehicleModel4WheelParameters::steeringMaxRate);

    pybind11::class_<VehicleModelBicycle>(m, "VehicleModel")
    .def(pybind11::init())
    .def("readConfig", &VehicleModelBicycle::readConfig)
    .def("forwardIntegrate", &VehicleModelBicycle::forwardIntegrate)
    .def("forwardIntegrateControlFrame", &forwardIntegrateControlFrame<VehicleModelBicycle>)
    .def("setSteeringSetpointFront", &VehicleModelBicycle::setSteeringSetpointFront)
    .def("setRpmSetpoints", &VehicleModelBicycle::setRpmSetpoints)
    .def("setMinTorques", &VehicleModelBicycle::setMinTorques)
    .def("setMaxTorques", &VehicleModelBicycle::setMaxTorques)
    .def("getPosition", &VehicleModelBicycle::getPosition)
    .def("getOrientation", &VehicleModelBicycle::getOrientation)
    .def("getVelocity", &VehicleModelBicycle::getVelocity)
    .def("getAcceleration", &VehicleModelBicycle::getAcceleration)
    .def("getAngularVelocity", &VehicleModelBicycle::getAngularVelocity)
    .def("getWheelspeeds", &VehicleModelBicycle::getWheelspeeds)
    .def("getWheelOrientations", &VehicleModelBicycle::getWheelOrientations)
    .def("getSteeringWheelAngle", &VehicleModelBicycle::getSteeringWheelAngle)
    .def("getTorques", &VehicleModelBicycle::getTorques)
    .def("setPosition", &VehicleModelBicycle::setPosition)
    .def("setOrientation", &VehicleModelBicycle::setOrientation);


    pybind11::class_<VehicleModel4Wheel>(m, "VehicleModel4Wheel")
    .def(pybind11::init())
    .def("readConfig", &VehicleModel4Wheel::readConfig)
    .def("getParameters", &VehicleModel4Wheel::getParameters)
    .def("configure", &VehicleModel4Wheel::configure)
    .def("forwardIntegrate", &VehicleModel4Wheel::forwardIntegrate)
    .def("forwardIntegrateControlFrame", &forwardIntegrateControlFrame<VehicleModel4Wheel>)
    .def("setSteeringSetpointFront", &VehicleModel4Wheel::setSteeringSetpointFront)
    .def("setRpmSetpoints", &VehicleModel4Wheel::setRpmSetpoints)
    .def("setMinTorques", &VehicleModel4Wheel::setMinTorques)
    .def("setMaxTorques", &VehicleModel4Wheel::setMaxTorques)
    .def("getPosition", &VehicleModel4Wheel::getPosition)
    .def("getOrientation", &VehicleModel4Wheel::getOrientation)
    .def("getVelocity", &VehicleModel4Wheel::getVelocity)
    .def("getAcceleration", &VehicleModel4Wheel::getAcceleration)
    .def("getAngularVelocity", &VehicleModel4Wheel::getAngularVelocity)
    .def("getWheelspeeds", &VehicleModel4Wheel::getWheelspeeds)
    .def("getWheelOrientations", &VehicleModel4Wheel::getWheelOrientations)
    .def("getSteeringWheelAngle", &VehicleModel4Wheel::getSteeringWheelAngle)
    .def("getTorques", &VehicleModel4Wheel::getTorques)
    .def("setPosition", &VehicleModel4Wheel::setPosition)
    .def("setOrientation", &VehicleModel4Wheel::setOrientation);

    pybind11::class_<ImuSensor>(m, "ImuSensor")
    .def(pybind11::init<double, double>())
    .def("readConfig", &ImuSensor::readConfig)
    .def("RunTick", &ImuSensor::RunTick)
    .def("getOldest", &ImuSensor::getOldest);

    pybind11::class_<Logger, std::shared_ptr<Logger>>(m, "Logger")
    .def(pybind11::init());

    pybind11::class_<CompetitionLogic>(m, "CompetitionLogic")
    .def(pybind11::init<std::shared_ptr<Logger>, Track&, MainConfig>())
    .def("performAllChecks", &CompetitionLogic::performAllChecks)
    .def("pointsInTrackConnected", &CompetitionLogic::pointsInTrackConnected)
    .def("fillReport", &CompetitionLogic::fillReport);

    typedef DeadTime<Wheels> wheelsDeadtime;
    pybind11::class_<wheelsDeadtime>(m, "WheelsDeadtime")
    .def(pybind11::init<double>())
    .def("getOldest", &wheelsDeadtime::getOldest)
    .def("availableDeadTime", &wheelsDeadtime::availableDeadTime)
    .def("addVal", &wheelsDeadtime::addVal);

    typedef DeadTime<double> scalarDeadtime;
    pybind11::class_<scalarDeadtime>(m, "ScalarDeadtime")
    .def(pybind11::init<double>())
    .def("getOldest", &scalarDeadtime::getOldest)
    .def("availableDeadTime", &scalarDeadtime::availableDeadTime)
    .def("addVal", &scalarDeadtime::addVal);

    pybind11::class_<VulkanCameraSim>(m, "VulkanCameraSim")
    .def(pybind11::init<int, int, const std::string&, const std::string&, const std::string&>(), pybind11::arg("width") = 306,
        pybind11::arg("height") = 256,
        pybind11::arg("modelRoot") = std::string("/root/workspace/pipeline/Models"),
        pybind11::arg("cameraConfigPath") = std::string(),
        pybind11::arg("carXacroPath") = std::string())
    .def("setAssetPaths", &VulkanCameraSim::setAssetPaths)
    .def("setTrackAndCones", &VulkanCameraSim::setTrackAndCones)
    .def("setShadowsEnabled", &VulkanCameraSim::setShadowsEnabled)
    .def("width", &VulkanCameraSim::width)
    .def("height", &VulkanCameraSim::height)
    .def("cameraNames", &VulkanCameraSim::cameraNames)
    .def("cameraEnabledFlags", &VulkanCameraSim::cameraEnabledFlags)
    .def("cameraRatesHz", &VulkanCameraSim::cameraRatesHz)
    .def("cameraDelayMeans", &VulkanCameraSim::cameraDelayMeans)
    .def("render", [](VulkanCameraSim& self, const Eigen::Vector3d& carPosition,
                        const Eigen::Vector3d& carOrientation,
                        float steeringAngle, const Eigen::Vector4d& wheelOrientations) {
        Wheels wheels;
        wheels.FL = wheelOrientations[0];
        wheels.FR = wheelOrientations[1];
        wheels.RL = wheelOrientations[2];
        wheels.RR = wheelOrientations[3];
        auto out = self.render(carPosition, carOrientation, steeringAngle, wheels);
        const int w = self.width();
        const int h = self.height();
        pybind11::list images;
        for (auto& img : out)
        {
            images.append(imageToNumpy(std::move(img), w, h));
        }
        return images;
    });
}
