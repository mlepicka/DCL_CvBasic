/*!
 * \file
 * \brief 
 * \author Marta Lepicka
 */

#ifndef CALCSTATISTICS_HPP_
#define CALCSTATISTICS_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"

#include <opencv2/opencv.hpp>
#include "Types/Objects3D/Object3D.hpp"

#include <Types/HomogMatrix.hpp>
#include <Types/CameraInfo.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace Processors {
namespace CalcStatistics {

/*!
 * \class CalcStatistics
 * \brief CalcStatistics processor class.
 *
 * 
 */
class CalcStatistics: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	CalcStatistics(const std::string & name = "CalcStatistics");

	/*!
	 * Destructor
	 */
	virtual ~CalcStatistics();

	/*!
	 * Prepare components interface (register streams and handlers).
	 * At this point, all properties are already initialized and loaded to 
	 * values set in config file.
	 */
	void prepareInterface();

protected:

	/*!
	 * Connects source to given device.
	 */
	bool onInit();

	/*!
	 * Disconnect source from device, closes streams, etc.
	 */
	bool onFinish();

	/*!
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();

	// Input data streams
	Base::DataStreamIn<Types::HomogMatrix,Base::DataStreamBuffer::Newest> in_homog_matrix;
	Base::DataStreamIn<Types::HomogMatrix,Base::DataStreamBuffer::Newest> in_homog_matrix_right;
	Base::DataStreamIn<Base::UnitType> in_trigger;

	// Output data streams
	Base::DataStreamOut<Types::HomogMatrix> out_homog_matrix;

	// Handlers

	// Properties
	///  Property - path to save calculation results
	Base::Property<std::string> prop_calc_path;

	// Handlers
	Base::EventHandler<CalcStatistics> h_calculate;

	Types::HomogMatrix cumulated_homog_matrix;
	cv::Mat_<double> cumulated_rvec;
	cv::Mat_<double> cumulated_tvec;
	cv::Mat_<double> cumulated_axis;
	double cumulated_fi;

	cv::Mat_<double> avg_rvec;
	cv::Mat_<double> avg_tvec;
	cv::Mat_<double> avg_axis;
	double avg_fi;
	int counter;

	double sum_deviation_fi;
	double sum_deviation_axis_X;
	double sum_deviation_axis_Y;
	double sum_deviation_axis_Z;

	void calculate();

};

} //: namespace CalcStatistics
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("CalcStatistics", Processors::CalcStatistics::CalcStatistics)

#endif /* CALCSTATISTICS_HPP_ */
