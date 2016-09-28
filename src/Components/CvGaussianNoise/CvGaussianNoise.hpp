/*!
 * \file
 * \brief 
 * \author Mort
 */

#ifndef CVGAUSSIANNOISE_HPP_
#define CVGAUSSIANNOISE_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"

#include <opencv2/opencv.hpp>


namespace Processors {
namespace CvGaussianNoise {

/*!
 * \class CvGaussianNoise
 * \brief CvGaussianNoise processor class.
 *
 * Component for generate noisy gray image
 */
class CvGaussianNoise: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	CvGaussianNoise(const std::string & name = "CvGaussianNoise");

	/*!
	 * Destructor
	 */
	virtual ~CvGaussianNoise();

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
	Base::DataStreamIn< cv::Mat, Base::DataStreamBuffer::Newest > in_img;
	
	// Output data streams
	Base::DataStreamOut< cv::Mat> out_img;
	
	// Handlers

	// Properties

	Base::Property<bool> pass_through;
	
	Base::Property<float> mean;

	Base::Property<float> sigma;

	// Handlers

	void generate_noise();

};

} //: namespace CvGaussianNoise
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("CvGaussianNoise", Processors::CvGaussianNoise::CvGaussianNoise)

#endif /* CVGAUSSIANNOISE_HPP_ */
