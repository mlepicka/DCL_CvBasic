#ifndef CVKAZE_HPP_
#define CVKAZE_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "Types/Features.hpp"

#include <opencv2/opencv.hpp>

#if (CV_MAJOR_VERSION == 2)
#if (CV_MINOR_VERSION > 3)
#include <opencv2/nonfree/features2d.hpp>
#endif
#endif



namespace Processors {
namespace CvKAZE {

using namespace cv;

/*!
 * \class CvKAZE
 * \brief CvKAZE processor class.
 *
 * CvKAZE processor.
 */
class CvKAZE: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	CvKAZE(const std::string & name = "CvKAZE");

	/*!
	 * Destructor
	 */
	virtual ~CvKAZE();

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

	/*!
	 * Event handler function.
	 */
	void calculate_features();

	/// Input data stream
	Base::DataStreamIn <Mat> in_img;

	/// Output data stream containing extracted features
	Base::DataStreamOut <Types::Features> out_features;

	/// Output data stream containing feature descriptors
	Base::DataStreamOut <cv::Mat> out_descriptors;

	///  Property - path to save calculation results
	Base::Property<std::string> prop_calc_path;
	///  Property - set functions for calculate diffusivity in non linear space
	Base::Property<int> prop_diffusivity;

};

} //: namespace CvKAZE
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("CvKAZE", Processors::CvKAZE::CvKAZE)

#endif /* CVSIFT_HPP_ */
