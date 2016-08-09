/*!
 * \file
 * \brief
 * \author Tomek Kornuta,,,
 */

#include <memory>
#include <string>

#include "CvKAZE.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

#include <opencv2/features2d.hpp>
namespace Processors {
namespace CvKAZE {

CvKAZE::CvKAZE(const std::string & name) :
		Base::Component(name)  {

}

CvKAZE::~CvKAZE() {
}

void CvKAZE::prepareInterface() {
	// Register handlers with their dependencies.
	registerHandler("onNewImage", boost::bind(&CvKAZE::onNewImage, this));
	addDependency("onNewImage", &in_img);

	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_features", &out_features);
	registerStream("out_descriptors", &out_descriptors);
}

bool CvKAZE::onInit() {

	return true;
}

bool CvKAZE::onFinish() {
	return true;
}

bool CvKAZE::onStop() {
	return true;
}

bool CvKAZE::onStart() {
	return true;
}

void CvKAZE::onNewImage()
{
	//tu property czy kaze czy akaze
	//kaze nie extended

	LOG(LTRACE) << "CvKAZE::onNewImage\n";
	try {
		// Input: a grayscale image.
		cv::Mat input = in_img.read();

		std::vector<cv::KeyPoint> keypoints;
		Mat descriptors;

		Ptr<KAZE> kaze = KAZE::create();
		kaze->detectAndCompute(input, cv::noArray(), keypoints, descriptors);

		// Write results to outputs.
	    Types::Features features(keypoints);
		out_features.write(features);
		out_descriptors.write(descriptors);
	} catch (...) {
		LOG(LERROR) << "CvKAZE::onNewImage failed\n";
	}
}


} //: namespace CvKAZE
} //: namespace Processors
