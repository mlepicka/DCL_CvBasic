/*!
 * \file
 * \brief
 * \author Tomek Kornuta,,,
 */

#include <memory>
#include <string>

#include "CvSIFT.hpp"
#include "Common/Logger.hpp"
#include "Common/Timer.hpp"
#include <sstream>
#include <fstream>
#include <boost/bind.hpp>

#if CV_MAJOR_VERSION == 2
#if CV_MINOR_VERSION > 3
#include <opencv2/nonfree/features2d.hpp>
#endif
#elif CV_MAJOR_VERSION == 3
#include <opencv2/nonfree/features2d.hpp>
#endif

namespace Processors {
namespace CvSIFT {

CvSIFT::CvSIFT(const std::string & name) :
		Base::Component(name), prop_calc_path("Calculations.path",
				std::string(".")) {
	registerProperty(prop_calc_path);

}

CvSIFT::~CvSIFT() {
}

void CvSIFT::prepareInterface() {
	// Register handlers with their dependencies.
	registerHandler("onNewImage", boost::bind(&CvSIFT::onNewImage, this));
	addDependency("onNewImage", &in_img);

	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_features", &out_features);
	registerStream("out_descriptors", &out_descriptors);
}

bool CvSIFT::onInit() {

	return true;
}

bool CvSIFT::onFinish() {
	return true;
}

bool CvSIFT::onStop() {
	return true;
}

bool CvSIFT::onStart() {
	return true;
}

void CvSIFT::onNewImage() {
	LOG(LTRACE)<< "CvSIFT::onNewImage\n";
	try {
		// Input: a grayscale image.
		cv::Mat input = in_img.read();
		std::ofstream feature_calc_time;

		if(!string(prop_calc_path).empty()) {
			feature_calc_time.open((string(prop_calc_path)+string("czas_wyznaczenia_cech_sift.txt")).c_str(), ios::out|ios::app);
		}
		Common::Timer timer;

		timer.restart();
		//-- Step 1: Detect the keypoints.
		cv::SiftFeatureDetector detector(0,4);
		std::vector<cv::KeyPoint> keypoints;
		detector.detect(input, keypoints);

		//-- Step 2: Calculate descriptors (feature vectors).
		cv::SiftDescriptorExtractor extractor;
		Mat descriptors;
		extractor.compute( input, keypoints, descriptors);

		if(!string(prop_calc_path).empty()) {
			feature_calc_time << timer.elapsed() << endl;
		}
		// Write results to outputs.
		Types::Features features(keypoints);
		features.type = "SIFT";
		out_features.write(features);
		out_descriptors.write(descriptors);
	} catch (...) {
		LOG(LERROR) << "CvSIFT::onNewImage failed\n";
	}
}

} //: namespace CvSIFT
} //: namespace Processors
