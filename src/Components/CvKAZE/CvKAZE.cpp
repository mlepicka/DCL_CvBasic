#include <memory>
#include <string>

#include "CvKAZE.hpp"
#include "Common/Logger.hpp"
#include "Common/Timer.hpp"
#include <sstream>
#include <fstream>
#include <boost/bind.hpp>

#include <opencv2/features2d.hpp>
#include "Types/KAZE.hpp"
#include "Types/AKAZE.hpp"

namespace Processors {
namespace CvKAZE {

CvKAZE::CvKAZE(const std::string & name) :
		Base::Component(name), prop_calc_path("Calculations_path",std::string(".")) {
	registerProperty(prop_calc_path);
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

		std::ofstream feature_calc_time;
		feature_calc_time.open((string(prop_calc_path)+string("czas_wyznaczenia_cech.txt")).c_str(), ios::out|ios::app);

		std::vector<cv::KeyPoint> keypoints;
		Mat descriptors;
		Common::Timer timer;

		timer.restart();
		Ptr<AKAZE> kaze = AKAZE::create(); //create(false);
		kaze->detectAndCompute(input, cv::noArray(), keypoints, descriptors);

		feature_calc_time << timer.elapsed() << endl;

		// Write results to outputs.
	    Types::Features features(keypoints, "KAZE");

	    LOG(LTRACE) << "CvKaze:: features found: " << features.features.size() << " descriptors found: " << descriptors.size() << "\n";
	    //features.type = "KAZE";
	    LOG(LTRACE) <<"FEATURES TYPE: " << features.type;
		out_features.write(features);
		out_descriptors.write(descriptors);

			} catch (...) {
		LOG(LERROR) << "CvKAZE::onNewImage failed\n";
	}
}


} //: namespace CvKAZE
} //: namespace Processors
