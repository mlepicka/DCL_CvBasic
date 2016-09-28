/*!
 * \file
 * \brief
 * \author Mort
 */

#include <memory>
#include <string>

#include "CvGaussianNoise.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Processors {
namespace CvGaussianNoise {

CvGaussianNoise::CvGaussianNoise(const std::string & name) :
		Base::Component(name), pass_through("pass_through",0), mean("GaussianNoise.mean",0) , sigma("GaussianNoise.sigma",0)   {
			registerProperty(pass_through);
			registerProperty(mean);
			registerProperty(sigma);
}

CvGaussianNoise::~CvGaussianNoise() {
}

void CvGaussianNoise::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_img", &in_img);
    registerStream("out_img", &out_img);

	// Register handlers
	registerHandler("generate_noise", boost::bind(&CvGaussianNoise::generate_noise, this));
	addDependency("generate_noise", &in_img);
}

bool CvGaussianNoise::onInit() {

	return true;
}

bool CvGaussianNoise::onFinish() {
	return true;
}

bool CvGaussianNoise::onStop() {
	return true;
}

bool CvGaussianNoise::onStart() {
	return true;
}

void CvGaussianNoise::generate_noise(){
	CLOG(LINFO) << "in generate_noise()";
	cv::Mat greyMat, colorMat, result;
	colorMat =in_img.read();
	
	cv::cvtColor(colorMat, greyMat, CV_BGR2GRAY);
	
	CLOG(LINFO) << "in generate_noise() " <<greyMat.type();
	
	cv::Mat noise = cv::Mat(greyMat.size(), greyMat.type());
	result = greyMat;
	
	
	cv::randn(noise, 0, 10);

	noise.convertTo(noise, greyMat.type());
	
	result = result + noise;
	
	//cv::imshow("OUTPUT",result);
	//cvWaitKey(0);

	if(!pass_through){
		out_img.write(result);
	}else{
		out_img.write(colorMat);
	}
	
	
	
}

} //: namespace CvGaussianNoise
} //: namespace Processors
