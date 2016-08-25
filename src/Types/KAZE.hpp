#ifndef KAZE_HPP_
#define KAZE_HPP_

#include <vector>
#include <opencv2/core/core.hpp>
#include "kaze/KAZEFeatures.h"
#include "kaze/KAZEFeatures.cpp"

using namespace cv;
namespace cv
{
    class KAZE 
    {
    public:
		    enum
    {
        DIFF_PM_G1 = 0,
        DIFF_PM_G2 = 1,
        DIFF_WEICKERT = 2,
        DIFF_CHARBONNIER = 3
    };
		
    public:
        KAZE(bool _extended, bool _upright, float _threshold, int _octaves,
                   int _sublevels, int _diffusivity)
        : extended(_extended)
        , upright(_upright)
        , threshold(_threshold)
        , octaves(_octaves)
        , sublevels(_sublevels)
        , diffusivity(_diffusivity)
        {
        }

        virtual ~KAZE() {}

        void setExtended(bool extended_) { extended = extended_; }
        bool getExtended() const { return extended; }

        void setUpright(bool upright_) { upright = upright_; }
        bool getUpright() const { return upright; }

        void setThreshold(double threshold_) { threshold = (float)threshold_; }
        double getThreshold() const { return threshold; }

        void setNOctaves(int octaves_) { octaves = octaves_; }
        int getNOctaves() const { return octaves; }

        void setNOctaveLayers(int octaveLayers_) { sublevels = octaveLayers_; }
        int getNOctaveLayers() const { return sublevels; }

        void setDiffusivity(int diff_) { diffusivity = diff_; }
        int getDiffusivity() const { return diffusivity; }

        // returns the descriptor size in bytes
        int descriptorSize() const
        {
            return extended ? 128 : 64;
        }

        // returns the descriptor type
        int descriptorType() const
        {
            return CV_32F;
        }

        // returns the default norm type
        int defaultNorm() const
        {
            return NORM_L2;
        }

        void detectAndCompute(InputArray image, InputArray mask,
                              std::vector<KeyPoint>& keypoints,
                              OutputArray descriptors,
                              bool useProvidedKeypoints=false)
        {
            cv::Mat img = image.getMat();
            if (img.channels() > 1)
                cvtColor(image, img, COLOR_BGR2GRAY);

            Mat img1_32;
            if ( img.depth() == CV_32F )
                img1_32 = img;
            else if ( img.depth() == CV_8U )
                img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
            else if ( img.depth() == CV_16U )
                img.convertTo(img1_32, CV_32F, 1.0 / 65535.0, 0);

            CV_Assert( ! img1_32.empty() );

            KAZEOptions options;
            options.img_width = img.cols;
            options.img_height = img.rows;
            options.extended = extended;
            options.upright = upright;
            options.dthreshold = threshold;
            options.omax = octaves;
            options.nsublevels = sublevels;
            options.diffusivity = diffusivity;

            KAZEFeatures impl(options);
            impl.Create_Nonlinear_Scale_Space(img1_32);

            if (!useProvidedKeypoints)
            {
                impl.Feature_Detection(keypoints);
            }

            if (!mask.empty())
            {
                cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
            }

            if( descriptors.needed() )
            {
                Mat& desc = descriptors.getMatRef();
                impl.Feature_Description(keypoints, desc);

                CV_Assert((!desc.rows || desc.cols == descriptorSize()));
                CV_Assert((!desc.rows || (desc.type() == descriptorType())));
            }
        }

        void write(FileStorage& fs) const
        {
            fs << "extended" << (int)extended;
            fs << "upright" << (int)upright;
            fs << "threshold" << threshold;
            fs << "octaves" << octaves;
            fs << "sublevels" << sublevels;
            fs << "diffusivity" << diffusivity;
        }

        void read(const FileNode& fn)
        {
            extended = (int)fn["extended"] != 0;
            upright = (int)fn["upright"] != 0;
            threshold = (float)fn["threshold"];
            octaves = (int)fn["octaves"];
            sublevels = (int)fn["sublevels"];
            diffusivity = (int)fn["diffusivity"];
        }
        
		static Ptr<KAZE> create(bool extended=false, bool upright=false,
                                    float threshold = 0.001f,
                                    int nOctaves = 4, int nOctaveLayers = 4,
                                    int diffusivity = KAZE::DIFF_WEICKERT)
                          {
			return makePtr<KAZE>(extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity);
		}

        bool extended;
        bool upright;
        float threshold;
        int octaves;
        int sublevels;
        int diffusivity;
    };


}
#endif
