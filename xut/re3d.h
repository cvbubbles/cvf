#pragma once

#include"def.h"
#include"opencv2/core.hpp"
#include"CVRender/cvrender.h"
#include"re3d/base/base.h"

_XUT_BEG
using namespace cv;
using namespace re3d;

inline void loadModelSet(ModelSet &modelSet, const std::string &modelsDir, const std::string &modelSetName, const char *idPattern=nullptr, bool searchRecursive = false, const std::string &filter = ".ply;.3ds;.obj;.stl")
{
	auto modelInfos = modelInfosFromDir(modelsDir, modelSetName, searchRecursive, filter);
	if (idPattern)
	{
		int n = modelInfosParseIDs(modelInfos, idPattern);
		CV_Assert(n == modelInfos.size());
	}
	modelSet.set(modelInfos, modelSetName, modelSetName);
}


class BaseRigidObject
{
public:
	float   score=0.f;

	int		modelIndex=-1; 

	cv::Rect  roi;

	RigidPose  pose;
};

template<typename _ObjT>
class BaseFrame
{
public:
	int    frameID;
	std::vector<_ObjT>  objs;
};

template<typename _ObjT>
inline const RigidPose& getRigidPose(_ObjT &obj)
{
	return obj.pose;
}
inline RigidPose getRigidPose(const re3d::Variable &v) {
	auto &vx=const_cast<re3d::Variable&>(v).get<std::vector<RigidPose>>();
	RigidPose pose;
	if (!vx.empty())
		pose = vx.front();
	return pose;
}

template<typename _ObjT>
inline cv::Mat renderResults(const cv::Mat &img, const std::vector<_ObjT> &objs, const Matx44f &mProj, ModelSet &modelSet, Scalar color=Scalar(255,0,0), bool _drawContour=true, bool _drawBlend=true, bool _drawScore=false, bool _drawBox=false)
{
	cv::Mat dimg = img.clone();
	bool useRandColor = color[0] < 0;

	for (auto &r : objs)
	{
		if (useRandColor)
			color = Scalar(rand() % 255, rand() % 255, rand() % 255);

		RigidPose pose = getRigidPose(r);

		CVRMats mats;
		mats.mModel = cvrm::fromR33T(pose.R, pose.t);
		mats.mProjection = mProj;

		auto modelPtr = modelSet.getModel(r.modelIndex);

		if (_drawContour || _drawBlend)
		{
			CVRModel &m3d = modelPtr->get3DModel();
			CVRender render(m3d);
			auto rr = render.exec(mats, img.size(), CVRM_IMAGE | CVRM_DEPTH, CVRM_DEFAULT, nullptr);
			//Mat1b mask = getRenderMask(rr.depth);
			Mat1b mask = rr.getMaskFromDepth();
			Rect roi = cv::get_mask_roi(DWHS(mask), 127);

			if (roi.empty())
				continue;

			if (_drawBlend)
			{
				Mat t;
				cv::addWeighted(dimg(roi), 0.5, rr.img(roi), 0.5, 0, t);
				t.copyTo(dimg(roi), mask(roi));
			}
			if (_drawContour)
			{
				std::vector<std::vector<Point> > cont;
				cv::findContours(mask, cont, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
				drawContours(dimg, cont, -1, color, 2, CV_AA);
			}
		}

		if (_drawBox)
		{
			cv::rectangle(dimg, r.roi, color, 2);
		}
		
		if (_drawScore)
		{
			char label[32];
			sprintf(label, "score=%.2f", r.score);
			cv::putText(dimg, label, cv::Point(r.roi.x + r.roi.width / 2, r.roi.y + r.roi.height / 2), cv::FONT_HERSHEY_PLAIN, 1.5, color, 2, CV_AA);
		}
	}
	return dimg;
}


inline Rect getProjectedROI(const std::vector<Point3f>& points, const Matx33f& K, const Matx33f& R, const Vec3f& t)
{
	CVRProjectorKRt prj(K, R, t);
	auto v2d = prj(points);
	return (Rect)getBoundingBox2D(v2d);
}

inline Matx33f getKInROI(Matx33f K, const Rect& roi, float scale = 1.f)
{
	return cvrm::getKInROI(K, roi, scale);
}

inline CVRResult renderPose(CVRender& render, Size viewSize, const RigidPose& pose, const Matx44f& mProj)
{
	CVRMats mats;
	mats.mProjection = mProj;
	mats.mModel = cvrm::fromR33T(pose.R, pose.t);

	return render.exec(mats, viewSize);
}


_XUT_END

