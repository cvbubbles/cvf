#pragma once

#include"def.h"
#include"re3d.h"
#include"opencv2/flann.hpp"
#include"BFC/stdf.h"
#include"BFC/portable.h"

_XUT_BEG

class PoseEvaluator
{
public:
	static std::vector<Point3f> sampleModelPoints(const std::vector<Point3f> &vertices, int maxPoints = 2000)
	{
		double step = __max(1.0, double(vertices.size()) / maxPoints);
		std::vector<Point3f> v;
		for (double i = 0; i < (double)vertices.size(); i += step)
		{
			int j = int(i + 0.5);
			if (j < vertices.size())
				v.push_back(vertices[j]);
		}
		return v;
	}
	static float calcModelDiameter(const std::vector<cv::Point3f>& points) 
	{
		float maxDist = 0;
		for (size_t i = 0; i < points.size(); ++i)
			for (size_t j = i + 1; j < points.size(); ++j)
			{
				Point3f dv = points[i] - points[j];
				float d = dv.dot(dv);
				if (d > maxDist)
					maxDist = d;
			}
		return sqrt(maxDist);
	}

	static std::vector<cv::Point2f>  _resampleAPCurve(const std::vector<cv::Point2f>& points, int curveSamples)
	{
		float delta = points.back().x / curveSamples;
		std::vector<Point2f>  samples;
		samples.push_back(points.front());
		size_t ptcur = 0;
		for (int i = 1; i < curveSamples; ++i)
		{
			float x = delta * i;
			while (ptcur + 1<points.size() && points[ptcur + 1].x < x)
				++ptcur;
			CV_Assert(x >= points[ptcur].x && x <= points[ptcur + 1].x);
			float w = (x - points[ptcur].x) / (points[ptcur + 1].x - points[ptcur].x);
			float y = (1.f - w) * points[ptcur].y + w * points[ptcur + 1].y;
			samples.push_back(Point2f(x, y));
		}
		samples.push_back(points.back());
		return samples;
	}

	static float calcScoreAUC(const std::vector<float>& _errs, float maxThreshold, std::vector<cv::Point2f> *curve = nullptr, int curveSamples = 100)
	{
		std::vector<double> rec(_errs.size() + 1);
		rec[0] = 0.;
		//memcpy(&rec[1], &_errs[0], sizeof(float) * _errs.size());
		for (size_t i = 0; i < _errs.size(); ++i)
			rec[i + 1] = _errs[i];
		std::sort(rec.begin(), rec.end());

		std::vector<double> prec(rec.size());
		prec[0] = 0.f;
		int n = 1;
		for (; n < rec.size(); ++n)
		{
			if (rec[n] < maxThreshold)
				prec[n] = double(n) / double(_errs.size());
			else
				break;
		}
		if (n != rec.size())
		{
			rec.resize(n);
			prec.resize(n);
		}
		rec.push_back(maxThreshold);
		prec.push_back(prec.back());

		std::vector<Point2f>  curvePoints;
		curvePoints.push_back(Point2f(0.f, 0.f));

		double ap = 0;
		for (int i = 1; i < (int)rec.size(); ++i)
		{
			ap += (rec[i] - rec[i - 1]) * prec[i];
			curvePoints.push_back(Point2f(float(rec[i]), prec[i]));
		}
		if (curve)
			*curve = _resampleAPCurve(curvePoints, curveSamples);

		return float(ap / maxThreshold);
	}
	static float calcErrorR(const cv::Matx33f &R1, const cv::Matx33f &R2)
	{
		cv::Matx33f tmp = R2.t() * R1;
		float trace = tmp(0, 0) + tmp(1, 1) + tmp(2, 2);
		float dr = (trace - 1) / 2;
		dr = __max(-1, __min(1, dr));
		return acos(dr) * 180 / CV_PI;
	}

	static float calcErrorT(const cv::Vec3f &t1, const cv::Vec3f &t2)
	{
		float l22 = pow(t1[0] - t2[0], 2) + pow(t1[1] - t2[1], 2) + pow(t1[2] - t2[2], 2);
		return sqrt(l22);
	}

	static std::vector<Point3f> loadPointsFromXYZFile(const std::string& file)
	{
		std::vector<Point3f>  points;
		FILE* fp = fopen(file.c_str(), "r");
		if (!fp)
			throw "file open failed:"+file;
		Point3f pt;
		while (fscanf(fp, "%f%f%f", &pt.x, &pt.y, &pt.z) == 3)
			points.push_back(pt);
		fclose(fp);
		return points;
	}
	static void savePointsToXYZFile(const std::string& file, const std::vector<Point3f> &points)
	{
		FILE* fp = fopen(file.c_str(), "w");
		if (!fp)
			throw "file open failed:"+file;
		for(auto &p : points)
		{
			fprintf(fp, "%f\t%f\t%f\n", p.x, p.y, p.z);
		}
		fclose(fp);
	}
private:
	std::vector<cv::Point3f>  _modelPoints;
	cv::flann::Index          _index;
	bool                      _indexBuilt=false;

	void _buildIndex()
	{
		if (!_indexBuilt && !_modelPoints.empty())
		{
			Mat _points((int)_modelPoints.size(), 3, CV_32FC1, &_modelPoints[0]);
			_index.build(_points, cv::flann::KDTreeIndexParams());
			_indexBuilt = true;
		}
	}
public:
	PoseEvaluator() = default;

	void loadModelPoints(std::string file, int maxPoints = -1, bool autoSaveAndLoadFromXYZ=false)
	{
		std::string xyzFile = ff::ReplacePathElem(file, "xyz", ff::RPE_FILE_EXTENTION);
		bool xyzFileExist = ff::pathExist(xyzFile);
		if (autoSaveAndLoadFromXYZ && xyzFileExist)
			file = xyzFile;

		std::vector<Point3f>  modelPoints;
		std::string ext = ff::GetFileExtention(file);
		ff::str2lower(ext);
		if (ext == "xyz")
		{
			modelPoints = loadPointsFromXYZFile(file);
			autoSaveAndLoadFromXYZ = false;
		}
		else
		{
			CVRModel model;
			model.load(file, 0);
			modelPoints = model.getVertices();
		}
		if (maxPoints > 0)
			_modelPoints = sampleModelPoints(modelPoints, maxPoints);
		else
			_modelPoints.swap(modelPoints);
		_indexBuilt = false;

		if (autoSaveAndLoadFromXYZ && !xyzFileExist)
		{
			savePointsToXYZFile(xyzFile, _modelPoints);
		}

	}
	bool empty() const
	{
		return _modelPoints.empty();
	}

	float calcADD(const cv::Matx33f& R1, const cv::Vec3f& t1, const cv::Matx33f& R2, const cv::Vec3f& t2)
	{
		double v = 0.0;
		for (auto& p : _modelPoints)
		{
			auto q1 = R1 * p + Point3f(t1);
			auto q2 = R2 * p + Point3f(t2);
			auto dv = q1 - q2;
			v += sqrt(dv.dot(dv));
		}
		return float(v / _modelPoints.size());
	}
	float calcPRJ(const cv::Matx33f& R1, const cv::Vec3f& t1, const cv::Matx33f& R2, const cv::Vec3f& t2, const cv::Matx33f& K)
	{
		double v = 0.0;
		CVRProjectorKRt prj1(K, R1, t1), prj2(K, R2, t2);
		for (auto& p : _modelPoints)
		{
			auto dv = prj1(p) - prj2(p);
			double d = dv.ddot(dv);
			v += sqrt(d);
		}
		return float(v / _modelPoints.size());
	}

	float calcADDS(const cv::Matx33f& R1, const cv::Vec3f& t1, const cv::Matx33f& R2, const cv::Vec3f& t2)
	{
		std::vector<Point3f>  vq(_modelPoints.size());
		for (size_t i = 0; i < _modelPoints.size(); ++i)
			vq[i] = R2 * _modelPoints[i] + Point3f(t2);

		float v = 0.0;
		for (auto& p : _modelPoints)
		{
			auto q1 = R1 * p + Point3f(t1);
			float dmin = FLT_MAX;
			for (auto& q2 : vq)
			{
				auto dv = q1 - q2;
				float d = dv.dot(dv);
				if (d < dmin)
					dmin = d;
			}
			v += sqrt(dmin);
		}
		return v / _modelPoints.size();
	}
	float calcADDSfast(const cv::Matx33f& R1, const cv::Vec3f& t1, const cv::Matx33f& R2, const cv::Vec3f& t2)
	{
		cv::Matx33f R2inv = R2.inv();
		cv::Matx33f R = R2inv*R1;
		cv::Point3f t = R2inv*Point3f(t1 - t2);

		std::vector<Point3f>  vq(_modelPoints.size());
		for (size_t i = 0; i < _modelPoints.size(); ++i)
			vq[i] = R * _modelPoints[i] + t;

		this->_buildIndex();
		cv::Mat vqMat((int)vq.size(), 3, CV_32FC1, &vq[0]);
		
		Mat idx, dists;
		_index.knnSearch(vqMat, idx, dists, 1);

		CV_Assert(dists.type() == CV_32FC1);
		const float *distsData = dists.ptr<float>();

		float v = 0.0;
		for (size_t i = 0; i < vq.size(); ++i)
			v += sqrt(distsData[i]);

		return v / vq.size();
	}
};






_XUT_END


