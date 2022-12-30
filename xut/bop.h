#pragma once

#include"re3d.h"
#include"json.h"
#include"eval_3d_pose.h"

_XUT_BEG
//_NS_BEG(bop)

inline Matx33f getYCBVCameraK()
{//1066.778, 1067.487, 312.9869, 241.3109
	cv::Matx33f K = cv::Matx33f::zeros();
	K(0, 0) = 1066.778;
	K(0, 2) = 312.9869;
	K(1, 1) = 1067.487;
	K(1, 2) = 241.3109;
	K(2, 2) = 1.0f;
	return K;
}

class BOPDataset
{
public:
	class Model
	{
	public:
		float		diameter;
		std::string modelFile;
	private:
		std::shared_ptr<CVRModel>  _modelPtr;
		std::shared_ptr<PoseEvaluator>  _evalPtr;
	public:
		CVRModel& getModel()
		{
			if (!_modelPtr)
				_modelPtr = std::make_shared<CVRModel>(this->modelFile);
			return *_modelPtr;
		}
		PoseEvaluator& getEvaluator()
		{
			if (!_evalPtr)
			{
				_evalPtr = std::make_shared<PoseEvaluator>();
				_evalPtr->loadModelPoints(this->modelFile, 2000, true);
			}
			return *_evalPtr;
		}
	};
public:
	template<typename _FrameT>
	static void doLoadSceneGT(const std::string& file, std::map<int, _FrameT>& frames)
	{
		frames.clear();
		nlohmann::ordered_json jf = jsonLoad(file);
		for (auto& jx : jf.items())
		{
			int frameID= atoi(jx.key().c_str());
			
			auto& fx = frames[frameID];
			fx.frameID = frameID;
			auto jobjs = jx.value();
			fx.objs.resize(jobjs.size());
			for (size_t i = 0; i < jobjs.size(); ++i)
			{
				auto& jv = jobjs[i];
				auto& obj = fx.objs[i];
				obj.modelIndex = jv["obj_id"].get<int>();
				jsonGetArray(jv["cam_R_m2c"], obj.pose.R.val, 9);
				jsonGetArray(jv["cam_t_m2c"], obj.pose.t.val, 3);
				obj.score = 1.f;
			}
		}
	}
	static Mat1f doLoadDepth(const std::string& file, float scale = 0.1f)
	{
		Mat dimg = imread(file, cv::IMREAD_UNCHANGED);
		int step = dimg.step;
		CV_Assert(dimg.type() == CV_16UC1);
		Mat1f depth;
		dimg.convertTo(depth, CV_32F);
		return depth * scale;
	}

	static std::map<int, Model>  doLoadModels(const std::string& modelDir)
	{
		std::map<int, Model> infos;
		auto jfInfos = jsonLoad(modelDir + "/models_info.json");
		for (auto& jx : jfInfos.items())
		{
			int mid = atoi(jx.key().c_str());
			auto& mi = infos[mid];
			auto& jv = jx.value();
			mi.diameter = jv["diameter"].get<float>();
			mi.modelFile = modelDir + ff::StrFormat("/obj_%06d.ply", mid);
		}
		return infos;
	}
	struct CameraInfo
	{
		Matx33f  K;
		float    depthScale;
	};
	static CameraInfo  doLoadFirstCameraInfo(const std::string &jfile)
	{
		CameraInfo cam;
		auto jf = jsonLoad(jfile);
		for (auto& jx : jf)
		{
			jsonGetArray(jx["cam_K"], cam.K.val, 9);
			cam.depthScale = jx["depth_scale"].get<float>();
		}
		return cam;
	}
public:

	struct Scene
	{
	public:
		std::string dir;
	private:
		CameraInfo  _camInfo;
	public:
		Scene(const std::string& dir_)
			:dir(dir_)
		{
			_camInfo.depthScale = -1.f;
		}
		std::string colorFile(int fid)
		{
			return dir + ff::StrFormat("rgb/%06d.png", fid);
		}
		std::string depthFile(int fid)
		{
			return dir + ff::StrFormat("depth/%06d.png", fid);
		}

		template<typename _FrameT>
		void loadGT(std::map<int, _FrameT>& frames)
		{
			doLoadSceneGT(this->dir + "scene_gt.json", frames);
		}

		Mat loadColorImage(int fid)
		{
			return cv::imread(this->colorFile(fid), cv::IMREAD_COLOR);
		}
		Mat1f loadDepthImage(int fid, float scale = 0.1f)
		{
			return doLoadDepth(this->depthFile(fid), scale);
		}
		const CameraInfo& getCameraInfo() 
		{
			if (_camInfo.depthScale < 0)
			{
				_camInfo = doLoadFirstCameraInfo(dir + "/scene_camera.json");
			}
			return _camInfo;
		}
	};

	std::string droot;
private:
	std::map<int, Model> _models;
public:
	BOPDataset(const std::string &droot_)
		:droot(droot_)
	{}
	std::string sceneDir(const std::string& sceneName)
	{
		return droot + "/test/" + sceneName + "/";
	}
	std::string modelsDir()
	{
		return droot + "/models/";
	}
	Scene getScene(const std::string& sceneName)
	{
		return Scene(this->sceneDir(sceneName));
	}
	std::map<int, Model>& getModels()
	{
		if (_models.empty())
			_models = doLoadModels(this->modelsDir());
		return _models;
	}
};


//_NS_END(bop)
_XUT_END


