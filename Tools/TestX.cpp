#include"appstd.h"
#include<map>
#include"CVX/bfsio.h"
#include"opencv2/calib3d.hpp"
#include<iostream>
using namespace ff;
using namespace std;

_CMDI_BEG

void test_mat_nd()
{
	Mat3f m(3, 2);

	int sizes[] = { 2,3,4,5 };
	m.create(3, sizes);
	Size sz = m.size();
	int v = m.rows;
	auto mx=m.reshape(4);
	
	return;
}

void test_homography()
{
	const int vp[] = { 431, 490, 616, 125, 799, 216, 626, 571 };
	const int vq[] = { 432, 491, 615, 126,	800, 217, 625, 570 };
	std::vector<Point2f> p, q;
	for (int i = 0; i < 4; ++i)
	{
		p.emplace_back(vp[i * 2], vp[i * 2 + 1]);
		q.emplace_back(vq[i * 2], vq[i * 2 + 1]);
	}

	//for (int i = 0; i < 3; ++i)
	//{
	//	p.push_back((p[i] + p[i + 1])*0.5);
	//	q.push_back((q[i] + q[i + 1])*0.5);
	//}
	//p.push_back((p[0] + p[3])*0.5);
	//q.push_back((q[0] + q[3])*0.5);

	Mat H = cv::findHomography(p, q);
	std::cout << H << std::endl;
	Point2f s(545.39, 319.545);
	auto dp=cv::transH(s, (double*)H.data);
	std::cout << dp << std::endl;
	
	Mat dimg = Mat3b::zeros(1000, 1000);
	for (auto x : p)
		cv::circle(dimg, Point(x), 3,Scalar(255,255,255));
	cv::circle(dimg, Point(s), 3, Scalar(0, 255, 255),-1);
	imshow("dimg", dimg);

	Mat1b mask = Mat1b::zeros(dimg.size());
	std::vector<std::vector<Point>> poly(1);
	poly[0] = cv::cvtPoint(p);
	cv::fillPoly(mask,poly,Scalar(255));

	Mat1f error = Mat1f::zeros(dimg.size());
	for_each_2c(DWHN1(error), DN1(mask), [H](float &err, uchar m, double x, double y) {
		if (m != 0)
		{
			double dx, dy;
			transH(x, y, dx, dy, (double*)H.data);
			double d = fabs(dx - x) + fabs(dy - y);
			if (d > 2)
			{
				err = __min(d - 2, 3.0) / 3.0;
			}
		}
	});
	imshowx("error", error);

	cv::cvxWaitKey();
}

CMD_BEG()
CMD0("test.homography", test_homography)
CMD_END()


void test_load_ply_pointcloud()
{
	std::string file = R"(f:\scene_dense.ply)";
	CVRModel model(file);
	mdshow("model", model);
	cvxWaitKey();
}

CMD_BEG()
CMD0("test.load_ply_pointcloud", test_load_ply_pointcloud)
CMD_END()


void test_show_model_in_display_1()
{
	float  modelSize = 0.25;  //模型的尺寸，单位米（下同）
	cv::Size2f  displaySize(0.62, 0.35);  //显示器宽高

	/*世界坐标系：假设摄像头放在显示器上边缘中心，世界坐标系原点是摄像头位置，X轴向右，Y轴向上，Z轴向外
	*/
	cv::Point3f modelPosition(0, -displaySize.height/2, 0); //模型中心在世界坐标系的位置
	cv::Point3f viewPostion(0, 0, 2.f); //人眼视点位置

	std::string file = R"(F:\SDUicloudCache\re3d\test\cat.obj)";
	CVRModel model(file);
	CVRMats mats;

	//把模型缩放到指定大小，并把中心平移到modelPosition
	{
		//缩放
		model.setSceneTransformation(model.calcStdPose());
		Vec3f vsize=model.getSizeBB();
		float scale = modelSize/vsize[1];
		model.setSceneTransformation(cvrm::scale(scale));

		//平移, mats.mModeli是对模型的初始变换，可以理解成model-view变换的一部分
		auto t = modelPosition - (Point3f)model.getCenter();
		mats.mModeli = cvrm::translate(t.x, t.y, t.z);
	}
	
	//根据视点位置设置视图变换
	mats.mView = cvrm::lookat(viewPostion.x, viewPostion.y, viewPostion.z, modelPosition.x, modelPosition.y, modelPosition.z, 0, 1, 0);

	//渲染人眼视野下观察到的图像
	Size viewSize(1920/2, 1080/2); //图像大小
	mats.mProjection = cvrm::fromK(cvrm::defaultK(viewSize, 1.5f),viewSize,0.1,100); //投影变换，这里用defaultK假设一个人眼的内参

	//mats.mProjection = mats.mProjection*cvrm::ortho(-0.5, 0.5, -0.5, 0.5, -1, 1);

	CVRender render(model);
	auto rr=render.exec(mats, viewSize); //渲染人眼视野下观察到的图像 rr.img

	Mat vdisp = rr.img.clone();

	std::vector<Point3f> displayCorners3D; //显示器4个角在世界坐标系的三维坐标
	std::vector<Point2f>  displayCornersInHumanView; //显示器4个角在人眼画面下的投影坐标

	{
		float w = displaySize.width, h = displaySize.height;
		displayCorners3D = { Point3f(-w / 2,0,0),Point3f(w / 2,0,0),Point3f(w / 2,-h,0),Point3f(-w / 2,-h,0) };

		//对相机坐标系下的三维点p的投影可以直接用K*p得到，其中K是3*3内参矩阵，和CVRProjector的结果是一致的
		//如果p是模型上一点，则先用modelView矩阵进行变换
		CVRProjector prj(mats.mView, mats.mProjection, viewSize);
		prj.project(displayCorners3D, displayCornersInHumanView); //从世界坐标系投影到人眼画面
	}
	
	//绘制显示器在人眼画面下的四边形区域
	{
		std::vector<std::vector<Point>> poly(1);
		poly[0] = cvtPoint(displayCornersInHumanView);
		cv::polylines(vdisp, poly, true, Scalar(0, 255, 255), 2, CV_AA);
	}

	std::vector<Point2f> realDisplayCorners;//显示器屏幕坐标系下4个角点的坐标
	{
		//这里用图像坐标，还是用归一化的[-1,1]的坐标？理论上应该是一样的，只要和后面正交投影一致就行……
		float w(viewSize.width), h(viewSize.height);
		realDisplayCorners = { Point2f(0,0),Point2f(w,0),Point2f(w,h),Point2f(0,h) };
	}

	//计算显示器从人眼画面和屏幕坐标系的homography变换
	Matx33f H = cv::findHomography(displayCornersInHumanView, realDisplayCorners, 0);
	Matx44f Hx = Matx44f::eye();
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			Hx(i, j) = H(i, j);
	
	//设置新的投影变换并渲染显示器上的显示图像
	//!!!变换不能这么设置，需要在shader里实现。mProjection投影后的结果是在[-1,1]的OpenGL屏幕坐标系，并且是齐次坐标，不能直接乘上H。
	mats.mProjection = mats.mProjection*Hx; 
	rr = render.exec(mats, viewSize);

	imshow("vdisp", vdisp);
	imshow("disp", rr.img);

	cv::waitKey();
}


void test_show_model_in_display_2()
{
	float  modelSize = 0.25;  //模型的尺寸，单位米（下同）
	cv::Size2f  displaySize(0.62, 0.35);  //显示器宽高
	float boxDepth = 0.5f;  

	/*世界坐标系：假设摄像头放在显示器上边缘中心，世界坐标系原点是摄像头位置，X轴向右，Y轴向上，Z轴向外
	*/
	cv::Point3f modelPosition(0, -displaySize.height / 2, 0); //模型中心在世界坐标系的位置
	cv::Point3f viewPostion(0, 0, 2.f); //人眼视点位置

	std::string file = R"(F:\SDUicloudCache\re3d\test\cat.obj)";
	CVRModel model(file);
	CVRMats mats;

	//把模型缩放到指定大小，并把中心平移到modelPosition
	{
		//缩放
		model.setSceneTransformation(model.calcStdPose());
		Vec3f vsize = model.getSizeBB();
		float scale = modelSize / vsize[1];
		model.setSceneTransformation(cvrm::scale(scale));

		//平移, mats.mModeli是对模型的初始变换，可以理解成model-view变换的一部分
		auto t = modelPosition - (Point3f)model.getCenter();
		//mats.mModeli = cvrm::translate(t.x, t.y, t.z);
		model.setSceneTransformation(cvrm::translate(t.x, t.y, t.z));
	}

	//根据视点位置设置视图变换
	mats.mView = cvrm::lookat(viewPostion.x, viewPostion.y, viewPostion.z, modelPosition.x, modelPosition.y, modelPosition.z, 1, 1, 0);


	//渲染人眼视野下观察到的图像
	Size viewSize(1920 / 2, 1080 / 2); //图像大小
	Matx33f K = cvrm::defaultK(viewSize, 1.5f);
	mats.mProjection = cvrm::fromK(K, viewSize, 0.1, 100); //投影变换，这里用defaultK假设一个人眼的内参

	CVRender render(model);
	auto rr = render.exec(mats, viewSize); //渲染人眼视野下观察到的图像 rr.img

	Mat vdisp = rr.img.clone();

	std::vector<Point3f> displayCorners3D; //显示器4个角在世界坐标系的三维坐标
	std::vector<Point2f>  displayCornersInHumanView; //显示器4个角在人眼画面下的投影坐标

	{
		float w = displaySize.width, h = displaySize.height, z=boxDepth;
		displayCorners3D = { Point3f(-w / 2,0,0),Point3f(w / 2,0,0),Point3f(w / 2,-h,0),Point3f(-w / 2,-h,0),
			Point3f(-w / 2,0,-z),Point3f(w / 2,0, -z),Point3f(w / 2,-h, -z),Point3f(-w / 2,-h, -z)
		};

		CVRProjector prj(mats.mView, mats.mProjection, viewSize);
		prj.project(displayCorners3D, displayCornersInHumanView); //从世界坐标系投影到人眼画面
	}

	//绘制显示器在人眼画面下的四边形区域
	{
		std::vector<std::vector<Point>> poly(1);
		poly[0] = cvtPoint(displayCornersInHumanView);
		poly[0].resize(4);
		cv::polylines(vdisp, poly, true, Scalar(0, 255, 255), 2, CV_AA);
	}

	std::vector<Point2f> displayCornersInDisplay;
	Mat rvec, tvec;
	{
		std::vector<Point2f>  displayCornersInHumanView4 = displayCornersInHumanView;
		displayCornersInHumanView4.resize(4);

		float w(viewSize.width), h(viewSize.height);
		std::vector<Point2f> realDisplayCorners = { Point2f(0,0),Point2f(w,0),Point2f(w,h),Point2f(0,h) };
		Matx33f H = cv::findHomography(displayCornersInHumanView4, realDisplayCorners, 0);
		
		cv::solvePnP(displayCorners3D, displayCornersInDisplay, K, noArray(), rvec, tvec);
	}

	mats.mView = cvrm::fromRT(rvec, tvec);
	mats.mProjection = cvrm::fromK(K, viewSize, 0.1, 100);
	rr = render.exec(mats, viewSize);

	{
		CVRProjector prj(mats.mView, mats.mProjection, viewSize);
		std::vector<Point2f>  projected;
		prj.project(displayCorners3D, projected); 
		
		printf("reprj error:");
		for (size_t i = 0; i < projected.size(); ++i)
		{
			auto dp = projected[i] - displayCornersInDisplay[i];
			float err = sqrt(dp.dot(dp));
			printf("%2.1f ", err);
		}
		printf("\n");
	}

	imshow("vdisp", vdisp);
	imshow("disp", rr.img);

	cv::waitKey();
}


CMD_BEG()
CMD0("test.show_model_in_display", test_show_model_in_display_1)
CMD_END()


_CMDI_END



