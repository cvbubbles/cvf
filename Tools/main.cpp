
#include"appstd.h"
#include"CVX/codec.h"

using ff::exec;

int main()
{
	/*ff::setCurrentDirectory(R"(F:\data\3D装配数据\ruisong-data\车架配准数据\data\CJE-030\)");
	Mat img=cv::readBMP("DepthImage.bmp");
	CV_Assert(img.channels() == 4);*/

	cvrInit();

	//exec("tools.calib_camera");
	//exec("tools.os.list_3dmodels");

	//exec("examples.render.load_and_show_model");
	//exec("examples.render.render_to_image");
	//exec("examples.render.set_GL_matrix");
	//exec("examples.render.sample_sphere_views");
	//exec("examples.render.ortho_projection");
	//exec("test.render.set_rigid_mats");
	//exec("test.render.depth_precision");
	//
	//exec("tools.render.show_model_file");
	//exec("tools.render.show_models_drag_drop");
	//exec("tools.render.render_model_as_video");
	//exec("tools.render.set_model_pose");

	//exec("tools.render.render_6dpose_results");

	//exec("tools.render.build_scene");
	//exec("tools.render.verify_colmap_reconstruction");
	 
	//exec("tools.voc.render_3d_models_as_detection_dataset");

	//exec("tools.bop.gen_views_dataset");
	//exec("tools.bop.show_bop_gt");
	//exec("tools.bop.show_bop_scene");
	//exec("tools.rbot.show_rbot_gt");

	//exec("test.net_call");
	//exec("tools.netcall.det2d");
	//exec("tools.netcall.det3d");
	exec("tools.netcall.superglue");
	
	//exec("test.homography");

	return 0;
}

#include"BFC/log.h"

int main2()
{
	printf("#001\n");
	cvrInit("-display :0.0");
	//cvrInit();

	printf("#002\n");
	// return 0;
	std::string modelFile = "./bottle2.3ds";
	//std::string modelFile="/fan/dev/prj-c1/1100-Re3DX/TestRe3DX/3ds/bottle2.3ds";

	CVRModel model(modelFile);
	CVRender render(model);
	const int W = 500;
	CVRMats mats(model, Size(W, W));

	CVRResult r = render.exec(mats, Size(500, 500));

	double etbeg = ff::loget();
	for(int i=0; i<10; ++i)
		r = render.exec(mats, Size(W, W));
	printf("time=%.4f\n", (ff::loget() - etbeg) / 10);

	imwrite("./dimg.jpg", r.img);

	return 0;
}



