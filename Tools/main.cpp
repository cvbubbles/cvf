
#include"appstd.h"
#include"CVX/codec.h"

using ff::exec;


int main()
{
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
	//exec("test.render.measure_model");
	//exec("test.load_ply_pointcloud");
	//exec("test.show_model_in_display");
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

	//exec("test.netcall");
	//exec("tools.netcall.det2d");
	//exec("tools.netcall.det3d");
	//exec("tools.netcall.superglue");
	//exec("tools.netcall.raft");
	exec("tools.netcall.crestereo");
	
	//exec("test.homography");

	return 0;
}



