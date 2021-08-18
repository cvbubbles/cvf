
#include"appstd.h"
#include"BFC/netcall.h"


_CMDI_BEG

void on_det2d()
{
	

	ff::NetcallServer serv("10.102.32.173", 8011);

#if 0
	cv::VideoCapture vid;
	vid.open("../data/det2d-5.avi");
	Mat img;
	while (vid.read(img))
	{
#else
	//std::string dir = R"(F:\store\datasets\BOP\ycbv_test_bop19\test\000048\rgb\)";
	std::string dir = "f:/rgb/";

	std::vector<string> files;
	ff::listFiles(dir, files);
	for(auto f : files)
	{
		Mat img = cv::imread(dir+f);
#endif
		//resize(img, img, img.size() / 2);
		Mat rgb;
		cvtColor(img, rgb, CV_BGR2RGB);

		ff::NetObjs objs = {
			{ "cmd","run" },
			{ "img",ff::ObjStream::fromImage(rgb,".png") }
		};

		ff::NetObjs dobjs = serv.call(objs);

		auto labels = dobjs["labels"].get<std::vector<std::string>>();
		auto bboxes = dobjs["bboxes"].get<Mat1f>();
		auto scores = dobjs["scores"].get<std::vector<double>>();

		Mat dimg = img.clone();
		for (int i = 0; i < (int)labels.size(); ++i)
		{
			if (scores[i] > 0.1f)
			{
				const float *bb = bboxes.ptr<float>(i);
				//int x = int(bb[0]);
				//cv::Rect bbox(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]));
				cv::Rect bbox(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]);
				cv::rectangle(dimg, bbox, Scalar(0, 255, 255), 2);
			}
			//break;
		}
		imshow("dimg", dimg);
		if (cv::waitKey(0) == 'q')
			break;
	}
	serv.sendExit();
}

CMD_BEG()
CMD0("tools.netcall.det2d", on_det2d)
CMD_END()


_CMDI_END

