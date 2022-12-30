#pragma once

#include"opencv2/highgui.hpp"
#include"opencv2/core.hpp"
#include"opencv2/imgproc.hpp"
//#include"opencv2/xfeatures2d.hpp"
#include"opencv2/calib3d.hpp"
#include"opencv2/video.hpp"
#include"BFC/portable.h"
#include"BFC/bfstream.h"
#include"BFC/stdf.h"
#include"CVX/bfsio.h"
#include"CVX/core.h"
#include<iostream>
#include<time.h>
using namespace std;
using namespace cv;

#ifndef _BFCS_API
#define _BFCS_API
#endif
#include"BFC/commands.h"

#include"CVX/gui.h"

#ifndef CVRENDER_STATIC
#define CVRENDER_STATIC
#endif

#include"CVRender/cvrender.h"

#include"re3d/base/base.h"
using re3d::RigidPose;

#ifndef _STATIC_BEG
#define _STATIC_BEG namespace{
#define _STATIC_END }
#endif _STATIC_BEG

#ifndef _VX_BEG
#define _VX_BEG(vx) namespace vx{
#define _VX_END() }
#endif



