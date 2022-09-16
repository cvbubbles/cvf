
#include"_cvrender.h"

#include"cvrmodel.h"

#include "assimp/scene.h"
#include "assimp/cimport.h"
#include "assimp/postprocess.h"
#include"assimp/Exporter.hpp"

#include"BFC/err.h"
#include"BFC/stdf.h"
#include"BFC/portable.h"
#include"BFC/argv.h"
using namespace cv;

CVRRendable::~CVRRendable()
{}
void CVRRendable::setVisible(bool visible)
{
	_visible = visible;
}

static Matx44f cvtMatx(const aiMatrix4x4 &m)
{
	Matx44f dm;
	static_assert(sizeof(dm) == sizeof(m), "");
	memcpy(&dm, &m, sizeof(dm));
	return dm.t(); //transpose the result
}

class _SceneImpl
	:public CVRModel
{
public:
	static std::vector<MeshPtr> loadMeshes(aiScene *scene)
	{
		std::vector<MeshPtr>  meshes(scene->mNumMeshes);

		for (uint mi = 0; mi < scene->mNumMeshes; ++mi)
		{
			const aiMesh* mesh = scene->mMeshes[mi];

			MeshPtr  dmesh(new Mesh);

			std::unique_ptr<char[]> _vmask(new char[mesh->mNumVertices]);
			char *vmask = _vmask.get();
			memset(vmask, 0, mesh->mNumVertices);

			uint maxFaceIndices = 0;
			for (uint t = 0; t < mesh->mNumFaces; ++t)
			{
				const struct aiFace* face = &mesh->mFaces[t];
				if (face->mNumIndices > maxFaceIndices)
					maxFaceIndices = face->mNumIndices;
			}
			maxFaceIndices += 1;
			std::unique_ptr<int[]> _vmap(new int[maxFaceIndices]);
			int *vmap = _vmap.get();
			memset(vmap, 0xff, sizeof(int)*maxFaceIndices);

			int nFaceType = 0;

			for (uint t = 0; t < mesh->mNumFaces; ++t)
			{
				const struct aiFace* face = &mesh->mFaces[t];
				int faceType = vmap[face->mNumIndices];
				if (faceType == -1)
				{
					faceType = vmap[face->mNumIndices] = nFaceType++;
					dmesh->faces.push_back(Mesh::FaceType());
					dmesh->faces.back().numVertices = (int)face->mNumIndices;
				}
				auto &v = dmesh->faces[faceType].indices;
				for (uint i = 0; i < face->mNumIndices; i++)		// go through all vertices in face
				{
					v.push_back(face->mIndices[i]);
					vmask[face->mIndices[i]] = 1;
				}
			}
			dmesh->vertices.reserve(mesh->mNumVertices);
			for (uint i = 0; i < mesh->mNumVertices; ++i)
			{
				const aiVector3D &v = mesh->mVertices[i];
				dmesh->vertices.push_back(Point3f(v.x, v.y, v.z));
			}
			//if (mesh->mNormals)
			if(mesh->HasNormals())
			{
				dmesh->normals.reserve(mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
				{
					const aiVector3D &v = mesh->mNormals[i];
					dmesh->normals.push_back(Point3f(v.x, v.y, v.z));
				}
			}
			if (mesh->mNumVertices > 0)
			{
				dmesh->verticesMask.resize(mesh->mNumVertices);
				memcpy(&dmesh->verticesMask[0], vmask, mesh->mNumVertices);
			}

			if (mesh->HasTextureCoords(0))
			{
				//dmesh->textureCoords.resize(1);
				auto &v = dmesh->textureCoords;
				v.resize(mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
				{
					auto &c = mesh->mTextureCoords[0][i];
					v[i] = Point3f(c.x, c.y, c.z);
				}
			}

			if (mesh->mColors[0])
			{
				dmesh->colors.resize(mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
				{
					auto &c = mesh->mColors[0][i];
					dmesh->colors[i] = cv::Vec4f(c.r, c.g, c.b, c.a);
				}
			}

			dmesh->materialIndex = mesh->mMaterialIndex;

			meshes[mi] = dmesh;
		}
		return meshes;
	}
	static std::vector<MaterialPtr> loadMaterials(aiScene *scene, const std::string &modelDir)
	{
		std::vector<MaterialPtr>  materials(scene->mNumMaterials);

		for (uint mi = 0; mi < scene->mNumMaterials; ++mi)
		{
			MaterialPtr dmat(new Material);
			auto *mtl = scene->mMaterials[mi];

			int nTex;
			if ((nTex = mtl->GetTextureCount(aiTextureType_DIFFUSE)) > 0)
			{
				aiString path;	// filename
				for (int i = 0; i < nTex; ++i)
				{
					if (mtl->GetTexture(aiTextureType_DIFFUSE, i, &path) == AI_SUCCESS)
					{
						Texture tex;
						tex.name = std::string(path.data);
						tex.fullPath = ff::CatDirectory(modelDir, tex.name);
						dmat->textures.push_back(tex);
					}
				}
			}
			materials[mi] = dmat;
		}
		return materials;
	}
	static  NodePtr loadNodes(aiScene *scene, aiNode *node)
	{
		if (!node)
			return NodePtr(nullptr);

		NodePtr dnode(new Node);

		uint numChildren=node->mNumChildren;
		if (numChildren > 0)
		{
			dnode->children.resize(numChildren);
			for (uint i = 0; i < numChildren; ++i)
			{
				dnode->children[i] = loadNodes(scene, node->mChildren[i]);
			}
		}

		dnode->transformation = cvtMatx(node->mTransformation);

		dnode->meshes.reserve(node->mNumMeshes);
		for (uint i = 0; i < node->mNumMeshes; ++i)
			dnode->meshes.push_back(node->mMeshes[i]);

		dnode->name = std::string(node->mName.data);

		return dnode;
	}
};


void CVRModel::load(const std::string &file, int postProLevel, const std::string &options)
{
	uint postPro = postProLevel == 0 ? 0 :
		postProLevel == 1 ? aiProcessPreset_TargetRealtime_Fast :
		postProLevel == 2 ? aiProcessPreset_TargetRealtime_Quality :
		aiProcessPreset_TargetRealtime_MaxQuality;

	aiScene *scene = (aiScene*)aiImportFile(file.c_str(), postPro);
	if (!scene)
		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, file.c_str());
	else
		this->clear();

	std::string modelDir = ff::GetDirectory(ff::getFullPath(file));

	_this->meshes = _SceneImpl::loadMeshes(scene);
	_this->materials = _SceneImpl::loadMaterials(scene, modelDir);
	_this->root = _SceneImpl::loadNodes(scene, scene->mRootNode);

	_this->modelFile = ff::getFullPath(file);

	aiReleaseImport(scene);
	++_this->updateVersion;
}


const CVRModel::Infos& CVRModel::getInfos() const
{
	if (!_this->infosPtr||_this->infosUpdateVersion!=_this->updateVersion)
	{
		_this->infosPtr = InfosPtr(new Infos);

		size_t nVertices = 0;
		for (auto &m : _this->meshes)
			nVertices += m->vertices.size();
		
		auto &vertices = _this->infosPtr->vertices;
		vertices.reserve(nVertices);
		this->forAllVertices([&vertices](const Point3f &p) {
			vertices.push_back(p);
		});

		if (!vertices.empty())
		{
			Point3f  vmin, vmax;
			vmin = vmax = vertices.front();
			
			auto setmm = [](float &vmin, float &vmax, float x) {
				if (x < vmin)
					vmin = x;
				else if (x > vmax)
					vmax = x;
			};

			for (auto &p : vertices)
			{
				setmm(vmin.x, vmax.x, p.x);
				setmm(vmin.y, vmax.y, p.y);
				setmm(vmin.z, vmax.z, p.z);
			}
			_this->infosPtr->bboxMin = vmin;
			_this->infosPtr->bboxMax = vmax;
			_this->infosPtr->center = (vmax + vmin)*0.5f;
		}
		_this->infosUpdateVersion = _this->updateVersion;
	}
	return *_this->infosPtr;
}


std::string CVRModel::_newTextureName(const std::string &nameBase)
{
	std::vector<std::string>  matched;
	for (auto &m : _this->materials)
	{
		for (auto &t : m->textures)
		{
			if (t.name.size() >= nameBase.size() && strncmp(t.name.c_str(), nameBase.c_str(), nameBase.size()) == 0)
				matched.push_back(t.name);
		}
	}
	if (matched.empty())
		return nameBase + ".png";

	for (int i = 1; ; ++i)
	{
		std::string name = nameBase+ff::StrFormat("_%d.png", i);
		if (std::find(matched.begin(), matched.end(), name) == matched.end())
			return name;
	}
	return "";
}

void CVRModel::addNodes(NodePtr root, std::vector<MeshPtr>  &meshes, std::vector<MaterialPtr> &materials)
{
	if (!_this->root)
	{
		_this->root = root;
		_this->meshes = meshes;
		_this->materials = materials;
		return;
	}

	int offset = (int)this->_this->materials.size();
	_this->materials.insert(_this->materials.end(), materials.begin(), materials.end());
	for (auto &m : meshes)
		m->materialIndex += offset;

	offset = (int)this->_this->meshes.size();
	_this->meshes.insert(_this->meshes.end(), meshes.begin(), meshes.end());
	_forAllNodes(root.get(), [offset](Node *node, const Matx44f &) {
		for (auto &mi : node->meshes)
			mi += offset;
	});

	root->transformation = _this->root->transformation.inv()*root->transformation;
	_this->root->children.push_back(root);
	++_this->updateVersion;
}

void CVRModel::addNodes(const std::string &name, MeshPtr mesh, MaterialPtr material, const cv::Matx44f &mT)
{
	std::vector<MeshPtr> meshes;
	meshes.push_back(mesh);

	std::vector<MaterialPtr> materials;
	materials.push_back(material);

	NodePtr root(new Node);
	root->meshes.push_back(0);
	root->transformation = mT;
	root->name = name;

	this->addNodes(root, meshes, materials);
}

void CVRModel::addQuad(const cv::Point3f points[4], const cv::Mat &texImage, const std::string &name)
{
	Texture tex;
	tex.name = this->_newTextureName(name);
	tex.image = texImage;

	MaterialPtr mat(new Material);
	mat->textures.push_back(tex);

	MeshPtr mesh(new Mesh);
	mesh->vertices.insert(mesh->vertices.end(), points, points + 4);
	mesh->faces.resize(1);
	auto &ft = mesh->faces[0];
	ft.numVertices = 4;
	for (int i = 0; i < 4; ++i)
		ft.indices.push_back(i);

	const float tc[][2] = { { 0.f,0.f },{ 1.f,0.f },{ 1.f,1.f },{ 0.f,1.f } };
	mesh->textureCoords.resize(4);
	for (int i = 0; i < 4; ++i)
		mesh->textureCoords[i] = Point3f(tc[i][0], tc[i][1], 0.f);
	mesh->materialIndex = 0;

	//this->addNodes(mesh, mat, name);
	this->addNodes(name, mesh, mat);
}

void CVRModel::addCuboid(const std::string &name, const cv::Vec3f &size, const std::vector<cv::Mat> &texImages, const std::string &options, const cv::Matx44f &mT)
{
	ff::CommandArgSet args(options);

	float dx = size[0] * 0.5f, dy = size[1] * 0.5f, dz = size[2] * 0.5f;
	Point3f corners[] = {
		{dx,dy,dz},{-dx,dy,dz},{-dx,-dy,dz},{dx,-dy,dz},
		{ dx,dy,-dz },{ -dx,dy,-dz },{ -dx,-dy,-dz },{ dx,-dy,-dz },
	};

	int vv[][8] = {
		{3,7,4,0, 2,6,5,1},{ 0,3,7,4, 5,6,2,1 },{3,7,4,0, 6,2,1,5}, //xyz, xzy
		{0,4,5,1, 3,7,6,2},{ 1,0,4,5, 6,7,3,2 },{0,4,5,1, 7,3,2,6}, //yxz, yzx
		{1,2,3,0, 5,6,7,4},{ 0,1,2,3, 7,6,5,4 }, {1,2,3,0, 6,5,4,7},  //zxy, zyx
	};

	struct DSize
	{
		float sz;
		char  axis;
	}
	vsize[3] = { {size[0],'x'},{size[1],'y'},{size[2],'z'} };

	std::string order = args.getd<std::string>("order", "auto");

	if (order == "auto")
	{
		std::sort(vsize, vsize + 3, [](const DSize &a, const DSize &b) {
			return a.sz > b.sz;
		});
	}
	else
	{
		DSize t[3];
		int n = 0;
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				if (vsize[j].axis == order[i])
				{
					t[n++] = vsize[j]; vsize[j].axis = -1;
					break;
				}
			}
		}
		CV_Assert(n == 3);//invalid order
		memcpy(vsize, t, sizeof(t));
	}
	//std::swap(vsize[0], vsize[1]);


	char tag[4] = { vsize[0].axis,vsize[1].axis,vsize[2].axis, '\0' };

	struct Config
	{
		const char *tag;
		int  imain, isub;
	}
	vconfig[6] = {
		{"xyz",0,1},{"xzy",0,2},{"yxz",3,4},{"yzx",3,5},{"zxy",6,7},{"zyx",6,8}
	};

	Config cfg;
	for(auto &c : vconfig)
		if (strncmp(c.tag, tag, 3) == 0)
		{
			cfg = c; break;
		}

	auto getEdgeLen = [&corners](int i, int j) {
		Point3f dv = corners[i] - corners[j];
		return sqrt(dv.dot(dv));
	};

	const int *v = vv[cfg.imain];
	float a = getEdgeLen(v[0], v[1]), b = getEdgeLen(v[1], v[2]), c = vsize[1].sz / vsize[0].sz*vsize[2].sz;
	CV_Assert(__min(a, b) == vsize[2].sz && __max(a, b) == vsize[1].sz);

	bool lookInside = args.getd<bool>("lookInside",false);
	auto getImage = [lookInside, &texImages](int i) {
		Mat img = texImages[i];
		if (lookInside)
			flip(img, img, 1);
		return img;
	};

	float texHeight = vsize[0].sz, texWidth = (a + b + c) * 2;
	Mat img = getImage(0);
	if (img.rows > img.cols)
		cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);

	a /= texWidth; b /= texWidth; c /= texWidth;

	int dwidth = int(img.rows*texWidth / texHeight + 0.5);
	if (texImages.size() > 1)
	{
		float vw[] = { a,b,a,b,c,c };
		int j = 5;
		Mat dimg(img.rows, dwidth, CV_8UC3);
		int x = dwidth;
		for (int i = (int)texImages.size() - 1; i >= 1&&j>=0; --i,--j)
		{
			int roiWidth = int(vw[j] * dwidth+0.5);
			Rect roi(x - roiWidth, 0, roiWidth, img.rows);
			Mat curImg = getImage(i);
			CV_Assert(curImg.type() == CV_8UC3);
			if (curImg.rows > curImg.cols && roi.height < roi.width)
				cv::rotate(curImg, curImg, cv::ROTATE_90_COUNTERCLOCKWISE);
			cv::resize(curImg, dimg(roi), roi.size());
			x = roi.x;
		}
		if (x > 0)
		{
			if (img.cols < x)
				cv::resize(img, img, Size(x, img.rows));
			cv::copyMem(img(Rect(0, 0, x, img.rows)), dimg(Rect(0, 0, x, img.rows)));
		}
		img = dimg;
	}
	else
	{
		if (img.cols > dwidth)
			img = img(Rect(0, 0, dwidth, img.rows)).clone();
	}

	int maxTexSize = args.getd<int>("maxTexSize", 2048 * 10);
	if (__max(img.cols, img.rows) > maxTexSize)
	{
		double scale = double(maxTexSize) / __max(img.cols, img.rows);
		img = imscale(img, scale, cv::INTER_LINEAR);
	}

	MeshPtr mesh(new Mesh);
	mesh->materialIndex = 0;

	const int *vx = vv[cfg.imain];
	for (int i = 0; i < 8; ++i)
		mesh->vertices.push_back(corners[vx[i]]);
	
	vx = vv[cfg.isub];
	for (int i = 0; i < 8; ++i)
		mesh->vertices.push_back(corners[vx[i]]);

	mesh->vertices.push_back(mesh->vertices[4]);
	mesh->vertices.push_back(mesh->vertices[0]);
	
	float tc[][2] = {
		{ 0,0 },{ a,0 },{ a + b,0 },{ a + b + a,0 },
		{ 0,1 },{ a,1 },{ a + b, 1 },{ a + b + a, 1 },
		{ (a + b) * 2,0 },{ (a + b) * 2,1.0f },{ (a + b) * 2 + c,1.0f },{ (a + b) * 2 + c,0.f },
		{ (a + b) * 2+c,0.f },{ (a + b) * 2+c,1.f },{ (a + b+c) * 2, 1.f },{ (a + b +c) * 2,0.f },
		{ (a + b) * 2,1.f },{ (a + b) * 2,0 }
	};
	for (int i = 0; i < 18; ++i)
		mesh->textureCoords.push_back(Point3f(tc[i][0], 1.f - tc[i][1], 0.f));

	mesh->faces.resize(1);
	auto &ft = mesh->faces[0];
	ft.numVertices = 4;
	ft.indices = { 0,4,5,1, 1,5,6,2, 2,6,7,3, 3,7,16,17,  8,9,10,11, 12,13,14,15 };

	//std::reverse(ft.indices.begin(), ft.indices.end());

	if (args.getd<bool>("notop", false))
	{
		size_t i = 0;
		auto isTop = [&mesh](size_t i) {
			return mesh->vertices[mesh->faces[0].indices[i]].y > 0.f;
		};

		for (; i < ft.indices.size(); i += 4)
		{
			if (isTop(i) && isTop(i + 1) && isTop(i + 2) && isTop(i + 3))
				break;
		}
		CV_Assert(i < ft.indices.size());
		ft.indices.erase(ft.indices.begin() + i, ft.indices.begin() + i + 4);
	}

	Texture tex;
	tex.name = this->_newTextureName(name);
	tex.image = img;

	MaterialPtr mat(new Material);
	mat->textures.push_back(tex);

	this->addNodes(name, mesh, mat, mT);
}

cv::Matx44f CVRModel::calcStdPose()
{
	const std::vector<Point3f>  &vtx = this->getInfos().vertices;

	Mat mvtx(vtx.size(), 3, CV_32FC1, (void*)&vtx[0]);
	cv::PCA pca(mvtx, noArray(), PCA::DATA_AS_ROW);

	Vec3f mean = pca.mean;
	Matx33f ev = pca.eigenvectors;

	//swap x-y so that the vertical directional is the longest dimension
	Vec3f vx(-ev(1, 0), -ev(1, 1), -ev(1, 2));
	Matx44f R = cvrm::rotate(&vx[0], &ev(0, 0), &ev(2, 0));
	if (determinant(R) < 0)
	{//reverse the direction of Z axis
		for (int i = 0; i < 4; ++i)
			R(i, 2) *= -1;
	}
	//std::cout << "det=" << determinant(R) << std::endl;

	return cvrm::translate(-mean[0], -mean[1], -mean[2])*R;
}

void CVRModel::setSceneTransformation(const cv::Matx44f &mT, bool multiplyCurrent)
{
	if (_this->root)
	{
		if (multiplyCurrent)
			_this->root->transformation = _this->root->transformation*mT;
		else
			_this->root->transformation = mT;
		
		++_this->updateVersion;
	}
}


Matx44f CVRModel::getUnitize(const cv::Vec3f &center, const cv::Vec3f &bbMin, const cv::Vec3f &bbMax)
{
	float tmp = 0;
	for (int i = 0; i < 3; ++i)
		tmp = __max(bbMax[i] - bbMin[i], tmp);
	tmp = 2.f / tmp;

	return cvrm::translate(-center[0], -center[1], -center[2]) * cvrm::scale(tmp, tmp, tmp);
}

Matx44f CVRModel::getUnitize() const
{
	if (!*this)
		return cvrm::I();

	Vec3f bbMin, bbMax;
	this->getBoundingBox(bbMin, bbMax);
	return getUnitize(this->getCenter(), bbMin, bbMax);
}


//====================================================================


struct _Texture
{
	GLuint texID;
};

typedef std::map<std::string, _Texture> TexMap;


class CVRModel::_Render
{
public:
	//CVRModel     *_site=nullptr;
	TexMap       _texMap;
	GLuint		_sceneList = 0;
	int			_sceneListRenderFlags = -1;
	uint        _sceneListVersion = 0;

public:
	~_Render()
	{
		this->clear();
	}
	void clear()
	{
		if (!_texMap.empty() || _sceneList != 0)
		{
			cvrCall([this](int) {
				for (auto &t : _texMap)
				{
					if (t.second.texID > 0)
						glDeleteTextures(1, &t.second.texID);
				}
				_texMap.clear();


				if (_sceneList != 0)
				{
					glDeleteLists(_sceneList, 1);
					_sceneList = 0;
					_sceneListRenderFlags = -1;
				}
			});
			cvrWaitFinish();
		}
	}

	_Texture& _getTexture(CVRModel  *_site, const CVRModel::Texture &tex)
	{
		auto itr = _texMap.find(tex.name);
		if (itr == _texMap.end())
		{
			cv::Mat img = tex.image;
			if (img.empty())
			{
				std::string imgFile = tex.fullPath;
				if (imgFile.empty())
					imgFile = ff::GetDirectory(_site->getFile()) + tex.name;

				img = cv::imread(imgFile, cv::IMREAD_ANYCOLOR);
				CV_Assert(img.type() == CV_8UC3);
				if (img.empty())
					printf("error: failed to load %s\n", imgFile.c_str());
			}
			CV_Assert(img.type() == CV_8UC3);

			makeSizePower2(img);


			GLuint texID = loadGLTexture(img);
			_texMap[tex.name].texID = texID;
			itr = _texMap.find(tex.name);
		}
		return itr->second;
	}

	void render_node(CVRModel  *_site, CVRModel::Node *node, const Matx44f &mT, int flags)
	{
	//	flags = CVRM_ENABLE_LIGHTING;

		auto *scene = _site->_this.get();

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glMultMatrixf(mT.val);

		for (auto mi : node->meshes)
		{
			auto meshPtr = scene->meshes[mi];

			glBindTexture(GL_TEXTURE_2D, 0);

			auto &mat = scene->materials[meshPtr->materialIndex];
			bool hasTexture = false;
			if ((flags&CVRM_ENABLE_TEXTURE) && !mat->textures.empty())
			{
				//auto &t = mTex[mat->textures.front().path];
				auto &t = this->_getTexture(_site, mat->textures.front());
				glBindTexture(GL_TEXTURE_2D, t.texID);
				hasTexture = true;
			}

			bool onlyTexture = hasTexture && (flags&CVRM_TEXTURE_NOLIGHTING);

			if (!meshPtr->normals.empty() && (flags&CVRM_ENABLE_LIGHTING) && !onlyTexture)
				glEnable(GL_LIGHTING);
			else
				glDisable(GL_LIGHTING);

			const Point3f *normals = meshPtr->normals.empty() ? nullptr : &meshPtr->normals[0];
			const Point3f *texCoords = (flags&CVRM_ENABLE_TEXTURE) && !meshPtr->textureCoords.empty() ? &meshPtr->textureCoords[0] : nullptr;
			const Vec4f   *colors = meshPtr->colors.empty() ? nullptr : &meshPtr->colors[0];

			if (colors)
				glEnable(GL_COLOR_MATERIAL);
			else
				glDisable(GL_COLOR_MATERIAL);

			for (auto &faceType : meshPtr->faces)
			{
				GLenum face_mode;

				switch (faceType.numVertices)
				{
				case 1: face_mode = GL_POINTS; break;
				case 2: face_mode = GL_LINES; break;
				case 3: face_mode = GL_TRIANGLES; break;
				case 4: face_mode = GL_QUADS; break;
				default: face_mode = GL_POLYGON; break;
				}

				const int *v = &faceType.indices[0];
				int nFaces = (int)faceType.indices.size() / faceType.numVertices;

				for (int f = 0; f < nFaces; ++f, v += faceType.numVertices)
				{
					glBegin(face_mode);

					for (int i = 0; i < faceType.numVertices; i++)		// go through all vertices in face
					{
						int vertexIndex = v[i];	// get group index for current index

						if (onlyTexture)
							glColor4f(1, 1, 1, 1);
						else
						{
							if (colors)
								glColor4fv(&colors[vertexIndex][0]);

							if (normals)
								glNormal3fv(&normals[vertexIndex].x);
						}

						if ((flags&CVRM_ENABLE_TEXTURE) && !meshPtr->textureCoords.empty())		//HasTextureCoords(texture_coordinates_set)
						{
							glTexCoord2f(texCoords[vertexIndex].x, /*1.0 -*/ texCoords[vertexIndex].y); //mTextureCoords[channel][vertex]
						}

						glVertex3fv(&meshPtr->vertices[vertexIndex].x);
					}
					glEnd();
				}
			}
		}

		glPopMatrix();
	}


	void render(CVRModel *site, int flags)
	{
#if 1
		if (_sceneList == 0)
		{
			_sceneList = glGenLists(1);
			_sceneListRenderFlags = flags - 1;//make them unequal to trigger re-compile
		}

		if (_sceneListRenderFlags != flags || _sceneListVersion != site->getUpdateVersion())
		{
			glNewList(_sceneList, GL_COMPILE);
			/* now begin at the root node of the imported data and traverse
			the scenegraph by multiplying subsequent local transforms
			together on GL's matrix stack. */
			site->forAllNodes([this, site, flags](CVRModel::Node *node, Matx44f &mT) {
				render_node(site, node, mT, flags);
			});
			glEndList();

			_sceneListRenderFlags = flags;
			_sceneListVersion = site->getUpdateVersion();
		}

		glCallList(_sceneList);

		checkGLError();
#else
		site->forAllNodes([this, site, flags](CVRModel::Node *node, Matx44f &mT) {
			render_node(_site, node, mT, flags);
#endif
	}
};


CVRModel::CVRModel()
	:_this(new This), _render(new _Render)
{}
void CVRModel::clear()
{
	*this = CVRModel();
}
void CVRModel::render(const Matx44f &sceneModelView, int flags)
{
	if (*this)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(sceneModelView.val);

		if (this->isVisible())
		{
			_render->render(this, flags);
		}
	}
}


//=================================================================================


CVRModelEx::CVRModelEx(const CVRModel &_model, const Matx44f &_mModeli, const Matx44f &_mModel)
	:model(_model), mModeli(_mModeli), mModel(_mModel)
{}

void CVRModelEx::render(const Matx44f &sceneModelView, int flags)
{
	Matx44f mx = mModeli*mModel*sceneModelView;
	model.render(mx, flags);
}

void CVRModelArray::render(const Matx44f &sceneModelView, int flags)
{
	for (size_t i = 0; i < _v.size(); ++i)
		_v[i].render(sceneModelView, flags);
}

void CVRRendableArray::render(const Matx44f &sceneModelView, int flags)
{
	for (size_t i = 0; i < _v.size(); ++i)
		_v[i]->render(sceneModelView, flags);
}


