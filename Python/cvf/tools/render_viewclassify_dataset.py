
import cvf.cvrender as cvr

def gen_views(nViews, nViewSamples, marginRatio=0.3):
    assert(nViewSamples>=nViews)
    views=cvr.sampleSphere(nViews)
    samples=cvr.sampleSphere(nViewSamples)

    viewClusters=[]
    for i in range(0,len(views)):
        viewClusters.append([])

    for s in samples:
        mcos=-1
        mi=0
        for i,v in enumerate(views):
            sv_cos=s[0]*v[0]+s[1]*v[1]+s[2]*v[2]
            if sv_cos>mcos:
                mcos=sv_cos
                mi=i
        viewClusters[mi].append((s,mcos))
    #for v in viewClusters:
    for i,v in enumerate(viewClusters):
        v.sort(key=lambda x:x[1],reverse=True)
        viewClusters[i]=v[0:int(len(v)*(1-marginRatio))]

    return viewClusters

import math
import random
import numpy as np

def render_viewclassify_ds(modelFile, outDir, nViews, nImagesPerView):
    model=cvr.CVRModel(modelFile)
    modelCenter=np.array(model.getCenter())
    sizeBB=model.getSizeBB()
    maxBBSize=max(sizeBB)
    unitScale=2.0/maxBBSize
    eyeDist=4.0/unitScale
    fscale=1.5
    viewSize=[500,500]

    viewClusters=gen_views(nViews,3000)
    
    for ci,viewCluster in enumerate(viewClusters):

        for ii in range(0,nImagesPerView):
            viewDir=viewCluster[int(random.uniform(0,len(viewCluster)))][0]
            viewDir=np.array(viewDir)

            eyePos=modelCenter+viewDir*eyeDist
            mats=cvr.CVRMats()
            mats.mModel=cvr.lookat(eyePos[0],eyePos[1],eyePos[2],modelCenter[0],modelCenter[1],modelCenter[2],0,0,1)
            mats.mProjection=cvr.perspective(viewSize[1]*fscale, viewSize, max(1,eyeDist-maxBBSize), eyeDist+maxBBSize)

            angle=2*math.pi*ii/nImagesPerView
            mats.mView = cvr.rotate(angle, [0.0, 0.0, 1.0])

            render=cvr.CVRender(model)
            rr=render.exec(mats,viewSize)
            




def main():
    modelFile='/home/aa/data/3dmodels/3ds-model/plane2/plane2.ply'
    outDir='/home/aa/data/3dgen/viewclassify_01'
    render_viewclassify_ds(modelFile,outDir,10,100)


if __name__=='__main__':
    main()
