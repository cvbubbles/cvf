
from numpy.core.fromnumeric import shape
import cvf.bfc as bfc
import cvf.cvrender as cvr
import cv2

'''composite a set of image objects with a background image
harmonizeF : transform fg color with respect to bg
degradeF : add random smooth and noise to fg, specified with maxSmoothSigma and maxNoiseStd
'''
def composite_images(bg, fgList, harmonizeF=True, degradeF=True, maxSmoothSigma=1.0, maxNoiseStd=5.0):
    compositer=cvr.Compositer()
    compositer.init(bg)
    for fg in fgList:
        compositer.addLayer(fg['img'],  # fg image
                            fg['mask'], # mask of fg in [0,255]
                            fg['roi'],  # ROI of fg in the background image, the fg image will be resized as necessary
                            -1,harmonizeF,degradeF,maxSmoothSigma,maxNoiseStd)
    return compositer.getComposite()


#load an image object and composite with a new background
def load_with_new_bg(fgFile, maskFile, bgFile, maxBgBorderWidth=20):
    fg=cv2.imread(fgFile,cv2.IMREAD_COLOR)
    mask=cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)

    #crop redudant background region in fg
    fg,mask=cvr.cropImageRegion(fg,mask,maxBgBorderWidth)

    bg=cv2.imread(bgFile,cv2.IMREAD_COLOR)
    dsize=(fg.shape[1],fg.shape[0])
    bg=cv2.resize(bg,dsize)

    fgList=[{'img':fg, 'mask':mask, 'roi':[0,0,dsize[0],dsize[1]]   
    }]
    return composite_images(bg,fgList)

if __name__=='__main__':
    imdir= r'f:/home/aa/data/3dgen/viewclassify_01/0001/'
    bgImgFile=r'f:/home/aa/data/plane.png'
    dimg=load_with_new_bg(imdir+'img/0001.png',imdir+'mask/0001.png',bgImgFile)
    cv2.imshow('dimg',dimg)
    cv2.waitKey()
    #cv2.imwrite('/home/aa/data/out.jpg',dimg)


