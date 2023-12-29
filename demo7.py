import cv2
import numpy as np
from skimage import data

def fast_glcm(img):
    vmin = 0
    vmax = 255
    nbit = 8
    kernel_size = 5
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img):
    '''
    calc glcm mean
    '''
    nbit = 8
    h,w = img.shape
    glcm = fast_glcm(img)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    return mean


def fast_glcm_std(img):
    '''
    calc glcm std
    '''
    nbit = 8
    h,w = img.shape
    glcm = fast_glcm(img)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img):
    '''
    calc glcm contrast
    '''
    nbit = 8
    h,w = img.shape
    glcm = fast_glcm(img)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img):
    '''
    calc glcm dissimilarity
    '''
    nbit = 8
    h,w = img.shape
    glcm = fast_glcm(img)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img):
    '''
    calc glcm homogeneity
    '''
    nbit = 8
    h,w = img.shape
    glcm = fast_glcm(img )
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img):
    '''
    calc glcm asm, energy
    '''

    nbit = 8

    h,w = img.shape
    glcm = fast_glcm(img )
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img )
    max_  = np.max(glcm, axis=(0,1))
    return max_


def fast_glcm_entropy(img):
    '''
    calc glcm entropy
    '''
    ks = 5
    glcm = fast_glcm(img )
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent


if __name__ == '__main__':
    img = data.camera() #
    h,w = img.shape

    glcm_mean = fast_glcm_mean(img)
    print(glcm_mean)

    glcm_std = fast_glcm_std(img)
    print(glcm_std)

    glcm_contrast = fast_glcm_contrast(img)
    print(glcm_contrast)

    glcm_dissimilarity = fast_glcm_dissimilarity(img)
    print(glcm_dissimilarity)

    glcm_homogeneity = fast_glcm_homogeneity(img)
    print(glcm_homogeneity)

    glcm_ASM = fast_glcm_ASM(img)
    print(glcm_ASM)

    glcm_max = fast_glcm_max(img)
    print(glcm_max)

    glcm_entropy = fast_glcm_entropy(img)
    print(glcm_entropy)


img = cv2.imread("1.webp")#读入一张图像
img = cv2.resize(img,(1080,720))#图像缩放
cv2.imshow("original",img)#显示图像

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)#灰度化：彩色图像转为灰度图像

#使用SIFT
sift = cv2.xfeatures2d.SIFT_create()#SIFT对象创建                    
keypoints, descriptor = sift.detectAndCompute(gray,None)#对整张图片进行检测与匹配

cv2.drawKeypoints(image = img,
                  outImage = img,
                  keypoints = keypoints,
                  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color = (51,163,236))
cv2.imshow("SIFT",img)#显示图像

#使用SURF
img = cv2.imread("1.webp")#读入一张图像
img = cv2.resize(img,(1080,720))#图像缩放

surf = cv2.xfeatures2d.SURF_create()#SURF对象创建
keypoints, descriptor = surf.detectAndCompute(gray,None)#对整张图片进行检测与匹配

cv2.drawKeypoints(image = img,
                  outImage = img,
                  keypoints = keypoints,
                  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  color = (51,163,236))
cv2.imshow("SURF",img)#显示图像

cv2.waitKey(0)
cv2.destroyAllWindows()