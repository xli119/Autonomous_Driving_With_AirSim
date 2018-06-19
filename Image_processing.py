# -*- coding: UTF-8 -*-

from PIL import Image
from PIL import ImageEnhance
import glob, os

# 原始图像
'''
image = Image.open("C:/Users/Roy/Desktop/img_1636.png")
print(image.format)
image.show()


# 亮度增强
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)
image_brightened.show()

# 色度增强
enh_col = ImageEnhance.Color(image)
color = 1.5
image_colored = enh_col.enhance(color)
image_colored.show()


# 锐度增强
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 3.0
image_sharped = enh_sha.enhance(sharpness)
image_sharped.show()

# 对比度增强
enh_con = ImageEnhance.Contrast(image)
contrast = 2.0
image_contrasted = enh_con.enhance(contrast)
image_contrasted.save("C:/Users/Roy/Desktop/pic/normal1/img_1636_copy.png")
print(image_contrasted.format)
image_contrasted.show()
'''
def timage():
    for files in glob.glob('C:/Users/Roy/Desktop/Summer_project/AutonomousDrivingCookbook/EndToEndLearningRawData/data_raw/swerve_3/images/*.png'):
        filepath,filename = os.path.split(files)
        filterame,exts = os.path.splitext(filename)
        #输出路径
        opfile = 'C:/Users/Roy/Desktop/pic/swerve3/'
        #判断opfile是否存在，不存在则创建
        if (os.path.isdir(opfile)==False):
            os.mkdir(opfile)
        im = Image.open(files)
        enh_con = ImageEnhance.Contrast(im)
        contrast = 2.0
        image_contrasted = enh_con.enhance(contrast)
        image_contrasted.save(opfile + filterame + '.png')


if __name__ == '__main__':
    timage()
'''
## 写给老梁：
加入testmodel
NO.5 cell:
def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    contrast = 2.0
    image_rgba = ImageEnhance.Contrast(image_rgba).enhance(contrast)
    
    return image_rgba[76:135,0:255,0:3].astype(float)
'''
