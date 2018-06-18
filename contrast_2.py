import cv2 as cv

for i in range(6407):
    num = str(i)
    img = cv.imread('C://Users/gnehc/Desktop/cv/img_' + num + '.png')
    # cv.imshow('im1', img)
    rows, cols, channels = img.shape
    dst = img.copy()

    a = 90
    b = 90
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color1 = img[i, j][c] + a
                color2 = img[i, j][c] - b
                if color1 > 255:
                    dst[i, j][c] = 255
                elif color2 < 0:
                    dst[i, j][c] = 0
    # cv.imshow('dst', dst)
    cv.imwrite('C://Users/gnehc/Desktop/cv/result/test_' + num + '.png', dst)

# cv.waitKey(0)
# cv.destroyAllWindows()
