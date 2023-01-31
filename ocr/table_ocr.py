import base64
import json
import cv2
import cv2 as cv
import numpy as np
import os
# from skimage import io,color,exposure,morphology,transform,img_as_float
# import matplotlib.pyplot as plt
# import skimage.transform as st
# import os
# from PIL import Image
# import imagehash
# import shutil
# import time
# import copy
# import skimage
# from imagesplit import splitImg
# from paddleocr import PaddleOCR
# import pytesseract
from cell_ocr import cell_ocr

DEBUG_MODE = True
RESIZE_MODE = True


def showAndSave(img, showName, savePath):
    cv2.imshow(showName, img)
    cv2.waitKey(0)
    cv2.imwrite(savePath, img)  # 将二值像素点生成图片保存


def cv_show(name, img):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 240, 120)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 1280
    height_new = 720
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new



def table_ocr(imgFilePath, maxCellNum=10, DEBUG_MODE=False):      # maxCellNum 一个表格最多能识别的单元格最大数量

    # 读取图片，并转灰度图
    image = cv2.imread(imgFilePath, 1)
    if DEBUG_MODE:
        print(image.shape)

    if RESIZE_MODE:
        image = img_resize(image)
        print(image.shape)

    if DEBUG_MODE:
        showAndSave(image, 'Original table', 'img/debug/OriginalTable.png')

    # 灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    # ret,binary = cv2.threshold(~gray, 127, 255, cv2.THRESH_BINARY)
    if DEBUG_MODE:
        showAndSave(binary, 'binary', 'img/debug/binary.png')

    rows, cols = binary.shape
    scale = 40

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedCol = cv2.dilate(eroded, kernel, iterations=1)
    if DEBUG_MODE:
        showAndSave(dilatedCol, 'dilatedCol', 'img/debug/dilatedCol.png')

    # 识别竖线
    scale = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedRow = cv2.dilate(eroded, kernel, iterations=1)
    if DEBUG_MODE:
        showAndSave(dilatedRow, 'dilatedRow', 'img/debug/dilatedRow.png')

    # 标识表格
    merge = cv2.add(dilatedCol, dilatedRow)
    if DEBUG_MODE:
        showAndSave(merge, 'merge table', 'img/debug/merge.png')

    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedCol, dilatedRow)
    if DEBUG_MODE:
        showAndSave(bitwiseAnd, 'bitwiseAnd', 'img/debug/bitwiseAnd.png')

    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    if DEBUG_MODE:
        showAndSave(merge2, "remove table lines", 'img/debug/removetablelines.png')

    # 画出表格线
    imageCopy = image.copy()
    contours, hierarchy = cv2.findContours(merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if DEBUG_MODE:
        cv2.drawContours(imageCopy, contours, -1, (0, 0, 255), 3)
        showAndSave(imageCopy, "findContours", 'img/debug/findContours.png')

    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwiseAnd > 0)

    mylisty = []  # 纵坐标
    mylistx = []  # 横坐标

    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
    # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    myxs = np.sort(xs)
    for i in range(len(myxs) - 1):
        if (myxs[i + 1] - myxs[i] > 20):
            mylistx.append(myxs[i])
        i = i + 1
    mylistx.append(myxs[i])  # 要将最后一个点加入

    i = 0
    myys = np.sort(ys)
    for i in range(len(myys) - 1):
        if (myys[i + 1] - myys[i] > 10):
            mylisty.append(myys[i])
        i = i + 1
    mylisty.append(myys[i])  # 要将最后一个点加入

    if DEBUG_MODE:
        print('mylisty=', mylisty, 'length=', len(mylisty))
        print('mylistx=', mylistx, 'length=', len(mylistx))

    # 分割表格
    leny = len(mylisty)
    lenx = len(mylistx)
    result = {
        "state": "识别失败",
        "result": None
              }
    if leny != 23 or lenx != 15:
        return result

    cellNum = 0  # 已经识别的单元格数量
    resList = []
    for i in range(leny - 1):
        if i + 1 == 6 or i + 1 == 17:  # 牙齿标号行不识别
            continue
        tempRowList = []
        for j in range(lenx - 1):
            cellNum += 1

            # 在分割时，第一个参数为y坐标，第二个参数为x坐标
            padding = 20      # 减小ROI识别框范围
            ROI = binary[mylisty[i]+padding//2:mylisty[i+1]-padding//2, mylistx[j]+padding:mylistx[j+1]-padding]

            # 如果单元格全为0，不使用ocr程序，直接将返回空字符串
            if (ROI == 0).all():
                ROI_res = ""
                tempRowList.append(ROI_res)
                continue

            # 使用ocr程序识别，并将结果返回
            padding = -3  # 增大ROI识别框范围
            ROI = image[mylisty[i]+padding:mylisty[i+1]-padding, mylistx[j]+padding:mylistx[j+1]-padding]
            ROI_str = cv2.imencode('.jpg', ROI)[1].tobytes()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
            ROI_b64 = base64.b64encode(ROI_str)  # 编码成base64
            #ROI_res = "1"
            ROI_res = cell_ocr(ROI_b64)  # ocr识别
            tempRowList.append(ROI_res)

            if DEBUG_MODE:
                ROI_name = str(cellNum)+"--"+str(ROI_res)+".png"
                print(ROI_name)
                cv.imwrite("img/debug/"+ROI_name, ROI)
                # cv_show(ROI_name, ROI)

            if cellNum > maxCellNum:
                break

        resList.append(tempRowList)


    result = {
        "state": "识别成功",
        "result": resList
    }
    return result


def itemFilter(res: dict):  # 基于规则替换的过滤器
    if res["state"] == "识别成功":
        indexList = ["FI", "BOP", "M", "CAL", "PD"]
        indexList = indexList + list(reversed(indexList)) + indexList + list(reversed(indexList))
        for i in range(len(res["result"])):
            itemName = indexList[i]
            tmpList = []
            for s in res["result"][i]:
                s = str(s).strip()
                if itemName == "FI" or itemName == "BOP":
                    s = s.replace('一', '-')
                    s = s.replace('十', '+')
                if itemName == "M":
                    s = s.replace('工', 'I')
                    s = s.replace('T', 'I')
                tmpList.append(s)
            res["result"][i] = tmpList
    return res


def printTableResult(res: dict):
    if res["state"] == "识别成功":
        indexList = ["FI ", "BOP", "M  ", "CAL", "PD "]
        indexList = indexList + list(reversed(indexList)) + indexList + list(reversed(indexList))
        print("表格识别结果:")
        for i in range(len(res["result"])):
            print(indexList[i], res["result"][i])

if __name__ == '__main__':
    imgFilePath = "img/table9.jpg"
    outputPath1 = "img/debug/table9.json"
    outputPath2 = "img/debug/table9_replace.json"
    res = table_ocr(imgFilePath, maxCellNum=500, DEBUG_MODE=True)

    with open(outputPath1, "w") as f:
        json.dump(res, f)
    printTableResult(res)

    res = itemFilter(res)
    with open(outputPath2, "w") as f:
        json.dump(res, f)
    printTableResult(res)

    # print(json.dumps(res))

