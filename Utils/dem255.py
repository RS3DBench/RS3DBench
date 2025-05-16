import os
import time
from scipy.ndimage import binary_erosion, binary_dilation
from matplotlib import pyplot as plt
from osgeo import gdal
import numpy as np
import tifffile
from PIL import Image
import rasterio
from tqdm import tqdm
from scipy.ndimage import sobel
import sys

sys.setrecursionlimit(100000)


def main():
    sourceDir = rf"..\DEM-{targetName}"
    targetDir = rf"..\DEM-{targetName}_255"
    rgbDir = rf"..\Image\png-stretched-{targetName}"
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    # for i, file in enumerate(os.listdir(sourceDir)):
    rgbList = os.listdir(rgbDir)
    # maxList = []
    for i in tqdm(range(len(os.listdir(sourceDir)))):
    # for i in tqdm(range(501, 8000)):
        # if i == 84:
        #     print("debug")
        if f"tile_{i}.png" not in rgbList:
            print(f"skip {i}, not in {rgbDir}")
            continue
        file = f"tile_{i}.tif"
        image = gdal.Open(os.path.join(sourceDir, file)).ReadAsArray()
        if len(image.shape) > 2:
            print(f"skip {file}, shape={image.shape}")
            continue
        else:
            # imgArray, temp = removeNegative(image, file)
            imgArray, temp = image, []
            # imgArray = linearRevome(image, 5)

        imgArray = ((imgArray - imgArray.min()) / (imgArray.max() - imgArray.min()) * 255).astype(np.uint8)
        img = Image.fromarray(imgArray)
        img.save(os.path.join(targetDir, file))
        print(f"save {file}, {i + 1}/{len(os.listdir(sourceDir))}")

    # print(sorted(maxList))


def linearRevome(imgArray, threshold):
    sumPixelCount = imgArray.size
    originalMin = imgArray.min()
    unique_values, counts = np.unique(imgArray, return_counts=True)
    # 排序
    frequency_dict = dict(zip(unique_values, counts))
    s, target, tempMin = 0, sumPixelCount * threshold / 100, 0
    for key in sorted(frequency_dict.keys()):
        s += frequency_dict[key]
        if s >= target:
            tempMin = key
            break
    if originalMin != tempMin:
        print(f"debug, originalMin={originalMin}, tempMin={tempMin}")
    imgArray[imgArray < tempMin] = tempMin

    return imgArray


def getSpeed():
    sourceDir = r"..\preSplitImages-Mediterranean"
    time1, time2, time3 = 0, 0, 0
    for i in range(len(os.listdir(sourceDir))):
        file = f"tile_{i}.tif"
        if len(tifffile.imread(os.path.join(sourceDir, file)).shape) != 3:
            continue
        timeI = time.time()
        image1 = tifffile.imread(os.path.join(sourceDir, file))
        time1 += time.time() - timeI
        timeI = time.time()
        image2 = gdal.Open(os.path.join(sourceDir, file)).ReadAsArray()
        time2 += time.time() - timeI
        timeI = time.time()
        image3 = np.array(rasterio.open(os.path.join(sourceDir, file), 'r').read())
        time3 += time.time() - timeI
        # assert np.array_equal(image1, image2) and np.array_equal(image1, image3)
        print(f"check {file}, {i + 1}/{len(os.listdir(sourceDir))}")
    print(f"time1={time1}, time2={time2}, time3={time3}")


def removeNegative(imgArray, fileName):
    if imgArray.min() >= 0 or fileName in exceptionDict[targetName]:
        # print(f"skip {fileName}, min={imgArray.min()}")
        return imgArray, []
    raise Exception("error")

    minThreshold = -8
    if abs(imgArray.max() / imgArray.min()) <= 100:
        # print(f"debug, {fileName}, min={imgArray.min()}, max={imgArray.max()}")
        minThreshold = max(minThreshold, int(-imgArray.max() / 100))

    mask = np.zeros_like(imgArray)
    negativeCountArray = []
    # for i in range(imgArray.shape[0]):
    #     for j in range(imgArray.shape[1]):
    #         if imgArray[i, j] < minThreshold and mask[i, j] == 0:
    #             negativeCount = findNegativeCount(imgArray, mask, i, j)
    #             negativeCountArray.append(negativeCount)
    #             if negativeCount > 400:
    #                 print(f"debug, negativeCount={negativeCount}, {fileName}")

    while imgArray.min() < minThreshold:
        cloneImgArray = imgArray.copy()
        for i in range(imgArray.shape[0]):
            for j in range(imgArray.shape[1]):
                if imgArray[i, j] < minThreshold and mask[i, j] == 0:
                    n, s = 0, 0
                    for ii in range(i - 1, i + 2):
                        for jj in range(j - 1, j + 2):
                            if 0 <= ii < imgArray.shape[0] and 0 <= jj < imgArray.shape[1] and cloneImgArray[
                                ii, jj] >= minThreshold:
                                n += 1
                                s += cloneImgArray[ii, jj]
                    if n != 0:
                        imgArray[i, j] = int(s / n)

    return imgArray, negativeCountArray


def findNegativeCount(imgArray, mask, i, j):
    mask[i, j] = 1
    count = 1
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            if 0 <= x < imgArray.shape[0] and 0 <= y < imgArray.shape[1] and imgArray[x, y] < -8 and mask[x, y] == 0:
                count += findNegativeCount(imgArray, mask, x, y)
    return count


def checkZero():
    sourceDir = rf"..\DEM-{targetName}"
    negativeSumDict, negativeCountDict = {}, {}
    for i in range(len(os.listdir(sourceDir))):
        file = f"tile_{i}.tif"
        # file = "tile_476.tif"
        image = tifffile.imread(os.path.join(sourceDir, file))
        if len(image.shape) > 2:
            print(f"skip {file}, shape={image.shape}")
            continue
        if np.min(image) < -8:
            negativeSumDict[file] = np.sum(image < 0)
            # print(f"min={np.min(image)}, {file}")
        if np.min(image) < 0:
            negativeCountDict[np.min(image)] = negativeCountDict.get(np.min(image), 0) + 1
    negativeSumDict = dict(sorted(negativeSumDict.items(), key=lambda x: x[1], reverse=True))
    print(negativeSumDict)
    # plt.bar(negativeSumDict.keys(), negativeSumDict.values())
    plt.bar(negativeCountDict.keys(), negativeCountDict.values())
    plt.show()


def detect_dirty_data(imgArray):
    imgArray = imgArray.astype(np.int64)
    grad_x = sobel(imgArray, axis=0)
    grad_y = sobel(imgArray, axis=1)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def divertDEM255():
    targetDir = fr"..\DEM-{targetName}_255"
    sourceDir = fr"..\DEM-{targetName}"
    os.makedirs(targetDir, exist_ok=True)

    for dem in os.listdir(sourceDir):
        if dem.endswith(".tif"):
            try:
                imgArray = gdal.Open(os.path.join(sourceDir, dem)).ReadAsArray()
                imgArray = ((imgArray - imgArray.min()) / (imgArray.max() - imgArray.min()) * 255).astype(np.uint8)
                Image.fromarray(imgArray).save(os.path.join(targetDir, dem))
            except Exception as e:
                print(f"error={e}, {dem}")


def renameDEM2Tile():
    targetDir = r"..\preSplitDEM-Mediterranean"
    for i, file in enumerate(os.listdir(targetDir)):
        # print(file, end=",")
        os.rename(os.path.join(targetDir, file), os.path.join(targetDir, f"tile_{i}.tif"))
        # print(f"rename {file} to tile_{i}.tif")


if __name__ == '__main__':
    targetName = "Mediterranean"
    exceptionDict = {
        "Japan+Korea": ["tile_1223.tif", "tile_1377.tif"],
        "Southeast-Asia": ["tile_695.tif", "tile_11350.tif", "tile_11357.tif", "tile_688.tif", "tile_2155.tif", "tile_2168.tif",
                   "tile_4109.tif", "tile_12035.tif", "tile_12485.tif", "tile_12485.tif", "tile_13769.tif"],
        "Mediterranean": ["..."],
    }
    # getSpeed()
    main()
    # checkZero()
    # divertDEM255()
    # renameDEM2Tile()
