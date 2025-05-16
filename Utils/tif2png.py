import os
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from Utils.linearExtend import linear_stretch
from Utils.getDeletedIndex import deletedIndexDict


def main():
    targetAreaName = "Australia"
    targetDir = r"..\Image\tif-" + targetAreaName
    outputDir = r"..\Image\png-stretched-" + targetAreaName
    os.makedirs(outputDir, exist_ok=True)

    normalization255 = False
    rgbSingleNormalized = True
    original = False
    linearExtend = 1

    # cloudArray = [4, 5, 8, 9, 10, 21, 23, 34, 53, 60, 67, 68, 70, 73, 80, 86, 93, 94, 101]
    tempDict = {}

    for i in tqdm(range(len(os.listdir(targetDir)))):
    # for i in tqdm(range(58839)):
        # if i == 240:
        #     print("debug")
        # if i < 7219:
        #     print(f"skip {i}")
        #     continue
        if i in deletedIndexDict[targetAreaName] or not os.path.exists(os.path.join(targetDir, f"tile_{i}.tif")):
            print(f"skip {i}")
            continue
        file = f"tile_{i}.tif"
        # if file == "tile_1062.tif":
        #     print("debug")
        # if not file.endswith(".tif"):
        #     continue
        image = tifffile.imread(os.path.join(targetDir, file))
        if len(image.shape) != 3:
            imgArray = image[..., :3]
        else:
            imgArray = image
        # tempArray.append([np.max(imgArray), np.min(imgArray), np.max(imgArray) - np.min(imgArray)])
        # if np.max(imgArray) - np.min(imgArray) < 0.1:
        #     print(f"skip {file}, max-min={np.max(imgArray) - np.min(imgArray)}")
        #     continue
        # temp = np.sum(np.all(imgArray >= 250, axis=-1))
        if len(imgArray.shape) < 3:
            # print(f"imgArray.shape = {imgArray.shape} !")
            continue
        if imgArray.max() == 0:
            print(f"delete {file}, max=0")
            os.remove(os.path.join(targetDir, file))
            continue

        if normalization255:
            imgArray = ((imgArray - imgArray.min()) / (imgArray.max() - imgArray.min()) * 255).astype(np.uint8)
        elif rgbSingleNormalized:
            r, g, b = imgArray[:, :, 0], imgArray[:, :, 1], imgArray[:, :, 2]
            r_normalized = (r - r.min()) / (r.max() - r.min()) * 255
            g_normalized = (g - g.min()) / (g.max() - g.min()) * 255
            b_normalized = (b - b.min()) / (b.max() - b.min()) * 255
            imgArray = np.stack([r_normalized, g_normalized, b_normalized], axis=-1).astype(np.uint8)
        elif original:
            # imgArray = (imgArray * 255).astype(np.uint8)
            imgArray = imgArray.astype(np.uint8)

        lightness1 = 0.299 * imgArray[:, :, 0] + 0.587 * imgArray[:, :, 1] + 0.114 * imgArray[:, :, 2]
        std1 = np.std(lightness1)
        tempDict[i] = [lightness1.mean(), std1]

        if linearExtend > 0:
            imgArray = linear_stretch(imgArray, min_percentile=linearExtend, max_percentile=100 - linearExtend)
            lightness2 = 0.299 * imgArray[:, :, 0] + 0.587 * imgArray[:, :, 1] + 0.114 * imgArray[:, :, 2]
            std2 = np.std(lightness2)
            print(f"{linearExtend}%, {i} {lightness1.mean()}, {std1}；"
                  f"{linearExtend}%, {i} {lightness2.mean()}, {std2}")

        Image.fromarray(imgArray).save(os.path.join(outputDir, file.replace(".tif", ".png")))



if __name__ == '__main__':
    main()
