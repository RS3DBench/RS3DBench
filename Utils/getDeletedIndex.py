import os


def main():
    maxIndex = 1750
    targetAreaNeme = ""
    targetDir = rf"../Image/png-stretched-{targetAreaNeme}"
    deletedIndex = []
    i = 0
    while i <= maxIndex:
        filename = f"tile_{i}.tif"
        if not os.path.exists(os.path.join(targetDir, filename)):
            deletedIndex.append(i)
        i += 1
    print(deletedIndex)


if __name__ == '__main__':
    main()

deletedIndexDict = {
    "Switzerland0.5": [],
    "Australia": [1, 10, 13, "..."],
    "Switzerland2": [],
    "Mediterranean": [8, 15, "..."],
    "Southeast_Asia": [0, 1, 2, 4, 5, "..."],
    "Japan+Korea": [0, 1, 2, 3, "..."]

}
