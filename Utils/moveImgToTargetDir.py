import os
import shutil


def main():
    thread_num = 50
    targertAreaName = "Switzerland0.5"
    basicDirs = [f"../preSplitImages-{targertAreaName}", f"../DEM-{targertAreaName}", f"../Image/tif-{targertAreaName}"]
    for dir in basicDirs:
        os.makedirs(dir, exist_ok=True)
        for i in range(thread_num):
            try:
                for filename in os.listdir(dir + f"_{i}"):
                    shutil.move(dir + "_" + str(i) + "/" + filename, dir)
                    print("move " + dir + "_" + str(i) + "/" + filename + " " + dir)
                # 删除空文件夹
                os.rmdir(dir + f"_{i}")
            except:
                print()

if __name__ == '__main__':
    main()