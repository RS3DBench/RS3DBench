import math
import os
import re
import time
import asyncio
import aiohttp
import threading

import tifffile
from pyproj import Proj, transform
import ee
import geemap
import rasterio
import rasterio.features
import rasterio.warp
from osgeo import gdal
import numpy as np
from datetime import datetime


class GEEDownloader:
    def __init__(self, thread_num, start_index):
        self.thread_num = thread_num
        self.start_index = start_index
        self.download_counts = [[] for i in range(thread_num)]
        self.download_times = []
        self.proxies = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890',
        }

        # Initialize required directories for each thread
        self.init_directories()

        # Constants
        self.SCALE = -1
        self.DATA_SIZE = 512
        # self.DATA_SIZE = 500
        self.SATELLITE_SR = "COPERNICUS/S2_HARMONIZED"
        # self.SATELLITE_SR = "SKYSAT/GEN-A/PUBLIC/ORTHO/RGB"

        self.total_files = len(os.listdir(f'preSplitDEM-{targetAreaName}'))
        self.start_time = time.time()

        self.print_lock = threading.Lock()

    def init_directories(self):
        base_dirs = [f'preSplitImages-{targetAreaName}', f'Image/tif-{targetAreaName}', f'DEM-{targetAreaName}']
        for i in range(self.thread_num):
            for base_dir in base_dirs:
                dir_path = f"{base_dir}_{i}"
                os.makedirs(dir_path, exist_ok=True)

    def maskS2clouds(self, image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
            qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    def removeZeroTiles(self, filename):
        delete = False
        with rasterio.open(filename, 'r') as ds:
            arr = np.array(ds.read())
            if np.amax(arr) == 0:
                delete = True
        if delete:
            os.remove(filename)
            print(f"File: {filename} removed, it contained only zeros.")
        return delete

    def print_progress(self, thread_id, dem_index):
        with self.print_lock:
            total_processed = sum([len(i) for i in self.download_counts])
            elapsed_time = time.time() - self.start_time

            if total_processed > 0:
                avg_time_per_file = elapsed_time / total_processed
                remaining_files = self.total_files - (self.start_index + total_processed)
                estimated_time = remaining_files * avg_time_per_file

                print(f"\nProgress Update [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:")
                print(f"Thread {thread_id} processing DEM {dem_index}")
                print(f"Total processed: {total_processed} files")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Estimated remaining time: {estimated_time:.2f} seconds ({estimated_time / 3600:.2f} hours)")
                print(f"Progress: {((dem_index - self.start_index) / self.total_files * 100):.2f}%")

                print(f"\nThread Progress:{self.download_counts}")
                print("\n" + "=" * 50)

    async def process_single_dem(self, thread_id, dem_index, resize):
        try:
            filename = f"tile_{dem_index}.tif"
            dem_path = f"preSplitDEM-{targetAreaName}/{filename}"

            start_time = time.time()

            # Get DEM geometry and coordinates
            with rasterio.open(dem_path) as dataset:
                mask = dataset.dataset_mask()
                if not mask.all():
                    raise "Image contains nodata values. Please remove them before processing."
                for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
                    geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=20)
                    # geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:2056', geom, precision=20)

                self.SCALE = 30.95
                # self.SCALE = 0.5
                print("Thread", thread_id, "DEM", dem_index, "SCALE:", self.SCALE)

            reprojected_geom = ee.Geometry(geom).transform('EPSG:4326')
            # reprojected_geom = ee.Geometry(geom).transform('EPSG:2056')

            # Download RGB image
            output_rgb = f"preSplitImages-{targetAreaName}_{thread_id}/{filename[:-4]}.tif"

            dataset = ee.ImageCollection(self.SATELLITE_SR) \
                .filterDate('2021-01-01', '2023-12-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .map(self.maskS2clouds) \
                .select(['B4', 'B3', 'B2'])
            image = dataset.reduce('median')

            # dataset = ee.ImageCollection(self.SATELLITE_SR) \
            #     .filterDate('2021-01-01', '2023-12-31') \
            #     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            #     .map(self.maskS2clouds) \
            #     .select(['R', 'G', 'B'])
            # image = dataset.reduce('median')
            try:
                returnJson = await asyncio.to_thread(
                    geemap.ee_export_image,
                    image,
                    filename=output_rgb,
                    scale=self.SCALE,
                    region=reprojected_geom,
                    file_per_band=False,
                    proxies=self.proxies,
                    timeout=1000,
                )

                if returnJson and returnJson['error']['message'] == 'User memory limit exceeded.':
                    os.system(f"gdal_translate -of GTIFF -epo -srcwin 0, 0, 512, 512 {dem_path} {output_rgb}")
                    print(f"Thread {thread_id}: 生成空白tif文件 for DEM {dem_index}")
            except Exception as e:
                print(f"Download error in thread {thread_id} for DEM {dem_index}: {str(e)}")
                os.system(f"gdal_translate -of GTIFF -epo -srcwin 0, 0, 512, 512 {dem_path} {output_rgb}")
                print(f"Thread {thread_id}: 生成空白tif文件 for DEM {dem_index} (after error)")

            # if not os.path.exists(output_rgb):
            #     try:
            #         returnJson = await asyncio.to_thread(
            #             geemap.ee_export_image,
            #             ee.Image('Switzerland/SWISSIMAGE/orthos/10cm/2017'),
            #             filename=output_rgb,
            #             scale=self.SCALE,
            #             region=reprojected_geom,
            #             file_per_band=False,
            #             proxies=self.proxies,
            #             timeout=1000,
            #         )
            #
            #         if returnJson and returnJson['error']['message'] == 'User memory limit exceeded.':
            #             os.system(f"gdal_translate -of GTIFF -epo -srcwin 0, 0, 512, 512 {dem_path} {output_rgb}")
            #             print(f"Thread {thread_id}: generate empty tif for DEM {dem_index}")
            #     except Exception as e:
            #         print(f"Download error in thread {thread_id} for DEM {dem_index}: {str(e)}")
            #         os.system(f"gdal_translate -of GTIFF -epo -srcwin 0, 0, 512, 512 {dem_path} {output_rgb}")
            #         print(f"Thread {thread_id}: generate empty tif for DEM {dem_index} (after error)")
            if not os.path.exists(output_rgb):
                print(f"Failed to download RGB for thread {thread_id}, DEM {dem_index}")
                return False

            if os.path.getsize(output_rgb) >= 20240:
                print(f"Thread {thread_id}: {dem_index} RGB size is {os.path.getsize(output_rgb)} bytes")

            # Split both DEM and RGB
            await asyncio.to_thread(self.split_images, dem_path, output_rgb, thread_id, dem_index, resize)
            self.download_counts[thread_id].append(dem_index)

            processing_time = time.time() - start_time
            self.download_times.append(processing_time)
            self.print_progress(thread_id, dem_index)

            return True

        except Exception as e:
            print(f"Error in thread {thread_id} processing DEM {dem_index}: {e}")
            return False

    def split_images(self, dem_path, rgb_path, thread_id, dem_index, resize=False):
        tile_size = self.DATA_SIZE

        # Open DEM to get dimensions
        ds = gdal.Open(dem_path)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize

        if resize:
            dem_output = f"DEM-{targetAreaName}_{thread_id}/tile_{dem_index}.tif"
            cmd_dem = f"gdal_translate -of GTIFF -epo -outsize {tile_size} {tile_size} {dem_path} {dem_output}"
            os.system(cmd_dem)

            rgb_output = f"Image/tif-{targetAreaName}_{thread_id}/tile_{dem_index}.tif"
            cmd_rgb = f"gdal_translate -of GTIFF -epo -outsize {tile_size} {tile_size} {rgb_path} {rgb_output}"
            os.system(cmd_rgb)

        else:
            # Split both DEM and RGB
            for i in range(0, xsize, tile_size):
                for j in range(0, ysize, tile_size):
                    if i + tile_size > xsize or j + tile_size > ysize:
                        continue

                    # Split DEM
                    dem_output = f"DEM-{targetAreaName}_{thread_id}/tile_{dem_index}.tif"
                    cmd_dem = f"gdal_translate -of GTIFF -epo -srcwin {i}, {j}, {tile_size}, {tile_size} {dem_path} {dem_output}"
                    os.system(cmd_dem)

                    if not self.removeZeroTiles(dem_output):
                        # Split RGB
                        rgb_output = f"Image/tif-{targetAreaName}_{thread_id}/tile_{dem_index}.tif"
                        rgbShape = tifffile.imread(rgb_path).shape
                        print(f"Thread {thread_id}: RGB shape: {rgbShape}")
                        # if rgbShape[0] < 512 or rgbShape[1] < 512:
                        #     print(f"Thread {thread_id}: RGB shape too small, using fallback for DEM {dem_index}")
                        #     f = open("log.txt", 'a')
                        #     f.write(f"Thread {thread_id}: {dem_index} RGB shape is {rgbShape}\n")
                        #     f.close()
                        #     os.system(f"gdal_translate -of GTIFF -epo -srcwin 0, 0, 512, 512 {rgb_path} {rgb_output}")
                        cmd_rgb = f"gdal_translate -of GTIFF -epo -srcwin {i}, {j}, {tile_size}, {tile_size} {rgb_path} {rgb_output}"
                        os.system(cmd_rgb)
                    else:
                        os.system(f"gdal_translate -of GTIFF -epo -srcwin 0, 0, 512, 512 {rgb_path} {rgb_output}")
                        f = open("log.txt", 'a')
                        f.write(f"Thread {thread_id}: {dem_index} DEM contains only zeros\n")
                        f.close()
                        print(f"File: {dem_output} removed, it contained only zeros.")

    async def process_batch(self, thread_id):
        if "Switzerland" in targetAreaName:
            resize = True
        else:
            resize = False
        # resize = False
        while True:
            dem_index = self.start_index + thread_id + self.thread_num * len(self.download_counts[thread_id])
            print(f"Thread {thread_id} processing DEM {dem_index}")

            # Check if we've processed all files
            if dem_index >= len(os.listdir(f'preSplitDEM-{targetAreaName}/')):
                break

            success = await self.process_single_dem(thread_id, dem_index, resize)
            if not success:
                print(f"Failed to process DEM {dem_index} in thread {thread_id}")

            await asyncio.sleep(1)

    async def run(self):
        print(f"Starting download with {self.thread_num} threads at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting from index: {self.start_index}")
        print(f"Total files to process: {self.total_files - self.start_index}")
        print("=" * 50)

        tasks = []
        for thread_id in range(self.thread_num):
            task = asyncio.create_task(self.process_batch(thread_id))
            tasks.append(task)

        await asyncio.gather(*tasks)

        print("\nDownload completed!")
        print(f"Total time elapsed: {time.time() - self.start_time:.2f} seconds")
        print(f"Files processed: {self.download_counts}")


def removeZeroTiles(filename):
    delete = False
    with rasterio.open(filename, 'r') as ds:
        arr = np.array(ds.read())  # read all raster values
        if np.amax(arr) == 0:
            delete = True
    if delete:
        os.remove(filename)
        print("File:" + filename + " removed, it contained only zeros.")
    return delete


def splitDEM():
    tile_size_x = 512
    tile_size_y = 512
    k = 0
    output_filename = 'tile_'
    path = f'DataDEM-{targetAreaName}/'
    out_path = f'preSplitDEM-{targetAreaName}/'
    os.makedirs(out_path, exist_ok=True)
    for i, filename in enumerate(os.listdir(path)):
        # if i < 5809:
        #     continue
        # if not os.path.exists(rf"Image/tif-{targetAreaName}" + f"/tile_{i}.tif"):
        #     print(f"Skipping {filename}")
        #     continue
        print(f"Processing {i + 1}/{len(os.listdir(path))}")
        dem = filename
        ds = gdal.Open(path + dem)
        arr = np.array(ds.ReadAsArray())
        print(f"Processing {dem}, max: {np.max(arr)}, min: {np.min(arr)}")
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                if i + tile_size_x > xsize or j + tile_size_y > ysize:
                    print(f"size error!,{filename} size is ({xsize}, {ysize})")
                    continue
                com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i) + ", " + str(j) + ", " + str(
                    tile_size_x) + ", " + str(tile_size_y) + " " + path + str(dem) + " " + str(out_path) + str(
                    output_filename) + str(k) + ".tif"
                out = os.system(com_string)
                if out == 0:
                    ret = removeZeroTiles(str(out_path) + str(output_filename) + str(k) + ".tif")
                    if not ret:
                        k += 1
                        arrI = tifffile.imread(str(out_path) + str(output_filename) + str(k - 1) + ".tif")
                        print(f"getting {output_filename}{k}, max: {np.max(arrI)}, min: {np.min(arrI)}")


def main():
    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project="your-project-id")

    splitDEM()

    # Create and run downloader
    thread_num = 50  # Adjust this number based on your system's capabilities
    start_index = 0  # Your starting index
    downloader = GEEDownloader(thread_num, start_index)
    #
    # # Run the async process
    asyncio.run(downloader.run())


if __name__ == "__main__":
    targetAreaName = 'Switzerland0.5'
    main()