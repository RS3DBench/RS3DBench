import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pyvista as pv
from scipy.ndimage import zoom


def read_dem(dem_path):
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
    return dem_data


def read_rgb(rgb_path):
    rgb_image = Image.open(rgb_path)
    rgb_data = np.array(rgb_image)
    return rgb_data


def upsample_data(data, target_shape):
    zoom_factors = np.array(target_shape) / np.array(data.shape[:2])

    if data.ndim == 2:
        upsampled_data = zoom(data, (zoom_factors[0], zoom_factors[1]), order=3)
    elif data.ndim == 3:
        upsampled_data = zoom(data, (zoom_factors[0], zoom_factors[1], 1), order=3)
    else:
        raise ValueError("Unsupported data dimensions")

    return upsampled_data


def create_point_cloud(dem_data, rgb_data):
    height, width = dem_data.shape

    # 生成x, y坐标
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    z = dem_data.flatten()

    r = rgb_data[..., 0].flatten()
    g = rgb_data[..., 1].flatten()
    b = rgb_data[..., 2].flatten()

    points = np.vstack((x, y, z)).T
    colors = np.vstack((r, g, b)).T

    return points, colors


def smooth_and_reconstruct_surface(points, colors):
    cloud = pv.PolyData(points)

    cloud.point_data['RGB'] = colors / 255.0

    smooth_cloud = cloud.smooth(n_iter=50, relaxation_factor=0.1)

    surface = smooth_cloud.delaunay_2d()

    return surface


def create_3d_model_with_pyvista(points, colors):
    surface = smooth_and_reconstruct_surface(points, colors)

    plotter = pv.Plotter()
    plotter.add_mesh(surface, scalars='RGB', rgb=True, show_edges=False, opacity=1.0)
    plotter.set_background('white')
    title_text = "Ground Truth"
    plotter.add_text(title_text, position='upper_left', font_size=20, color='black')
    plotter.show()


def main(dem_path, rgb_path):
    dem_data = read_dem(dem_path)
    rgb_data = read_rgb(rgb_path)

    target_shape = (1024, 1024)
    dem_data_up = upsample_data(dem_data, target_shape)
    rgb_data_up = upsample_data(rgb_data, target_shape)

    points, colors = create_point_cloud(dem_data_up, rgb_data_up)

    create_3d_model_with_pyvista(points, colors)


index = 8

dem_path = f'3ddatatest/figure1/tile_{index}.tif'
rgb_path = f'3ddatatest/figure1/tile_{index}.png'
main(dem_path, rgb_path)
