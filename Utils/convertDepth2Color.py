import os
from glob import glob

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from matplotlib import cm


def colorize_tifs(input_dir: str,
                  output_dir: str = None,
                  cmap_name: str = 'inferno'):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tif_list = glob(os.path.join(input_dir, '*.tif'))
    else:
        tif_list = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.tif') or file.endswith('.png'):
                    tif_list.append(os.path.join(root, file))

    cmap = cm.get_cmap(cmap_name)

    for src_path in tif_list:
        if src_path.endswith('.tif'):
            with rasterio.open(src_path) as src:
                depth = src.read(1, resampling=Resampling.nearest).astype(np.float32)
                if src.nodata is not None:
                    mask = (depth == src.nodata)
                    depth[mask] = np.nan
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                depth = 1 - depth
                colored = cmap(depth)
                rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        elif src_path.endswith('.png'):
            img = Image.open(src_path)
            depth = np.array(img, dtype=np.float32)
            if len(depth.shape) > 2:
                if (depth[:, :, 0] != depth[:, :, 1]).any() or (depth[:, :, 0] != depth[:, :, 2]).any():
                    print(f'skip {src_path}, shape={depth.shape}')
                    continue
                else:
                    depth = depth[:, :, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = 1 - depth
            colored = cmap(depth)
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        if output_dir:
            dst_path = os.path.join(output_dir,
                                    os.path.basename(src_path).replace('.tif', '.png').replace('.png', '_color.png'))
        else:
            dst_dir = os.path.join(os.path.dirname(src_path)+'_color')
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir,
                                    os.path.basename(src_path).replace('.tif', '.png').replace('.png', '_color.png'))

        Image.fromarray(rgb).save(dst_path)
        print(f'→ saved {dst_path}')


if __name__ == '__main__':
    colorize_tifs(input_dir='../Compare',
                  output_dir='',
                  cmap_name='Spectral')
