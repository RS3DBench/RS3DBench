import os

import numpy as np
from PIL import Image


def linear_stretch(image_array, preChannel=False, min_percentile=1, max_percentile=99):
    if not preChannel:
        lower_percentile = np.percentile(image_array, min_percentile)
        upper_percentile = np.percentile(image_array, max_percentile)

        clipped_array = np.clip(image_array, lower_percentile, upper_percentile)

        stretched_array = np.uint8((clipped_array - lower_percentile) / (upper_percentile - lower_percentile) * 255)

        return stretched_array
    else:
        r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        r_lower_percentile, r_upper_percentile = np.percentile(r, [1, 99])
        g_lower_percentile, g_upper_percentile = np.percentile(g, [1, 99])
        b_lower_percentile, b_upper_percentile = np.percentile(b, [0.3, 99.7])

        r_clipped = np.clip(r, r_lower_percentile, r_upper_percentile)
        g_clipped = np.clip(g, g_lower_percentile, g_upper_percentile)
        b_clipped = np.clip(b, b_lower_percentile, b_upper_percentile)

        r_stretched = np.uint8((r_clipped - r_lower_percentile) / (r_upper_percentile - r_lower_percentile) * 255)
        g_stretched = np.uint8((g_clipped - g_lower_percentile) / (g_upper_percentile - g_lower_percentile) * 255)
        b_stretched = np.uint8((b_clipped - b_lower_percentile) / (b_upper_percentile - b_lower_percentile) * 255)

        return np.stack([r_stretched, g_stretched, b_stretched], axis=-1)


def main():
    input_dir = r"../preSplitImages-old"
    output_dir = r"../preSplitImages-old-stretched"
    preChannel = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(input_dir, filename)

            image = Image.open(image_path)
            image_array = np.array(image)

            tempArray = np.array([np.mean(image_array[:, :, 0]), np.mean(image_array[:, :, 1]),
                                  np.mean(image_array[:, :, 2])])
            if tempArray.max() > 200 and tempArray.min() < 0.01:
                print(f"skip {filename}, tempArray={tempArray}")
                continue

            stretched_image = Image.fromarray(linear_stretch(image_array, preChannel))
            output_path = os.path.join(output_dir, filename)
            stretched_image.save(output_path)


if __name__ == '__main__':
    main()
