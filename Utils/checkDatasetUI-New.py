import copy
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
import numpy as np
from osgeo import gdal
import tifffile


class ImageViewer(QMainWindow):
    def __init__(self, folder_paths):
        super().__init__()
        self.setWindowTitle("Image Dataset Viewer")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)

        self.image_grid = QVBoxLayout()
        self.image_grid.setSpacing(20)
        self.row_layouts = []

        self.current_mouse_pos = None
        self.focused_label = None
        self.focused_row = None
        self.focused_col = None

        self.image_labels = [[], []]
        self.current_images = [[], []]
        self.magnifier_labels = [[], []]
        self.filename_labels = [[], []]
        self.pixel_labels = [[], []]

        self.current_index = 0
        self.total_images = 0

        self.xBoxLength = 100
        self.yBoxLength = 100
        self.zoom_factor = 2

        self.folder_paths = folder_paths

        self.setFocusPolicy(Qt.StrongFocus)

        self.init_ui()

    def enterEvent(self, event):
        # When mouse enters the window, set focus to receive key events
        self.setFocus()
        super().enterEvent(event)

    def keyPressEvent(self, event):
        if self.focused_label and self.current_mouse_pos:
            # Get current position
            pos = self.current_mouse_pos
            new_pos = QPoint(pos.x(), pos.y())

            # Update position based on key press
            if event.key() == Qt.Key_W or event.key() == Qt.Key_Up:
                new_pos.setY(max(0, pos.y() - 1))
            elif event.key() == Qt.Key_S or event.key() == Qt.Key_Down:
                new_pos.setY(min(self.focused_label.height() - 1, pos.y() + 1))
            elif event.key() == Qt.Key_A or event.key() == Qt.Key_Left:
                new_pos.setX(max(0, pos.x() - 1))
            elif event.key() == Qt.Key_D or event.key() == Qt.Key_Right:
                new_pos.setX(min(self.focused_label.width() - 1, pos.x() + 1))

            # If position changed, update cursor and trigger magnifier
            if new_pos != pos:
                # Convert to global coordinates for cursor movement
                # Create a mock event object
                class MockEvent:
                    def __init__(self, pos):
                        self._pos = pos

                    def pos(self):
                        return self._pos

                # Update current position and trigger magnifier update
                self.current_mouse_pos = new_pos
                mock_event = MockEvent(new_pos)
                self.show_magnifier(mock_event,
                                    self.focused_row,
                                    self.focused_col)

    def init_ui(self):
        for row in range(2):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)
            self.row_layouts.append(row_layout)
            self.image_grid.addLayout(row_layout)

        self.main_layout.addLayout(self.image_grid)

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 10, 0, 10)

        self.prev_50_btn = QPushButton("<<", self)
        self.next_50_btn = QPushButton(">>", self)
        self.prev_btn = QPushButton("←", self)
        self.next_btn = QPushButton("→", self)
        self.index_label = QLabel("0/0", self)

        self.prev_50_btn.clicked.connect(lambda: self.jump_images(-100))
        self.next_50_btn.clicked.connect(lambda: self.jump_images(100))
        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)

        control_layout.addStretch()
        control_layout.addWidget(self.prev_50_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.index_label)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.next_50_btn)
        control_layout.addStretch()

        self.main_layout.addLayout(control_layout)
        self.load_images()

    def show_current_images(self):
        self.current_images = [[], []]

        first_index = self.current_index * 2
        second_index = first_index + 1


        indices = []
        if first_index < len(self.valideIndexes):
            indices.append(first_index)
        if second_index < len(self.valideIndexes):
            indices.append(second_index)

        for row, img_index in enumerate(indices):
            for i in range(len(self.folder_paths)):
                filename = f"tile_{self.valideIndexes[img_index]}.tif"

                deep_array = None
                if not os.path.exists(os.path.join(self.folder_paths[i], filename)):
                    filename = f"tile_{self.valideIndexes[img_index]}.png"
                    image_path = os.path.join(self.folder_paths[i], filename)
                    pixmap = QPixmap(image_path)
                    image = Image.open(image_path)
                    pixmap = pixmap.scaled(pixmap.width(), pixmap.height(), Qt.KeepAspectRatio)
                else:
                    image_path = os.path.join(self.folder_paths[i], filename)
                    image_array = gdal.Open(image_path).ReadAsArray()
                    if len(image_array.shape) != 3:
                        deep_array = copy.deepcopy(image_array)
                        image_array = ((image_array - np.min(image_array)) / (
                                np.max(image_array) - np.min(image_array)) * 255)
                        image_array = np.stack([image_array] * 3, axis=2)
                    else:
                        # image_array = ((image_array - np.min(image_array)) / (
                        #         np.max(image_array) - np.min(image_array)) * 255)
                        r, g, b = image_array[0], image_array[1], image_array[2]
                        r_normalized = (r - r.min()) / (r.max() - r.min()) * 255
                        g_normalized = (g - g.min()) / (g.max() - g.min()) * 255
                        b_normalized = (b - b.min()) / (b.max() - b.min()) * 255
                        image_array = np.stack([r_normalized, g_normalized, b_normalized], axis=-1).astype(np.uint8)
                        # image_array = image_array.transpose(1, 2, 0)
                    image = Image.fromarray(image_array.astype(np.uint8))
                    pixmap = self.array_to_pixmap(image)

                self.image_labels[row][i].setPixmap(pixmap)
                self.current_images[row].append(image)
                if deep_array is not None:
                    self.current_images[row][i].depth_array = deep_array
                    self.filename_labels[row][i].setText(filename + f"({deep_array.min():.1f},{deep_array.max():.1f})")
                else:
                    self.filename_labels[row][i].setText(filename)

        for row in range(len(indices), 2):
            for i in range(len(self.folder_paths)):
                self.image_labels[row][i].clear()
                self.filename_labels[row][i].setText("")
                self.current_images[row] = []

        self.total_images = len(self.valideIndexes)
        self.index_label.setText(f"{self.current_index * 2 + 1}~{self.current_index * 2 + 2} / {self.total_images}")

    def load_images(self):
        if len(self.folder_paths) < 1:
            return

        for row in range(2):
            for label in self.image_labels[row]:
                self.row_layouts[row].removeWidget(label)
                label.deleteLater()
            self.image_labels[row].clear()
            self.magnifier_labels[row].clear()
            self.filename_labels[row].clear()
            self.pixel_labels[row].clear()

        try:
            original_image_size = Image.open(
                os.path.join(self.folder_paths[0], os.listdir(self.folder_paths[0])[0])).size
        except:
            original_image_size = tifffile.imread(
                os.path.join(self.folder_paths[0], os.listdir(self.folder_paths[0])[0])).shape[:2]

        images_per_row = len(self.folder_paths)

        for row in range(2):
            for i in range(images_per_row):
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setSpacing(0)
                container_layout.setContentsMargins(0, 0, 0, 0)

                label = QLabel()
                label.setFixedSize(original_image_size[0], original_image_size[1])
                label.setAlignment(Qt.AlignCenter)
                label.setMouseTracking(True)
                label.mouseMoveEvent = lambda event, r=row, idx=i: self.show_magnifier(event, r, idx)
                label.leaveEvent = lambda event, r=row, idx=i: self.hide_magnifier(event, r, idx)
                self.image_labels[row].append(label)
                container_layout.addWidget(label)

                filename_label = QLabel()
                filename_label.setAlignment(Qt.AlignCenter)
                filename_label.setMaximumHeight(20)
                self.filename_labels[row].append(filename_label)
                container_layout.addWidget(filename_label)

                mag_label = QLabel(label)
                mag_label.hide()
                self.magnifier_labels[row].append(mag_label)

                pixel_label = QLabel(mag_label)
                pixel_label.setStyleSheet(
                    "QLabel { color: white; font-weight: bold; background-color: rgba(0, 0, 0, 180); }")
                pixel_label.hide()
                self.pixel_labels[row].append(pixel_label)

                self.row_layouts[row].addWidget(container)

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')
        min_images = float('inf')
        min_index = -1

        for i, folder in enumerate(self.folder_paths):
            image_paths = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)])
            if len(image_paths) < min_images:
                min_images = len(image_paths)
                min_index = i

        self.valideIndexes = sorted(
            [int(i.split("tile_")[1].split(".")[0]) for i in os.listdir(self.folder_paths[min_index])])
        self.total_images = (len(self.valideIndexes) + 1) // 2
        self.current_index = 0
        self.show_current_images()

    def show_magnifier(self, event, row, idx):
        self.focused_label = self.image_labels[row][idx]
        self.focused_row = row
        self.focused_col = idx
        self.current_mouse_pos = event.pos()
        if not self.current_images[row]:
            return

        pos = event.pos()
        image = self.current_images[row][idx]

        label_size = self.image_labels[row][idx].size()
        image_size = image.size
        scale_x = image_size[0] / label_size.width()
        scale_y = image_size[1] / label_size.height()

        x = int(pos.x() * scale_x)
        y = int(pos.y() * scale_y)

        half_box_x = self.xBoxLength // 2
        half_box_y = self.yBoxLength // 2

        padded_image = Image.new('RGB', (image_size[0] + self.xBoxLength,
                                         image_size[1] + self.yBoxLength),
                                 (0, 0, 0))
        padded_image.paste(image, (half_box_x, half_box_y))

        padded_x = x + half_box_x
        padded_y = y + half_box_y

        left = max(0, padded_x - half_box_x)
        top = max(0, padded_y - half_box_y)
        right = min(padded_image.size[0], padded_x + half_box_x)
        bottom = min(padded_image.size[1], padded_y + half_box_y)

        for current_col in range(len(self.magnifier_labels[row])):
            if current_col < len(self.current_images[row]):
                current_padded = Image.new('RGB', padded_image.size, (0, 0, 0))
                current_padded.paste(self.current_images[row][current_col], (half_box_x, half_box_y))

                region = current_padded.crop((left, top, right, bottom))
                region = region.resize((self.xBoxLength * self.zoom_factor,
                                        self.yBoxLength * self.zoom_factor))

                region_np = np.array(region)

                center_x = self.xBoxLength * self.zoom_factor // 2
                center_y = self.yBoxLength * self.zoom_factor // 2

                cross_color = [100, 200, 255]
                alpha = 1
                for i in range(max(0, center_x - 10), min(region_np.shape[1], center_x + 11)):
                    if 0 <= center_y < region_np.shape[0]:

                        region_np[center_y, i] = region_np[center_y, i] * (1 - alpha) + np.array(cross_color) * alpha


                for i in range(max(0, center_y - 10), min(region_np.shape[0], center_y + 11)):
                    if 0 <= center_x < region_np.shape[1]:

                        region_np[i, center_x] = region_np[i, center_x] * (1 - alpha) + np.array(cross_color) * alpha

                height, width, channel = region_np.shape
                bytes_per_line = 3 * width
                qImg = QImage(region_np.data, width, height, bytes_per_line,
                              QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)

                mag_label = self.magnifier_labels[row][current_col]
                mag_label.setPixmap(pixmap)

                mag_x = pos.x() + 20
                mag_y = pos.y() + 20

                if mag_x + pixmap.width() > label_size.width():
                    mag_x = pos.x() - 20 - pixmap.width()
                if mag_y + pixmap.height() > label_size.height():
                    mag_y = pos.y() - 20 - pixmap.height()

                mag_label.setGeometry(mag_x, mag_y,
                                      pixmap.width(), pixmap.height() + 20)

                if hasattr(self.current_images[row][current_col], 'depth_array'):
                    if 0 <= y < image_size[1] and 0 <= x < image_size[0]:
                        depth_value = self.current_images[row][current_col].depth_array[y, x]
                        pixel_text = f"Depth: {depth_value:.2f}"
                    else:
                        pixel_text = "Out of bounds"
                else:
                    if 0 <= y < image_size[1] and 0 <= x < image_size[0]:
                        pixel_value = np.array(self.current_images[row][current_col])[y, x]
                        pixel_text = f"RGB: {pixel_value[0]}, {pixel_value[1]}, {pixel_value[2]}"
                    else:
                        pixel_text = "Out of bounds"

                pixel_label = self.pixel_labels[row][current_col]
                pixel_label.setText(pixel_text)
                pixel_label.setGeometry(0, pixmap.height(), pixmap.width(), 20)
                pixel_label.show()

                mag_label.show()

    def jump_images(self, offset):
        new_index = self.current_index + offset // 2
        max_index = (len(self.valideIndexes) - 1) // 2
        new_index = max(0, min(new_index, max_index))
        if new_index != self.current_index:
            self.current_index = new_index
            self.show_current_images()

    def hide_magnifier(self, event, row, idx):
        self.focused_label = None
        self.focused_row = None
        self.focused_col = None
        self.current_mouse_pos = None
        for i in range(len(self.magnifier_labels[row])):
            self.magnifier_labels[row][i].hide()
            self.pixel_labels[row][i].hide()

    @staticmethod
    def array_to_pixmap(image):
        image_np = np.array(image)

        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif len(image_np.shape) == 3 and image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]

        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        qImage = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qImage)

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_images()

    def show_next(self):
        next_first_index = (self.current_index + 1) * 2
        if next_first_index < len(self.valideIndexes):
            for row in range(2):
                for i in range(len(self.current_images[row])):
                    if hasattr(self.current_images[row][i], 'depth_array'):
                        del self.current_images[row][i].depth_array
            self.current_index += 1
            self.show_current_images()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    targetAreaName = "Mediterranean"
    folder_paths = [
        fr"..\Image\png-stretched-{targetAreaName}",
        fr"..\Image\tif-{targetAreaName}",
        fr"..\DEM-{targetAreaName}",
        # fr"..\Mask-{targetAreaName}",
        # fr"..\Gradient-{targetAreaName}",
        # fr"..\DEM-{targetAreaName}_255",
        # fr"..\DEM-{targetAreaName}_color",
        # fr"..\DEM-{targetAreaName}",
    ]
    viewer = ImageViewer(folder_paths)
    viewer.show()
    sys.exit(app.exec_())
