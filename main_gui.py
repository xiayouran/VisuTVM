# -*- coding: utf-8 -*-
# Author  : lipengyu
# Email   : juancai.li@qq.com
# Datetime: 2025/3/1 17:09
# Filename: main_gui.py
# >>> 解决splash启动画面不关闭问题
from contextlib import suppress

with suppress(ModuleNotFoundError):
    import pyi_splash
    pyi_splash.close()
# <<<

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import shutil
import glob
import ctypes
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QFileDialog, QLabel, QVBoxLayout, QHBoxLayout,
                             QCheckBox, QStatusBar, QGridLayout, QGroupBox,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QDesktopWidget)
from PyQt5.QtSvg import QGraphicsSvgItem
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QWheelEvent, QPainter, QIcon

from utils import visu_relay_ir, visu_relay_ir_single
import visutvm_icon


class ScalableSVGView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # 启用拖动模式
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 缩放时以鼠标为中心
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale_factor = 1.0  # 初始缩放比例

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        if event.angleDelta().y() > 0:
            self.scale(1.1, 1.1)  # 放大
            self.scale_factor *= 1.1
        else:
            self.scale(0.9, 0.9)  # 缩小
            self.scale_factor *= 0.9

    def reset_view(self):
        """重置视图"""
        self.resetTransform()
        self.scale_factor = 1.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.svg_list = []  # 存储处理后的svg文件路径
        self.initUI()

    def initUI(self):
        # 设置窗口图标
        self.setWindowIcon(QIcon(':/visutvm.ico'))

        # 窗口设置
        self.setWindowTitle('TVM计算图可视化软件')
        self.setGeometry(100, 100, 1024, 768)

        # 将窗口居中显示
        self.center_window()

        # 主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 数据输入区域
        input_group = QGroupBox("数据输入")
        input_group.setStyleSheet("QGroupBox { color: white; font-size: 16px; }")
        input_layout = QHBoxLayout(input_group)
        input_layout.setContentsMargins(15, 15, 15, 15)
        input_layout.setSpacing(20)

        # 上传按钮
        self.upload_btn = QPushButton('📁 上传计算图拓扑文件')
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.clicked.connect(self.upload_files)
        input_layout.addWidget(self.upload_btn)

        # 复选框
        self.checkbox = QCheckBox('启用高级处理')
        self.checkbox.setStyleSheet("color: white; font-size: 14px; padding-left: 10px;")
        input_layout.addWidget(self.checkbox)
        input_layout.addStretch(1)  # 填充剩余空间，避免右侧空白

        # 固定数据输入区域高度
        input_group.setFixedHeight(100)  # 设置固定高度
        input_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 结果显示区域
        result_group = QGroupBox("结果显示")
        result_group.setStyleSheet("QGroupBox { color: white; font-size: 16px; }")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(15, 25, 15, 15)

        # 动态显示区域
        self.display_container = QWidget()
        self.display_layout = QGridLayout(self.display_container)
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        self.display_layout.setSpacing(20)

        # 下载按钮区域
        self.download_container = QWidget()
        self.download_layout = QHBoxLayout(self.download_container)
        self.download_layout.setContentsMargins(0, 0, 0, 0)

        # 将动态区域和下载按钮添加到结果显示区域
        result_layout.addWidget(self.display_container)
        result_layout.addWidget(self.download_container)

        # 组合所有组件
        main_layout.addWidget(input_group)
        main_layout.addWidget(result_group)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 样式美化
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D2D;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #AAAAAA;
            }
            QScrollArea {
                border: 2px solid #404040;
                border-radius: 5px;
                background: #1E1E1E;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 12px;
            }
            QStatusBar {
                background-color: #404040;
                color: #FFFFFF;
            }
            QGroupBox {
                border: 2px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

    def center_window(self):
        """将窗口居中显示"""
        screen = QDesktopWidget().screenGeometry()  # 获取屏幕的几何信息
        size = self.geometry()  # 获取窗口的几何信息
        self.move(
            (screen.width() - size.width()) // 2,  # 水平居中
            (screen.height() - size.height()) // 2  # 垂直居中
        )

    def upload_files(self):
        """处理文件上传"""
        # 清空之前的显示结果
        self.clear_display()

        # 清空生成的缓存文件
        self.clear_tmp()

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, '选择计算图拓扑文件', '',
            'TXT Files (*.txt)'
        )

        if file_paths:
            self.status_bar.showMessage(f'上传了 {len(file_paths)} 个文件，正在处理中...')
            self.set_ui_processing(True)

            # 获取复选框状态，并传递给文件处理函数
            with_tensor = self.checkbox.isChecked()

            # 模拟处理逻辑
            # QTimer.singleShot(2000, lambda: self.process_file(file_paths, with_tensor))
            self.process_file(file_paths, with_tensor)

    def process_file(self, file_list: list, with_tensor: bool = False):
        """对接VisuTVM的处理逻辑"""
        try:
            self.svg_list = []
            save_name = os.path.basename(file_list[0]).split('_')[0]
            if len(file_list) == 2:
                visu_relay_ir(
                    bp_file=file_list[1],
                    ap_file=file_list[0],
                    save_name=save_name,
                    with_info=with_tensor
                )
            else:
                visu_relay_ir_single(
                    ir_file=file_list[0],
                    save_name=save_name,
                    with_info=with_tensor
                )
            gen_svg_list = glob.glob(os.path.join('output', f'visu_{save_name}_relay_ir*.svg'))
            for i, file_path in enumerate(gen_svg_list):
                file_name = os.path.basename(file_path)
                self.svg_list.append(file_path)

                # 更新显示
                self.add_svg_display(file_path, i, file_name)

            self.status_bar.showMessage('处理完成', 5000)
            self.set_ui_processing(False)

        except Exception as e:
            self.status_bar.showMessage(f'处理失败: {str(e)}', 5000)
            self.set_ui_processing(False)

    def add_svg_display(self, file_path: str, index: int, name: str):
        """添加 SVG 显示和下载按钮"""
        # 创建 QGraphicsView 和 QGraphicsScene
        svg_view = ScalableSVGView()
        scene = QGraphicsScene()
        svg_item = QGraphicsSvgItem(file_path)
        scene.addItem(svg_item)
        svg_view.setScene(scene)

        # 显示区域
        svg_container = QVBoxLayout()
        svg_container.addWidget(QLabel(f"输出文件 {name}"))
        svg_container.addWidget(svg_view)

        # 下载按钮
        download_btn = QPushButton(f'⬇️ 下载 SVG {name}')
        download_btn.setFixedHeight(35)
        download_btn.clicked.connect(lambda: self.download_svg(file_path))

        # 添加到布局
        self.display_layout.addLayout(svg_container, index // 2, index % 2)
        self.download_layout.addWidget(download_btn)

    def download_svg(self, file_path):
        """处理文件下载"""
        try:
            file_name = os.path.basename(file_path)
            path, _ = QFileDialog.getSaveFileName(
                self, '保存SVG文件',
                file_name, 'SVG Files (*.svg)'
            )
            if path:
                shutil.copy(file_path, path)
                self.status_bar.showMessage('SVG文件保存成功', 3000)

        except Exception as e:
            self.status_bar.showMessage(f'保存失败: {str(e)}', 5000)

    def clear_display(self):
        """清空显示和下载按钮"""
        # 清空显示区域
        while self.display_layout.count():
            item = self.display_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

        # 清空下载按钮
        while self.download_layout.count():
            item = self.download_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def clear_layout(self, layout):
        """递归清空布局中的所有控件"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def clear_tmp(self):
        """清空生成的缓存文件"""
        cur_path = os.path.dirname(__file__)
        tmp_path = os.path.join(cur_path, 'output')
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

    def set_ui_processing(self, processing):
        """更新界面处理状态"""
        self.upload_btn.setEnabled(not processing)
        self.upload_btn.setText('🔄 处理中...' if processing else '📁 上传计算图拓扑文件')
        QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置现代字体
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
