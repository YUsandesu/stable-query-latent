import sys
import os
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, 
                             QMessageBox, QLineEdit, QLabel, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ==========================================
# 关键修改：全局配置 Matplotlib 以支持中日文显示
# 按照优先级查找字体：优先使用微软雅黑，其次是黑体，再到日文的 Meiryo 和 Yu Gothic
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Meiryo', 'Yu Gothic', 'sans-serif']
# 解决更换字体后坐标轴负号（-）显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# ==========================================

class GenericDataVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.dfs = {}  # 使用字典存放多个 Pandas DataFrame，键为文件名
        self.initUI()

    def initUI(self):
        self.setWindowTitle('通用数据可视化工具 (Generic Data Visualizer)')
        self.resize(850, 750)

        main_layout = QVBoxLayout()

        # 1. 顶部按钮区域
        btn_layout = QHBoxLayout()
        self.btn_import = QPushButton('导入 CSV (Import CSV)')
        self.btn_import.clicked.connect(self.import_csv)
        
        self.btn_render = QPushButton('渲染图表 (Render Plot)')
        self.btn_render.clicked.connect(self.render_plot)
        self.btn_render.setEnabled(False)

        self.btn_save = QPushButton('保存图表 (Save Chart)')
        self.btn_save.clicked.connect(self.save_chart)
        self.btn_save.setEnabled(False)

        btn_layout.addWidget(self.btn_import)
        btn_layout.addWidget(self.btn_render)
        btn_layout.addWidget(self.btn_save)
        main_layout.addLayout(btn_layout)

        # 2. 图表文本设置区域 (支持中日文输入)
        settings_group = QGroupBox("图表文本设置 (Chart Text Settings)")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("图表标题 (Title):"), 0, 0)
        # 默认文字也换成包含中日文的示例
        self.input_title = QLineEdit("自定义数据图表 / カスタムデータグラフ")
        settings_layout.addWidget(self.input_title, 0, 1)

        settings_layout.addWidget(QLabel("X轴标签 (X-Axis):"), 0, 2)
        self.input_xlabel = QLineEdit("X轴 / X軸")
        settings_layout.addWidget(self.input_xlabel, 0, 3)

        settings_layout.addWidget(QLabel("Y轴标签 (Y-Axis):"), 0, 4)
        self.input_ylabel = QLineEdit("数值 / 数値")
        settings_layout.addWidget(self.input_ylabel, 0, 5)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # 3. 中间选项栏
        self.list_widget = QListWidget()
        self.list_widget.setToolTip("勾选需要绘制的数据列。双击文字可以自定义图例名称。")
        main_layout.addWidget(self.list_widget)

        # 4. 底部绘图区域
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, '导入 CSV 文件', '', 'CSV Files (*.csv)')
        
        if fname:
            try:
                filename = os.path.basename(fname)
                
                if filename in self.dfs:
                    QMessageBox.information(self, "提示", f"文件 '{filename}' 已经导入过了。")
                    return
                
                df = pd.read_csv(fname)
                self.dfs[filename] = df
                
                for col in df.columns:
                    display_text = f"{filename} - {col}"
                    item = QListWidgetItem(display_text)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setData(Qt.ItemDataRole.UserRole, (filename, col))
                    self.list_widget.addItem(item)
                
                self.btn_render.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "读取错误", f"无法解析该 CSV 文件:\n{e}")

    def render_plot(self):
        if not self.dfs:
            return
            
        selected_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                filename, col = item.data(Qt.ItemDataRole.UserRole)
                custom_label = item.text() 
                selected_items.append((filename, col, custom_label))
                
        if not selected_items:
            QMessageBox.warning(self, "提示", "请至少在列表栏勾选一个需要显示的数据列！")
            return

        self.ax.clear()
        
        for filename, col, custom_label in selected_items:
            df = self.dfs[filename]
            
            x_data = df.index
            if 'Epoch' in df.columns:
                x_data = df['Epoch']
            elif 'Step' in df.columns:
                x_data = df['Step']
                
            self.ax.plot(x_data, df[col], marker='.', label=custom_label)
            
        # 读取输入框中的多语言文字并设置
        self.ax.set_title(self.input_title.text())
        self.ax.set_xlabel(self.input_xlabel.text())
        self.ax.set_ylabel(self.input_ylabel.text())
        
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas.draw()
        self.btn_save.setEnabled(True)

    def save_chart(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, '保存图表', '',
            'PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;PDF Files (*.pdf);;SVG Files (*.svg)'
        )

        if fname:
            try:
                self.figure.savefig(fname, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"图表已保存至:\n{fname}")
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存图表时出错:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GenericDataVisualizer()
    ex.show()
    sys.exit(app.exec())