import sys
import os
import shutil
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QFileDialog, 
                             QHBoxLayout, QStackedWidget, QScrollArea, QGridLayout, 
                             QSlider, QSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QMouseEvent, QPainter, QColor, QPen

# --- 設定 ---
DEFAULT_BLUR_THRESHOLD = 150.0  
DEFAULT_SIMILARITY_THRESHOLD = 85 
DELETE_FOLDER_NAME = "Deleted_Images"
SELECT_FOLDER_NAME = "Selected_Images"

# --- 解析スレッド ---
class ImageAnalyzer(QThread):
    result_ready = pyqtSignal(str, float, float, QImage)
    progress_update = pyqtSignal(int, int)

    def __init__(self, files):
        super().__init__()
        self.files = files
        self.is_running = True

    def calculate_blur(self, img):
        h, w = img.shape[:2]
        target_w = 640
        if w > target_w:
            scale = target_w / w
            img_small = cv2.resize(img, (target_w, int(h * scale)))
        else:
            img_small = img
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_similarity(self, img1, img2):
        if img1 is None or img2 is None: return 0.0
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def run(self):
        total = len(self.files)
        last_img_sim = None
        
        for i, fpath in enumerate(self.files):
            if not self.is_running: break
            
            img = cv2.imread(fpath)
            if img is None: continue

            h, w = img.shape[:2]
            thumb_w = 240
            scale = thumb_w / w
            thumb_h = int(h * scale)
            thumb_cv = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            thumb_rgb = cv2.cvtColor(thumb_cv, cv2.COLOR_BGR2RGB)
            thumb_qimg = QImage(thumb_rgb.data, thumb_w, thumb_h, 3*thumb_w, QImage.Format.Format_RGB888).copy()

            sim_score = 0.0
            img_sim_curr = cv2.resize(img, (100, int(100*h/w)))
            if i > 0 and last_img_sim is not None:
                sim_score = self.calculate_similarity(img_sim_curr, last_img_sim)
            last_img_sim = img_sim_curr

            blur_score = self.calculate_blur(img)
            self.result_ready.emit(fpath, blur_score, sim_score, thumb_qimg)
            self.progress_update.emit(i + 1, total)

    def stop(self):
        self.is_running = False
        self.wait()

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

class PhotoReviewApp(QWidget):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.delete_path = os.path.join(folder_path, DELETE_FOLDER_NAME)
        self.select_path = os.path.join(folder_path, SELECT_FOLDER_NAME)
        
        self.files = sorted([os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        self.current_idx = 0
        self.image_data = {f: {
            'score': 0.0, 'sim_prev': 0.0, 'to_delete': False, 'to_select': False, 
            'group_id': -1, 'is_blur': False
        } for f in self.files}
        
        self.group_counts = {}
        self.thumbnail_cache = {}
        self.grid_widgets = {}
        
        # 【高速化】 現在表示中の「枠線なし」画像を保持する変数
        self.current_base_pixmap = None 

        self.initUI()
        
        if self.files:
            self.show_current_image()
            self.start_analysis()

    def initUI(self):
        self.setWindowTitle(f'選別ツール - {os.path.basename(self.folder_path)}')
        self.showMaximized()
        self.setStyleSheet("background-color: #111; color: #EEE;")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 画像エリア
        self.stack = QStackedWidget()
        
        # 1枚表示
        self.page_single = QWidget()
        single_layout = QVBoxLayout(self.page_single)
        single_layout.setContentsMargins(0,0,0,0)
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        single_layout.addWidget(self.image_display)

        # グリッド表示
        self.page_grid = QWidget()
        grid_layout_outer = QVBoxLayout(self.page_grid)
        grid_layout_outer.setContentsMargins(0,0,0,0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")
        self.grid_content = QWidget()
        self.grid_content.setStyleSheet("background-color: #222;")
        self.grid_layout = QGridLayout(self.grid_content)
        self.grid_layout.setSpacing(5)
        self.grid_layout.setContentsMargins(5,5,5,5)
        self.scroll_area.setWidget(self.grid_content)
        grid_layout_outer.addWidget(self.scroll_area)

        self.stack.addWidget(self.page_single)
        self.stack.addWidget(self.page_grid)
        main_layout.addWidget(self.stack, stretch=1)

        # --- コントロールパネル ---
        control_panel = QWidget()
        control_panel.setStyleSheet("background-color: #222; border-top: 1px solid #444;")
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 5, 10, 5)

        # ピント設定
        blur_group = QGroupBox("ピント判定")
        blur_group.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 5px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        blur_layout = QHBoxLayout(blur_group)
        
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(0, 500)
        self.slider_blur.setValue(int(DEFAULT_BLUR_THRESHOLD))
        self.slider_blur.valueChanged.connect(self.refresh_judgments_instantly) # ここを高速版に変更
        self.slider_blur.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        self.spin_blur = QSpinBox()
        self.spin_blur.setRange(0, 500)
        self.spin_blur.setValue(int(DEFAULT_BLUR_THRESHOLD))
        self.spin_blur.valueChanged.connect(self.slider_blur.setValue)
        self.slider_blur.valueChanged.connect(self.spin_blur.setValue)
        self.spin_blur.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        blur_layout.addWidget(QLabel("緩"))
        blur_layout.addWidget(self.slider_blur)
        blur_layout.addWidget(QLabel("厳"))
        blur_layout.addWidget(self.spin_blur)

        # 連写設定
        sim_group = QGroupBox("連写判定")
        sim_group.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 5px; font-weight: bold; }")
        sim_layout = QHBoxLayout(sim_group)

        self.slider_sim = QSlider(Qt.Orientation.Horizontal)
        self.slider_sim.setRange(50, 100)
        self.slider_sim.setValue(DEFAULT_SIMILARITY_THRESHOLD)
        self.slider_sim.valueChanged.connect(self.refresh_judgments_instantly) # ここを高速版に変更
        self.slider_sim.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.spin_sim = QSpinBox()
        self.spin_sim.setRange(50, 100)
        self.spin_sim.setSuffix("%")
        self.spin_sim.setValue(DEFAULT_SIMILARITY_THRESHOLD)
        self.spin_sim.valueChanged.connect(self.slider_sim.setValue)
        self.slider_sim.valueChanged.connect(self.spin_sim.setValue)
        self.spin_sim.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        sim_layout.addWidget(QLabel("別物"))
        sim_layout.addWidget(self.slider_sim)
        sim_layout.addWidget(QLabel("同一"))
        sim_layout.addWidget(self.spin_sim)

        control_layout.addWidget(blur_group)
        control_layout.addWidget(sim_group)
        
        # ステータス
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("padding-left: 20px;")
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        main_layout.addWidget(control_panel)
        self.setLayout(main_layout)

    def start_analysis(self):
        self.worker = ImageAnalyzer(self.files)
        self.worker.result_ready.connect(self.handle_analysis_result)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.start()

    @pyqtSlot(str, float, float, QImage)
    def handle_analysis_result(self, fpath, blur_score, sim_score, thumb_qimg):
        self.image_data[fpath]['score'] = blur_score
        self.image_data[fpath]['sim_prev'] = sim_score
        self.thumbnail_cache[fpath] = QPixmap.fromImage(thumb_qimg)
        self.apply_judgment_logic(fpath) # ロジックのみ適用
        
        if self.files[self.current_idx] == fpath:
            self.update_overlay_only() # 現在表示中ならオーバーレイ更新

    # --- 高速更新ロジック ---
    def refresh_judgments_instantly(self):
        """スライダー操作時に呼ばれる。画像の再読み込みはしない。"""
        # 1. 計算ロジックを全件に適用 (数値比較だけなので高速)
        for fpath in self.files:
            self.apply_judgment_logic(fpath, update_group_id=False) 
        
        # 2. グループID再計算 (シーケンシャル処理)
        curr_group_id = 0
        sim_threshold = self.slider_sim.value() / 100.0
        
        if self.files:
            self.image_data[self.files[0]]['group_id'] = 0

        for i in range(1, len(self.files)):
            fpath = self.files[i]
            prev_sim = self.image_data[fpath]['sim_prev']
            if prev_sim >= sim_threshold:
                self.image_data[fpath]['group_id'] = curr_group_id
            else:
                curr_group_id += 1
                self.image_data[fpath]['group_id'] = curr_group_id
        
        # 3. カウント集計
        ids = [v['group_id'] for v in self.image_data.values()]
        self.group_counts = {i: ids.count(i) for i in set(ids)}

        # 4. 画面更新 (ここがポイント: 画像再読み込みを避ける)
        if self.stack.currentIndex() == 0:
            self.update_overlay_only() # 枠線だけ書き直す
        else:
            self.refresh_grid_styles() # グリッドの枠線更新

    def apply_judgment_logic(self, fpath, update_group_id=True):
        """データの判定のみ行う"""
        data = self.image_data[fpath]
        blur_th = self.slider_blur.value()
        data['is_blur'] = (data['score'] < blur_th)

        if update_group_id:
            idx = self.files.index(fpath)
            if idx > 0:
                sim_th = self.slider_sim.value() / 100.0
                if data['sim_prev'] >= sim_th:
                    prev_path = self.files[idx-1]
                    data['group_id'] = self.image_data[prev_path]['group_id']
                else:
                    data['group_id'] = self.image_data[self.files[idx-1]]['group_id'] + 1
            else:
                data['group_id'] = 0
            
            ids = [v['group_id'] for v in self.image_data.values() if v['group_id'] != -1]
            if ids:
                self.group_counts = {i: ids.count(i) for i in set(ids)}

    @pyqtSlot(int, int)
    def update_progress(self, current, total):
        pass

    # --- 描画関連 (高速化の肝) ---
    def show_current_image(self):
        """ディスクから画像を読み込み、ベース画像として保持する"""
        if not self.files: return
        fpath = self.files[self.current_idx]
        
        img_bgr = cv2.imread(fpath)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            qimg = QImage(img_rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
            
            # ここでリサイズしてベース画像を作成
            self.current_base_pixmap = QPixmap.fromImage(qimg).scaled(
                self.page_single.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            # 枠線を描画してセット
            self.update_overlay_only()
        
        self.update_status_label()

    def update_overlay_only(self):
        """保持しているベース画像に、現在の判定状況で枠線を上書きして表示する"""
        if self.current_base_pixmap is None: return
        
        # ベース画像をコピー（元の画像を汚さないため）
        pixmap_with_hud = self.current_base_pixmap.copy()
        fpath = self.files[self.current_idx]
        
        # 描画実行
        self.draw_hud_on_pixmap(pixmap_with_hud, fpath)
        self.image_display.setPixmap(pixmap_with_hud)

    def draw_hud_on_pixmap(self, pixmap, fpath):
        painter = QPainter(pixmap)
        data = self.image_data[fpath]
        
        color, width = QColor(0,0,0,0), 0
        gid = data['group_id']
        is_burst = self.group_counts.get(gid, 0) > 1
        
        if data['to_delete']:
            color, width = QColor(255, 50, 50), 10
        elif data['to_select']:
            color, width = QColor(50, 255, 50), 10
        elif data['is_blur']:
            color, width = QColor(255, 170, 0), 10
        elif is_burst:
            color, width = QColor(0, 255, 255), 10
            
        if width > 0:
            pen = QPen(color)
            pen.setWidth(width)
            # 枠線を描画
            painter.setPen(pen)
            painter.drawRect(0, 0, pixmap.width(), pixmap.height())

        # HUDテキスト
        painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        info = f" {os.path.basename(fpath)} | Score: {data['score']:.0f} "
        painter.setPen(Qt.GlobalColor.white)
        painter.fillRect(0, 0, 700, 60, QColor(0,0,0,160))
        painter.drawText(10, 40, info)
        
        status = ""
        if data['to_delete']: status = "[削除対象]"
        elif data['to_select']: status = "[★選抜★]"
        elif data['is_blur']: status = "[ピンボケ]"
        elif is_burst: status = "[連写グループ]"
        
        if status:
            painter.setPen(color)
            painter.drawText(720, 40, status)
        painter.end()

    def update_status_label(self):
        if not self.files: return
        idx = self.current_idx + 1
        total = len(self.files)
        self.status_label.setText(f"枚数: {idx}/{total} | 操作: [S]一覧 [D]削除 [E]選抜 [F]実行")

    # --- グリッド関連 ---
    def build_grid_view(self):
        for i in reversed(range(self.grid_layout.count())): 
            w = self.grid_layout.itemAt(i).widget()
            if w: w.setParent(None)
        
        self.grid_widgets = {}
        col_count = 5 
        
        for i, fpath in enumerate(self.files):
            pixmap = self.thumbnail_cache.get(fpath, QPixmap(200,150))
            lbl = ClickableLabel()
            lbl.setPixmap(pixmap)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(220, pixmap.height() + 10)
            lbl.setStyleSheet(self.get_grid_style(fpath))
            lbl.clicked.connect(lambda idx=i: self.switch_to_single_view(idx))
            
            self.grid_layout.addWidget(lbl, i // col_count, i % col_count)
            self.grid_widgets[fpath] = lbl

    def get_grid_style(self, fpath):
        data = self.image_data[fpath]
        is_burst = self.group_counts.get(data['group_id'], 0) > 1
        
        if data['to_delete']: return "border: 5px solid #F33; background: #400;"
        if data['to_select']: return "border: 5px solid #3F3; background: #040;"
        if data['is_blur']: return "border: 5px solid #FA0; background: #430;"
        if is_burst: return "border: 5px solid #0FF; background: #033;"
        return "border: 2px solid #444;"

    def refresh_grid_styles(self):
        for fpath, w in self.grid_widgets.items():
            w.setStyleSheet(self.get_grid_style(fpath))

    def switch_to_single_view(self, index):
        self.current_idx = index
        self.stack.setCurrentIndex(0)
        self.show_current_image()

    def resizeEvent(self, event):
        if hasattr(self, 'stack') and self.stack.currentIndex() == 0:
            self.show_current_image()
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if not self.files: return
        fpath = self.files[self.current_idx]

        if key == Qt.Key.Key_S:
            if self.stack.currentIndex() == 0:
                self.build_grid_view()
                self.stack.setCurrentIndex(1)
            else:
                self.stack.setCurrentIndex(0)
                self.show_current_image()
        
        elif key == Qt.Key.Key_D:
            self.image_data[fpath]['to_delete'] = not self.image_data[fpath]['to_delete']
            if self.image_data[fpath]['to_delete']:
                self.image_data[fpath]['to_select'] = False
            
            if self.stack.currentIndex() == 0: self.update_overlay_only() # ここも高速化
            else: self.refresh_grid_styles()
            
        elif key == Qt.Key.Key_E:
            self.image_data[fpath]['to_select'] = not self.image_data[fpath]['to_select']
            if self.image_data[fpath]['to_select']:
                self.image_data[fpath]['to_delete'] = False
            
            if self.stack.currentIndex() == 0: self.update_overlay_only() # ここも高速化
            else: self.refresh_grid_styles()

        elif key == Qt.Key.Key_Right:
            if self.current_idx < len(self.files)-1: 
                self.current_idx += 1
                if self.stack.currentIndex()==0: self.show_current_image()
        elif key == Qt.Key.Key_Left:
            if self.current_idx > 0: 
                self.current_idx -= 1
                if self.stack.currentIndex()==0: self.show_current_image()
        
        elif key == Qt.Key.Key_F:
            self.execute_actions()

    def execute_actions(self):
        if not os.path.exists(self.delete_path): os.makedirs(self.delete_path)
        if not os.path.exists(self.select_path): os.makedirs(self.select_path)
        
        to_process = []
        for f in self.files:
            if self.image_data[f]['to_delete']:
                to_process.append((f, "delete"))
            elif self.image_data[f]['to_select']:
                to_process.append((f, "select"))
        
        if not to_process: return

        count_del, count_sel = 0, 0
        for fpath, action in to_process:
            try:
                fname = os.path.basename(fpath)
                if action == "delete":
                    shutil.move(fpath, os.path.join(self.delete_path, fname))
                    count_del += 1
                elif action == "select":
                    shutil.move(fpath, os.path.join(self.select_path, fname))
                    count_sel += 1
                
                self.files.remove(fpath)
                del self.image_data[fpath]
                if fpath in self.thumbnail_cache: del self.thumbnail_cache[fpath]
            except Exception: pass

        if self.current_idx >= len(self.files): self.current_idx = max(0, len(self.files)-1)
        if self.stack.currentIndex() == 0: self.show_current_image()
        else: self.build_grid_view()
        
        self.status_label.setText(f"完了: 削除{count_del}枚, 選抜{count_sel}枚")

if __name__ == '__main__':
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    path = QFileDialog.getExistingDirectory(None, "画像フォルダ")
    if path:
        ex = PhotoReviewApp(path)
        ex.show()
        sys.exit(app.exec())