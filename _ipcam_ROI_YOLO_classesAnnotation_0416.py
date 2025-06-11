import os
import sys
import cv2
import numpy as np
import subprocess
from datetime import datetime, timedelta
import threading
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from PyQt5.QtCore import QTimer, Qt, QRect, QUrl, pyqtSignal, QThread, QCoreApplication, pyqtSlot, QMutex, QDateTime
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QSpinBox, QGroupBox, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QFileDialog, QDoubleSpinBox, QListWidget, QListWidgetItem, QFormLayout, 
    QLineEdit)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from ultralytics import YOLO
import imageio_ffmpeg
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QTabWidget, QTextEdit
from PyQt5.QtGui import QIcon
import json

def resource_path(relative_path):
    """獲取資源檔案的正確路徑，無論是開發環境還是 PyInstaller 打包後"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

SETTINGS_FILE = resource_path("config/settings.json")
ROI_FILE = resource_path("config/roi_settings.json")


def apply_global_font(self):
    font = QFont()
    font.setPointSize(16)
    QApplication.instance().setFont(font)
    self.setFont(font)

    # 全部 widget 設定
    for widget in self.findChildren(QWidget):
        widget.setFont(font)

    # 額外放大 QPushButton 的字體
    for btn in self.findChildren(QPushButton):
        btn_font = QFont()
        btn_font.setPointSize(20)  # 讓按鈕更醒目
        btn.setFont(btn_font)

    print("[🆗] 全域字體套用完畢，按鈕字體大小提升至 20pt")



# def send_line_message(message, channel_access_token, to_user_id):
#     url = "https://api.line.me/v2/bot/message/push"
    
#     # 確保 message 一定為字串
#     if not isinstance(message, str):
#         message = str(message)
    
#     # 也可以在此強制將 message 轉為 UTF-8 bytes 再解碼回來
#     # message = message.encode('utf-8').decode('utf-8')

#     payload = {
#         "to": to_user_id,
#         "messages": [
#             {
#                 "type": "text",
#                 "text": message
#             }
#         ]
#     }

#     # 設定 Header 明確指定使用 UTF-8
#     headers = {
#         "Content-Type": "application/json; charset=UTF-8",
#         "Authorization": f"Bearer {channel_access_token}"
#     }

#     # 1. 使用 json.dumps(...) 時，指定 ensure_ascii=False，避免自動跳脫非 ASCII 字元
#     # 2. encode('utf-8') 將字串轉為 UTF-8 bytes
#     json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')

#     # 以 data 參數傳入 UTF-8 bytes，確保不會被默認編碼成其他編碼
#     response = requests.post(url, headers=headers, data=json_payload)
#     if response.status_code == 200:
#         print("LINE 推播訊息成功")
#     else:
#         print("LINE 推播訊息失敗：", response.text)

# 例如在 MainWindow 中修改您的 send_alert_line() 函式：

# 取得 ffmpeg 路徑與設定環境變數
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = r"C:\AI\yolo\ROI_Detection_2024_12_31\yolov_env\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
QCoreApplication.addLibraryPath(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"])
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# ---------------- InferenceThread：獨立推論執行緒 ----------------
class InferenceThread(QThread):
    # 傳出推論結果的訊號
    inference_done = pyqtSignal(object)
    
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._busy = False
        self._mutex = QMutex()
    
    @pyqtSlot(np.ndarray)
    def run_inference(self, frame):
        # 若目前忙碌則略過這幀
        self._mutex.lock()
        if self._busy:
            self._mutex.unlock()
            return
        self._busy = True
        self._mutex.unlock()
        
        try:
            # 這裡採用普通推論（predict）模式，不使用追蹤功能
            results = self.model(frame, imgsz=640, conf=0.5, iou=0.5, verbose=False)
        except Exception as e:
            print("推論錯誤:", e)
            results = None

        # 傳回結果
        self.inference_done.emit(results)
        
        self._mutex.lock()
        self._busy = False
        self._mutex.unlock()

class VideoWidget(QLabel):
    """自訂的視訊顯示元件，繼承自 QLabel，支援繪製 ROI 和顯示偵測結果。"""

    roi_changed = pyqtSignal()  # 當 ROI 改變時發出信號

    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.setMouseTracking(True)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_rects = []         # 儲存所有已畫的 ROI
        self.current_rect = None    # 正在繪製的 ROI
        self.color = QColor(0, 255, 0)
        self.pen_width = 3

        self.show_alert = False
        self.detections = []
        self.intruder_info = None  # ✅ 新增：警報資訊資料

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # If Shift not held, optionally clear existing ROIs (start a new selection)
            if not (event.modifiers() & Qt.ShiftModifier):
                self.roi_rects.clear()
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.current_rect = QRect(self.start_point, self.end_point)

        if event.button() == Qt.RightButton:
            self.roi_rects.clear()
            self.current_rect = None
            self.update()
            self.roi_changed.emit()


    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.current_rect = QRect(self.start_point, self.end_point)
            self.update()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, event.pos()).normalized()
            if rect.isValid():
                self.roi_rects.append(rect)
            self.current_rect = None
            self.update()
            self.roi_changed.emit()  # signal to inform ROI list changed

    def paintEvent(self, event):
        super(VideoWidget, self).paintEvent(event)
        painter = QPainter(self)

        for idx, roi in enumerate(self.roi_rects):
            if roi is not None and not roi.isNull():
                pen = painter.pen()
                pen.setColor(self.color)
                pen.setWidth(self.pen_width)
                painter.setPen(pen)
                painter.drawRect(roi)

                painter.setFont(QFont("Arial", 12))
                label = f"貨櫃{idx+1}"
                x = roi.x()
                y = roi.y() - 5 if roi.y() > 10 else roi.y() + 15
                painter.drawText(x, y, label)

        if self.current_rect is not None and not self.current_rect.isNull():
            pen = painter.pen()
            pen.setColor(self.color)
            pen.setWidth(self.pen_width)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        for det in self.detections:
            x1, y1, x2, y2, cls_name, conf = det
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            pen = painter.pen()
            pen.setColor(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.setFont(QFont("Arial", 12))
            text = f"{cls_name} {conf:.2f}"
            painter.drawText(x1, y1 - 10, text)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            pen.setColor(QColor(255, 0, 0))
            pen.setWidth(5)
            painter.setPen(pen)
            painter.drawPoint(center_x, center_y)

        # ✅ 顯示紅字警告文字（若有）
        if self.show_alert and self.intruder_info:
            painter.setPen(QColor(255, 0, 0))
            painter.setFont(QFont("Arial", 24))
            alert_text = "警告：\n"
            for roi_idx, info_dict in self.intruder_info.items():
                label = f"貨櫃{roi_idx+1}: "
                person_count = 0
                intruder_count = 0

                for cat, info in info_dict.items():
                    if cat == "person":
                        person_count += info['count']
                    else:
                        intruder_count += info['count']

                if person_count > 0:
                    label += f"person:{person_count}  "
                if intruder_count > 0:
                    label += f"入侵者:{intruder_count}"

                alert_text += label + "\n"

            painter.drawText(10, 60, alert_text)  # 往下移動 30px

# class VideoCaptureThread(QThread):
#     """獨立執行緒處理 RTSP 影像擷取"""
#     frame_ready = pyqtSignal(np.ndarray)  # 發送影像至主執行緒

#     def __init__(self, rtsp_url):
#         super().__init__()
#         self.rtsp_url = rtsp_url
#         self.running = True
#         self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

#         # 設定 OpenCV 緩存最小化，減少延遲
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.cap.set(cv2.CAP_PROP_FPS, 30)
#         # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     def run(self):
#         """持續讀取影像並發送到主執行緒"""
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 self.frame_ready.emit(frame)  # 發送影像訊號至主執行緒

#     def stop(self):
#         """停止影像擷取"""
#         self.running = False
#         self.cap.release()
#         self.quit()
#         self.wait()

class FFMPEGStreamThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, str)  # 傳 frame + timestamp string
    
    def __init__(self, rtsp_url, width, height):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.running = True

    def run(self):
        # 將 -vf 與 scale=... 分成兩個參數
        ffmpeg_cmd = [
            ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.width}:{self.height}",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-an",
            "-"
        ]
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        frame_size = self.width * self.height * 3  # 更新 frame_size 為新解析度
        while self.running:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) == frame_size:
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 精確到毫秒
                self.frame_ready.emit(frame, timestamp)
        process.terminate()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# --- 修改監控畫面元件，禁用繪製功能 ---
class ReadOnlyVideoWidget(VideoWidget):
    def mousePressEvent(self, event):
        pass
    def mouseMoveEvent(self, event):
        pass
    def mouseReleaseEvent(self, event):
        pass

# ---------------- 主視窗類別 ----------------
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.apply_global_font()
        self.setWindowIcon(QIcon(resource_path("icons/cctv-camera.png")))

       # 建立播放列表並加入警報音檔案
        self.playlist = QMediaPlaylist()
        alert_url = QUrl.fromLocalFile(resource_path("alert.mp3"))
        self.playlist.addMedia(QMediaContent(alert_url))
        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)  # 設定為循環播放

        self.player = QMediaPlayer()
        self.player.setPlaylist(self.playlist)
        self.detection_history_by_roi = {}  # e.g., {0: {'person': False, 'dog': False, ...}, 1: {...}, ...}
        self.intruder_info_by_roi = {}     # e.g., {0: {'person': {'count': 0, 'start_time': None, 'alert_sent': False}, ...}, 1: {...}, ...}
        self.tracked_objects_by_roi = {}         # 結構：{ roi_idx: { 'person': [record, ...], 'dog': [...], ... }, ... }
        self.intruder_info_by_roi = {}           # 結構：{ roi_idx: { 'person': {'count': 0, 'start_time': None, 'alert_sent': False}, ... }, ... }在此版本中，每筆記錄只在首次檢測到物件從 ROI 外進入時建立，並設置狀態為
        self.match_threshold = 100
        self.screenshot_taken = False
        self.show_alert = False
        # --- 將 self.video_monitor 改為 ReadOnlyVideoWidget 實例 ---
        self.video_monitor = ReadOnlyVideoWidget()
        self.video_roi = VideoWidget()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_all_alerts)
        self.timer.start(1000)  # 每1秒檢查一次警報
        
        self.ser = None  # 用於與 Arduino 通訊的串口
        self.relay_pin = 7  # 與 Arduino 連接的繼電器引腳
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.yolo_model = None
        self.connect_to_arduino()  # 在初始化時連接 Arduino
        self.init_ui()
        self.detect_motion = False
        self.frame_counter = 0
        # self.intruder_check_timer = QTimer()
        # self.intruder_check_timer.timeout.connect(self.check_intruder_timeouts)
        # self.intruder_check_timer.start(1000)
        self.confidence_threshold = 0.1
        self.alert_threshold_seconds = 10
        self.selected_classes = set()
        self.video_roi.roi_changed.connect(self.on_roi_changed)
        self.screenshot_taken = False
        self.last_frame = None  # 用於存放最新影像

        # 新增：推論執行緒（模型加載後建立）
        self.inference_thread = None  # 等載入模型後建立
        self.detect_interval = 2  # 調整推論頻率，依硬體資源而定

        self.last_save_time = None
        self.save_cooldown = timedelta(seconds=5)

        self.tabs = QTabWidget()
        # 主監控畫面 tab
        self.monitor_tab = QWidget()
        self.roi_tab = QWidget()
        self.initial_tab = QWidget()

        self.init_monitor_ui()  # 初始化主畫面內容
        self.init_initial_settings_ui()
        self.init_roi_ui()      # 初始化 ROI 畫面內容

        self.tabs.addTab(self.monitor_tab, "監控畫面")
        self.tabs.addTab(self.roi_tab, "ROI 設定")
        self.tabs.addTab(self.initial_tab, "初始設定")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        # ✅ 所有 UI 都設置好後再自動啟動攝影機
        QTimer.singleShot(500, self.start_camera)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: white;
                font-family: "Microsoft JhengHei";
                font-size: 25px;
            }
            QPushButton {
                background-color: #2d89ef;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 25px;
            }
            QPushButton:hover {
                background-color: #1a5fb4;
                font-size: 25px;
            }
            QPushButton:pressed {
                background-color: #104e9c;
                font-size: 25px;
            }
            QComboBox, QDoubleSpinBox, QListWidget {
                background-color: #333333;
                color: white;
                border: 1px solid #666;
                padding: 4px;
                font-size: 25px;
            }
            QTabWidget::pane {
                border-top: 2px solid #2d89ef;
                position: absolute;
                top: -0.5em;
                font-size: 25px;
            }
            QTabBar::tab {
                background: #2d2d2d;
                color: white;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 25px;
            }
            QTabBar::tab:selected {
                background: #2d89ef;
                font-size: 25px;
            }
            QSpinBox {
                font-size: 25px;
            }
        """)
        self.load_initial_settings()
        
        self.recipient_email = ""

        # self.is_recording = False
        # self.video_writer = None
        # self.record_fps = 30  # 可根據實際情況設定

    def init_ui(self):
        # self.video_widget = VideoWidget()
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(['攝影機0', '攝影機1', '攝影機2'])
        self.resolution_selector = QComboBox()
        self.resolution_selector.addItems(['640x480', '800x600', '1024x768', '1280x720', '1920x1080'])
        self.resolution_selector.setCurrentText('1920x1080')  # 預設解析度

        # self.start_button = QPushButton('開啟攝影機')
        self.detect_button = QPushButton('開始偵測')
        self.load_yolo_button = QPushButton('載入 YOLO 模型')
        self.detect_button.setEnabled(False)
        self.load_yolo_button.setEnabled(False)
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 0.9)
        self.confidence_spinbox.setSingleStep(0.1)
        self.confidence_spinbox.setValue(0.5)
        self.confidence_spinbox.setDecimals(1)
        self.confidence_spinbox.setSuffix(' 信心值')
        self.confidence_spinbox.setMinimumWidth(120)
        self.confidence_spinbox.setStyleSheet("font-size:16px;")
        self.class_list_widget = QListWidget()
        self.class_list_widget.setFixedWidth(150)
        self.class_list_widget.itemChanged.connect(self.on_class_item_changed)

        # 新增：控制繼電器的按鈕
        self.relay_on_button = QPushButton("開啟警示燈")
        self.relay_off_button = QPushButton("關閉警示燈")

        self.stop_alert_button = QPushButton("關閉警報聲")
        self.stop_alert_button.clicked.connect(self.player.stop)

        # 連接按鈕事件
        self.relay_on_button.clicked.connect(self.turn_on_relay)
        self.relay_off_button.clicked.connect(self.turn_off_relay)

        h_layout_controls = QHBoxLayout()
        h_layout_controls.addWidget(self.camera_selector)
        h_layout_controls.addWidget(self.resolution_selector)
        # h_layout_controls.addWidget(self.start_button)
        h_layout_controls.addWidget(self.load_yolo_button)
        h_layout_controls.addWidget(self.detect_button)
        h_layout_controls.addWidget(self.confidence_spinbox)
        h_layout_controls.addWidget(self.relay_on_button)  # 加入開啟繼電器按鈕
        h_layout_controls.addWidget(self.relay_off_button)  # 加入關閉繼電器按鈕
        h_layout_controls.addWidget(self.stop_alert_button)  # 新增的關閉警報聲按鈕

        h_layout_main = QHBoxLayout()
        h_layout_main.addWidget(self.video_roi)
        h_layout_main.addWidget(self.class_list_widget)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout_controls)
        v_layout.addLayout(h_layout_main)
        # self.setLayout(v_layout)
        # self.start_button.clicked.connect(self.start_camera)
        self.detect_button.clicked.connect(self.start_detection)
        self.load_yolo_button.clicked.connect(self.load_yolo_model)
        self.confidence_spinbox.valueChanged.connect(self.update_confidence_threshold)

        # ✅ 修改所有按鈕圖示載入路徑
        # self.start_button.setIcon(QIcon(resource_path("icons/play.png")))
        self.load_yolo_button.setIcon(QIcon(resource_path("icons/upload.png")))
        self.detect_button.setIcon(QIcon(resource_path("icons/eye.png")))
        self.relay_on_button.setIcon(QIcon(resource_path("icons/power_on.png")))
        self.relay_off_button.setIcon(QIcon(resource_path("icons/power_off.png")))
        self.stop_alert_button.setIcon(QIcon(resource_path("icons/stop.png")))

        # # ✅ 圖示素材（icons/）已產生，放入以下檔案：
        # # 🎥 開啟攝影機
        # with open("icons/play.png", "wb") as f:
        #     f.write(requests.get("https://img.icons8.com/ios-filled/50/ffffff/play.png").content)
        # # ⬆️ 載入 YOLO 模型
        # with open("icons/upload.png", "wb") as f:
        #     f.write(requests.get("https://img.icons8.com/ios-filled/50/ffffff/upload.png").content)
        # # 👁 開始偵測
        # with open("icons/eye.png", "wb") as f:
        #     f.write(requests.get("https://img.icons8.com/ios-filled/50/ffffff/visible.png").content)
        # # 🔌 開啟繼電器
        # with open("icons/power_on.png", "wb") as f:
        #     f.write(requests.get("https://img.icons8.com/ios-filled/50/ffffff/flash-on.png").content)
        # # ⚡ 關閉繼電器
        # with open("icons/power_off.png", "wb") as f:
        #     f.write(requests.get("https://img.icons8.com/ios-filled/50/ffffff/flash-off.png").content)
        # # 🛑 停止警報聲
        # with open("icons/stop.png", "wb") as f:
        #     f.write(requests.get("https://img.icons8.com/ios-filled/50/ffffff/stop.png").content)
    
    # --- 原 monitor tab 現在只顯示影像畫面 ---
    def init_monitor_ui(self):
        layout = QVBoxLayout()
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_monitor)
        layout.addLayout(video_layout)
        self.monitor_tab.setLayout(layout)  
       # 👉 上方按鈕列
        self.manual_capture_button = QPushButton("📸 手動擷取影像")
        self.manual_capture_button.clicked.connect(self.save_manual_frame)
        layout.addWidget(self.manual_capture_button, alignment=Qt.AlignLeft)

        # self.record_button = QPushButton("🎥 開始錄影")
        # self.record_button.setCheckable(True)
        # self.record_button.clicked.connect(self.toggle_recording)
        # layout.addWidget(self.record_button)  # 加到合適的 layout

    def save_manual_frame(self):
        if hasattr(self, "last_frame") and self.last_frame is not None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_capture/manual_{now}.jpg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(filename, self.last_frame)
            print(f"[📸] 手動擷取影像儲存至：{filename}")
        else:
            print("[⚠️] 無法擷取：目前尚未接收到影像")

    # --- 初始設定 tab 美化版 ---
    def init_initial_settings_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        button_width = 250
        button_height = 36
        combo_width = 160
        spinbox_width = 80

        # 固定按鈕寬高
        for btn in [self.load_yolo_button, self.detect_button, self.relay_on_button, self.relay_off_button, self.stop_alert_button]:
            btn.setFixedWidth(button_width)
            btn.setFixedHeight(button_height)

        # 固定選單與數值設定元件大小
        self.camera_selector.setFixedWidth(combo_width)
        self.resolution_selector.setFixedWidth(combo_width)
        self.confidence_spinbox.setFixedWidth(spinbox_width)

        self.class_list_widget.setFixedWidth(140)

        # 警報秒數輸入欄位（可設定為 1 秒 ～ 3600 秒）
        self.alert_spinbox = QSpinBox()
        self.alert_spinbox.setRange(1, 3600)  # 1 秒到 1 小時
        # self.alert_spinbox.setValue(self.alert_threshold_seconds)
        self.alert_spinbox.valueChanged.connect(self.update_alert_threshold)
        self.alert_spinbox.setFixedWidth(80)

        form_layout.addRow("📷 選擇攝影機：", self.camera_selector)
        form_layout.addRow("🖥️ 選擇解析度：", self.resolution_selector)
        form_layout.addRow("🧠 載入 YOLO 模型：", self.load_yolo_button)
        # form_layout.addRow("👁️ 偵測控制：", self.detect_button)
        form_layout.addRow("🎯 設定信心值：", self.confidence_spinbox)
        form_layout.addRow("⏱️ 警報秒數設定：", self.alert_spinbox)
        # form_layout.addRow("⚡ 繼電器開啟：", self.relay_on_button)
        form_layout.addRow("⚡ 繼電器關閉：", self.relay_off_button)
        form_layout.addRow("🔊 停止警報聲：", self.stop_alert_button)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("輸入接收警報通知的 Email")
        self.email_input.setFixedWidth(400)
        form_layout.addRow("📧 警報通知信箱：", self.email_input)
        form_layout.addRow("📚 類別選擇：", self.class_list_widget)

        # 確定/取消按鈕區塊
        button_layout = QHBoxLayout()
        btn_save = QPushButton("✅ 確定設定")
        btn_cancel = QPushButton("❌ 取消變更")
        btn_save.setFixedWidth(200)
        btn_save.setFixedHeight(40)
        btn_cancel.setFixedWidth(200)
        btn_cancel.setFixedHeight(40)

        button_layout.addWidget(btn_save)
        button_layout.addWidget(btn_cancel)

        btn_save.clicked.connect(self.save_initial_settings)
        btn_cancel.clicked.connect(self.cancel_initial_settings)

        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        self.initial_tab.setLayout(layout)
    
    def update_alert_threshold(self, value):
        self.alert_threshold_seconds = value
        print(f"[🔧] 警報秒數已更新為：{value} 秒")
    # --- ROI 設定 UI ---
    def init_roi_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("請於此畫面進行 ROI 區域設定："))
        layout.addWidget(self.video_roi)

        button_layout = QHBoxLayout()
        btn_save = QPushButton("✅ 儲存 ROI")
        btn_cancel = QPushButton("❌ 清除 ROI")
        btn_save.setFixedWidth(160)
        btn_cancel.setFixedWidth(160)
        btn_save.clicked.connect(self.confirm_roi_settings)
        btn_cancel.clicked.connect(self.clear_roi_settings)
        button_layout.addWidget(btn_save)
        button_layout.addWidget(btn_cancel)

        layout.addLayout(button_layout)
        self.roi_tab.setLayout(layout)

        self.video_roi.roi_changed.connect(self.sync_roi_to_monitor)
        self.load_roi_settings()  # ⬅ 自動載入 ROI

    # --- 儲存與載入 ROI 設定（含標籤名稱） ---
    def confirm_roi_settings(self):
        roi_list = []
        for idx, roi in enumerate(self.video_roi.roi_rects):
            roi_list.append({
                "x": roi.x(),
                "y": roi.y(),
                "w": roi.width(),
                "h": roi.height(),
                "label": f"貨櫃{idx + 1}"
            })
        if not roi_list:
            print("[!] 沒有 ROI 可儲存")
            return
        os.makedirs(os.path.dirname(ROI_FILE), exist_ok=True)
        with open(ROI_FILE, "w", encoding="utf-8") as f:
            json.dump(roi_list, f, indent=2, ensure_ascii=False)
        print(f"[✔] 已儲存 ROI 設定，共 {len(roi_list)} 區")

    def load_roi_settings(self):
        if not os.path.exists(ROI_FILE):
            return

        if os.path.getsize(ROI_FILE) == 0:
            print("[!] ROI 設定檔是空的，略過載入")
            return

        try:
            with open(ROI_FILE, "r", encoding="utf-8") as f:
                roi_data = json.load(f)
        except json.JSONDecodeError:
            print("[!] ROI 設定檔格式錯誤，無法載入")
            return

        self.video_roi.roi_rects.clear()
        for item in roi_data:
            rect = QRect(item["x"], item["y"], item["w"], item["h"])
            self.video_roi.roi_rects.append(rect)

        # --- 關鍵：同步給 video_monitor 顯示 ---
        self.video_monitor.roi_rects = list(self.video_roi.roi_rects)
        self.video_monitor.update()

        self.video_roi.update()
        print(f"[↩] 已載入 ROI 設定，共 {len(self.video_roi.roi_rects)} 區，並同步顯示到監控畫面")

    def clear_roi_settings(self):
        self.video_roi.roi_rects.clear()
        self.video_roi.update()
        self.sync_roi_to_monitor()  # 同步清除到主畫面
        self.confirm_roi_settings()  # 立即清空寫入檔案

        # ✅ 清除所有警報與記錄
        self.intruder_info_by_roi.clear()
        self.tracked_objects_by_roi.clear()
        self.video_monitor.intruder_info = {}
        self.video_monitor.show_alert = False
        self.video_monitor.color = QColor(0, 255, 0)
        self.video_monitor.update()

        print("✅ 已清除所有 ROI、警報與物件追蹤資訊")



    # 解析度變更時自動重啟攝影機
    def restart_camera(self):
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()

        selected_camera_idx = self.camera_selector.currentIndex()
        resolution_text = self.resolution_selector.currentText()
        width, height = map(int, resolution_text.split('x'))

        self.capture = cv2.VideoCapture(selected_camera_idx)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.video_monitor.setFixedSize(width, height)
        self.video_roi.setFixedSize(width, height)

        self.current_resolution = resolution_text
        print(f"[🔄] 攝影機已重新啟動：{resolution_text} 並更新顯示大小")

    def apply_global_font(self):
        apply_global_font(self)

    # 同步生效並儲存設定
    def save_initial_settings(self):
        # --- 即時同步更新系統參數 ---
        selected_camera_idx = self.camera_selector.currentIndex()
        resolution_text = self.resolution_selector.currentText()
        confidence_value = self.confidence_spinbox.value()
        alert_seconds = self.alert_spinbox.value()
        print(f"[DEBUG] email 輸入框內容：{self.email_input.text()}")

        # 解析度同步並重啟攝影機
        self.restart_camera()

        # 信心值與警報秒數同步
        self.confidence_threshold = confidence_value
        self.alert_threshold_seconds = alert_seconds

        # 類別同步
        self.selected_classes.clear()
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                self.selected_classes.add(item.text())

        if self.email_input:
            self.recipient_email = self.email_input.text().strip()
        else:
            self.recipient_email = ""

        settings = {
            "selected_camera": selected_camera_idx,
            "resolution": resolution_text,
            "confidence_threshold": confidence_value,
            "alert_threshold_seconds": alert_seconds,
            "selected_classes": list(self.selected_classes),
            "model_path": getattr(self, 'last_model_path', None),
            "recipient_email": self.recipient_email  # ✅ 一定要加入這行
        }

        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

        print("[✔] 設定已同步套用並儲存至 config/settings.json")


    def cancel_initial_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            print("[!] 尚無設定檔可取消還原")
            return
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # 還原設定值
        self.camera_selector.setCurrentIndex(settings.get("selected_camera", 0))
        self.resolution_selector.setCurrentText(settings.get("resolution", "1920x1080"))
        self.confidence_spinbox.setValue(settings.get("confidence_threshold", 0.5))
        self.alert_spinbox.setValue(settings.get("alert_threshold_seconds", 60))
        # 還原類別選取
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            if item.text() in settings.get("selected_classes", []):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

        print("[↩] 已還原初始設定（從 config/settings.json）")
    # ✅ 自動載入設定檔（在 __init__ 結尾呼叫）
    def load_initial_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            print("[!] 尚無設定檔可自動載入")
            return

        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)

        self.camera_selector.setCurrentIndex(settings.get("selected_camera", 0))
        self.resolution_selector.setCurrentText(settings.get("resolution", "1920x1080"))
        self.confidence_spinbox.setValue(settings.get("confidence_threshold", 0.5))
        self.alert_threshold_seconds = settings.get("alert_threshold_seconds", 10)
        self.recipient_email = settings.get("recipient_email", "")
        if hasattr(self, "email_input") and self.email_input:
            self.email_input.setText(self.recipient_email)

         # ✅ 若元件已存在，則套用設定值（這段放在外層）
        if hasattr(self, "alert_spinbox"):
            self.alert_spinbox.setValue(self.alert_threshold_seconds)
        if hasattr(self, "confidence_spinbox"):
            self.confidence_spinbox.setValue(self.confidence_threshold)

        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            if item.text() in settings.get("selected_classes", []):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

        model_path = settings.get("model_path")
        if model_path and os.path.exists(model_path):
            try:
                self.yolo_model = YOLO(model_path).to('cuda')
                self.load_yolo_button.setText('YOLO 模型已載入')
                self.load_yolo_button.setEnabled(False)
                self.update_class_list()
                self.inference_thread = InferenceThread(self.yolo_model)
                self.inference_thread.inference_done.connect(self.update_detections)
                self.last_model_path = model_path
                print(f"[✔] 自動載入 YOLO 模型：{model_path}")
            except Exception as e:
                print(f"[✘] 自動載入模型失敗：{e}")

        self.recipient_email = settings.get("recipient_email", "")
        if self.email_input:
            self.email_input.setText(self.recipient_email)

    # --- ROI 同步函式 ---
    def sync_roi_to_monitor(self):
        self.video_monitor.roi_rects = list(self.video_roi.roi_rects)  # 同步 ROI 區域
        self.video_monitor.update()

    @staticmethod
    def compute_iou(boxA, boxB):
        # boxA, boxB 格式皆為 (x1, y1, x2, y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def update_confidence_threshold(self, value):
        """當用戶調整信心值時，更新推論的閾值"""
        self.confidence_threshold = value  # 更新信心值閾值
        print(f"信心值閾值已設為：{self.confidence_threshold}")

    def on_class_item_changed(self, item):
        class_name = item.text()
        if item.checkState() == Qt.Checked:
            self.selected_classes.add(class_name)
        else:
            self.selected_classes.discard(class_name)

    def start_camera(self):
        # 從解析度下拉式選單讀取使用者選擇
        resolution = self.resolution_selector.currentText()  # e.g. "1920x1080"
        width_str, height_str = resolution.split('x')
        width, height = int(width_str), int(height_str)

        rtsp_url = "rtsp://admin:123456@192.168.226.201:554/profile1"

        if hasattr(self, "video_thread") and self.video_thread.isRunning():
            self.video_thread.stop()

        self.video_thread = FFMPEGStreamThread(rtsp_url, width, height)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()

        self.detect_button.setEnabled(True)
        self.load_yolo_button.setEnabled(True)

        print(f"✅ RTSP 攝影機已成功連接，解析度: {width}x{height}")

    def load_yolo_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "選擇 YOLO 權重檔案", "", "PyTorch Files (*.pt)", options=options)
        if file_name:
            try:
                self.yolo_model = YOLO(file_name).to('cuda')
                self.last_model_path = file_name  # ✅ 新增這行
                self.load_yolo_button.setText('YOLO 模型已載入')
                self.load_yolo_button.setEnabled(False)
                self.update_class_list()
                # 建立推論執行緒並連線訊號
                self.inference_thread = InferenceThread(self.yolo_model)
                self.inference_thread.inference_done.connect(self.update_detections)
            except Exception as e:
                print(f"載入 YOLO 模型失敗：{e}")
                self.load_yolo_button.setText('載入失敗，重試')

    def update_class_list(self):
        self.class_list_widget.clear()
        self.selected_classes.clear()
        
        # 只顯示這三個類別
        target_classes = {"person", "dog", "cat"}

        if self.yolo_model is not None:
            class_names = self.yolo_model.names  # dict: {0: 'person', 1: 'bicycle', ...}
            for cls_id, cls_name in class_names.items():
                if cls_name in target_classes:
                    item = QListWidgetItem(cls_name)
                    item.setCheckState(Qt.Checked)
                    self.selected_classes.add(cls_name)
                    self.class_list_widget.addItem(item)

    def start_detection(self):
        if self.video_roi.rect is not None and not self.video_roi.rect.isNull():
            self.detect_motion = True
        else:
            self.detect_motion = False
            self.detect_button.setText('請先繪製 ROI')
            QTimer.singleShot(2000, lambda: self.detect_button.setText('開始偵測'))

    def on_roi_changed(self):
        if self.detect_motion:
            self.player.stop()
            self.video_monitor.show_alert = False
            self.screenshot_taken = False

        # 新增：根據目前繪製的 ROI 數量初始化多 ROI 的偵測資料結構
        num_rois = len(self.video_roi.roi_rects)
        self.detection_history_by_roi = {i: {} for i in range(num_rois)}
        self.intruder_info_by_roi = {i: {} for i in range(num_rois)}
        # 若希望預先初始化某些類別（例如 person, dog, cat），可取消下列註解
        # for i in range(num_rois):
        #     for cls in ['person', 'dog', 'cat']:
        #         self.detection_history_by_roi[i][cls] = False
        #         self.intruder_info_by_roi[i][cls] = {'count': 0, 'start_time': None, 'alert_sent': False}
        
        print(f"已初始化 {num_rois} 個 ROI 的偵測與警報資料結構。")


    def capture_screenshot_for_roi(self, roi, cls_name=None):
        if self.last_frame is None or cls_name is None:
            return
        x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
        roi_frame = self.last_frame[y:y+h, x:x+w]
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        folder_path = os.path.join("img", "images", cls_name)
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f"{cls_name}_{timestamp}.jpg")
        cv2.imwrite(filename, roi_frame)
        print(f"已儲存圖像至：{filename}")

    # --- 儲存影像與 YOLO 格式標記（含非同步 + 閾值 + 冷卻時間）---
    def save_yolo_dataset_frame(self, frame, detections):
        threading.Thread(target=self._save_yolo_dataset_frame_worker, args=(frame.copy(), detections)).start()

    def _save_yolo_dataset_frame_worker(self, frame, detections):
        now = datetime.now()
        if self.last_save_time and (now - self.last_save_time) < self.save_cooldown:
            return

        filtered_detections = [d for d in detections if d[5] >= self.confidence_threshold]
        if not filtered_detections:
            return

        self.last_save_time = now
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")

        height, width, _ = frame.shape

        # 為每個類別建立對應圖檔與標註檔
        for x1, y1, x2, y2, cls_name, conf in filtered_detections:
            if cls_name not in self.yolo_model.names.values():
                continue
            class_id = list(self.yolo_model.names.values()).index(cls_name)

            # 建立類別資料夾
            img_dir = os.path.join("dataset", "images", "train", cls_name)
            label_dir = os.path.join("dataset", "labels", "train", cls_name)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            filename = f"yolo_{cls_name}_{timestamp}"
            img_file = os.path.join(img_dir, f"{filename}.jpg")
            label_file = os.path.join(label_dir, f"{filename}.txt")

            # 儲存影像（同一張圖可多次儲存，但不同類別）
            cv2.imwrite(img_file, frame)

            # 儲存標註
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            with open(label_file, "w") as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            print(f"[✓] 儲存 {cls_name} 圖像至 {img_file}")

    def update_frame(self, frame):
        self.last_frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # 同時更新兩個 widget
        self.video_monitor.setPixmap(QPixmap.fromImage(q_image))
        self.video_roi.setPixmap(QPixmap.fromImage(q_image))

        self.frame_counter += 1
        # 每 detect_interval 幀送一幀給推論執行緒
        if self.frame_counter % self.detect_interval == 0 and self.yolo_model and self.inference_thread:
            # 呼叫推論執行緒（透過槽呼叫）
            self.inference_thread.run_inference(frame)

    @pyqtSlot(object)
    # ✅ 修補 update_detections()：目標離開畫面時自動清除框線

    def update_detections(self, results):
        # ✅ 若沒有偵測結果，自動清除殘留框
        if results is None or len(results[0].boxes) == 0:
            self.video_monitor.detections = []
            self.video_monitor.update()
            return

        boxes = results[0].boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                cls_name = self.yolo_model.names[cls_id]
                if conf >= self.confidence_threshold and cls_name in self.selected_classes:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    detections.append((x1, y1, x2, y2, cls_name, conf))

        # ✅ 更新畫面框線
        self.video_monitor.detections = detections  
        self.video_monitor.update()

        if detections:
                self.save_yolo_dataset_frame(self.last_frame, detections)

        # ✅ 進行警報與 ROI 邏輯判斷
        self.update_intruder_info()

        any_detection = False
        for roi_idx, info_dict in self.intruder_info_by_roi.items():
            for cat, info in info_dict.items():
                if info['count'] > 0:
                    any_detection = True
                    break
            if any_detection:
                break

        self.video_monitor.show_alert = any_detection
        self.video_monitor.update()
        # self.handle_recording(self.last_frame)

    def is_point_inside_roi(self, x, y, roi):
        return roi.left() <= x <= roi.right() and roi.top() <= y <= roi.bottom()

    def update_intruder_info(self):
        # 用 ROI 畫面來當作標準 ROI 座標來源
        roi_rects = self.video_roi.roi_rects
        if not roi_rects:
            return
        now = datetime.now()
        num_rois = len(roi_rects)
        
        # 確保每個 ROI 的資料結構已初始化
        for roi_idx in range(num_rois):
            if roi_idx not in self.tracked_objects_by_roi:
                self.tracked_objects_by_roi[roi_idx] = {}
            if roi_idx not in self.intruder_info_by_roi:
                self.intruder_info_by_roi[roi_idx] = {}
        
        # 針對每筆偵測結果，計算物件中心點，並依據每個 ROI 更新（僅根據進入與離開事件更新計數）
        for (x1, y1, x2, y2, cls_name, conf) in self.video_monitor.detections:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            for roi_idx, roi in enumerate(roi_rects):
                detection_inside = self.is_point_inside_roi(center[0], center[1], roi)
                # 初始化該 ROI 下此類別的記錄與入侵資訊
                if cls_name not in self.tracked_objects_by_roi[roi_idx]:
                    self.tracked_objects_by_roi[roi_idx][cls_name] = []
                if cls_name not in self.intruder_info_by_roi[roi_idx]:
                    self.intruder_info_by_roi[roi_idx][cls_name] = {'count': 0, 'start_time': None, 'alert_sent': False}
                
                records = self.tracked_objects_by_roi[roi_idx][cls_name]
                if not records:
                    # 若無記錄，且偵測在 ROI 內，視為進入事件
                    if detection_inside:
                        new_record = {
                            'center': center,
                            'start_time': now,
                            'last_seen': now,
                            'alert_sent': False,
                            'inside': True
                        }
                        records.append(new_record)
                        self.intruder_info_by_roi[roi_idx][cls_name]['count'] += 1
                        self.intruder_info_by_roi[roi_idx][cls_name]['start_time'] = now
                        # 當物件首次進入 ROI 時，截圖該 ROI 畫面
                        self.capture_screenshot_for_roi(roi, cls_name)
                    # 如果不在 ROI 外則不做更新
                else:
                    matched = False
                    for record in records:
                        # 使用中心點距離判斷是否為同一物件
                        distance = np.linalg.norm(np.array(center) - np.array(record['center']))
                        if distance < self.match_threshold:
                            matched = True
                            # 若記錄原本不在 ROI 內，而現在偵測到在內，則視為進入事件
                            if detection_inside and not record.get('inside', False):
                                record['inside'] = True
                                record['start_time'] = now
                                self.intruder_info_by_roi[roi_idx][cls_name]['count'] += 1
                                self.intruder_info_by_roi[roi_idx][cls_name]['start_time'] = now
                                # 進入 ROI 時也可選擇截圖（根據需求，只截圖第一次進入即可）
                                self.capture_screenshot_for_roi(roi, cls_name)
                            # 若記錄原本在內，但現在偵測到不在內，則視為離開事件
                            elif (not detection_inside) and record.get('inside', False):
                                record['inside'] = False
                                self.intruder_info_by_roi[roi_idx][cls_name]['count'] = max(
                                    0, self.intruder_info_by_roi[roi_idx][cls_name]['count'] - 1)
                                if self.intruder_info_by_roi[roi_idx][cls_name]['count'] == 0:
                                    self.intruder_info_by_roi[roi_idx][cls_name]['start_time'] = None
                            # 更新中心點與最後偵測時間
                            record['center'] = center
                            record['last_seen'] = now
                            break
                    if not matched:
                        # 如果未匹配到記錄且偵測在 ROI 內，視為新物件進入
                        if detection_inside:
                            new_record = {
                                'center': center,
                                'start_time': now,
                                'last_seen': now,
                                'alert_sent': False,
                                'inside': True
                            }
                            records.append(new_record)
                            self.intruder_info_by_roi[roi_idx][cls_name]['count'] += 1
                            self.intruder_info_by_roi[roi_idx][cls_name]['start_time'] = now
                            self.capture_screenshot_for_roi(roi, cls_name)
                        # 若偵測在 ROI 外則不作變更

        # 更新 ROI 輪廓顏色：若任一 ROI 中有入侵者（count > 0），設為紅色；否則設為綠色
        any_detection = False
        for roi_idx, info_dict in self.intruder_info_by_roi.items():
            for info in info_dict.values():
                if info['count'] > 0:
                    any_detection = True
                    break
            if any_detection:
                break
        self.video_monitor.color = QColor(255, 0, 0) if any_detection else QColor(0, 255, 0)
        
        # 分別檢查每個 ROI 的警報狀態
        for roi_idx in range(num_rois):
            self.check_alerts(roi_idx)
        
        self.video_monitor.intruder_info = self.intruder_info_by_roi




    def is_crossing_boundary(self, center_x, center_y, roi):
        """檢查物件中心點是否落在 ROI 區域內"""
        return roi.left() <= center_x <= roi.right() and roi.top() <= center_y <= roi.bottom()
    
    def check_all_alerts(self):
        num_rois = len(self.video_roi.roi_rects)
        for roi_idx in range(num_rois):
            self.check_alerts(roi_idx)

    def check_alerts(self, roi_idx):
        now = datetime.now()
        if roi_idx not in self.intruder_info_by_roi:
            return
        for cls_name, info in self.intruder_info_by_roi[roi_idx].items():
            if info['count'] > 0:
                if info['start_time'] is None:
                    info['start_time'] = now
                elapsed = now - info['start_time']
                if elapsed >= timedelta(seconds=self.alert_threshold_seconds) and not info['alert_sent']:
                    self.trigger_alert(roi_idx, cls_name, info['count'])
                    info['alert_sent'] = True
                elif elapsed < timedelta(seconds=self.alert_threshold_seconds) and info['alert_sent']:
                    info['alert_sent'] = False

    def trigger_alert(self, roi_idx, cls_name, count):
        container_label = f"貨櫃{roi_idx+1}"
        alert_time = datetime.now()
        print(f"[警告] {container_label} 中類別 {cls_name} 停留超過 {self.alert_threshold_seconds} 秒，數量: {count}。")
        # 播放警報聲、發送郵件、控制硬體
        if self.player.state() != QMediaPlayer.PlayingState:
            self.player.play()
        # 呼叫非同步發送郵件函式
        self.send_alert_email_async(cls_name, count, container_label, alert_time)
        self.turn_on_relay()

    def send_alert_email_async(self, cat, count, container_number, alert_time):
        def _worker():
            self.send_alert_email(cat, count, container_number, alert_time)
        thread = threading.Thread(target=_worker)
        thread.start()

    def send_alert_email(self, cat, count, container_number, alert_time):
        # 格式化入侵時間
        alert_time_str = alert_time.strftime("%Y/%m/%d %H:%M:%S")
        print(f"[警告] {container_number} 中類別 {cat} 停留超過 {self.alert_threshold_seconds} 秒，數量: {count}，時間: {alert_time_str}。")
        
        sender_email = "wwe99008@gmail.com"
        receiver_email = self.recipient_email or "default@example.com"
        app_password = "vssj jpma tkkw ctth"
        subject = "ROI 入侵警報"
        body = (
            f"警報事件通報\n"
            f"入侵時間：{alert_time_str}\n"
            f"入侵物體類別：{cat}\n"
            f"生物檢測：{cat}（數量：{count}）\n"
            f"貨櫃編號：{container_number}\n"
            "請立即確認現場狀況。"
        )
        
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("警報信件已寄出。")


    def connect_to_arduino(self):
        """自動連接到 Arduino 串口"""
        try:
            # 列出所有可用的 COM 埠
            ports = list(serial.tools.list_ports.comports())
            com_port = None
            
            # 打印所有串口的描述，查看是否有 Arduino
            for port in ports:
                print(f"Port: {port.device}, Description: {port.description}")

            # 查找 Arduino 所在的 COM 埠
            for port in ports:
                if 'USB-SERIAL CH340' in port.description:  # 查找 Arduino 的串口描述
                    com_port = port.device
                    break

            if com_port is not None:
                # 成功找到 Arduino 的 COM 埠，連接到該埠
                self.ser = serial.Serial(com_port, 9600, timeout=1)
                print(f"已成功連接到 Arduino，串口: {com_port}")
            else:
                print("未能找到 Arduino 的 COM 埠")
        except Exception as e:
            print(f"連接 Arduino 失敗：{e}")

    def turn_on_relay(self):
        """開啟繼電器"""
        if self.ser is not None:
            self.ser.write(b'0')  # 傳送 '1' 開啟繼電器
            print("繼電器已開啟")
        else:
            print("串口未連接") 

    def turn_off_relay(self):
        """關閉繼電器"""
        if self.ser is not None:
            self.ser.write(b'1')  # 傳送 '0' 關閉繼電器
            print("繼電器已關閉")
        else:
            print("串口未連接")

    def closeRelayEvent(self, event):
        """關閉應用程式時關閉串口"""
        if self.ser is not None:
            self.ser.close()
            print("串口已關閉")
        event.accept()

    #         
    # # 例如在 MainWindow 中修改您的 send_alert_line() 函式：
    # def send_alert_line(self, cat, count):
    #     # 建議也一律使用 UTF-8 處理字串
    #     message = f"[Warning] Category {cat} in ROI has been present for over 10 seconds, count: {count}. Please check the scene!"

    #     # 請替換為您自己的 Channel Access Token 與目標用戶 ID
    #     channel_access_token = "CefxTuO71Gd9VzfqjtDy4bpJKPzyOKLJwZHmVbzR4nnyJhoUngZAVzgQ12I4rixl88rGxOPAaq6FTuNl8C9VFEpIlNE/EuKEdfDFDXgKLUFoyfGbeAP1o/pDqe90Ci5gh6mqF/tU6N3D7+uip5j+gdB04t89/1O/w1cDnyilFU="
    #     to_user_id = "14hunter"
    #     send_line_message(message, channel_access_token, to_user_id)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.player.stop()

    def toggle_recording(self):
        if self.record_button.isChecked():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        now = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        filename = f"record_{now}.mov"
        filepath = os.path.join("recordings", filename)
        os.makedirs("recordings", exist_ok=True)

        if self.last_frame is None:
            print("尚未收到影像")
            return
        height, width, _ = self.last_frame.shape
        print(f"開始錄影，影像尺寸：{width}x{height}")

        self.video_writer = cv2.VideoWriter(
            filepath,
            cv2.VideoWriter_fourcc(*'XVID'),  # 或 'mp4v' for mp4
            self.record_fps,
            (width, height)
        )
        self.is_recording = True
        # self.show_status(f"🎬 錄影中... 儲存為 {filename}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            # self.show_status("⏹️ 錄影已停止")
        self.is_recording = False
    
    def handle_recording(self, frame):
        if self.is_recording and self.video_writer and frame is not None:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('攝影機偵測程式')
    window.show()
    sys.exit(app.exec_())