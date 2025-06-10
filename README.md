# 🛡️ Electronic-Fence (電子圍籬監控系統)

> Real-time, multi-ROI intrusion detection with YOLOv8 & PyQt5  
> Version `v1.0-handover` | Last update 2025-06-10

![screenshot](docs/assets/monitor_ui.png)

---

## Table of Contents
1. [Project Motivation](#project-motivation)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Quick Start](#quick-start)
   
---

## Project Motivation
貨櫃與戶外邊界環境常有 **人員／動物入侵** 風險。本專案提供一套 **即時影像監控 + 多 ROI 入侵警示** 解決方案，  
透過 **Ultralytics YOLOv8** 模型判斷物件「中心點」是否落於 ROI，並在停留超過閾值時觸發 **聲光警示、LINE 通報** 等多重告警，適用於工地、農場、倉儲等場域。

---

## Features
- **Multi-ROI**：GUI 手繪多邊形 / 矩形 ROI，中心點才計入。
- **Per-class dwell timer**：同類目目標各自計時，逾時觸發告警。
- **Alerts**  
  - 本地 mp3 警報 (QMediaPlayer)  
  - USB Relay → Siren / Beacon  
  - LINE Messaging API & E-mail
- **User-friendly PyQt5 GUI**：即時影像、ROI 編輯、模型熱載入、錄影/截圖。
- **Config-as-JSON**：所有參數熱更新並落檔 (`config/*.json`)。
- **Lightweight deployment**：單一 EXE (PyInstaller)；模型以 Git LFS 管理。
- **Extensible**：模組化架構，易於插入追蹤演算法、Docker 化、Grafana 監控等。

---

## Architecture
```text
RTSP Camera → FFMPEGStreamThread ─┐
                                  ▼
                        InferenceThread (YOLOv8)
                                  ▼
             ROI Timer & Alert Manager (Qt signals)
                                  ▼
        ┌───────────────┬───────────────┬───────────────┐
        │ Sound (mp3)   │  Relay (USB)  │  LINE / E-mail│
        └───────────────┴───────────────┴───────────────┘
GUI (PyQt5)  ←───────────── Frame & Meta ───────────────┘
```

---

## Quick Star
```
# 1 / Clone repository  (with Git LFS)
git clone --recurse-submodules https://github.com/nicelala/electronic-fence.git
cd electronic-fence
git lfs pull          # 下載 YOLOv8 權重

# 2 / Create & activate virtualenv
python -m venv venv
venv\\Scripts\\activate   # Linux: source venv/bin/activate

# 3 / Install dependencies
pip install -r requirements.txt

# 4 / Copy config templates
cp config/settings.template.json   config/settings.json
cp config/roi_settings.template.json  config/roi_settings.json
# → 填入 RTSP、Line Token、Email 等參數

# 5 / Run
python _ipcam_ROI_YOLO_classesAnnotation_0416.py
```
