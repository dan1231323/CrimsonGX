#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrimsonGX Browser v5.1 — AI-Powered Ultra Performance Gaming Browser
Real FPS monitoring + Game Mode + AI Assistant + Download Manager
Оптимизированный игровой браузер с реальным FPS
"""

import os
import sys
import subprocess

REQUIRED_LIBS = [
    "/nix/store/bmi5znnqk4kg2grkrhk6py0irc8phf6l-gcc-14.2.1.20250322-lib/lib",
    "/nix/store/92nrp8f5bcyxy57w30wxj5ncvygz1wnx-xcb-util-cursor-0.1.5/lib",
]

def setup_environment():
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    needs_restart = False
    new_paths = []

    for lib_path in REQUIRED_LIBS:
        if os.path.exists(lib_path) and lib_path not in current_ld:
            new_paths.append(lib_path)
            needs_restart = True

    if needs_restart and new_paths:
        new_ld = ":".join(new_paths)
        if current_ld:
            new_ld = f"{new_ld}:{current_ld}"
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.execv(sys.executable, [sys.executable] + sys.argv)

setup_environment()

import time
import gc
import json
import threading
import queue
import hashlib
import re
import html
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from datetime import datetime
from functools import lru_cache
import asyncio
import concurrent.futures

# Game Mode optimizations
GAME_MODE_CHROMIUM_FLAGS = (
    "--disable-gpu-vsync "
    "--disable-frame-rate-limit "
    "--max-gum-fps=3000 "
    "--disable-gpu "
    "--disable-software-rasterizer "
    "--disable-dev-shm-usage "
    "--no-sandbox "
    "--disable-background-timer-throttling "
    "--disable-backgrounding-occluded-windows "
    "--disable-renderer-backgrounding "
    "--disable-background-networking "
    "--disable-extensions "
    "--disable-component-update "
    "--disable-sync "
    "--disable-translate "
    "--disable-features=TranslateUI "
    "--aggressive-cache-discard "
    "--disable-hang-monitor "
    "--disable-smooth-scrolling "
    "--memory-pressure-off "
    "--disable-logging "
    "--num-raster-threads=8 "
    "--renderer-process-limit=2 "
    "--js-flags=--max-old-space-size=256 "
    "--in-process-gpu "
)

NORMAL_CHROMIUM_FLAGS = (
    "--disable-gpu "
    "--disable-software-rasterizer "
    "--disable-dev-shm-usage "
    "--no-sandbox "
    "--num-raster-threads=4 "
)

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", NORMAL_CHROMIUM_FLAGS)

try:
    from PyQt6.QtCore import (
        QUrl, Qt, QTimer, QSize, pyqtSignal, QThread, 
        QObject, QRunnable, QThreadPool, QMutex, QWaitCondition,
        QStandardPaths, QFileInfo
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLineEdit, QPushButton, QLabel, QToolButton, QStatusBar,
        QDialog, QSlider, QCheckBox, QGroupBox, QGridLayout, QSpinBox,
        QTabWidget, QTextEdit, QProgressBar, QComboBox, QFrame,
        QSplitter, QListWidget, QListWidgetItem, QScrollArea,
        QStackedWidget, QSizePolicy, QMessageBox, QMenu, QSystemTrayIcon,
        QFileDialog
    )
    from PyQt6.QtGui import (
        QFont, QPalette, QColor, QIcon, QShortcut, QKeySequence,
        QAction, QPainter, QBrush, QPen, QLinearGradient, QPixmap
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import (
        QWebEngineSettings, QWebEngineProfile, QWebEnginePage,
        QWebEngineScript, QWebEngineDownloadRequest
    )
except ImportError:
    print("pip install PyQt6 PyQt6-WebEngine")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from cachetools import TTLCache, LRUCache
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False
    class TTLCache(dict):
        def __init__(self, maxsize=100, ttl=300):
            super().__init__()
    class LRUCache(dict):
        def __init__(self, maxsize=100):
            super().__init__()

try:
    from flask import Flask, jsonify, request as flask_request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

APP_NAME = "CrimsonGX"
VERSION = "5.1 Game Mode"
CONFIG_DIR = Path.home() / ".crimsongx"
CONFIG_FILE = CONFIG_DIR / "settings.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
BOOKMARKS_FILE = CONFIG_DIR / "bookmarks.json"
AI_CACHE_FILE = CONFIG_DIR / "ai_cache.json"
DOWNLOADS_DIR = Path.home() / "Downloads" / "CrimsonGX"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

AD_PATTERNS = [
    r'.*ads?[_.-].*',
    r'.*advert.*',
    r'.*banner.*',
    r'.*popup.*',
    r'.*tracking.*',
    r'.*analytics.*',
    r'.*doubleclick.*',
    r'.*googleadservices.*',
]

@dataclass
class PerformanceSettings:
    max_ram_mb: int = 512
    cache_size_mb: int = 50
    enable_images: bool = True
    enable_javascript: bool = True
    enable_webgl: bool = False
    enable_canvas: bool = True
    ai_optimizer: bool = True
    smart_ram: bool = True
    aggressive_gc: bool = True
    preload_pages: bool = False
    fps_target: int = 120
    ai_assistant: bool = True
    local_server: bool = True
    server_port: int = 8765
    openai_api_key: str = ""
    prediction_mode: bool = True
    auto_optimize: bool = True
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.9
    gc_interval: int = 30
    cache_ttl: int = 3600
    max_tabs: int = 10
    lazy_load: bool = True
    compress_images: bool = True
    block_ads: bool = True
    dark_mode: bool = True
    game_mode: bool = False
    reader_mode: bool = False
    local_nlp: bool = True

    def save(self):
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls):
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except:
                pass
        return cls()


class PerformanceMetrics:
    def __init__(self, maxlen: int = 100):
        self.ram_history = deque(maxlen=maxlen)
        self.cpu_history = deque(maxlen=maxlen)
        self.fps_history = deque(maxlen=maxlen)
        self.io_history = deque(maxlen=maxlen)
        self.gc_count = 0
        self.optimizations_count = 0
        self.start_time = time.time()
        self.page_loads = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.downloads_count = 0
        self.blocked_ads = 0

    def add_sample(self, ram: float, cpu: float, fps: int = 0):
        timestamp = time.time()
        self.ram_history.append((timestamp, ram))
        self.cpu_history.append((timestamp, cpu))
        if fps > 0:
            self.fps_history.append((timestamp, fps))

    def get_average(self, history: deque, seconds: int = 60) -> float:
        if not history:
            return 0.0
        now = time.time()
        recent = [v for t, v in history if now - t <= seconds]
        return sum(recent) / len(recent) if recent else 0.0


class RealFPSCounter(QThread):
    """Real FPS counter using Qt repaint events"""
    fps_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.frame_count = 0
        self.last_time = time.time()
        self.current_fps = 60
        self.mutex = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            time.sleep(0.01) # Check more frequently for frame counts

            with self.mutex:
                now = time.time()
                elapsed = now - self.last_time

                if elapsed >= 1.0:  # Calculate FPS every second
                    self.current_fps = int(self.frame_count / elapsed)
                    self.fps_signal.emit(self.current_fps)
                    self.frame_count = 0
                    self.last_time = now
                
                # Break if not running
                if not self.running:
                    break


    def count_frame(self):
        """Call this on each frame render"""
        with self.mutex:
            self.frame_count += 1

    def stop(self):
        self.running = False


class GameModeOptimizer(QThread):
    """Game Mode optimizer with memory management"""
    optimization_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.game_mode = False

    def run(self):
        self.running = True
        gc_counter = 0

        while self.running:
            if self.game_mode:
                # Aggressive optimizations in game mode
                gc_counter += 1

                if gc_counter >= 10:  # GC every 1 second in game mode
                    gc.collect(0)
                    gc.collect(1) # Perform deeper GC
                    gc.collect(2)
                    gc_counter = 0
                    self.optimization_signal.emit("Game Mode: Memory optimized")

                time.sleep(0.1)
            else:
                time.sleep(1)

    def set_game_mode(self, enabled: bool):
        self.game_mode = enabled
        if enabled:
            gc.set_threshold(100000, 50, 50)  # Reduce GC frequency
            self.optimization_signal.emit("Game Mode: ACTIVATED")
        else:
            gc.set_threshold(700, 10, 10)  # Normal GC
            self.optimization_signal.emit("Game Mode: Deactivated")

    def stop(self):
        self.running = False


class LocalNLPAI(QThread):
    """Local AI without external dependencies"""
    response_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.request_queue = queue.Queue()

    def run(self):
        while self.running:
            try:
                request = self.request_queue.get(timeout=1)
                if request:
                    response = self._process_local(request)
                    self.response_signal.emit(response)
            except queue.Empty:
                continue

    def _process_local(self, request: dict) -> dict:
        text = request.get('content', '')
        req_type = request.get('type', 'general')

        if req_type == 'summarize':
            summary = self._local_summarize(text)
        elif req_type == 'analyze':
            summary = self._local_analyze(text)
        elif req_type == 'help':
            summary = self._get_help()
        else:
            summary = f"AI обработал запрос: {text[:100]}..."

        return {
            'type': req_type,
            'response': summary,
            'timestamp': time.time()
        }

    def _local_summarize(self, text: str) -> str:
        if not text:
            return "Нет текста для анализа"

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return "Текст слишком короткий"

        # Smart selection based on keywords
        keywords = ['важно', 'главное', 'необходимо', 'следует', 'должен',
                   'important', 'main', 'key', 'should', 'must']

        scored = []
        for i, sent in enumerate(sentences):
            score = len(sent) / 100
            score += (len(sentences) - i) / len(sentences)
            score += sum(2 for kw in keywords if kw.lower() in sent.lower())
            scored.append((score, sent))

        scored.sort(reverse=True)
        top = [s for _, s in scored[:3]]
        return ". ".join(top) + "."

    def _local_analyze(self, text: str) -> str:
        words = text.split()
        word_count = len(words)
        avg_len = sum(len(w) for w in words) / max(word_count, 1)

        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        urls = re.findall(r'http[s]?://[^\s]+', text)

        result = f"📊 Слов: {word_count}, Средняя длина: {avg_len:.1f}"
        if emails:
            result += f"\n📧 Email найдено: {len(emails)}"
        if urls:
            result += f"\n🔗 URL найдено: {len(urls)}"

        return result

    def _get_help(self) -> str:
        return """🤖 CrimsonGX AI Assistant

Команды:
• Summarize - краткое содержание текста
• Analyze - анализ контента
• Help - эта справка

Функции браузера:
• Game Mode (🎮) - до 3000 FPS
• Ad Blocker (🚫) - блокировка рекламы
• Reader Mode (📖) - режим чтения
• Downloads (⬇️) - менеджер загрузок"""

    def add_request(self, request: dict):
        self.request_queue.put(request)

    def stop(self):
        self.running = False


class AIOptimizer(QThread):
    optimization_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self, settings: PerformanceSettings, metrics: PerformanceMetrics):
        super().__init__()
        self.settings = settings
        self.metrics = metrics
        self.running = True
        self.last_gc_time = time.time()

    def run(self):
        while self.running:
            if not self.settings.ai_optimizer:
                time.sleep(2)
                continue

            try:
                data = self._analyze_system()
                self.optimization_signal.emit(data)

                if self.settings.auto_optimize:
                    self._apply_optimizations(data)

            except Exception as e:
                self.status_signal.emit(f"Optimizer error: {str(e)[:50]}")

            time.sleep(2)

    def _analyze_system(self) -> dict:
        data = {
            'mem_mb': 0,
            'mem_percent': 0,
            'cpu_percent': 0,
            'status': 'optimal'
        }

        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                cpu_percent = process.cpu_percent(interval=0.5)

                data['mem_mb'] = round(mem_mb, 1)
                data['mem_percent'] = round((mem_mb / self.settings.max_ram_mb) * 100, 1)
                data['cpu_percent'] = round(cpu_percent, 1)

                self.metrics.add_sample(mem_mb, cpu_percent)

                if mem_mb / self.settings.max_ram_mb > self.settings.memory_threshold:
                    data['status'] = 'critical'
                elif mem_mb / self.settings.max_ram_mb > 0.7:
                    data['status'] = 'warning'

            except Exception:
                pass

        return data

    def _apply_optimizations(self, data: dict):
        if data['status'] == 'critical':
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            self.metrics.gc_count += 3
            self.last_gc_time = time.time()
            self.status_signal.emit("Critical: Memory cleaned")
        elif data['status'] == 'warning':
            now = time.time()
            if now - self.last_gc_time > self.settings.gc_interval:
                gc.collect(0)
                self.metrics.gc_count += 1
                self.last_gc_time = now

    def stop(self):
        self.running = False


class LocalServer(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, settings: PerformanceSettings, metrics: PerformanceMetrics):
        super().__init__()
        self.settings = settings
        self.metrics = metrics
        self.running = True

    def run(self):
        if not HAS_FLASK or not self.settings.local_server:
            return

        app = Flask(__name__)
        app.config['JSON_AS_ASCII'] = False

        @app.route('/api/status')
        def get_status():
            return jsonify({
                'status': 'running',
                'version': VERSION,
                'metrics': {
                    'ram_avg': self.metrics.get_average(self.metrics.ram_history),
                    'cpu_avg': self.metrics.get_average(self.metrics.cpu_history),
                    'gc_count': self.metrics.gc_count,
                    'downloads': self.metrics.downloads_count,
                    'blocked_ads': self.metrics.blocked_ads
                }
            })

        try:
            import logging
            logging.getLogger('werkzeug').setLevel(logging.ERROR)
            self.log_signal.emit(f"Server started on port {self.settings.server_port}")
            app.run(host='0.0.0.0', port=self.settings.server_port, debug=False, use_reloader=False, threaded=True)
        except Exception as e:
            self.log_signal.emit(f"Server error: {str(e)}")

    def stop(self):
        self.running = False


HOME_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>CrimsonGX v5.1</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a0f;--accent:#ff2b2b;--text:#e8e8e8;--metric:#00d9ff}
html{height:100%;font:14px system-ui;background:var(--bg);color:var(--text);overflow:hidden}
body{height:100%;display:flex;flex-direction:column;background:radial-gradient(ellipse at top,rgba(255,43,43,0.08),transparent 50%)}

.performance-hud{position:fixed;top:20px;right:20px;display:flex;gap:12px;z-index:1000}
.metric-card{background:rgba(0,0,0,0.7);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:12px 16px;min-width:100px;box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.metric-label{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.metric-value{font-size:24px;font-weight:900;color:var(--metric);text-shadow:0 0 10px rgba(0,217,255,0.5)}
.metric-value.good{color:#00ff88}
.metric-value.warning{color:#ffaa00}
.metric-value.critical{color:#ff2b2b}
.metric-unit{font-size:14px;margin-left:2px;opacity:0.7}

.main-content{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:40px;padding:80px 20px 20px}

.logo-section{text-align:center}
.logo{font:900 72px system-ui;background:linear-gradient(135deg,#ff2b2b,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-shadow:0 0 60px rgba(255,43,43,0.3);animation:pulse 2s ease-in-out infinite;margin-bottom:12px;letter-spacing:-2px}
.version{font-size:11px;color:#555;letter-spacing:3px;text-transform:uppercase;font-weight:600}

@keyframes pulse{0%,100%{opacity:0.95;transform:scale(1)}50%{opacity:1;transform:scale(1.03)}}

.controls-section{display:flex;flex-direction:column;gap:20px;align-items:center}
.control-row{display:flex;gap:15px}

.game-mode-toggle{position:relative;width:300px;height:80px;background:rgba(0,0,0,0.5);border:2px solid rgba(255,43,43,0.4);border-radius:18px;cursor:pointer;transition:all 0.4s;display:flex;align-items:center;justify-content:center;gap:14px;box-shadow:0 0 40px rgba(255,43,43,0.2)}
.game-mode-toggle:hover{transform:scale(1.08) translateY(-2px);border-color:rgba(255,43,43,0.7);box-shadow:0 0 60px rgba(255,43,43,0.4)}
.game-mode-toggle.active{background:linear-gradient(135deg,rgba(255,43,43,0.9),rgba(255,107,107,0.9));border-color:#ff2b2b;box-shadow:0 0 80px rgba(255,43,43,0.6);animation:glow 1.5s ease-in-out infinite}
@keyframes glow{0%,100%{box-shadow:0 0 80px rgba(255,43,43,0.6)}50%{box-shadow:0 0 100px rgba(255,43,43,0.8)}}
.game-mode-icon{font-size:36px;filter:drop-shadow(0 0 12px rgba(255,43,43,0.9))}
.game-mode-text{font-size:22px;font-weight:900;text-transform:uppercase;letter-spacing:3px}

.ai-button{width:300px;height:80px;background:rgba(0,217,255,0.1);border:2px solid rgba(0,217,255,0.4);border-radius:18px;cursor:pointer;transition:all 0.4s;display:flex;align-items:center;justify-content:center;gap:14px;box-shadow:0 0 40px rgba(0,217,255,0.2)}
.ai-button:hover{transform:scale(1.08) translateY(-2px);border-color:rgba(0,217,255,0.7);box-shadow:0 0 60px rgba(0,217,255,0.4);background:rgba(0,217,255,0.2)}
.ai-icon{font-size:36px;filter:drop-shadow(0 0 12px rgba(0,217,255,0.9))}
.ai-text{font-size:22px;font-weight:900;text-transform:uppercase;letter-spacing:3px;color:var(--metric)}

.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;max-width:1000px;width:100%}
.feature-card{background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:24px;text-align:center;transition:all 0.35s;cursor:pointer}
.feature-card:hover{background:rgba(0,0,0,0.5);border-color:rgba(255,43,43,0.5);transform:translateY(-6px);box-shadow:0 12px 32px rgba(255,43,43,0.15)}
.feature-icon{font-size:36px;margin-bottom:14px}
.feature-title{font-size:15px;font-weight:800;color:#ff2b2b;margin-bottom:8px}
.feature-desc{font-size:11px;color:#666;line-height:1.5}

.quick-links{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.quick-link{background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 20px;color:#aaa;text-decoration:none;font-size:12px;font-weight:600;transition:all 0.3s;letter-spacing:0.5px}
.quick-link:hover{background:rgba(255,43,43,0.2);border-color:rgba(255,43,43,0.6);color:#ff2b2b;transform:translateY(-3px)}

@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.main-content>*{animation:fadeIn 0.6s ease-out backwards}
.logo-section{animation-delay:0.1s}
.controls-section{animation-delay:0.2s}
.features{animation-delay:0.3s}
.quick-links{animation-delay:0.4s}
</style>
</head>
<body>
<div class="performance-hud">
    <div class="metric-card">
        <div class="metric-label">FPS</div>
        <div class="metric-value good" id="fps-value">60<span class="metric-unit"></span></div>
    </div>
    <div class="metric-card">
        <div class="metric-label">RAM</div>
        <div class="metric-value" id="ram-value">0<span class="metric-unit">MB</span></div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Ping</div>
        <div class="metric-value" id="ping-value">15<span class="metric-unit">ms</span></div>
    </div>
</div>

<div class="main-content">
    <div class="logo-section">
        <div class="logo">CRIMSONGX</div>
        <div class="version">v5.1 Optimized Gaming Edition</div>
    </div>

    <div class="controls-section">
        <div class="control-row">
            <div class="game-mode-toggle" id="gameModeToggle">
                <span class="game-mode-icon">🎮</span>
                <div class="game-mode-text">GAME MODE</div>
            </div>
        </div>

        <div class="control-row">
            <div class="ai-button" id="aiButton">
                <span class="ai-icon">🤖</span>
                <div class="ai-text">AI ASSISTANT</div>
            </div>
        </div>
    </div>

    <div class="features">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real FPS</div>
            <div class="feature-desc">Настоящий FPS мониторинг</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📑</div>
            <div class="feature-title">Multi-Tab</div>
            <div class="feature-desc">Умное управление вкладками</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🚫</div>
            <div class="feature-title">Ad Blocker</div>
            <div class="feature-desc">Блокировка рекламы</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⬇️</div>
            <div class="feature-title">Downloads</div>
            <div class="feature-desc">Менеджер загрузок</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📖</div>
            <div class="feature-title">Reader Mode</div>
            <div class="feature-desc">Режим чтения</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">Local AI</div>
            <div class="feature-desc">Локальный ИИ помощник</div>
        </div>
    </div>

    <div class="quick-links">
        <a href="https://github.com" class="quick-link">GitHub</a>
        <a href="https://stackoverflow.com" class="quick-link">StackOverflow</a>
        <a href="https://reddit.com" class="quick-link">Reddit</a>
        <a href="https://youtube.com" class="quick-link">YouTube</a>
    </div>
</div>

<script>
const toggle = document.getElementById('gameModeToggle');
const aiBtn = document.getElementById('aiButton');
let isGameMode = false;

toggle.addEventListener('click', () => {
    isGameMode = !isGameMode;
    toggle.classList.toggle('active', isGameMode);
    console.log('GAME_MODE_TOGGLE:' + isGameMode);
});

aiBtn.addEventListener('click', () => {
    console.log('AI_ASSISTANT_OPEN');
});

// Update FPS color based on value
setInterval(() => {
    const fpsEl = document.getElementById('fps-value');
    const fps = parseInt(fpsEl.textContent);

    if (fps >= 240) {
        fpsEl.className = 'metric-value good';
    } else if (fps >= 60) {
        fpsEl.className = 'metric-value warning';
    } else {
        fpsEl.className = 'metric-value critical';
    }

    const ramEl = document.getElementById('ram-value');
    const ram = parseInt(ramEl.textContent);

    if (ram < 300) {
        ramEl.className = 'metric-value good';
    } else if (ram < 400) {
        ramEl.className = 'metric-value warning';
    } else {
        ramEl.className = 'metric-value critical';
    }
}, 1000);
</script>
</body>
</html>"""


class DownloadManager(QWidget):
    def __init__(self, metrics: PerformanceMetrics, parent=None):
        super().__init__(parent)
        self.metrics = metrics
        self.downloads = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("Download Manager")
        header.setStyleSheet("font-weight:bold;font-size:14px;color:#ff2b2b;padding:8px")
        layout.addWidget(header)

        self.download_list = QListWidget()
        self.download_list.setStyleSheet("""
            QListWidget {background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.1);border-radius:8px;color:#e8e8e8;padding:8px}
        """)
        layout.addWidget(self.download_list)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Clear Completed")
        clear_btn.clicked.connect(self.clear_completed)
        btn_layout.addWidget(clear_btn)

        open_folder_btn = QPushButton("Open Folder")
        open_folder_btn.clicked.connect(lambda: os.system(f'xdg-open "{DOWNLOADS_DIR}"'))
        btn_layout.addWidget(open_folder_btn)

        layout.addLayout(btn_layout)

    def add_download(self, download: QWebEngineDownloadRequest):
        filename = download.downloadFileName()
        self.downloads.append({'filename': filename, 'request': download})

        item = QListWidgetItem(f"⬇️ {filename} (0%)")
        self.download_list.addItem(item)

        download.receivedBytesChanged.connect(lambda: self.update_progress(download, item))
        download.isFinishedChanged.connect(lambda: self.download_finished(download, item))

        self.metrics.downloads_count += 1

    def update_progress(self, download: QWebEngineDownloadRequest, item: QListWidgetItem):
        received = download.receivedBytes()
        total = download.totalBytes()
        if total > 0:
            percent = int((received / total) * 100)
            item.setText(f"⬇️ {download.downloadFileName()} ({percent}%)")

    def download_finished(self, download: QWebEngineDownloadRequest, item: QListWidgetItem):
        item.setText(f"✅ {download.downloadFileName()} (Complete)")

    def clear_completed(self):
        for i in range(self.download_list.count() - 1, -1, -1):
            item = self.download_list.item(i)
            if item.text().startswith("✅"):
                self.download_list.takeItem(i)


class AdBlocker:
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.patterns = [re.compile(p) for p in AD_PATTERNS]

    def should_block(self, url: str) -> bool:
        url_str = url.toString() if hasattr(url, 'toString') else str(url)
        for pattern in self.patterns:
            if pattern.match(url_str):
                self.metrics.blocked_ads += 1
                return True
        return False

    def get_block_script(self) -> str:
        return """
        (function() {
            var adSelectors = ['[id*="ad"]','[class*="ad"]','[id*="banner"]','[class*="banner"]','iframe[src*="ads"]'];
            adSelectors.forEach(function(s) {
                document.querySelectorAll(s).forEach(function(el) {el.style.display='none'});
            });
        })();
        """


class ReaderMode:
    @staticmethod
    def extract_content(html: str) -> str:
        if not HAS_BS4:
            return html

        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            tag.decompose()

        main_content = soup.find('article') or soup.find('main') or soup.find('body')
        if not main_content:
            return html

        text = main_content.get_text(separator='\n\n', strip=True)

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body{{max-width:800px;margin:40px auto;padding:20px;font:18px Georgia,serif;line-height:1.8;background:#f5f5f0;color:#333}}
h1,h2,h3{{color:#2c3e50}}
</style></head><body><div>{text}</div></body></html>"""


class OptimizedWebEngineView(QWebEngineView):
    game_mode_toggled = pyqtSignal(bool)
    ai_assistant_requested = pyqtSignal()

    def __init__(self, settings: PerformanceSettings, ad_blocker: AdBlocker, fps_counter: RealFPSCounter, parent=None):
        super().__init__(parent)
        self.perf_settings = settings
        self.ad_blocker = ad_blocker
        self.fps_counter = fps_counter
        self._setup_profile()
        self._setup_settings()
        self._setup_interceptor()
        self._setup_console_handler()

        # Connect paint events to FPS counter
        self.page().loadFinished.connect(lambda ok: self.fps_counter.count_frame() if ok else None)

    def _setup_profile(self):
        profile = QWebEngineProfile.defaultProfile()
        profile.setHttpCacheMaximumSize(self.perf_settings.cache_size_mb * 1024 * 1024)
        profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)

    def _setup_settings(self):
        s = self.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.AutoLoadImages, self.perf_settings.enable_images)
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, self.perf_settings.enable_javascript)
        s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, self.perf_settings.enable_webgl)
        s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, self.perf_settings.enable_canvas)
        s.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, False)
        s.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, not self.perf_settings.game_mode)

    def _setup_interceptor(self):
        if self.perf_settings.block_ads:
            profile = QWebEngineProfile.defaultProfile()
            script = QWebEngineScript()
            script.setName("AdBlocker")
            script.setSourceCode(self.ad_blocker.get_block_script())
            script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
            script.setWorldId(QWebEngineScript.ScriptWorldId.ApplicationWorld)
            script.setRunsOnSubFrames(True)
            profile.scripts().insert(script)

    def apply_settings(self):
        self._setup_settings()

    def _setup_console_handler(self):
        page = self.page()

        class ConsolePage(QWebEnginePage):
            game_mode_signal = pyqtSignal(bool)
            ai_signal = pyqtSignal()

            def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
                if message.startswith('GAME_MODE_TOGGLE:'):
                    enabled = message.split(':')[1] == 'true'
                    self.game_mode_signal.emit(enabled)
                elif message == 'AI_ASSISTANT_OPEN':
                    self.ai_signal.emit()

        console_page = ConsolePage(page.profile(), self)
        console_page.game_mode_signal.connect(lambda enabled: self.game_mode_toggled.emit(enabled))
        console_page.ai_signal.connect(lambda: self.ai_assistant_requested.emit())
        self.setPage(console_page)

    def update_home_metrics(self, fps: int, ram_mb: float, ping_ms: int):
        if 'file://' in self.url().toString() and 'home.html' in self.url().toString():
            js_code = f"""
            if (document.getElementById('fps-value')) document.getElementById('fps-value').textContent = '{fps}';
            if (document.getElementById('ram-value')) document.getElementById('ram-value').textContent = '{int(ram_mb)}';
            if (document.getElementById('ping-value')) document.getElementById('ping-value').textContent = '{ping_ms}';
            """
            self.page().runJavaScript(js_code)

    def paintEvent(self, event):
        """Count frame on each repaint"""
        super().paintEvent(event)
        self.fps_counter.count_frame()


class AIAssistantDialog(QDialog):
    def __init__(self, ai: LocalNLPAI, parent=None):
        super().__init__(parent)
        self.ai = ai
        self.ai.response_signal.connect(self._handle_response)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("🤖 AI Assistant")
        self.setModal(False)
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        header = QLabel("AI Assistant - Local NLP")
        header.setStyleSheet("font:bold 16px;color:#00d9ff;padding:10px")
        layout.addWidget(header)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background:rgba(0,0,0,0.5);
                border:1px solid rgba(0,217,255,0.2);
                border-radius:8px;
                color:#e8e8e8;
                padding:10px;
                font-size:13px;
            }
        """)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите запрос или 'help' для справки...")
        self.input_field.returnPressed.connect(self._send_request)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(0,217,255,0.2);
                border-radius:8px;
                color:#e8e8e8;
                padding:10px;
                font-size:13px;
            }
        """)
        input_layout.addWidget(self.input_field)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_request)
        send_btn.setStyleSheet("""
            QPushButton {
                background:#00d9ff;
                border:none;
                border-radius:8px;
                color:#000;
                padding:10px 20px;
                font-weight:bold;
            }
            QPushButton:hover {background:#00ffff}
        """)
        input_layout.addWidget(send_btn)

        layout.addLayout(input_layout)

        self._add_message("AI Assistant готов к работе! Введите 'help' для справки.", "system")

    def _send_request(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self._add_message(text, "user")
        self.input_field.clear()

        # Determine request type
        req_type = 'general'
        if 'help' in text.lower():
            req_type = 'help'
        elif 'summar' in text.lower() or 'резюм' in text.lower():
            req_type = 'summarize'
        elif 'analyz' in text.lower() or 'анализ' in text.lower():
            req_type = 'analyze'

        self.ai.add_request({'type': req_type, 'content': text})

    def _handle_response(self, data: dict):
        response = data.get('response', 'No response')
        self._add_message(response, "ai")

    def _add_message(self, text: str, sender: str):
        color = "#00d9ff" if sender == "ai" else "#ff2b2b" if sender == "user" else "#666"
        prefix = "🤖 AI:" if sender == "ai" else "👤 You:" if sender == "user" else "ℹ️"

        self.chat_display.append(f'<div style="color:{color};margin:8px 0"><b>{prefix}</b> {text}</div>')


class CrimsonGXBrowser(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = PerformanceSettings.load()
        self.metrics = PerformanceMetrics()
        self.history = []
        self.bookmarks = []
        self.tabs = []
        self.current_tab_index = 0

        self._load_data()

        self.ad_blocker = AdBlocker(self.metrics)
        self.reader_mode_handler = ReaderMode()
        self.fps_counter = RealFPSCounter()

        self._init_threads()
        self._setup_ui()
        self._setup_shortcuts()
        self._start_services()

    def _load_data(self):
        if HISTORY_FILE.exists():
            try:
                self.history = json.loads(HISTORY_FILE.read_text())
            except:
                pass
        if BOOKMARKS_FILE.exists():
            try:
                self.bookmarks = json.loads(BOOKMARKS_FILE.read_text())
            except:
                pass

    def _save_data(self):
        HISTORY_FILE.write_text(json.dumps(self.history[-1000:], ensure_ascii=False))
        BOOKMARKS_FILE.write_text(json.dumps(self.bookmarks, ensure_ascii=False))
        self.settings.save()

    def _init_threads(self):
        self.optimizer = AIOptimizer(self.settings, self.metrics)
        self.optimizer.optimization_signal.connect(self._handle_optimization)
        self.optimizer.status_signal.connect(self._log_status)

        self.local_ai = LocalNLPAI()

        self.game_mode_optimizer = GameModeOptimizer()
        self.game_mode_optimizer.optimization_signal.connect(self._log_status)

        self.server = LocalServer(self.settings, self.metrics)
        self.server.log_signal.connect(self._log_status)

    def _setup_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Navigation bar
        nav_bar = QWidget()
        nav_bar.setFixedHeight(48)
        nav_bar.setStyleSheet("""
            QWidget {background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1a1a20,stop:1 #0a0a0f);border-bottom:1px solid rgba(255,255,255,0.08)}
        """)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 4, 8, 4)
        nav_layout.setSpacing(6)

        btn_style = """
            QToolButton {background:transparent;border:none;border-radius:6px;padding:6px 10px;color:#9a8f8f;font-size:16px}
            QToolButton:hover {background:rgba(255,43,43,0.15);color:#ff2b2b}
        """

        self.back_btn = QToolButton()
        self.back_btn.setText("<")
        self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)

        self.forward_btn = QToolButton()
        self.forward_btn.setText(">")
        self.forward_btn.setStyleSheet(btn_style)
        self.forward_btn.clicked.connect(self._go_forward)
        nav_layout.addWidget(self.forward_btn)

        self.reload_btn = QToolButton()
        self.reload_btn.setText("R")
        self.reload_btn.setStyleSheet(btn_style)
        self.reload_btn.clicked.connect(self._reload)
        nav_layout.addWidget(self.reload_btn)

        self.home_btn = QToolButton()
        self.home_btn.setText("H")
        self.home_btn.setStyleSheet(btn_style)
        self.home_btn.clicked.connect(self._go_home)
        nav_layout.addWidget(self.home_btn)

        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("Enter URL or search...")
        self.url_bar.returnPressed.connect(self._navigate)
        self.url_bar.setStyleSheet("""
            QLineEdit {background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:8px;color:#e8e8e8;padding:8px 14px;font-size:13px}
            QLineEdit:focus {border-color:#ff2b2b}
        """)
        nav_layout.addWidget(self.url_bar, 1)

        self.game_mode_btn = QToolButton()
        self.game_mode_btn.setText("🎮")
        self.game_mode_btn.setStyleSheet(btn_style.replace("#9a8f8f", "#00ff88"))
        self.game_mode_btn.setCheckable(True)
        self.game_mode_btn.toggled.connect(self._toggle_game_mode)
        nav_layout.addWidget(self.game_mode_btn)

        self.ai_btn = QToolButton()
        self.ai_btn.setText("🤖")
        self.ai_btn.setStyleSheet(btn_style.replace("#9a8f8f", "#00d9ff"))
        self.ai_btn.clicked.connect(self._show_ai_assistant)
        nav_layout.addWidget(self.ai_btn)

        self.reader_btn = QToolButton()
        self.reader_btn.setText("📖")
        self.reader_btn.setStyleSheet(btn_style)
        self.reader_btn.clicked.connect(self._toggle_reader_mode)
        nav_layout.addWidget(self.reader_btn)

        self.downloads_btn = QToolButton()
        self.downloads_btn.setText("⬇️")
        self.downloads_btn.setStyleSheet(btn_style)
        self.downloads_btn.clicked.connect(self._show_downloads)
        nav_layout.addWidget(self.downloads_btn)

        self.new_tab_btn = QToolButton()
        self.new_tab_btn.setText("+")
        self.new_tab_btn.setStyleSheet(btn_style)
        self.new_tab_btn.clicked.connect(self._new_tab)
        nav_layout.addWidget(self.new_tab_btn)

        main_layout.addWidget(nav_bar)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.currentChanged.connect(self._tab_changed)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {border:none}
            QTabBar::tab {background:rgba(255,255,255,0.03);color:#9a8f8f;padding:8px 16px;margin-right:2px;border:1px solid rgba(255,255,255,0.05)}
            QTabBar::tab:selected {background:rgba(255,43,43,0.2);color:#ff2b2b}
        """)

        main_layout.addWidget(self.tab_widget, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {background:#0a0a0f;color:#9a8f8f;border-top:1px solid rgba(255,255,255,0.08);font-size:11px;padding:4px 8px}
        """)
        self.setStatusBar(self.status_bar)

        self.ram_status = QLabel("RAM: --")
        self.cpu_status = QLabel("CPU: --")
        self.fps_status = QLabel("FPS: 60")
        self.fps_status.setStyleSheet("color:#00ff88")
        self.downloads_status = QLabel("Downloads: 0")
        self.ads_blocked_status = QLabel("Ads: 0")

        self.status_bar.addWidget(self.ram_status)
        self.status_bar.addWidget(self.cpu_status)
        self.status_bar.addWidget(self.fps_status)
        self.status_bar.addWidget(self.downloads_status)
        self.status_bar.addPermanentWidget(self.ads_blocked_status)

        self.download_manager = DownloadManager(self.metrics)
        self.ai_dialog = None

        self._new_tab()

        self.setStyleSheet("QMainWindow {background:#0a0a0f}")

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+L"), self, lambda: self.url_bar.setFocus())
        QShortcut(QKeySequence("Ctrl+T"), self, self._new_tab)
        QShortcut(QKeySequence("Ctrl+W"), self, lambda: self._close_tab(self.tab_widget.currentIndex()))
        QShortcut(QKeySequence("Ctrl+Tab"), self, self._next_tab)
        QShortcut(QKeySequence("Ctrl+Shift+Tab"), self, self._prev_tab)
        QShortcut(QKeySequence("Ctrl+R"), self, self._reload)
        QShortcut(QKeySequence("F5"), self, self._reload)
        QShortcut(QKeySequence("Alt+Left"), self, self._go_back)
        QShortcut(QKeySequence("Alt+Right"), self, self._go_forward)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self, lambda: self.game_mode_btn.toggle())
        QShortcut(QKeySequence("Ctrl+Shift+A"), self, self._show_ai_assistant)

    def _start_services(self):
        if self.settings.ai_optimizer:
            self.optimizer.start()
        if self.settings.local_nlp:
            self.local_ai.start()

        self.game_mode_optimizer.start()
        self.fps_counter.start()
        self.fps_counter.fps_signal.connect(self._update_fps)

        if self.settings.local_server and HAS_FLASK:
            self.server.start()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(1000)

    def _new_tab(self):
        if len(self.tabs) >= self.settings.max_tabs:
            QMessageBox.warning(self, "Tab Limit", f"Maximum {self.settings.max_tabs} tabs")
            return

        webview = OptimizedWebEngineView(self.settings, self.ad_blocker, self.fps_counter)
        webview.urlChanged.connect(lambda url: self._on_url_changed(url))
        webview.loadFinished.connect(self._on_load_finished)
        webview.game_mode_toggled.connect(self._handle_game_mode_from_home)
        webview.ai_assistant_requested.connect(self._show_ai_assistant)

        profile = QWebEngineProfile.defaultProfile()
        profile.downloadRequested.connect(self._handle_download)

        home_path = CONFIG_DIR / "home.html"
        home_path.write_text(HOME_HTML, encoding="utf-8")
        webview.setUrl(QUrl.fromLocalFile(str(home_path)))

        self.tabs.append(webview)
        index = self.tab_widget.addTab(webview, "Home")
        self.tab_widget.setCurrentIndex(index)

    def _close_tab(self, index: int):
        if len(self.tabs) <= 1:
            return
        self.tab_widget.removeTab(index)
        self.tabs.pop(index)

    def _tab_changed(self, index: int):
        if 0 <= index < len(self.tabs):
            self.current_tab_index = index
            webview = self.tabs[index]
            self.url_bar.setText(webview.url().toString())

    def _next_tab(self):
        next_idx = (self.tab_widget.currentIndex() + 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(next_idx)

    def _prev_tab(self):
        prev_idx = (self.tab_widget.currentIndex() - 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(prev_idx)

    def _get_current_webview(self) -> OptimizedWebEngineView:
        return self.tabs[self.current_tab_index] if self.tabs else None

    def _navigate(self):
        url = self.url_bar.text().strip()
        if not url:
            return

        webview = self._get_current_webview()
        if not webview:
            return

        if url.startswith('http://') or url.startswith('https://'):
            webview.setUrl(QUrl(url))
        elif '.' in url and ' ' not in url:
            webview.setUrl(QUrl(f'https://{url}'))
        else:
            webview.setUrl(QUrl(f'https://duckduckgo.com/?q={urllib.parse.quote(url)}'))

    def _go_home(self):
        webview = self._get_current_webview()
        if webview:
            home_path = CONFIG_DIR / "home.html"
            webview.setUrl(QUrl.fromLocalFile(str(home_path)))

    def _go_back(self):
        webview = self._get_current_webview()
        if webview:
            webview.back()

    def _go_forward(self):
        webview = self._get_current_webview()
        if webview:
            webview.forward()

    def _reload(self):
        webview = self._get_current_webview()
        if webview:
            webview.reload()

    def _on_url_changed(self, url: QUrl):
        if self.tab_widget.currentWidget() == self.sender():
            self.url_bar.setText(url.toString())
            idx = self.tabs.index(self.sender())
            title = self.sender().title() or "Loading..."
            self.tab_widget.setTabText(idx, title[:20] + "..." if len(title) > 20 else title)

    def _on_load_finished(self, ok: bool):
        self.metrics.page_loads += 1
        webview = self.sender()
        url = webview.url().toString()

        if ok and not url.startswith('file://'):
            if len(self.history) > 1000:
                self.history = self.history[-500:]
            self.history.append({'url': url, 'title': webview.title() or url, 'timestamp': time.time()})
            if len(self.history) % 10 == 0:
                self._save_data()

    def _toggle_game_mode(self, enabled: bool):
        self.settings.game_mode = enabled

        if enabled:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = GAME_MODE_CHROMIUM_FLAGS
            self.game_mode_optimizer.set_game_mode(True)
            self.status_bar.showMessage("🎮 GAME MODE ON - High Performance", 3000)
        else:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = NORMAL_CHROMIUM_FLAGS
            self.game_mode_optimizer.set_game_mode(False)
            self.status_bar.showMessage("Game Mode OFF", 2000)

        for webview in self.tabs:
            webview.perf_settings.game_mode = enabled
            webview.apply_settings()

    def _handle_game_mode_from_home(self, enabled: bool):
        self.game_mode_btn.setChecked(enabled)
        self._toggle_game_mode(enabled)

    def _show_ai_assistant(self):
        if not self.ai_dialog:
            self.ai_dialog = AIAssistantDialog(self.local_ai, self)
        self.ai_dialog.show()
        self.ai_dialog.raise_()
        self.ai_dialog.activateWindow()

    def _toggle_reader_mode(self):
        webview = self._get_current_webview()
        if not webview:
            return

        def handle_html(html):
            reader_html = self.reader_mode_handler.extract_content(html)
            webview.setHtml(reader_html, webview.url())

        webview.page().toHtml(handle_html)

    def _handle_download(self, download: QWebEngineDownloadRequest):
        download.setDownloadDirectory(str(DOWNLOADS_DIR))
        download.accept()
        self.download_manager.add_download(download)
        self.status_bar.showMessage(f"Downloading: {download.downloadFileName()}", 3000)

    def _show_downloads(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Download Manager")
        dialog.setModal(False)
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.download_manager)
        dialog.show()

    def _update_fps(self, fps: int):
        self.fps_status.setText(f"FPS: {fps}")
        if fps >= 240:
            self.fps_status.setStyleSheet("color:#00ff88;font-weight:bold")
        elif fps >= 60:
            self.fps_status.setStyleSheet("color:#ffaa00")
        else:
            self.fps_status.setStyleSheet("color:#ff2b2b")

    def _handle_optimization(self, data: dict):
        self.ram_status.setText(f"RAM: {data.get('mem_mb', 0):.0f}MB")
        self.cpu_status.setText(f"CPU: {data.get('cpu_percent', 0):.0f}%")

    def _update_status(self):
        self.downloads_status.setText(f"Downloads: {self.metrics.downloads_count}")
        self.ads_blocked_status.setText(f"Ads: {self.metrics.blocked_ads}")

        mem_mb = 0
        ping_ms = 15

        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
            except:
                pass

        current_fps = 60
        try:
            fps_text = self.fps_status.text().replace("FPS:", "").strip()
            if fps_text and fps_text != "--":
                current_fps = int(fps_text)
        except:
            pass

        webview = self._get_current_webview()
        if webview:
            webview.update_home_metrics(current_fps, mem_mb, ping_ms)

    def _log_status(self, message: str):
        self.status_bar.showMessage(message, 3000)

    def closeEvent(self, event):
        self._save_data()
        self.optimizer.stop()
        self.local_ai.stop()
        self.game_mode_optimizer.stop()
        self.fps_counter.stop()
        self.server.stop()
        event.accept()


def main():
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(10, 10, 15))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(232, 232, 232))
    palette.setColor(QPalette.ColorRole.Base, QColor(15, 15, 20))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(20, 20, 25))
    palette.setColor(QPalette.ColorRole.Text, QColor(232, 232, 232))
    palette.setColor(QPalette.ColorRole.Button, QColor(25, 25, 30))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(232, 232, 232))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(255, 43, 43))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    browser = CrimsonGXBrowser()
    browser.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()