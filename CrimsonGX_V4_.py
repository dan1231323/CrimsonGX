#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrimsonGX Browser v4.0 — AI-Powered Ultra Performance Browser
Интеллектуальная оптимизация + ИИ-помощник + Мощный локальный сервер
Всё в одном файле на чистом Python
240+ FPS Ultra Optimized
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
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from datetime import datetime
from functools import lru_cache
import asyncio
import concurrent.futures

GAMING_CHROMIUM_FLAGS = (
    "--enable-gpu-rasterization "
    "--enable-zero-copy "
    "--enable-native-gpu-memory-buffers "
    "--ignore-gpu-blocklist "
    "--enable-accelerated-video-decode "
    "--enable-accelerated-mjpeg-decode "
    "--disable-software-rasterizer "
    "--disable-dev-shm-usage "
    "--disable-background-timer-throttling "
    "--disable-backgrounding-occluded-windows "
    "--disable-renderer-backgrounding "
    "--disable-background-networking "
    "--disable-extensions "
    "--disable-component-update "
    "--disable-sync "
    "--disable-translate "
    "--disable-features=TranslateUI,IsolateOrigins,site-per-process "
    "--no-sandbox "
    "--disable-setuid-sandbox "
    "--aggressive-cache-discard "
    "--disable-hang-monitor "
    "--disable-prompt-on-repost "
    "--disable-domain-reliability "
    "--disable-breakpad "
    "--enable-features=VaapiVideoDecoder,UseSkiaRenderer,CanvasOopRasterization "
    "--disable-smooth-scrolling "
    "--memory-pressure-off "
    "--disable-client-side-phishing-detection "
    "--disable-default-apps "
    "--disable-popup-blocking "
    "--disable-ipc-flooding-protection "
    "--disable-logging "
    "--disable-metrics "
    "--disable-metrics-reporting "
    "--max-gum-fps=240 "
    "--animation-duration-scale=0 "
    "--disable-frame-rate-limit "
    "--disable-gpu-vsync "
    "--num-raster-threads=4 "
    "--renderer-process-limit=4 "
    "--disable-reading-from-canvas "
    "--force-gpu-mem-available-mb=1024 "
    "--js-flags=--max-old-space-size=512 "
)
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", GAMING_CHROMIUM_FLAGS)

try:
    from PyQt6.QtCore import (
        QUrl, Qt, QTimer, QSize, pyqtSignal, QThread, 
        QObject, QRunnable, QThreadPool, QMutex, QWaitCondition
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLineEdit, QPushButton, QLabel, QToolButton, QStatusBar,
        QDialog, QSlider, QCheckBox, QGroupBox, QGridLayout, QSpinBox,
        QTabWidget, QTextEdit, QProgressBar, QComboBox, QFrame,
        QSplitter, QListWidget, QListWidgetItem, QScrollArea,
        QStackedWidget, QSizePolicy, QMessageBox, QMenu, QSystemTrayIcon
    )
    from PyQt6.QtGui import (
        QFont, QPalette, QColor, QIcon, QShortcut, QKeySequence,
        QAction, QPainter, QBrush, QPen, QLinearGradient, QPixmap
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import (
        QWebEngineSettings, QWebEngineProfile, QWebEnginePage,
        QWebEngineScript
    )
except ImportError:
    print("pip install PyQt6 PyQt6-WebEngine")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("pip install psutil - для мониторинга системы")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

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
            self.maxsize = maxsize
    class LRUCache(dict):
        def __init__(self, maxsize=100):
            super().__init__()
            self.maxsize = maxsize

try:
    from flask import Flask, jsonify, request as flask_request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

APP_NAME = "CrimsonGX"
VERSION = "4.0 AI Ultra"
CONFIG_DIR = Path.home() / ".crimsongx"
CONFIG_FILE = CONFIG_DIR / "settings.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
BOOKMARKS_FILE = CONFIG_DIR / "bookmarks.json"
AI_CACHE_FILE = CONFIG_DIR / "ai_cache.json"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

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
    
    def get_trend(self, history: deque) -> str:
        if len(history) < 5:
            return "stable"
        recent = [v for _, v in list(history)[-10:]]
        if len(recent) < 2:
            return "stable"
        diff = recent[-1] - recent[0]
        if diff > 10:
            return "increasing"
        elif diff < -10:
            return "decreasing"
        return "stable"
    
    def predict_memory_usage(self, seconds_ahead: int = 60) -> float:
        if len(self.ram_history) < 10:
            return self.get_average(self.ram_history)
        recent = [v for _, v in list(self.ram_history)[-20:]]
        if len(recent) < 2:
            return recent[-1] if recent else 0
        avg_change = (recent[-1] - recent[0]) / len(recent)
        prediction = recent[-1] + (avg_change * seconds_ahead / 3)
        return max(0, prediction)


class AIOptimizer(QThread):
    optimization_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)
    
    def __init__(self, settings: PerformanceSettings, metrics: PerformanceMetrics):
        super().__init__()
        self.settings = settings
        self.metrics = metrics
        self.running = True
        self.mutex = QMutex()
        self.optimization_queue = queue.Queue()
        self.last_gc_time = time.time()
        self.optimization_history = []
        
    def run(self):
        while self.running:
            if not self.settings.ai_optimizer:
                time.sleep(2)
                continue
            
            try:
                recommendations = self._analyze_system()
                self.optimization_signal.emit(recommendations)
                
                if self.settings.auto_optimize:
                    self._apply_optimizations(recommendations)
                    
            except Exception as e:
                self.status_signal.emit(f"Ошибка оптимизатора: {str(e)[:50]}")
            
            time.sleep(2)
    
    def _analyze_system(self) -> dict:
        recommendations = {
            'timestamp': time.time(),
            'mem_mb': 0,
            'mem_percent': 0,
            'cpu_percent': 0,
            'status': 'optimal',
            'actions': [],
            'predictions': {},
            'score': 100
        }
        
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                mem_percent = mem_mb / self.settings.max_ram_mb
                cpu_percent = process.cpu_percent(interval=0.5)
                
                recommendations['mem_mb'] = round(mem_mb, 1)
                recommendations['mem_percent'] = round(mem_percent * 100, 1)
                recommendations['cpu_percent'] = round(cpu_percent, 1)
                
                self.metrics.add_sample(mem_mb, cpu_percent)
                
                predicted_mem = self.metrics.predict_memory_usage(60)
                recommendations['predictions'] = {
                    'memory_1min': round(predicted_mem, 1),
                    'trend': self.metrics.get_trend(self.metrics.ram_history)
                }
                
                score = 100
                actions = []
                
                if mem_percent > self.settings.memory_threshold:
                    score -= 30
                    actions.append({
                        'type': 'memory_critical',
                        'action': 'gc_aggressive',
                        'priority': 'high',
                        'description': 'Критический уровень RAM - агрессивная очистка'
                    })
                    recommendations['status'] = 'critical'
                elif mem_percent > 0.7:
                    score -= 15
                    actions.append({
                        'type': 'memory_warning',
                        'action': 'gc_normal',
                        'priority': 'medium',
                        'description': 'Высокий уровень RAM - очистка кэша'
                    })
                    recommendations['status'] = 'warning'
                
                if cpu_percent > self.settings.cpu_threshold * 100:
                    score -= 25
                    actions.append({
                        'type': 'cpu_high',
                        'action': 'reduce_load',
                        'priority': 'high',
                        'description': 'Высокая нагрузка CPU - снижение активности'
                    })
                    if recommendations['status'] != 'critical':
                        recommendations['status'] = 'warning'
                
                if predicted_mem > self.settings.max_ram_mb:
                    score -= 20
                    actions.append({
                        'type': 'prediction_warning',
                        'action': 'preemptive_gc',
                        'priority': 'medium',
                        'description': f'Прогноз: превышение RAM через 1 мин ({predicted_mem:.0f}MB)'
                    })
                
                now = time.time()
                if now - self.last_gc_time > self.settings.gc_interval:
                    actions.append({
                        'type': 'scheduled_gc',
                        'action': 'gc_normal',
                        'priority': 'low',
                        'description': 'Плановая очистка мусора'
                    })
                
                recommendations['score'] = max(0, score)
                recommendations['actions'] = actions
                
            except Exception as e:
                recommendations['error'] = str(e)
        
        return recommendations
    
    def _apply_optimizations(self, recommendations: dict):
        for action in recommendations.get('actions', []):
            action_type = action.get('action', '')
            
            if action_type == 'gc_aggressive':
                self._aggressive_gc()
                self.metrics.optimizations_count += 1
                self.status_signal.emit("Агрессивная очистка памяти выполнена")
                
            elif action_type == 'gc_normal':
                gc.collect()
                self.metrics.gc_count += 1
                self.last_gc_time = time.time()
                
            elif action_type == 'preemptive_gc':
                gc.collect(0)
                gc.collect(1)
                
            self.optimization_history.append({
                'timestamp': time.time(),
                'action': action_type,
                'result': 'applied'
            })
    
    def _aggressive_gc(self):
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        self.metrics.gc_count += 3
        self.last_gc_time = time.time()
    
    def stop(self):
        self.running = False


class AIAssistant(QThread):
    response_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)
    
    def __init__(self, settings: PerformanceSettings):
        super().__init__()
        self.settings = settings
        self.running = True
        self.request_queue = queue.Queue()
        self.response_cache = TTLCache(maxsize=500, ttl=3600)
        self.openai_client = None
        self._init_openai()
        
    def _init_openai(self):
        if HAS_OPENAI and self.settings.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
                self.status_signal.emit("OpenAI подключен")
            except Exception as e:
                self.status_signal.emit(f"OpenAI ошибка: {str(e)[:30]}")
    
    def run(self):
        while self.running:
            try:
                request = self.request_queue.get(timeout=1)
                if request:
                    response = self._process_request(request)
                    self.response_signal.emit(response)
            except queue.Empty:
                continue
            except Exception as e:
                self.response_signal.emit({
                    'type': 'error',
                    'message': str(e),
                    'request_id': request.get('id', 'unknown') if request else 'unknown'
                })
    
    def _get_cache_key(self, request: dict) -> str:
        content = f"{request.get('type', '')}:{request.get('content', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _process_request(self, request: dict) -> dict:
        cache_key = self._get_cache_key(request)
        
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            cached['cached'] = True
            return cached
        
        request_type = request.get('type', 'general')
        content = request.get('content', '')
        
        if request_type == 'summarize':
            response = self._summarize_text(content)
        elif request_type == 'search':
            response = self._smart_search(content)
        elif request_type == 'optimize':
            response = self._get_optimization_tips(content)
        elif request_type == 'translate':
            response = self._translate_text(content, request.get('target_lang', 'en'))
        elif request_type == 'explain':
            response = self._explain_content(content)
        else:
            response = self._general_query(content)
        
        result = {
            'type': request_type,
            'request_id': request.get('id', 'unknown'),
            'response': response,
            'cached': False,
            'timestamp': time.time()
        }
        
        self.response_cache[cache_key] = result
        return result
    
    def _summarize_text(self, text: str) -> str:
        if self.openai_client:
            try:
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "Ты помощник для кратких резюме. Отвечай на русском языке."},
                        {"role": "user", "content": f"Сделай краткое резюме этого текста:\n\n{text[:4000]}"}
                    ],
                    max_completion_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return self._local_summarize(text)
        else:
            return self._local_summarize(text)
    
    def _local_summarize(self, text: str) -> str:
        if not text:
            return "Нет текста для анализа"
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:500] + "..." if len(text) > 500 else text
        
        word_freq = {}
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(w, 0) for w in words if len(w) > 3)
            sentence_scores.append((score, sentence))
        
        sentence_scores.sort(reverse=True)
        top_sentences = [s for _, s in sentence_scores[:3]]
        
        summary = ". ".join(top_sentences)
        return summary + "." if summary else "Не удалось создать резюме"
    
    def _smart_search(self, query: str) -> str:
        search_engines = {
            'Google': f'https://www.google.com/search?q={query}',
            'DuckDuckGo': f'https://duckduckgo.com/?q={query}',
            'Yandex': f'https://yandex.ru/search/?text={query}',
            'Bing': f'https://www.bing.com/search?q={query}'
        }
        
        tips = []
        if any(word in query.lower() for word in ['как', 'что', 'почему', 'зачем', 'how', 'what', 'why']):
            tips.append("Совет: Добавьте конкретные ключевые слова для точных результатов")
        if len(query.split()) < 3:
            tips.append("Совет: Более длинные запросы дают точные результаты")
        
        result = f"Поиск: {query}\n\nСсылки:\n"
        for name, url in search_engines.items():
            result += f"• {name}: {url}\n"
        
        if tips:
            result += "\n" + "\n".join(tips)
        
        return result
    
    def _get_optimization_tips(self, context: str) -> str:
        tips = [
            "Закройте неиспользуемые вкладки для экономии RAM",
            "Отключите изображения на тяжелых страницах",
            "Используйте режим чтения для статей",
            "Очистите кэш браузера регулярно",
            "Включите блокировщик рекламы для ускорения загрузки",
            "Используйте закладки вместо множества открытых вкладок"
        ]
        
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                if mem.percent > 80:
                    tips.insert(0, f"Системная память загружена на {mem.percent}% - закройте другие программы")
                cpu = psutil.cpu_percent()
                if cpu > 70:
                    tips.insert(0, f"CPU загружен на {cpu}% - подождите завершения процессов")
            except:
                pass
        
        return "Советы по оптимизации:\n\n" + "\n".join(f"• {tip}" for tip in tips[:5])
    
    def _translate_text(self, text: str, target_lang: str) -> str:
        if self.openai_client:
            try:
                lang_names = {'en': 'английский', 'ru': 'русский', 'de': 'немецкий', 'fr': 'французский'}
                lang_name = lang_names.get(target_lang, target_lang)
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": f"Переведи текст на {lang_name}. Отвечай только переводом."},
                        {"role": "user", "content": text[:2000]}
                    ],
                    max_completion_tokens=1000
                )
                return response.choices[0].message.content
            except:
                pass
        return f"Перевод недоступен без OpenAI API ключа\n\nОригинал:\n{text[:500]}"
    
    def _explain_content(self, content: str) -> str:
        if self.openai_client:
            try:
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "Объясни простыми словами на русском языке."},
                        {"role": "user", "content": f"Объясни это:\n\n{content[:3000]}"}
                    ],
                    max_completion_tokens=800
                )
                return response.choices[0].message.content
            except:
                pass
        return f"Для объяснений нужен OpenAI API ключ\n\nКонтент:\n{content[:300]}..."
    
    def _general_query(self, query: str) -> str:
        if self.openai_client:
            try:
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "Ты умный помощник браузера CrimsonGX. Отвечай на русском, кратко и по делу."},
                        {"role": "user", "content": query}
                    ],
                    max_completion_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Ошибка OpenAI: {str(e)[:100]}"
        
        responses = {
            'привет': 'Привет! Я ИИ-помощник CrimsonGX. Чем могу помочь?',
            'помощь': 'Я могу: суммаризировать страницы, искать информацию, давать советы по оптимизации.',
            'настройки': 'Откройте настройки через меню или нажмите Ctrl+,',
        }
        
        query_lower = query.lower()
        for key, response in responses.items():
            if key in query_lower:
                return response
        
        return "Для полноценных ответов добавьте OpenAI API ключ в настройках.\n\nДоступные команды:\n• /summarize - резюме страницы\n• /search [запрос] - умный поиск\n• /optimize - советы по оптимизации"
    
    def add_request(self, request: dict):
        request['id'] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.request_queue.put(request)
    
    def stop(self):
        self.running = False


class LocalServer(QThread):
    log_signal = pyqtSignal(str)
    
    def __init__(self, settings: PerformanceSettings, metrics: PerformanceMetrics):
        super().__init__()
        self.settings = settings
        self.metrics = metrics
        self.running = True
        self.app = None
        
    def run(self):
        if not HAS_FLASK or not self.settings.local_server:
            return
            
        self.app = Flask(__name__)
        self.app.config['JSON_AS_ASCII'] = False
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'status': 'running',
                'version': VERSION,
                'uptime': time.time() - self.metrics.start_time,
                'metrics': {
                    'ram_avg': self.metrics.get_average(self.metrics.ram_history),
                    'cpu_avg': self.metrics.get_average(self.metrics.cpu_history),
                    'gc_count': self.metrics.gc_count,
                    'optimizations': self.metrics.optimizations_count,
                    'page_loads': self.metrics.page_loads,
                    'cache_hits': self.metrics.cache_hits
                }
            })
        
        @self.app.route('/api/optimize', methods=['POST'])
        def trigger_optimize():
            gc.collect()
            self.metrics.gc_count += 1
            return jsonify({'status': 'optimized', 'gc_count': self.metrics.gc_count})
        
        @self.app.route('/api/settings', methods=['GET', 'POST'])
        def handle_settings():
            if flask_request.method == 'GET':
                return jsonify(asdict(self.settings))
            else:
                data = flask_request.json
                for key, value in data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
                self.settings.save()
                return jsonify({'status': 'updated'})
        
        @self.app.route('/api/history')
        def get_history():
            if HISTORY_FILE.exists():
                return jsonify(json.loads(HISTORY_FILE.read_text()))
            return jsonify([])
        
        try:
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            self.log_signal.emit(f"Сервер запущен на порту {self.settings.server_port}")
            self.app.run(
                host='127.0.0.1', 
                port=self.settings.server_port, 
                debug=False, 
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            self.log_signal.emit(f"Ошибка сервера: {str(e)}")
    
    def stop(self):
        self.running = False


HOME_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CrimsonGX Home</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0a0f;--bg2:#12050f;--accent:#ff2b2b;--accent2:#cc1b1b;
  --text:#e8e8e8;--muted:#9a8f8f;--glass:rgba(255,255,255,.02);
  --border:rgba(255,255,255,.06);--success:#00ff88;--warning:#ffaa00;
}
html{height:100%;font:14px/1.4 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeSpeed}
body{height:100%;overflow:hidden;display:flex;flex-direction:column;will-change:transform}

.top-bar{
  display:flex;align-items:center;gap:12px;padding:14px 20px;
  background:linear-gradient(135deg,rgba(255,43,43,.1),rgba(0,0,0,.3));
  border-bottom:1px solid var(--border);backdrop-filter:blur(12px);
}
.logo-section{display:flex;flex-direction:column;gap:2px}
.logo{font:800 24px/1 system-ui;color:var(--accent);letter-spacing:4px;
  text-shadow:0 0 15px rgba(255,43,43,.5),0 0 30px rgba(255,43,43,.3);
  animation:glow 3s ease-in-out infinite}
@keyframes glow{0%,100%{opacity:.9}50%{opacity:1;text-shadow:0 0 20px rgba(255,43,43,.7)}}
.subtitle{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase}
.version{font-size:9px;color:var(--accent);opacity:.7}

.stats{margin-left:auto;display:flex;gap:20px}
.stat{display:flex;flex-direction:column;align-items:center;gap:3px;
  padding:8px 12px;background:var(--glass);border:1px solid var(--border);border-radius:8px}
.stat-value{font:700 16px/1 monospace;color:var(--accent)}
.stat-value.optimal{color:var(--success)}
.stat-value.warning{color:var(--warning)}
.stat-value.critical{color:#ff4444}
.stat-label{color:var(--muted);font-size:8px;text-transform:uppercase;letter-spacing:1px}
.stat-bar{width:40px;height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden}
.stat-bar-fill{height:100%;background:var(--accent);transition:width .3s}

.search-box{
  display:flex;gap:8px;padding:10px 14px;background:var(--glass);
  border:1px solid var(--border);border-radius:12px;margin:16px 20px;
  transition:all .3s cubic-bezier(.4,0,.2,1);backdrop-filter:blur(8px);
}
.search-box:focus-within{border-color:var(--accent);box-shadow:0 0 25px rgba(255,43,43,.2);
  transform:translateY(-2px)}
.search-box input{
  flex:1;background:0;border:0;outline:0;color:var(--text);font-size:15px;padding:6px;
}
.search-box input::placeholder{color:var(--muted);opacity:.6}
.search-box button{
  background:linear-gradient(135deg,var(--accent),var(--accent2));border:0;
  padding:8px 20px;border-radius:8px;color:#fff;font:600 13px system-ui;
  cursor:pointer;transition:all .2s;box-shadow:0 3px 10px rgba(255,43,43,.4);
}
.search-box button:hover{transform:translateY(-2px) scale(1.02);box-shadow:0 5px 15px rgba(255,43,43,.5)}

.content{flex:1;overflow-y:auto;overflow-x:hidden;padding:0 20px 20px}
.content::-webkit-scrollbar{width:6px}
.content::-webkit-scrollbar-track{background:transparent}
.content::-webkit-scrollbar-thumb{background:var(--accent);border-radius:3px}

.section-title{font-size:12px;color:var(--muted);margin:24px 0 14px;font-weight:700;
  text-transform:uppercase;letter-spacing:2px;display:flex;align-items:center;gap:10px}
.section-title::before{content:'';width:4px;height:18px;background:linear-gradient(180deg,var(--accent),var(--accent2));border-radius:2px}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:24px}
.card{
  background:linear-gradient(135deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
  border:1px solid var(--border);border-radius:12px;padding:18px;cursor:pointer;
  transition:all .3s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden;
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),transparent);transform:scaleX(0);
  transition:transform .3s;transform-origin:left}
.card:hover{transform:translateY(-5px);border-color:rgba(255,43,43,.4);
  box-shadow:0 10px 30px rgba(255,43,43,.2)}
.card:hover::before{transform:scaleX(1)}
.card h3{font-size:15px;margin-bottom:8px;color:#ffd5d5}
.card p{font-size:11px;color:var(--muted);line-height:1.5}
.card-icon{font-size:24px;margin-bottom:8px}

.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:24px}
.feature{
  background:linear-gradient(135deg,rgba(255,43,43,.06),rgba(0,0,0,.15));
  border:1px solid rgba(255,43,43,.18);border-radius:12px;padding:16px;
  transition:all .3s;position:relative;overflow:hidden;
}
.feature::after{content:'';position:absolute;top:-50%;right:-50%;width:100%;height:100%;
  background:radial-gradient(circle,rgba(255,43,43,.12),transparent);
  transform:scale(0);transition:transform .5s}
.feature:hover::after{transform:scale(2.5)}
.feature:hover{border-color:rgba(255,43,43,.35)}
.feature h4{font-size:14px;color:var(--accent);margin-bottom:8px;font-weight:700}
.feature p{font-size:11px;color:var(--muted);line-height:1.5}

.ai-panel{
  background:linear-gradient(135deg,rgba(0,255,136,.05),rgba(0,0,0,.2));
  border:1px solid rgba(0,255,136,.2);border-radius:12px;padding:16px;margin-bottom:24px;
}
.ai-panel h3{color:var(--success);font-size:14px;margin-bottom:10px}
.ai-status{display:flex;gap:12px;flex-wrap:wrap}
.ai-status span{padding:6px 12px;background:rgba(0,255,136,.1);border-radius:6px;
  font-size:11px;color:var(--success)}

.footer{
  padding:14px 20px;background:rgba(10,10,15,.97);border-top:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;font-size:10px;color:var(--muted);
  backdrop-filter:blur(12px);
}
.tags{display:flex;gap:8px;flex-wrap:wrap}
.tag{padding:6px 12px;background:rgba(255,43,43,.1);border:1px solid rgba(255,43,43,.25);
  border-radius:16px;font-weight:600;transition:all .2s;font-size:10px}
.tag:hover{background:rgba(255,43,43,.2);transform:translateY(-1px)}
.tag.ai{background:rgba(0,255,136,.1);border-color:rgba(0,255,136,.25);color:var(--success)}

@media(max-width:768px){
  .grid{grid-template-columns:1fr 1fr}
  .features{grid-template-columns:1fr}
  .stats{gap:10px}
  .stat{padding:6px 8px}
}
</style>
</head>
<body>
<div class="top-bar">
  <div class="logo-section">
    <div class="logo">CRIMSONGX</div>
    <div class="subtitle">AI-Powered Ultra Performance</div>
    <div class="version">v4.0 AI Ultra</div>
  </div>
  <div class="stats">
    <div class="stat">
      <div class="stat-value" id="fps">--</div>
      <div class="stat-label">FPS</div>
      <div class="stat-bar"><div class="stat-bar-fill" id="fps-bar" style="width:0%"></div></div>
    </div>
    <div class="stat">
      <div class="stat-value" id="ram">--</div>
      <div class="stat-label">RAM MB</div>
      <div class="stat-bar"><div class="stat-bar-fill" id="ram-bar" style="width:0%"></div></div>
    </div>
    <div class="stat">
      <div class="stat-value" id="cpu">--</div>
      <div class="stat-label">CPU %</div>
      <div class="stat-bar"><div class="stat-bar-fill" id="cpu-bar" style="width:0%"></div></div>
    </div>
    <div class="stat">
      <div class="stat-value optimal" id="status">OK</div>
      <div class="stat-label">STATUS</div>
    </div>
  </div>
</div>

<form class="search-box" onsubmit="s(event)">
  <input id="q" placeholder="Поиск или URL... (Ctrl+L)" autocomplete="off" autofocus>
  <button type="submit">GO</button>
</form>

<div class="content">
  <div class="ai-panel">
    <h3>AI Assistant Ready</h3>
    <div class="ai-status">
      <span id="ai-optimizer">Optimizer: Active</span>
      <span id="ai-assistant">Assistant: Ready</span>
      <span id="ai-server">Server: Running</span>
    </div>
  </div>

  <div class="section-title">Quick Access</div>
  <div class="grid">
    <div class="card" onclick="g('https://youtube.com')"><div class="card-icon">▶</div><h3>YouTube</h3><p>Videos & Streams</p></div>
    <div class="card" onclick="g('https://github.com')"><div class="card-icon">◆</div><h3>GitHub</h3><p>Code & Projects</p></div>
    <div class="card" onclick="g('https://discord.com')"><div class="card-icon">◎</div><h3>Discord</h3><p>Voice & Chat</p></div>
    <div class="card" onclick="g('https://twitch.tv')"><div class="card-icon">◈</div><h3>Twitch</h3><p>Live Streams</p></div>
    <div class="card" onclick="g('https://reddit.com')"><div class="card-icon">◉</div><h3>Reddit</h3><p>Communities</p></div>
    <div class="card" onclick="g('https://steamcommunity.com')"><div class="card-icon">◐</div><h3>Steam</h3><p>Gaming</p></div>
  </div>
  
  <div class="section-title">CrimsonGX v4.0 Features</div>
  <div class="features">
    <div class="feature">
      <h4>AI Optimizer</h4>
      <p>Intelligent real-time RAM/CPU monitoring with predictive optimization and auto-cleanup</p>
    </div>
    <div class="feature">
      <h4>AI Assistant</h4>
      <p>Smart summarization, search tips, translations, and context-aware help</p>
    </div>
    <div class="feature">
      <h4>Local Server</h4>
      <p>Built-in API server for automation and external integrations</p>
    </div>
    <div class="feature">
      <h4>Smart RAM</h4>
      <p>Predictive memory management with preemptive garbage collection</p>
    </div>
    <div class="feature">
      <h4>Performance Metrics</h4>
      <p>Real-time FPS, RAM, CPU tracking with historical analysis</p>
    </div>
    <div class="feature">
      <h4>Ultra Optimization</h4>
      <p>Chromium flags tuned for maximum performance on any hardware</p>
    </div>
  </div>
</div>

<div class="footer">
  <div class="tags">
    <span class="tag">Ultra Fast</span>
    <span class="tag ai">AI Powered</span>
    <span class="tag">Smart RAM</span>
    <span class="tag">Local API</span>
  </div>
  <div>CrimsonGX v4.0 AI Ultra | Pure Python</div>
</div>

<script>
const FPS_TARGET=240;
let frameCount=0,lastTime=performance.now(),currentFPS=0;
let fpsHistory=[];

function measureFPS(now){
  frameCount++;
  const delta=now-lastTime;
  if(delta>=100){
    currentFPS=Math.round((frameCount*1000)/delta);
    fpsHistory.push(currentFPS);
    if(fpsHistory.length>10)fpsHistory.shift();
    const avgFPS=Math.round(fpsHistory.reduce((a,b)=>a+b,0)/fpsHistory.length);
    const el=document.getElementById('fps');
    el.textContent=avgFPS;
    el.className='stat-value '+(avgFPS>=120?'optimal':avgFPS>=60?'warning':'critical');
    document.getElementById('fps-bar').style.width=Math.min(100,(avgFPS/FPS_TARGET)*100)+'%';
    frameCount=0;
    lastTime=now;
  }
  requestAnimationFrame(measureFPS);
}
requestAnimationFrame(measureFPS);

if(performance.memory){
  setInterval(()=>{
    const m=Math.round(performance.memory.usedJSHeapSize/1048576);
    const el=document.getElementById('ram');
    el.textContent=m;
    el.className='stat-value '+(m<100?'optimal':m<300?'warning':'critical');
    document.getElementById('ram-bar').style.width=Math.min(100,m/5)+'%';
  },500);
}

function g(u){location.href=u}
function s(e){
  e.preventDefault();
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  location.href=q.match(/^https?:\/\//)||q.includes('.')&&!q.includes(' ')?
    (q.startsWith('http')?q:'https://'+q):
    'https://duckduckgo.com/?q='+encodeURIComponent(q);
}

document.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==='l'){e.preventDefault();document.getElementById('q').focus();document.getElementById('q').select()}
});

console.log('CrimsonGX v4.0 - 240+ FPS Ultra Mode Active');
</script>
</body>
</html>"""


class OptimizedWebEngineView(QWebEngineView):
    def __init__(self, settings: PerformanceSettings, parent=None):
        super().__init__(parent)
        self.perf_settings = settings
        self._setup_profile()
        self._setup_settings()
        
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
        s.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, False)
        s.setAttribute(QWebEngineSettings.WebAttribute.ErrorPageEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        
    def apply_settings(self):
        self._setup_settings()


class SettingsDialog(QDialog):
    def __init__(self, settings: PerformanceSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("CrimsonGX Settings")
        self.setModal(True)
        self.resize(700, 650)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        tabs.addTab(self.create_performance_tab(), "Performance")
        tabs.addTab(self.create_ai_tab(), "AI Settings")
        tabs.addTab(self.create_server_tab(), "Server")
        tabs.addTab(self.create_presets_tab(), "Presets")
        
        layout.addWidget(tabs)
        
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self.setStyleSheet("""
            QDialog {background:#0a0a0f;color:#e8e8e8}
            QTabWidget::pane {border:1px solid rgba(255,255,255,0.1);border-radius:8px;background:rgba(20,20,25,0.5);padding:10px}
            QTabBar::tab {background:rgba(255,255,255,0.03);color:#9a8f8f;padding:10px 20px;margin-right:2px;border:1px solid rgba(255,255,255,0.05);border-bottom:none;border-radius:8px 8px 0 0}
            QTabBar::tab:selected {background:rgba(255,43,43,0.2);color:#ff2b2b;border-color:rgba(255,43,43,0.3)}
            QGroupBox {border:1px solid rgba(255,255,255,0.12);border-radius:8px;margin-top:14px;padding:14px;color:#ffd5d5;background:rgba(255,255,255,0.02)}
            QGroupBox::title {subcontrol-origin:margin;subcontrol-position:top left;padding:0 8px;color:#ff2b2b}
            QLabel {color:#e8e8e8}
            QSlider::groove:horizontal {height:6px;background:rgba(255,255,255,0.08);border-radius:3px}
            QSlider::handle:horizontal {background:#ff2b2b;width:16px;height:16px;margin:-5px 0;border-radius:8px}
            QCheckBox {color:#e8e8e8;spacing:8px}
            QCheckBox::indicator {width:18px;height:18px;border:2px solid rgba(255,255,255,0.3);border-radius:4px;background:rgba(255,255,255,0.05)}
            QCheckBox::indicator:checked {background:#ff2b2b;border-color:#ff2b2b}
            QSpinBox,QLineEdit,QComboBox {background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);border-radius:6px;color:#e8e8e8;padding:6px 10px}
            QSpinBox:focus,QLineEdit:focus {border-color:#ff2b2b}
            QPushButton {background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);border-radius:8px;color:#e8e8e8;padding:10px 20px;font-weight:600}
            QPushButton:hover {background:rgba(255,43,43,0.2);border-color:rgba(255,43,43,0.4)}
        """)
    
    def create_performance_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        ram_group = QGroupBox("Memory Management")
        ram_layout = QGridLayout()
        
        ram_layout.addWidget(QLabel("Max RAM:"), 0, 0)
        self.ram_spin = QSpinBox()
        self.ram_spin.setRange(128, 8192)
        self.ram_spin.setSingleStep(128)
        self.ram_spin.setSuffix(" MB")
        self.ram_spin.setValue(self.settings.max_ram_mb)
        ram_layout.addWidget(self.ram_spin, 0, 1)
        
        ram_layout.addWidget(QLabel("Cache Size:"), 1, 0)
        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(10, 1000)
        self.cache_spin.setSingleStep(10)
        self.cache_spin.setSuffix(" MB")
        self.cache_spin.setValue(self.settings.cache_size_mb)
        ram_layout.addWidget(self.cache_spin, 1, 1)
        
        ram_layout.addWidget(QLabel("GC Interval:"), 2, 0)
        self.gc_spin = QSpinBox()
        self.gc_spin.setRange(10, 300)
        self.gc_spin.setSingleStep(10)
        self.gc_spin.setSuffix(" sec")
        self.gc_spin.setValue(self.settings.gc_interval)
        ram_layout.addWidget(self.gc_spin, 2, 1)
        
        ram_group.setLayout(ram_layout)
        layout.addWidget(ram_group)
        
        content_group = QGroupBox("Content Loading")
        content_layout = QVBoxLayout()
        
        self.images_cb = QCheckBox("Load Images")
        self.images_cb.setChecked(self.settings.enable_images)
        content_layout.addWidget(self.images_cb)
        
        self.js_cb = QCheckBox("Enable JavaScript")
        self.js_cb.setChecked(self.settings.enable_javascript)
        content_layout.addWidget(self.js_cb)
        
        self.webgl_cb = QCheckBox("Enable WebGL (GPU)")
        self.webgl_cb.setChecked(self.settings.enable_webgl)
        content_layout.addWidget(self.webgl_cb)
        
        self.canvas_cb = QCheckBox("Enable Canvas 2D")
        self.canvas_cb.setChecked(self.settings.enable_canvas)
        content_layout.addWidget(self.canvas_cb)
        
        self.lazy_cb = QCheckBox("Lazy Loading")
        self.lazy_cb.setChecked(self.settings.lazy_load)
        content_layout.addWidget(self.lazy_cb)
        
        content_group.setLayout(content_layout)
        layout.addWidget(content_group)
        
        layout.addStretch()
        return widget
    
    def create_ai_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        optimizer_group = QGroupBox("AI Optimizer")
        opt_layout = QVBoxLayout()
        
        self.ai_opt_cb = QCheckBox("Enable AI Optimizer")
        self.ai_opt_cb.setChecked(self.settings.ai_optimizer)
        opt_layout.addWidget(self.ai_opt_cb)
        
        self.auto_opt_cb = QCheckBox("Auto-apply Optimizations")
        self.auto_opt_cb.setChecked(self.settings.auto_optimize)
        opt_layout.addWidget(self.auto_opt_cb)
        
        self.smart_ram_cb = QCheckBox("Smart RAM Management")
        self.smart_ram_cb.setChecked(self.settings.smart_ram)
        opt_layout.addWidget(self.smart_ram_cb)
        
        self.aggressive_gc_cb = QCheckBox("Aggressive GC")
        self.aggressive_gc_cb.setChecked(self.settings.aggressive_gc)
        opt_layout.addWidget(self.aggressive_gc_cb)
        
        self.prediction_cb = QCheckBox("Predictive Mode")
        self.prediction_cb.setChecked(self.settings.prediction_mode)
        opt_layout.addWidget(self.prediction_cb)
        
        optimizer_group.setLayout(opt_layout)
        layout.addWidget(optimizer_group)
        
        assistant_group = QGroupBox("AI Assistant")
        asst_layout = QVBoxLayout()
        
        self.ai_asst_cb = QCheckBox("Enable AI Assistant")
        self.ai_asst_cb.setChecked(self.settings.ai_assistant)
        asst_layout.addWidget(self.ai_asst_cb)
        
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("OpenAI API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("sk-... (optional)")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setText(self.settings.openai_api_key)
        api_layout.addWidget(self.api_key_edit)
        asst_layout.addLayout(api_layout)
        
        asst_layout.addWidget(QLabel("Note: OpenAI API enables advanced features like translation and explanations"))
        
        assistant_group.setLayout(asst_layout)
        layout.addWidget(assistant_group)
        
        layout.addStretch()
        return widget
    
    def create_server_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        server_group = QGroupBox("Local Server")
        srv_layout = QVBoxLayout()
        
        self.server_cb = QCheckBox("Enable Local API Server")
        self.server_cb.setChecked(self.settings.local_server)
        srv_layout.addWidget(self.server_cb)
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self.settings.server_port)
        port_layout.addWidget(self.port_spin)
        port_layout.addStretch()
        srv_layout.addLayout(port_layout)
        
        srv_layout.addWidget(QLabel("API Endpoints:"))
        endpoints = QTextEdit()
        endpoints.setReadOnly(True)
        endpoints.setMaximumHeight(120)
        endpoints.setPlainText(
            f"GET  /api/status    - Browser status & metrics\n"
            f"POST /api/optimize  - Trigger optimization\n"
            f"GET  /api/settings  - Get current settings\n"
            f"POST /api/settings  - Update settings\n"
            f"GET  /api/history   - Browsing history"
        )
        srv_layout.addWidget(endpoints)
        
        server_group.setLayout(srv_layout)
        layout.addWidget(server_group)
        
        layout.addStretch()
        return widget
    
    def create_presets_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QVBoxLayout()
        
        ultra_btn = QPushButton("Ultra Performance")
        ultra_btn.clicked.connect(lambda: self.apply_preset('ultra'))
        presets_layout.addWidget(ultra_btn)
        
        balanced_btn = QPushButton("Balanced")
        balanced_btn.clicked.connect(lambda: self.apply_preset('balanced'))
        presets_layout.addWidget(balanced_btn)
        
        quality_btn = QPushButton("Quality")
        quality_btn.clicked.connect(lambda: self.apply_preset('quality'))
        presets_layout.addWidget(quality_btn)
        
        minimal_btn = QPushButton("Minimal (Text Only)")
        minimal_btn.clicked.connect(lambda: self.apply_preset('minimal'))
        presets_layout.addWidget(minimal_btn)
        
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)
        
        layout.addStretch()
        return widget
    
    def apply_preset(self, preset: str):
        presets = {
            'ultra': {'max_ram_mb': 256, 'cache_size_mb': 20, 'enable_images': False, 'enable_webgl': False, 'aggressive_gc': True},
            'balanced': {'max_ram_mb': 512, 'cache_size_mb': 50, 'enable_images': True, 'enable_webgl': False, 'aggressive_gc': True},
            'quality': {'max_ram_mb': 1024, 'cache_size_mb': 200, 'enable_images': True, 'enable_webgl': True, 'aggressive_gc': False},
            'minimal': {'max_ram_mb': 128, 'cache_size_mb': 10, 'enable_images': False, 'enable_javascript': False, 'enable_webgl': False}
        }
        
        if preset in presets:
            p = presets[preset]
            self.ram_spin.setValue(p.get('max_ram_mb', 512))
            self.cache_spin.setValue(p.get('cache_size_mb', 50))
            self.images_cb.setChecked(p.get('enable_images', True))
            self.js_cb.setChecked(p.get('enable_javascript', True))
            self.webgl_cb.setChecked(p.get('enable_webgl', False))
            self.aggressive_gc_cb.setChecked(p.get('aggressive_gc', True))
    
    def save_and_accept(self):
        self.settings.max_ram_mb = self.ram_spin.value()
        self.settings.cache_size_mb = self.cache_spin.value()
        self.settings.gc_interval = self.gc_spin.value()
        self.settings.enable_images = self.images_cb.isChecked()
        self.settings.enable_javascript = self.js_cb.isChecked()
        self.settings.enable_webgl = self.webgl_cb.isChecked()
        self.settings.enable_canvas = self.canvas_cb.isChecked()
        self.settings.lazy_load = self.lazy_cb.isChecked()
        self.settings.ai_optimizer = self.ai_opt_cb.isChecked()
        self.settings.auto_optimize = self.auto_opt_cb.isChecked()
        self.settings.smart_ram = self.smart_ram_cb.isChecked()
        self.settings.aggressive_gc = self.aggressive_gc_cb.isChecked()
        self.settings.prediction_mode = self.prediction_cb.isChecked()
        self.settings.ai_assistant = self.ai_asst_cb.isChecked()
        self.settings.openai_api_key = self.api_key_edit.text()
        self.settings.local_server = self.server_cb.isChecked()
        self.settings.server_port = self.port_spin.value()
        self.settings.save()
        self.accept()


class AIAssistantPanel(QWidget):
    def __init__(self, assistant: AIAssistant, parent=None):
        super().__init__(parent)
        self.assistant = assistant
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        header = QLabel("AI Assistant")
        header.setStyleSheet("font-weight:bold;font-size:14px;color:#00ff88;padding:4px")
        layout.addWidget(header)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                color: #e8e8e8;
                padding: 8px;
            }
        """)
        self.chat_history.setHtml("<i style='color:#666'>Type a message or use commands...</i>")
        layout.addWidget(self.chat_history)
        
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask AI... (/summarize, /search, /optimize)")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 6px;
                color: #e8e8e8;
                padding: 8px;
            }
            QLineEdit:focus {
                border-color: #00ff88;
            }
        """)
        input_layout.addWidget(self.input_field)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_message)
        send_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0,255,136,0.2);
                border: 1px solid rgba(0,255,136,0.3);
                border-radius: 6px;
                color: #00ff88;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: rgba(0,255,136,0.3);
            }
        """)
        input_layout.addWidget(send_btn)
        
        layout.addLayout(input_layout)
        
        self.assistant.response_signal.connect(self.handle_response)
        
    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
            
        self.input_field.clear()
        self.add_message("You", text, "#ff2b2b")
        
        if text.startswith('/summarize'):
            content = text[10:].strip()
            self.assistant.add_request({'type': 'summarize', 'content': content or 'current page'})
        elif text.startswith('/search'):
            query = text[7:].strip()
            self.assistant.add_request({'type': 'search', 'content': query})
        elif text.startswith('/optimize'):
            self.assistant.add_request({'type': 'optimize', 'content': ''})
        elif text.startswith('/translate'):
            parts = text[10:].strip().split(' ', 1)
            lang = parts[0] if parts else 'en'
            content = parts[1] if len(parts) > 1 else ''
            self.assistant.add_request({'type': 'translate', 'content': content, 'target_lang': lang})
        else:
            self.assistant.add_request({'type': 'general', 'content': text})
    
    def add_message(self, sender: str, text: str, color: str = "#e8e8e8"):
        current = self.chat_history.toHtml()
        if "<i style='color:#666'>" in current:
            current = ""
        timestamp = datetime.now().strftime("%H:%M")
        new_msg = f"<p><span style='color:{color};font-weight:bold'>{sender}</span> <span style='color:#666;font-size:10px'>{timestamp}</span><br>{html.escape(text)}</p>"
        self.chat_history.setHtml(current + new_msg)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
    
    def handle_response(self, response: dict):
        text = response.get('response', 'No response')
        cached = " (cached)" if response.get('cached') else ""
        self.add_message(f"AI{cached}", text, "#00ff88")


class StatusPanel(QWidget):
    def __init__(self, metrics: PerformanceMetrics, parent=None):
        super().__init__(parent)
        self.metrics = metrics
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        header = QLabel("Performance Monitor")
        header.setStyleSheet("font-weight:bold;font-size:14px;color:#ff2b2b;padding:4px")
        layout.addWidget(header)
        
        stats_widget = QWidget()
        stats_layout = QGridLayout(stats_widget)
        stats_layout.setSpacing(10)
        
        self.ram_label = QLabel("RAM: -- MB")
        self.cpu_label = QLabel("CPU: --%")
        self.gc_label = QLabel("GC: 0")
        self.opt_label = QLabel("Optimizations: 0")
        self.uptime_label = QLabel("Uptime: 0s")
        self.status_label = QLabel("Status: OK")
        self.status_label.setStyleSheet("color:#00ff88")
        
        stats_layout.addWidget(self.ram_label, 0, 0)
        stats_layout.addWidget(self.cpu_label, 0, 1)
        stats_layout.addWidget(self.gc_label, 1, 0)
        stats_layout.addWidget(self.opt_label, 1, 1)
        stats_layout.addWidget(self.uptime_label, 2, 0)
        stats_layout.addWidget(self.status_label, 2, 1)
        
        layout.addWidget(stats_widget)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                color: #9a8f8f;
                font-family: monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.log_area)
        
        layout.addStretch()
        
    def update_stats(self, data: dict):
        self.ram_label.setText(f"RAM: {data.get('mem_mb', 0):.1f} MB")
        self.cpu_label.setText(f"CPU: {data.get('cpu_percent', 0):.1f}%")
        self.gc_label.setText(f"GC: {self.metrics.gc_count}")
        self.opt_label.setText(f"Optimizations: {self.metrics.optimizations_count}")
        
        uptime = int(time.time() - self.metrics.start_time)
        mins, secs = divmod(uptime, 60)
        hours, mins = divmod(mins, 60)
        self.uptime_label.setText(f"Uptime: {hours}h {mins}m {secs}s")
        
        status = data.get('status', 'optimal')
        colors = {'optimal': '#00ff88', 'warning': '#ffaa00', 'critical': '#ff4444'}
        self.status_label.setText(f"Status: {status.upper()}")
        self.status_label.setStyleSheet(f"color:{colors.get(status, '#e8e8e8')}")
        
    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")


class CrimsonGXBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.settings = PerformanceSettings.load()
        self.metrics = PerformanceMetrics()
        self.history = []
        self.bookmarks = []
        
        self._load_data()
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
        
        self.assistant = AIAssistant(self.settings)
        self.assistant.status_signal.connect(self._log_status)
        
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
        
        nav_bar = QWidget()
        nav_bar.setFixedHeight(48)
        nav_bar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #1a1a20, stop:1 #0a0a0f);
                border-bottom: 1px solid rgba(255,255,255,0.08);
            }
        """)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 4, 8, 4)
        nav_layout.setSpacing(6)
        
        btn_style = """
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
                color: #9a8f8f;
                font-size: 16px;
            }
            QToolButton:hover {
                background: rgba(255,43,43,0.15);
                color: #ff2b2b;
            }
            QToolButton:pressed {
                background: rgba(255,43,43,0.25);
            }
        """
        
        self.back_btn = QToolButton()
        self.back_btn.setText("<")
        self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(lambda: self.webview.back())
        nav_layout.addWidget(self.back_btn)
        
        self.forward_btn = QToolButton()
        self.forward_btn.setText(">")
        self.forward_btn.setStyleSheet(btn_style)
        self.forward_btn.clicked.connect(lambda: self.webview.forward())
        nav_layout.addWidget(self.forward_btn)
        
        self.reload_btn = QToolButton()
        self.reload_btn.setText("R")
        self.reload_btn.setStyleSheet(btn_style)
        self.reload_btn.clicked.connect(lambda: self.webview.reload())
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
            QLineEdit {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                color: #e8e8e8;
                padding: 8px 14px;
                font-size: 13px;
                selection-background-color: #ff2b2b;
            }
            QLineEdit:focus {
                border-color: #ff2b2b;
                background: rgba(255,255,255,0.08);
            }
        """)
        nav_layout.addWidget(self.url_bar, 1)
        
        self.bookmark_btn = QToolButton()
        self.bookmark_btn.setText("*")
        self.bookmark_btn.setStyleSheet(btn_style)
        self.bookmark_btn.clicked.connect(self._toggle_bookmark)
        nav_layout.addWidget(self.bookmark_btn)
        
        self.ai_btn = QToolButton()
        self.ai_btn.setText("AI")
        self.ai_btn.setStyleSheet(btn_style.replace("#9a8f8f", "#00ff88"))
        self.ai_btn.clicked.connect(self._toggle_ai_panel)
        nav_layout.addWidget(self.ai_btn)
        
        self.settings_btn = QToolButton()
        self.settings_btn.setText("S")
        self.settings_btn.setStyleSheet(btn_style)
        self.settings_btn.clicked.connect(self._open_settings)
        nav_layout.addWidget(self.settings_btn)
        
        main_layout.addWidget(nav_bar)
        
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.webview = OptimizedWebEngineView(self.settings)
        self.webview.urlChanged.connect(self._on_url_changed)
        self.webview.loadFinished.connect(self._on_load_finished)
        self.webview.loadProgress.connect(self._on_load_progress)
        
        home_path = CONFIG_DIR / "home.html"
        home_path.write_text(HOME_HTML, encoding="utf-8")
        self.webview.setUrl(QUrl.fromLocalFile(str(home_path)))
        
        content_splitter.addWidget(self.webview)
        
        self.side_panel = QStackedWidget()
        self.side_panel.setMaximumWidth(350)
        self.side_panel.setMinimumWidth(280)
        
        self.ai_panel = AIAssistantPanel(self.assistant)
        self.status_panel = StatusPanel(self.metrics)
        
        self.side_panel.addWidget(self.ai_panel)
        self.side_panel.addWidget(self.status_panel)
        self.side_panel.setCurrentWidget(self.ai_panel)
        self.side_panel.hide()
        
        content_splitter.addWidget(self.side_panel)
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 0)
        
        main_layout.addWidget(content_splitter, 1)
        
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #0a0a0f;
                color: #9a8f8f;
                border-top: 1px solid rgba(255,255,255,0.08);
                font-size: 11px;
                padding: 4px 8px;
            }
        """)
        self.setStatusBar(self.status_bar)
        
        self.ram_status = QLabel("RAM: --")
        self.cpu_status = QLabel("CPU: --")
        self.fps_status = QLabel("Status: OK")
        self.fps_status.setStyleSheet("color:#00ff88")
        
        self.status_bar.addWidget(self.ram_status)
        self.status_bar.addWidget(self.cpu_status)
        self.status_bar.addPermanentWidget(self.fps_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setMaximumHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255,255,255,0.1);
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #ff2b2b;
                border-radius: 4px;
            }
        """)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.setStyleSheet("""
            QMainWindow {
                background: #0a0a0f;
            }
            QSplitter::handle {
                background: rgba(255,255,255,0.05);
                width: 2px;
            }
            QSplitter::handle:hover {
                background: #ff2b2b;
            }
        """)
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+L"), self, lambda: self.url_bar.setFocus())
        QShortcut(QKeySequence("Ctrl+R"), self, lambda: self.webview.reload())
        QShortcut(QKeySequence("F5"), self, lambda: self.webview.reload())
        QShortcut(QKeySequence("Alt+Left"), self, lambda: self.webview.back())
        QShortcut(QKeySequence("Alt+Right"), self, lambda: self.webview.forward())
        QShortcut(QKeySequence("Ctrl+D"), self, self._toggle_bookmark)
        QShortcut(QKeySequence("Ctrl+,"), self, self._open_settings)
        QShortcut(QKeySequence("Ctrl+Shift+A"), self, self._toggle_ai_panel)
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, self._toggle_status_panel)
        QShortcut(QKeySequence("Ctrl+H"), self, self._go_home)
        QShortcut(QKeySequence("Escape"), self, lambda: self.side_panel.hide())
        
    def _start_services(self):
        if self.settings.ai_optimizer:
            self.optimizer.start()
        if self.settings.ai_assistant:
            self.assistant.start()
        if self.settings.local_server and HAS_FLASK:
            self.server.start()
            
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(1000)
    
    def _navigate(self):
        url = self.url_bar.text().strip()
        if not url:
            return
            
        if url.startswith('http://') or url.startswith('https://'):
            self.webview.setUrl(QUrl(url))
        elif '.' in url and ' ' not in url:
            self.webview.setUrl(QUrl(f'https://{url}'))
        else:
            self.webview.setUrl(QUrl(f'https://duckduckgo.com/?q={url}'))
    
    def _go_home(self):
        home_path = CONFIG_DIR / "home.html"
        self.webview.setUrl(QUrl.fromLocalFile(str(home_path)))
    
    def _on_url_changed(self, url: QUrl):
        self.url_bar.setText(url.toString())
        
    def _on_load_finished(self, ok: bool):
        self.progress_bar.hide()
        self.metrics.page_loads += 1
        
        url = self.webview.url().toString()
        if ok and not url.startswith('file://'):
            self.history.append({
                'url': url,
                'title': self.webview.title() or url,
                'timestamp': time.time()
            })
    
    def _on_load_progress(self, progress: int):
        if progress < 100:
            self.progress_bar.show()
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.hide()
    
    def _toggle_bookmark(self):
        url = self.webview.url().toString()
        if url in [b['url'] for b in self.bookmarks]:
            self.bookmarks = [b for b in self.bookmarks if b['url'] != url]
            self.status_bar.showMessage("Bookmark removed", 2000)
        else:
            self.bookmarks.append({
                'url': url,
                'title': self.webview.title() or url,
                'timestamp': time.time()
            })
            self.status_bar.showMessage("Bookmark added", 2000)
    
    def _toggle_ai_panel(self):
        if self.side_panel.isVisible() and self.side_panel.currentWidget() == self.ai_panel:
            self.side_panel.hide()
        else:
            self.side_panel.setCurrentWidget(self.ai_panel)
            self.side_panel.show()
    
    def _toggle_status_panel(self):
        if self.side_panel.isVisible() and self.side_panel.currentWidget() == self.status_panel:
            self.side_panel.hide()
        else:
            self.side_panel.setCurrentWidget(self.status_panel)
            self.side_panel.show()
    
    def _open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.webview.apply_settings()
            self.status_bar.showMessage("Settings saved", 2000)
            
            if self.settings.ai_optimizer and not self.optimizer.isRunning():
                self.optimizer.start()
            if self.settings.ai_assistant and not self.assistant.isRunning():
                self.assistant.start()
    
    def _handle_optimization(self, data: dict):
        self.ram_status.setText(f"RAM: {data.get('mem_mb', 0):.0f}MB")
        self.cpu_status.setText(f"CPU: {data.get('cpu_percent', 0):.0f}%")
        
        status = data.get('status', 'optimal')
        colors = {'optimal': '#00ff88', 'warning': '#ffaa00', 'critical': '#ff4444'}
        self.fps_status.setText(f"Status: {status.upper()}")
        self.fps_status.setStyleSheet(f"color:{colors.get(status, '#e8e8e8')}")
        
        if self.side_panel.currentWidget() == self.status_panel:
            self.status_panel.update_stats(data)
    
    def _update_status(self):
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                cpu = process.cpu_percent()
                self.ram_status.setText(f"RAM: {mem_mb:.0f}MB")
                self.cpu_status.setText(f"CPU: {cpu:.0f}%")
            except:
                pass
    
    def _log_status(self, message: str):
        self.status_bar.showMessage(message, 3000)
        if self.side_panel.currentWidget() == self.status_panel:
            self.status_panel.add_log(message)
    
    def closeEvent(self, event):
        self._save_data()
        
        self.optimizer.stop()
        self.assistant.stop()
        self.server.stop()
        
        self.optimizer.wait(1000)
        self.assistant.wait(1000)
        
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
