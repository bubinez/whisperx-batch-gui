#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎙️ WhisperX GUI (для версии >=3.0)
- large-v3
- русский язык
- таймкоды по словам
- диаризация через официальный DiarizationPipeline
- логирование ошибок в error.log
"""
import os
import sys
import json
import traceback
import subprocess
import tempfile
from pathlib import Path

# Кэш моделей Hugging Face в .venv (если не задан HF_HUB_CACHE в run.bat)
_venv_cache = Path(__file__).parent.resolve() / ".venv" / "cache" / "huggingface" / "hub"
os.environ.setdefault("HF_HUB_CACHE", str(_venv_cache))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

import torch
import torchaudio
import whisperx

# Исправление для PyTorch 2.6+: разрешаем загрузку объектов из omegaconf и dict_keys
try:
    import collections
    import typing
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False

    # Список классов для разблокировки
    safe_globals = [
        typing.Any,
        collections.OrderedDict,
        collections.deque,
        collections.defaultdict,
        collections.Counter,
        type({}.keys()), # dict_keys
        list,
        dict,
        set,
        str,
        int,
        float,
        bool
    ]

    # Динамический импорт из omegaconf
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata, Node, Metadata
        from omegaconf.nodes import AnyNode, ValueNode, StringNode, IntegerNode, FloatNode, BooleanNode
        
        safe_globals.extend([
            ListConfig, DictConfig, 
            ContainerMetadata, Node, Metadata,
            AnyNode, ValueNode, StringNode, IntegerNode, FloatNode, BooleanNode
        ])
    except ImportError:
        pass

    if has_numpy:
        safe_globals.extend([
            np.dtype,
            np.core.multiarray._reconstruct,
            np.ndarray
        ])
    
    # Регистрация безопасных глобалов
    torch.serialization.add_safe_globals(safe_globals)
    
    # Глобальное отключение weights_only для torch.load
    # Это гарантирует работу даже если библиотека явно передает weights_only=True
    import functools
    original_load = torch.load
    def patched_load(*args, **kwargs):
        # Принудительно отключаем weights_only, так как мы доверяем локальным моделям
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
except Exception as e:
    print(f"Предупреждение при настройке safe_globals/patched_load: {e}")

# === Настройки ===
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "config.json"
RESULTS_DIR = SCRIPT_DIR / "results"
ERROR_LOG = SCRIPT_DIR / "error.log"
RESULTS_DIR.mkdir(exist_ok=True)

# Поддерживаемые форматы
SUPPORTED_FORMATS = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma", "*.aiff"]


def log_error(e: Exception):
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write("=== ОШИБКА ПРИ ОБРАБОТКЕ ===\n")
        f.write(str(e) + "\n\n")
        traceback.print_exc(file=f)
    print(f"Ошибка сохранена в: {ERROR_LOG}")


def load_config():
    # Настройки по умолчанию
    default_config = {
        "hf_token": "",
        "diarization_settings": {
            "min_speakers": 2,
            "max_speakers": 6,
            "cluster_method": "average",
            "threshold": 0.5
        },
        "whisper_settings": {
            "model": "large-v3",
            "batch_size": 8,
            "chunk_size": 30,
            # temperature удален, так как вызывает ошибку
        }
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
            # Объединяем загруженную конфигурацию с конфигурацией по умолчанию
            # Это гарантирует, что все ключи по умолчанию присутствуют
            # и новые настройки будут добавлены, если их нет в файле
            config = default_config.copy()
            config.update(loaded_config)
            # Дополнительная проверка для вложенных словарей
            for key in ["diarization_settings", "whisper_settings"]:
                if key in loaded_config and isinstance(loaded_config[key], dict):
                    config[key].update(loaded_config[key])
            # Удаляем 'temperature' из whisper_settings, если он там есть,
            # чтобы избежать ошибки TypeError при транскрибации.
            if "temperature" in config["whisper_settings"]:
                del config["whisper_settings"]["temperature"]
            return config
        except json.JSONDecodeError:
            # Если файл config.json поврежден, используем настройки по умолчанию
            print("Предупреждение: config.json поврежден или некорректен. Используются настройки по умолчанию.")
            return default_config
    return default_config

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def ensure_hf_token():
    config = load_config()
    if not config.get("hf_token") or config["hf_token"] == "your_token_here":
        root = tk.Tk()
        root.withdraw()
        token = simple_input_dialog("Введите ваш Hugging Face Token", 
                                    "Получите токен на https://huggingface.co/settings/tokens\n"
                                    "И примите лицензии на модели:\n"
                                    "• pyannote/speaker-diarization-3.1\n"
                                    "• pyannote/segmentation-3.0")
        if not token:
            messagebox.showerror("Ошибка", "Токен обязателен для диаризации!")
            sys.exit(1)
        config["hf_token"] = token.strip()
        save_config(config)
    return config["hf_token"]

def show_settings_dialog(current_config):
    """Диалог для настройки параметров диаризации"""
    win = tk.Toplevel()
    win.title("Настройки диаризации")
    win.geometry("600x800")
    win.minsize(500, 700)
    win.resizable(True, True)
    win.grab_set()
    win.focus_set()

    # Принудительное обновление компоновки окна
    win.update_idletasks() 

    # Создаем копию для работы, чтобы не изменять self.config напрямую до сохранения
    # Это позволяет отменить изменения, если пользователь нажмет "Закрыть" без сохранения
    temp_config = current_config.copy() 
    
    # Флаг, который будет указывать, были ли настройки сохранены
    settings_saved = False

    main_frame = ttk.Frame(win, padding=15)
    main_frame.pack(fill="both", expand=True)

    # УБРАН ЗАГОЛОВОК "Настройки диаризации"
    # Hugging Face Token
    ttk.Label(main_frame, text="Hugging Face Token:").pack(anchor="w", pady=2)
    hf_token_var = tk.StringVar(value=temp_config.get("hf_token", ""))
    hf_token_entry = ttk.Entry(main_frame, textvariable=hf_token_var, width=50)
    hf_token_entry.pack(anchor="w", pady=2)
    ttk.Label(main_frame, text="Необходим для диаризации (определения говорящих).", 
              font=("Arial", 8), foreground="gray", justify="left").pack(anchor="w", padx=10)

    # Минимальное количество говорящих
    ttk.Label(main_frame, text="Минимальное количество говорящих:").pack(anchor="w", pady=2)
    min_speakers_var = tk.StringVar(value=str(temp_config["diarization_settings"]["min_speakers"]))
    min_speakers_spin = ttk.Spinbox(main_frame, from_=1, to=10, width=10, textvariable=min_speakers_var)
    min_speakers_spin.pack(anchor="w", pady=2)

    # Максимальное количество говорящих
    ttk.Label(main_frame, text="Максимальное количество говорящих:").pack(anchor="w", pady=2)
    max_speakers_var = tk.StringVar(value=str(temp_config["diarization_settings"]["max_speakers"]))
    max_speakers_spin = ttk.Spinbox(main_frame, from_=1, to=20, width=10, textvariable=max_speakers_var)
    max_speakers_spin.pack(anchor="w", pady=2)

    # Метод кластеризации
    ttk.Label(main_frame, text="Метод кластеризации:").pack(anchor="w", pady=2)
    cluster_var = tk.StringVar(value=temp_config["diarization_settings"]["cluster_method"])
    cluster_combo = ttk.Combobox(main_frame, textvariable=cluster_var, 
                                values=["average", "centroid", "single", "complete"], width=15, state="readonly")
    cluster_combo.pack(anchor="w", pady=2)
    ttk.Label(main_frame, text="average: баланс (дефолт). complete: строго (для похожих голосов).\nsingle: мягко (может объединять). centroid: по центрам.", 
              font=("Arial", 8), foreground="gray", justify="left").pack(anchor="w", padx=10)


    # Порог
    ttk.Label(main_frame, text="Порог чувствительности (0.1-0.9):").pack(anchor="w", pady=2)
    threshold_var = tk.DoubleVar(value=temp_config["diarization_settings"]["threshold"])
    
    # Label для отображения текущего значения порога
    threshold_value_label = ttk.Label(main_frame, text=f"Текущее значение: {threshold_var.get():.2f}")
    threshold_value_label.pack(anchor="w", padx=10)

    def update_threshold_label(val):
        threshold_value_label.config(text=f"Текущее значение: {float(val):.2f}")
        # threshold_var.set(float(val)) # Это уже делается автоматически, если variable привязана

    threshold_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, variable=threshold_var, 
                                orient="horizontal", command=update_threshold_label)
    threshold_scale.pack(anchor="w", fill="x", pady=2)
    ttk.Label(main_frame, text="Меньше -> чаще разделяет голоса. Больше -> чаще объединяет.", 
              font=("Arial", 8), foreground="gray", justify="left").pack(anchor="w", padx=10)


    # Модель транскрибации Whisper
    ttk.Label(main_frame, text="Модель транскрибации:").pack(anchor="w", pady=2)
    model_var = tk.StringVar(value=temp_config["whisper_settings"].get("model", "large-v3"))
    whisper_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    model_combo = ttk.Combobox(main_frame, textvariable=model_var, values=whisper_models, width=15, state="readonly")
    model_combo.pack(anchor="w", pady=2)
    ttk.Label(main_frame, text="tiny/base: быстро. large-v3: макс. качество (требует больше VRAM).",
              font=("Arial", 8), foreground="gray", justify="left").pack(anchor="w", padx=10)

    # Batch size
    ttk.Label(main_frame, text="Batch size (1-32):").pack(anchor="w", pady=2)
    batch_size_var = tk.StringVar(value=str(temp_config["whisper_settings"]["batch_size"]))
    batch_size_spin = ttk.Spinbox(main_frame, from_=1, to=32, width=10, textvariable=batch_size_var)
    batch_size_spin.pack(anchor="w", pady=2)
    ttk.Label(main_frame, text="1-4: слабые GPU/CPU. 8-16: средние GPU (6-8ГБ). 16-32: мощные GPU (10ГБ+).", 
              font=("Arial", 8), foreground="gray", justify="left").pack(anchor="w", padx=10)


    # Chunk size
    ttk.Label(main_frame, text="Размер чанка (секунды):").pack(anchor="w", pady=2)
    chunk_size_var = tk.StringVar(value=str(temp_config["whisper_settings"]["chunk_size"]))
    chunk_size_spin = ttk.Spinbox(main_frame, from_=5, to=60, width=10, textvariable=chunk_size_var)
    chunk_size_spin.pack(anchor="w", pady=2)
    # ПОЯСНЕНИЕ ДЛЯ РАЗМЕРА ЧАНКА
    ttk.Label(main_frame, text="Больший чанк: лучше контекст, больше памяти.\nМеньший чанк: меньше памяти, может быть медленнее.", 
              font=("Arial", 8), foreground="gray", justify="left").pack(anchor="w", padx=10)

    def apply_settings():
        nonlocal settings_saved # Позволяет изменять переменную из внешней области видимости
        try:
            # Обновляем temp_config значениями из GUI
            temp_config["hf_token"] = hf_token_var.get().strip()
            temp_config["diarization_settings"]["min_speakers"] = int(min_speakers_var.get())
            temp_config["diarization_settings"]["max_speakers"] = int(max_speakers_var.get())
            temp_config["diarization_settings"]["cluster_method"] = cluster_var.get()
            temp_config["diarization_settings"]["threshold"] = float(threshold_var.get())
            temp_config["whisper_settings"]["model"] = model_var.get().strip()
            temp_config["whisper_settings"]["batch_size"] = int(batch_size_var.get())
            temp_config["whisper_settings"]["chunk_size"] = int(chunk_size_var.get())
            
            save_config(temp_config) # Сохраняем обновленный temp_config в файл
            settings_saved = True # Устанавливаем флаг, что сохранение произошло
            messagebox.showinfo("Настройки", "Настройки успешно сохранены!")
        except ValueError as e:
            messagebox.showerror("Ошибка", "Проверьте правильность введенных значений")
            settings_saved = False

    def close_dialog():
        win.destroy()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(pady=20, fill="x") # Добавлен fill="x" для лучшей компоновки кнопок
    
    # Новая кнопка "Сохранить" (без закрытия)
    ttk.Button(btn_frame, text="Сохранить", command=apply_settings, width=15).pack(side="left", padx=5, expand=True)
    # Новая кнопка "Закрыть" (без сохранения, если не нажали "Сохранить")
    ttk.Button(btn_frame, text="Закрыть", command=close_dialog, width=15).pack(side="left", padx=5, expand=True)

    # Обработка закрытия окна крестиком
    win.protocol("WM_DELETE_WINDOW", close_dialog)

    win.wait_window()
    
    # Возвращаем temp_config, только если было успешное сохранение
    return temp_config if settings_saved else None

def simple_input_dialog(title, message):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("500x200")
    win.resizable(False, False)
    win.grab_set()
    win.focus_set()

    tk.Label(win, text=message, wraplength=480, justify="left").pack(pady=10)
    entry = tk.Entry(win, width=60)
    entry.pack(pady=5)
    entry.focus()

    result = [None]
    def on_ok():
        result[0] = entry.get()
        win.destroy()
    def on_cancel():
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Отмена", command=on_cancel, width=10).pack(side="left", padx=5)

    win.wait_window()
    return result[0]

def open_results_folder():
    if sys.platform == "win32":
        os.startfile(RESULTS_DIR)
    elif sys.platform == "darwin":
        os.system(f'open "{RESULTS_DIR}"')
    else:
        os.system(f'xdg-open "{RESULTS_DIR}"')

def convert_audio_to_wav(input_path: Path, progress_callback) -> Path:
    """Конвертирует аудиофайл в WAV формат с помощью FFmpeg"""
    try:
        progress_callback(f"Конвертация {input_path.suffix} в WAV...")
        
        # Создаем временный файл
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / f"converted_{input_path.stem}.wav"
        
        # Команда FFmpeg для конвертации
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-y',  # Перезаписать если файл существует
            str(output_path)
        ]
        
        # Запускаем FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        
        if not output_path.exists():
            raise Exception("Конвертированный файл не создан")
            
        progress_callback("Конвертация завершена")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Конвертация заняла слишком много времени")
    except Exception as e:
        raise Exception(f"Ошибка конвертации: {str(e)}")

def load_audio_file(audio_path: Path, progress_callback):
    """Загружает аудиофайл, конвертируя при необходимости"""
    try:
        # Пытаемся загрузить напрямую
        progress_callback("Загрузка аудио...")
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return waveform, sample_rate
        
    except Exception as e:
        # Если не получилось, конвертируем в WAV
        progress_callback(f"Формат {audio_path.suffix} не поддерживается, конвертация...")
        
        # Проверяем наличие FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise Exception("FFmpeg не установлен. Установите FFmpeg и добавьте в PATH")

        # Конвертируем файл
        converted_path = convert_audio_to_wav(audio_path, progress_callback)

        # Загружаем конвертированный файл
        progress_callback("Загрузка конвертированного аудио...")
        waveform, sample_rate = torchaudio.load(str(converted_path))

        # Удаляем временный файл
        try:
            converted_path.unlink()
        except Exception as e:
            log_error(e)
            pass

        return waveform, sample_rate


# === Основная логика WhisperX (v3+) ===
def run_whisperx(audio_path: Path, hf_token: str, progress_callback):
    try:
        config = load_config()  # Всегда загружаем актуальный конфиг перед обработкой
        diarization_settings = config["diarization_settings"]
        whisper_settings = config["whisper_settings"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Выполняется на {device}")

        progress_callback("Загрузка модели Whisper...")
        model_name = whisper_settings.get("model", "large-v3")
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language="ru")

        # Загрузка аудио с возможной конвертацией
        waveform, sample_rate = load_audio_file(audio_path, progress_callback)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        audio = waveform.squeeze(0).numpy()

        progress_callback("Транскрибация...")
        # Удален параметр temperature, так как он вызывает ошибку TypeError
        result = model.transcribe(audio,
                                  batch_size=whisper_settings["batch_size"],
                                  chunk_size=whisper_settings["chunk_size"])

        progress_callback("Выравнивание по словам...")
        model_a, metadata = whisperx.load_align_model(language_code="ru", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        progress_callback("Диаризация (определение говорящих)...")
        from whisperx.diarize import DiarizationPipeline

        # Используем улучшенные настройки диаризации с обработкой ошибок загрузки модели
        try:
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
        except AttributeError as ae:
            if "'NoneType' object has no attribute 'to'" in str(ae):
                raise Exception(
                    "Ошибка загрузки модели диаризации. Вероятные причины:\n"
                    "1. Неверный Hugging Face Token в config.json\n"
                    "2. Вы не приняли условия использования моделей pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0 на сайте Hugging Face.\n"
                    "Пожалуйста, проверьте настройки и доступ к моделям."
                )
            raise ae
        except Exception as de:
            raise Exception(f"Ошибка при инициализации диаризации: {str(de)}")

        # Применяем настройки диаризации
        diarize_segments = diarize_model(
            audio,
            min_speakers=diarization_settings["min_speakers"],
            max_speakers=diarization_settings["max_speakers"]
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Сохранение результатов
        output_path = RESULTS_DIR / f"{audio_path.stem}_transcript.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result["segments"], f, ensure_ascii=False, indent=2)

        from datetime import timedelta

        def format_ts(s):
            td = timedelta(seconds=s)
            return str(td)[:-3].replace('.', ',').zfill(12)

        # SRT файл (с таймкодами)
        srt_path = RESULTS_DIR / f"{audio_path.stem}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], 1):
                speaker = seg.get("speaker", "SPEAKER_XX")
                text = seg.get("text", "").strip()
                if not text:
                    continue
                start = format_ts(seg["start"])
                end = format_ts(seg["end"])
                f.write(f'{i}\n{start} --> {end}\n{speaker}: {text}\n\n')

        # TXT файл (без таймкодов)
        txt_path = RESULTS_DIR / f"{audio_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                speaker = seg.get("speaker", "SPEAKER_XX")
                text = seg.get("text", "").strip()
                if text:
                    f.write(f'{speaker}: {text}\n')

        # DOC файл (форматированный текст для Word)
        doc_path = RESULTS_DIR / f"{audio_path.stem}_formatted.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            current_speaker = None
            for seg in result["segments"]:
                speaker = seg.get("speaker", "SPEAKER_XX")
                text = seg.get("text", "").strip()
                if not text:
                    continue

                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"{speaker}:\n")
                    current_speaker = speaker

                f.write(f"{text}\n")

        # Статистика по говорящим
        speakers = {}
        for seg in result["segments"]:
            speaker = seg.get("speaker", "SPEAKER_XX")
            if speaker in speakers:
                speakers[speaker] += 1
            else:
                speakers[speaker] = 1

        progress_callback("✅ Готово!")
        stats = f"Распознано говорящих: {len(speakers)}"
        return True, f"Созданы файлы:\n• {srt_path.name}\n• {txt_path.name}\n• {doc_path.name}\n• {output_path.name}\n\n{stats}"
    except Exception as e:
        log_error(e)
        progress_callback("❌ Ошибка! Подробности в error.log")
        return False, str(e)


# === GUI ===
class WhisperXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ WhisperX Транскрибатор (Пакетная обработка)")
        self.root.geometry("800x750")
        self.root.minsize(700, 600)
        self.root.resizable(True, True)

        self.queue = []
        self.queue_lock = threading.Lock()
        self.current_item_index = -1
        self.is_running = False
        self.cancel_event = threading.Event()
        self.hf_token = ensure_hf_token()
        self.config = load_config()
        
        # Данные для Drag n Drop
        self.drag_data = {"index": None, "y": 0}

        self.main_frame = ttk.Frame(root, padding=20)
        self.main_frame.pack(fill="both", expand=True)

        ttk.Label(self.main_frame, text="WhisperX Транскрибатор (Пакетная обработка)", font=("Arial", 12, "bold")).pack(pady=5)

        # Кнопка настроек
        settings_frame = ttk.Frame(self.main_frame)
        settings_frame.pack(pady=5, fill="x")
        ttk.Button(settings_frame, text="⚙️ Настройки диаризации", command=self.open_settings).pack(side="left")
        ttk.Button(settings_frame, text="📂 Открыть папку с результатами", command=open_results_folder).pack(side="right")

        # Очередь файлов
        queue_frame = ttk.LabelFrame(self.main_frame, text="Очередь файлов", padding=10)
        queue_frame.pack(fill="both", expand=True, pady=5)

        list_scroll = ttk.Scrollbar(queue_frame)
        list_scroll.pack(side="right", fill="y")

        self.queue_listbox = tk.Listbox(
            queue_frame, height=6, selectmode="single", # Для DND лучше single или ручное управление
            yscrollcommand=list_scroll.set, font=("Arial", 10),
            activestyle='none'
        )
        self.queue_listbox.pack(fill="both", expand=True, side="left")
        list_scroll.config(command=self.queue_listbox.yview)

        # Привязка событий для Drag n Drop
        self.queue_listbox.bind("<Button-1>", self.on_drag_start)
        self.queue_listbox.bind("<B1-Motion>", self.on_dragging)
        self.queue_listbox.bind("<ButtonRelease-1>", self.on_drag_drop)

        queue_btns = ttk.Frame(self.main_frame)
        queue_btns.pack(fill="x", pady=5)
        self.btn_add = ttk.Button(queue_btns, text="➕ Добавить файлы", command=self.select_file)
        self.btn_add.pack(side="left", padx=2)
        self.btn_remove = ttk.Button(queue_btns, text="❌ Удалить выбранные", command=self.remove_selected)
        self.btn_remove.pack(side="left", padx=2)
        self.btn_clear = ttk.Button(queue_btns, text="🧹 Очистить очередь", command=self.clear_queue)
        self.btn_clear.pack(side="left", padx=2)

        # Прогресс
        progress_frame = ttk.LabelFrame(self.main_frame, text="Прогресс", padding=10)
        progress_frame.pack(fill="x", pady=5)

        self.overall_label = ttk.Label(progress_frame, text="Всего: 0/0")
        self.overall_label.pack(anchor="w")
        self.overall_progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.overall_progress.pack(fill="x", pady=(0, 10))

        self.current_label = ttk.Label(progress_frame, text="Ожидание...", foreground="blue", wraplength=600)
        self.current_label.pack(anchor="w")
        self.current_progress = ttk.Progressbar(progress_frame, mode="indeterminate")
        # self.current_progress.pack(fill="x") # Будет паковаться программно

        # Кнопки управления
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill="x", pady=10)
        self.btn_start = ttk.Button(control_frame, text="▶️ Запустить пакетную обработку", command=self.start_transcribe)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=5)
        self.btn_stop = ttk.Button(control_frame, text="⏹️ Остановить", command=self.stop_transcribe, state="disabled")
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=5)

        # Информация о форматах и выходных файлах
        formats_text = "Поддерживаемые форматы:\n" + ", ".join([fmt.replace("*", "") for fmt in SUPPORTED_FORMATS])
        ttk.Label(self.main_frame, text=formats_text, foreground="blue", font=("Arial", 9)).pack(pady=5)

        info_text = "Создаваемые файлы:\n• .srt - субтитры с таймкодами\n• .txt - простой текст\n• _formatted.txt - форматированный текст\n• _transcript.json - полные данные"
        ttk.Label(self.main_frame, text=info_text, foreground="green", font=("Arial", 9)).pack(pady=5)

        ffmpeg_info = "⚠️ Для форматов M4A, AAC, WMA требуется FFmpeg\nСкачайте с ffmpeg.org и добавьте в PATH"
        ttk.Label(self.main_frame, text=ffmpeg_info, foreground="red", font=("Arial", 8)).pack(pady=5)

        # Советы по улучшению диаризации
        tips_text = "💡 Советы для лучшей диаризации:\n• Укажите точное количество говорящих в настройках\n• Используйте качественные записи без шума\n• Для 2-3 говорящих установите min=2, max=3"
        ttk.Label(self.main_frame, text=tips_text, foreground="purple", font=("Arial", 8), justify="left").pack(pady=5)

    def open_settings(self):
        """Открывает диалог настроек"""
        # Всегда загружаем актуальный конфиг перед открытием диалога
        self.config = load_config()
        # show_settings_dialog теперь возвращает конфиг, если он был сохранен, или None
        new_config = show_settings_dialog(self.config)
        if new_config:  # Если настройки были сохранены
            self.config = new_config
            self.hf_token = self.config.get("hf_token", "")
            if self.is_running:
                # Информируем пользователя о применении настроек к следующему файлу
                self.root.after(0, lambda: self.current_label.config(
                    text="⚙️ Настройки обновлены. Будут применены к следующему файлу в очереди."
                ))
        else:  # Если диалог был закрыт без сохранения
            messagebox.showinfo("Настройки", "Изменения не были сохранены.")

    def select_file(self):
        file_types = [("Аудио файлы", " ".join(SUPPORTED_FORMATS)), ("Все файлы", "*.*")]
        paths = filedialog.askopenfilenames(
            title="Выберите аудиофайлы",
            filetypes=file_types
        )
        if paths:
            with self.queue_lock:
                for p in paths:
                    path = Path(p)
                    if path not in self.queue:
                        self.queue.append(path)
                        self.queue_listbox.insert(tk.END, path.name)
            self.update_progress_labels()

    def remove_selected(self):
        selected_indices = list(self.queue_listbox.curselection())
        if not selected_indices:
            return

        with self.queue_lock:
            # Удаляем с конца, чтобы не сбить индексы при удалении
            for index in sorted(selected_indices, reverse=True):
                if index == self.current_item_index:
                    messagebox.showwarning("Внимание", f"Нельзя удалить файл '{self.queue[index].name}', так как он сейчас обрабатывается.")
                    continue
                
                self.queue.pop(index)
                self.queue_listbox.delete(index)
                
                # Если удалили элемент ПЕРЕД текущим, нужно сдвинуть индекс текущего
                if self.is_running and index < self.current_item_index:
                    self.current_item_index -= 1
        
        self.update_progress_labels()

    def clear_queue(self):
        with self.queue_lock:
            if self.is_running and self.current_item_index != -1:
                # Удаляем всё КРОМЕ текущего элемента
                current_file = self.queue[self.current_item_index]
                self.queue.clear()
                self.queue.append(current_file)
                
                self.queue_listbox.delete(0, tk.END)
                self.queue_listbox.insert(tk.END, f"▶️ {current_file.name}")
                self.current_item_index = 0
            else:
                self.queue.clear()
                self.queue_listbox.delete(0, tk.END)
                self.current_item_index = -1
        
        self.update_progress_labels()

    def update_progress_labels(self):
        with self.queue_lock:
            total = len(self.queue)
            current_display = self.current_item_index + 1 if self.current_item_index != -1 else 0
            self.overall_label.config(text=f"Всего: {current_display}/{total}")
            if total > 0:
                prog_val = (current_display / total) * 100
                self.overall_progress.config(value=prog_val)
            else:
                self.overall_progress.config(value=0)

    def stop_transcribe(self):
        if self.is_running:
            if messagebox.askyesno("Подтверждение", "Остановить обработку очереди? Текущий файл может быть завершен."):
                self.cancel_event.set()
                self.btn_stop.config(state="disabled")

    def start_transcribe(self):
        if not self.queue:
            messagebox.showwarning("Внимание", "Очередь пуста! Добавьте аудиофайлы.")
            return

        if self.is_running:
            return

        self.is_running = True
        self.cancel_event.clear()
        
        # Переключаем состояние кнопок
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        # Кнопки очереди теперь НЕ отключаются
        self.btn_add.config(state="normal")
        self.btn_remove.config(state="normal")
        self.btn_clear.config(state="normal")
        
        self.current_progress.pack(fill="x", pady=5)
        self.current_progress.start()

        thread = threading.Thread(target=self._transcribe_worker, daemon=True)
        thread.start()

    def _refresh_listbox_names(self):
        """Обновляет имена в listbox, убирая/добавляя индикатор текущего файла"""
        with self.queue_lock:
            # Сохраняем текущее выделение
            selected = self.queue_listbox.curselection()
            self.queue_listbox.delete(0, tk.END)
            for i, p in enumerate(self.queue):
                name = p.name
                if i == self.current_item_index:
                    name = f"▶️ {name}"
                else:
                    name = f"☰ {name}" # Иконка для перетаскивания
                self.queue_listbox.insert(tk.END, name)
            
            # Возвращаем выделение
            for idx in selected:
                if idx < self.queue_listbox.size():
                    self.queue_listbox.selection_set(idx)

    def on_drag_start(self, event):
        """Начало перетаскивания"""
        index = self.queue_listbox.nearest(event.y)
        if index < 0 or index >= len(self.queue):
            return
            
        if index == self.current_item_index:
            # Нельзя тащить текущий обрабатываемый файл
            self.drag_data["index"] = None
            return

        self.drag_data["index"] = index
        self.drag_data["y"] = event.y
        self.queue_listbox.selection_clear(0, tk.END)
        self.queue_listbox.selection_set(index)

    def on_dragging(self, event):
        """Процесс перетаскивания (визуальный фидбек)"""
        if self.drag_data["index"] is None:
            return
            
        target_index = self.queue_listbox.nearest(event.y)
        if target_index < 0 or target_index >= len(self.queue):
            return
            
        if target_index != self.queue_listbox.curselection()[0]:
            self.queue_listbox.selection_clear(0, tk.END)
            self.queue_listbox.selection_set(target_index)

    def on_drag_drop(self, event):
        """Завершение перетаскивания и реордеринг"""
        if self.drag_data["index"] is None:
            return
            
        from_idx = self.drag_data["index"]
        to_idx = self.queue_listbox.nearest(event.y)
        self.drag_data["index"] = None

        if from_idx == to_idx:
            return

        # Запрет на перемещение текущего обрабатываемого файла 
        # или перемещение ЧЕГО-ТО на его место (для простоты логики)
        if to_idx == self.current_item_index or from_idx == self.current_item_index:
            self._refresh_listbox_names()
            return

        with self.queue_lock:
            # Перемещаем в данных
            item = self.queue.pop(from_idx)
            self.queue.insert(to_idx, item)
            
            # Корректируем current_item_index если move повлиял на него
            if self.is_running and self.current_item_index != -1:
                # Если элемент прыгнул ЧЕРЕЗ текущий индекс
                if from_idx > self.current_item_index and to_idx <= self.current_item_index:
                    self.current_item_index += 1
                elif from_idx < self.current_item_index and to_idx >= self.current_item_index:
                    self.current_item_index -= 1

        self._refresh_listbox_names()
        self.update_progress_labels()

    def _transcribe_worker(self):
        success_count = 0
        error_count = 0
        
        def update_ui_status(msg, current_idx, total, current_name):
            self.root.after(0, lambda: self.current_label.config(text=msg))
            self.root.after(0, lambda: self.overall_label.config(text=f"Файл {current_idx + 1}/{total}: {current_name}"))
            if total > 0:
                prog_val = ((current_idx) / total) * 100
                self.root.after(0, lambda: self.overall_progress.config(value=prog_val))

        while True:
            # Выбираем следующий файл
            with self.queue_lock:
                self.current_item_index += 1
                if self.current_item_index >= len(self.queue) or self.cancel_event.is_set():
                    # Сбрасываем индекс, если закончили или прервали
                    if self.current_item_index >= len(self.queue):
                        self.current_item_index = -1
                    break
                
                audio_path = self.queue[self.current_item_index]
                total_files = len(self.queue)
                current_idx = self.current_item_index
                current_name = audio_path.name
                
                # Обновляем Listbox, чтобы показать ▶️
                self.root.after(0, self._refresh_listbox_names)

            if self.cancel_event.is_set():
                break
            
            update_ui_status(f"Обработка: {current_name}", current_idx, total_files, current_name)
            
            success, info = run_whisperx(audio_path, self.hf_token, 
                                        lambda msg: self.root.after(0, lambda: self.current_label.config(text=msg)))
            
            if success:
                success_count += 1
            else:
                error_count += 1

        # Завершение
        self.is_running = False
        self.root.after(0, lambda: self.current_progress.stop())
        self.root.after(0, lambda: self.current_progress.pack_forget())
        
        # Финальное обновление прогресса
        with self.queue_lock:
            total_at_end = len(self.queue)
            # Если мы закончили нормально, current_item_index будет -1 (из-за сброса выше)
            # или >= len(self.queue)
            prog_is_full = success_count + error_count >= total_at_end and total_at_end > 0
            self.root.after(0, lambda: self.overall_progress.config(value=100 if prog_is_full else self.overall_progress["value"]))
        
        # Восстанавливаем кнопки
        self.root.after(0, lambda: self.btn_start.config(state="normal"))
        self.root.after(0, lambda: self.btn_stop.config(state="disabled"))
        
        # Очищаем индикатор ▶️
        with self.queue_lock:
            self.current_item_index = -1
            self.root.after(0, self._refresh_listbox_names)

        report = f"Обработка завершена!\n\nУспешно: {success_count}\nОшибок: {error_count}"
        if self.cancel_event.is_set():
            report += "\nПроцесс был прерван пользователем."
            self.root.after(0, lambda: self.current_label.config(text="🛑 Обработка прервана"))
        else:
            self.root.after(0, lambda: self.current_label.config(text="✅ Готово"))
        
        self.root.after(0, lambda: messagebox.showinfo("Итог", report))


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperXGUI(root)
    root.mainloop()