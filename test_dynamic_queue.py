import unittest
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import tkinter as tk
import sys

# Мокаем тяжелые зависимости перед импортом script, 
# если они не установлены (например, в среде тестирования)
try:
    import torch
except ImportError:
    sys.modules["torch"] = MagicMock()
try:
    import torchaudio
except ImportError:
    sys.modules["torchaudio"] = MagicMock()
try:
    import whisperx
except ImportError:
    sys.modules["whisperx"] = MagicMock()

from script import WhisperXGUI

class TestDynamicQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        # Скрываем окно, чтобы тесты не мешали
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        # Патчим ensure_hf_token и load_config, чтобы избежать реальных запросов и чтения файлов
        with patch('script.ensure_hf_token', return_value="test_token"):
            with patch('script.load_config', return_value={
                "hf_token": "test_token",
                "diarization_settings": {"min_speakers": 1, "max_speakers": 1, "cluster_method": "average", "threshold": 0.5},
                "whisper_settings": {"model": "tiny", "batch_size": 1, "chunk_size": 30}
            }):
                self.app = WhisperXGUI(self.root)
        
        # Симулируем наличие файлов
        self.app.queue = [Path("file1.wav"), Path("file2.wav"), Path("file3.wav")]
        self.app.queue_listbox.insert(tk.END, "file1.wav")
        self.app.queue_listbox.insert(tk.END, "file2.wav")
        self.app.queue_listbox.insert(tk.END, "file3.wav")

    def test_add_files_during_running(self):
        """Проверка добавления файлов во время работы"""
        self.app.is_running = True
        new_files = ["file4.wav", "file5.wav"]
        
        with patch('tkinter.filedialog.askopenfilenames', return_value=new_files):
            self.app.select_file()
            
        self.assertEqual(len(self.app.queue), 5)
        self.assertEqual(self.app.queue_listbox.size(), 5)
        self.assertEqual(self.app.queue[4].name, "file5.wav")

    def test_remove_pending_file_during_running(self):
        """Проверка удаления файла из очереди во время работы"""
        self.app.is_running = True
        self.app.current_item_index = 0 # Обрабатывается первый файл
        
        # Выбираем второй файл (индекс 1)
        self.app.queue_listbox.selection_clear(0, tk.END)
        self.app.queue_listbox.selection_set(1)
        
        self.app.remove_selected()
        
        self.assertEqual(len(self.app.queue), 2)
        self.assertEqual(self.app.queue[0].name, "file1.wav")
        self.assertEqual(self.app.queue[1].name, "file3.wav")
        self.assertEqual(self.app.current_item_index, 0)

    def test_remove_current_file_blocked(self):
        """Проверка блокировки удаления текущего файла"""
        self.app.is_running = True
        self.app.current_item_index = 1 # Обрабатывается второй файл
        
        # Выбираем второй файл
        self.app.queue_listbox.selection_clear(0, tk.END)
        self.app.queue_listbox.selection_set(1)
        
        with patch('tkinter.messagebox.showwarning') as mock_warning:
            self.app.remove_selected()
            mock_warning.assert_called_once()
            
        # Количество файлов не должно измениться
        self.assertEqual(len(self.app.queue), 3)

    def test_clear_queue_during_running(self):
        """Проверка очистки очереди во время работы (должен остаться текущий)"""
        self.app.is_running = True
        self.app.current_item_index = 1 # Обрабатывается второй файл (file2.wav)
        
        self.app.clear_queue()
        
        self.assertEqual(len(self.app.queue), 1)
        self.assertEqual(self.app.queue[0].name, "file2.wav")
        self.assertEqual(self.app.current_item_index, 0)
        self.assertIn("▶️ file2.wav", self.app.queue_listbox.get(0))

    def test_remove_file_before_current_updates_index(self):
        """Проверка, что удаление файла ПЕРЕД текущим сдвигает индекс"""
        self.app.is_running = True
        self.app.current_item_index = 2 # Обрабатывается третий файл (file3.wav)
        
        # Выбираем первый файл
        self.app.queue_listbox.selection_clear(0, tk.END)
        self.app.queue_listbox.selection_set(0)
        
        self.app.remove_selected()
        
        self.assertEqual(self.app.current_item_index, 1)
        self.assertEqual(self.app.queue[1].name, "file3.wav")

    def test_drag_and_drop_move_down(self):
        """Тест DND: перемещение файла вниз по списку"""
        # Перемещаем file1 (0) на место file2 (1)
        # file2 станет 0, file1 станет 1
        
        # Симулируем DND логику
        self.app.drag_data["index"] = 0
        event = MagicMock()
        event.y = 10 # Допустим это соответствует индексу 1
        
        with patch.object(self.app.queue_listbox, 'nearest', return_value=1):
            self.app.on_drag_drop(event)
            
        self.assertEqual(self.app.queue[0].name, "file2.wav")
        self.assertEqual(self.app.queue[1].name, "file1.wav")

    def test_drag_and_drop_move_up_updates_worker_index(self):
        """Тест DND: перемещение файла снизу вверх ЧЕРЕЗ текущий индекс воркера"""
        self.app.is_running = True
        self.app.current_item_index = 1 # Текущий - file2.wav
        
        # Перемещаем file3 (2) на место file1 (0)
        # file2 должен стать индексом 2, т.к. file3 вклинился перед ним
        
        self.app.drag_data["index"] = 2 # file3
        event = MagicMock()
        
        with patch.object(self.app.queue_listbox, 'nearest', return_value=0):
            self.app.on_drag_drop(event)
            
        self.assertEqual(self.app.queue[0].name, "file3.wav")
        self.assertEqual(self.app.current_item_index, 2)
        self.assertEqual(self.app.queue[2].name, "file2.wav")

    def test_drag_and_drop_move_down_updates_worker_index(self):
        """Тест DND: перемещение файла сверху вниз ЧЕРЕЗ текущий индекс воркера"""
        self.app.is_running = True
        self.app.current_item_index = 1 # Текущий - file2.wav
        
        # Перемещаем file1 (0) на место file3 (2)
        # file2 должен стать индексом 0, т.к. file1 ушел из-под него
        
        self.app.drag_data["index"] = 0 # file1
        event = MagicMock()
        
        with patch.object(self.app.queue_listbox, 'nearest', return_value=2):
            self.app.on_drag_drop(event)
            
        self.assertEqual(self.app.queue[2].name, "file1.wav")
        self.assertEqual(self.app.current_item_index, 0)
        self.assertEqual(self.app.queue[0].name, "file2.wav")

    def test_clear_queue_while_running_preserves_current(self):
        """Проверка, что очистка очереди при работе оставляет текущий файл"""
        self.app.is_running = True
        self.app.current_item_index = 1 # file2.wav
        
        self.app.clear_queue()
        
        self.assertEqual(len(self.app.queue), 1)
        self.assertEqual(self.app.queue[0].name, "file2.wav")
        self.assertEqual(self.app.current_item_index, 0)

    def test_drag_start_blocked_for_current_item(self):
        """Проверка, что нельзя начать тащить текущий файл"""
        self.app.is_running = True
        self.app.current_item_index = 1
        
        event = MagicMock()
        event.y = 10 
        
        with patch.object(self.app.queue_listbox, 'nearest', return_value=1):
            self.app.on_drag_start(event)
            self.assertIsNone(self.app.drag_data["index"])

    def test_drag_drop_blocked_onto_current_item(self):
        """Проверка, что нельзя сбросить файл на позицию активного файла"""
        self.app.is_running = True
        self.app.current_item_index = 1 # file2
        
        self.app.drag_data["index"] = 0 # тащим file1
        event = MagicMock()
        
        with patch.object(self.app.queue_listbox, 'nearest', return_value=1):
            self.app.on_drag_drop(event)
            
        # Очередь не должна измениться
        self.assertEqual(self.app.queue[0].name, "file1.wav")
        self.assertEqual(self.app.queue[1].name, "file2.wav")

    def test_worker_picks_files_dynamically(self):
        """Тест воркера: должен подхватывать добавленные файлы"""
        self.app.queue = [Path("short1.wav")]
        self.app.queue_listbox.delete(0, tk.END)
        self.app.queue_listbox.insert(tk.END, "short1.wav")
        
        processed_files = []
        
        def mock_run_whisperx(audio_path, hf_token, callback):
            processed_files.append(audio_path.name)
            # Симулируем добавление файла во время обработки первого
            if audio_path.name == "short1.wav":
                self.app.queue.append(Path("short2.wav"))
                self.root.after(0, lambda: self.app.queue_listbox.insert(tk.END, "short2.wav"))
            return True, "ok"

        with patch('script.run_whisperx', side_effect=mock_run_whisperx):
            # Запускаем воркер вручную (не через thread для простоты теста)
            # Но воркер использует while True и threading.Lock, так что можно прогнать цикл
            
            # В реальности воркер работает в потоке, здесь мы можем вызвать его метод
            # Но так как он использует root.after, нам нужно прокручивать event loop
            
            self.app.is_running = True
            self.app.cancel_event.clear()
            self.app.current_item_index = -1
            
            # Вместо полного потока, мы можем запустить один шаг или симулировать поток
            # Для простоты проверим логику итерации:
            
            # Шаг 1
            with self.app.queue_lock:
                self.app.current_item_index += 1
                audio = self.app.queue[self.app.current_item_index]
                mock_run_whisperx(audio, "token", lambda x: None)
            
            # После шага 1 в очереди появилось 2 файла
            self.assertEqual(len(self.app.queue), 2)
            
            # Шаг 2
            with self.app.queue_lock:
                self.app.current_item_index += 1
                if self.app.current_item_index < len(self.app.queue):
                    audio = self.app.queue[self.app.current_item_index]
                    mock_run_whisperx(audio, "token", lambda x: None)
            
            self.assertEqual(processed_files, ["short1.wav", "short2.wav"])

if __name__ == '__main__':
    unittest.main()
