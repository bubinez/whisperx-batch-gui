import unittest
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

# Добавляем путь к основной директории
sys.path.append(str(Path(__file__).parent))
import script

class TestWhisperXBatchGUI(unittest.TestCase):
    """
    Тесты для GUI WhisperX с поддержкой пакетной обработки.
    Проверяют логику очереди, управление состоянием кнопок и рабочий процесс.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()  # Скрываем основное окно во время тестов

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        # Мокаем ensure_hf_token и load_config, чтобы избежать реальных запросов/чтения файлов
        self.mock_config = {
            "hf_token": "fake_token",
            "diarization_settings": {"min_speakers": 1, "max_speakers": 2, "cluster_method": "average", "threshold": 0.5},
            "whisper_settings": {"model": "tiny", "batch_size": 1, "chunk_size": 30}
        }
        
        with patch('script.ensure_hf_token', return_value="fake_token"), \
             patch('script.load_config', return_value=self.mock_config):
            self.app = script.WhisperXGUI(self.root)
        
        # Очищаем очередь перед каждым тестом
        self.app.clear_queue()

    def test_add_files_to_queue(self):
        """Тест добавления файлов в очередь"""
        fake_files = [
            str(Path("C:/test/audio1.mp3")),
            str(Path("C:/test/audio2.wav"))
        ]
        
        with patch('tkinter.filedialog.askopenfilenames', return_value=fake_files):
            self.app.select_file()
            
        self.assertEqual(len(self.app.queue), 2)
        self.assertEqual(self.app.queue_listbox.size(), 2)
        self.assertEqual(self.app.queue[0].name, "audio1.mp3")
        self.assertIn("audio2.wav", self.app.queue_listbox.get(1))

    def test_remove_selected_from_queue(self):
        """Тест удаления выбранных файлов из очереди"""
        # Добавляем 3 файла
        self.app.queue = [Path("a.mp3"), Path("b.mp3"), Path("c.mp3")]
        for f in self.app.queue:
            self.app.queue_listbox.insert(tk.END, f.name)
            
        # "Выбираем" второй файл (индекс 1)
        self.app.queue_listbox.selection_set(1)
        
        self.app.remove_selected()
        
        self.assertEqual(len(self.app.queue), 2)
        self.assertEqual(self.app.queue_listbox.size(), 2)
        self.assertEqual(self.app.queue[0].name, "a.mp3")
        self.assertEqual(self.app.queue[1].name, "c.mp3")

    @patch('script.run_whisperx')
    @patch('script.load_config')
    def test_batch_processing_workflow(self, mock_load_config, mock_run):
        """Сложный тест: симуляция полного цикла пакетной обработки"""
        mock_run.return_value = (True, "Success")
        mock_load_config.return_value = self.mock_config
        
        # 1. Подготовка очереди
        self.app.queue = [Path("test1.mp3"), Path("test2.mp3")]
        for f in self.app.queue:
            self.app.queue_listbox.insert(tk.END, f.name)
            
        # 2. Запуск.
        with patch('threading.Thread') as mock_thread:
            # Превращаем поток в синхронный вызов для теста
            def side_effect(target, daemon):
                target() # Выполняем воркер прямо сейчас
                return MagicMock()
            mock_thread.side_effect = side_effect
            
            # Запускаем
            self.app.start_transcribe()
            
        # Запускаем цикл tkinter, чтобы обработать .after() события
        self.root.update()

        # 3. Проверки
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(mock_run.call_args_list[0][0][0], Path("test1.mp3"))
        
        # Проверяем состояние кнопок
        self.assertEqual(str(self.app.btn_start['state']), 'normal')
        self.assertEqual(str(self.app.btn_stop['state']), 'disabled')

    def test_cancellation(self):
        """Тест прерывания процесса"""
        self.app.is_running = True
        self.app.btn_stop.config(state="normal")
        
        with patch('tkinter.messagebox.askyesno', return_value=True):
            self.app.stop_transcribe()
            
        self.assertTrue(self.app.cancel_event.is_set())
        self.assertEqual(str(self.app.btn_stop['state']), 'disabled')

    @patch('script.run_whisperx')
    @patch('script.load_config')
    def test_batch_error_handling(self, mock_load_config, mock_run):
        """Тест обработки ошибок (один падает, другой продолжает)"""
        mock_run.side_effect = [
            (False, "Error 1"),
            (True, "Success 2")
        ]
        mock_load_config.return_value = self.mock_config
        
        self.app.queue = [Path("fail.mp3"), Path("win.mp3")]
        for f in self.app.queue:
            self.app.queue_listbox.insert(tk.END, f.name)
            
        with patch('threading.Thread') as mock_thread:
            def side_effect(target, daemon):
                target()
                return MagicMock()
            mock_thread.side_effect = side_effect
            
            with patch('tkinter.messagebox.showinfo') as mock_info:
                self.app.start_transcribe()
                self.root.update()
                # Проверяем отчет в конце
                report = mock_info.call_args[0][1]
                self.assertIn("Успешно: 1", report)
                self.assertIn("Ошибок: 1", report)

if __name__ == "__main__":
    unittest.main()
