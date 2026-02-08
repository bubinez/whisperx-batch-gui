import unittest
import os
import sys
from pathlib import Path
import torch

# Добавляем путь к основной директории, чтобы импортировать script.py
sys.path.append(str(Path(__file__).parent))

import script

class TestFullDiarizationProcess(unittest.TestCase):
    """
    Интеграционный тест для проверки полного процесса транскрибации и диаризации.
    Использует реальный аудиофайл и настройки из config.json.
    """

    def setUp(self):
        self.audio_path = Path(r"C:\PyProj\win-gui-whisperx-main\test_materials\1925.aac")
        self.hf_token = script.load_config().get("hf_token")
        
        if not self.audio_path.exists():
            self.skipTest(f"Аудиофайл не найден: {self.audio_path}")
        
        if not self.hf_token or self.hf_token == "your_token_here":
            self.skipTest("Hugging Face Token не настроен в config.json")

    def test_run_whisperx_full_cycle(self):
        print(f"\nЗапуск полной обработки для: {self.audio_path.name}")
        
        # Минимальные настройки для теста (модель tiny)
        config = script.load_config()
        config["whisper_settings"]["model"] = "tiny"
        script.save_config(config)
        
        def progress_callback(msg):
            try:
                print(f"[PROGRESS] {msg}")
            except UnicodeEncodeError:
                # Если консоль не поддерживает юникод (например, CP1251 на Windows), 
                # заменяем проблемные символы
                clean_msg = msg.encode('ascii', errors='replace').decode('ascii')
                print(f"[PROGRESS] {clean_msg}")

        success, result_info = script.run_whisperx(
            self.audio_path, 
            self.hf_token, 
            progress_callback
        )
        
        if not success:
            # Читаем лог ошибки для подробностей
            error_details = ""
            if os.path.exists("error.log"):
                with open("error.log", "r", encoding="utf-8") as f:
                    error_details = f.read()
            self.fail(f"Обработка завершилась с ошибкой: {result_info}\nПодробности из error.log:\n{error_details}")
        
        print(f"Результат: {result_info}")
        self.assertTrue(success)
        self.assertIn("Созданы файлы", result_info)
        
        # Проверяем наличие выходных файлов
        base_name = self.audio_path.stem
        results_dir = Path("results")
        expected_files = [
            results_dir / f"{base_name}.srt",
            results_dir / f"{base_name}.txt",
            results_dir / f"{base_name}_formatted.txt",
            results_dir / f"{base_name}_transcript.json",
        ]
        
        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"Выходной файл не найден: {file_path}")
            print(f"Проверен файл: {file_path}")

if __name__ == "__main__":
    unittest.main()
