# 🎙️ whisperx-batch-gui

**Advanced Windows GUI for WhisperX with dynamic batch management, drag-and-drop queue, and speaker diarization support.**

> **Topics:** `whisperx`, `gui`, `transcription`, `diarization`, `python-3-13`, `windows-app`, `batch-processing`

A powerful graphical user interface for **WhisperX**, designed for batch transcription and speaker diarization of audio files. The project is optimized for Windows and supports dynamic queue management.

## ✨ Key Features

- **📦 Batch Processing**: Add any number of audio files to the queue.
- **⚡ Dynamic Queue**: Add or remove files directly while the processing is active.
- **🖱️ Drag-and-Drop**: Easily reorder files in the queue by dragging (using the `☰` handle icon).
- **🗣️ Diarization**: Automatic speaker identification (requires Hugging Face Token).
- **📝 Word-level Timestamps**: Highly accurate timestamps for every single word.
- **⚙️ Flexible Settings**: 
  - Choose Whisper models (tiny, base, small, medium, large-v2, large-v3).
  - Configure diarization parameters (number of speakers, sensitivity threshold).
  - Manage batch size and chunk size to optimize for your VRAM.
- **📄 Multiple Output Formats**: Save results in `.srt`, `.txt`, `.json`, and formatted text modes.

## 🚀 Installation

The project uses `uv` for automatic Python version management and dependency handling.

1. **Download the repository**.
2. **Install FFmpeg**:
   - Download from [ffmpeg.org](https://ffmpeg.org/).
   - Add the `bin` folder path to your system's `PATH` environment variable.
3. **Run the Installer**:
   - Execute `install.bat`. It will automatically download the correct Python version (3.13), create a virtual environment, and install all dependencies.
4. **Launch the App**:
   - Use `run.bat` for daily usage.

## 🛠️ Hugging Face Setup

For diarization to work, you need to obtain an access token:

1. Create an account on [Hugging Face](https://huggingface.co/).
2. Accept the model licenses (Accept License):
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a token in your profile settings (Access Tokens) and paste it into the application ("Diarization Settings" button).

## 🧪 Testing

The project includes an automated test suite to verify GUI logic and queue management.

To run tests:
```bash
python -m unittest test_dynamic_queue.py
```
*Note: Tests include mocks for heavy libraries (torch, whisperx), so they can be run even without ML dependencies installed.*

## 💻 Technical Details

- **PyTorch 2.6+ Compatibility**: The application includes specific fixes to allow loading model weights in recent PyTorch versions.
- **Multi-threading**: Audio processing runs in a background thread, keeping the GUI responsive.
- **Caching**: Hugging Face models are stored inside the `.venv/cache` folder, making it easy to keep your system clean.

---

### Special Thanks To:
- [oiik/win-gui-whisperx](https://github.com/oiik/win-gui-whisperx) — for the original implementation ideas.
- [Habr: Batch Transcription](https://habr.com/ru/articles/953320/) | [Habr: WhisperX Guide](https://habr.com/ru/articles/948894/)
