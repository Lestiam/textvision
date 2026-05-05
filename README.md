# TextVision

> Real-time digital magnifier with OCR and text-to-speech, built for low-vision users.

TextVision turns a webcam — or any portion of your screen — into an accessible reading aid.
It magnifies, applies high-contrast filters, recognizes text with Tesseract, and reads it
out loud. Everything runs locally; no cloud calls.

## Features

- **Two capture sources** — webcam or screen region, switchable from a single icon toggle.
- **Cursor magnifier overlay** — a topmost window that follows your mouse and shows the
  area under it already enlarged and post-processed.
- **Adaptive visual pipeline** — zoom 1×–10×, CLAHE adaptive contrast, brightness and
  contrast trims, color filters (high contrast B/W, black-on-yellow, yellow-on-black,
  inverted, grayscale).
- **Robust OCR** — auto-upscaling, denoising, Otsu/adaptive binarization, deskew, and
  multi-PSM scoring; runs off the UI thread.
- **Offline text-to-speech** via Windows SAPI5.
- **Minimalist dark UI** with large hit-targets, visible keyboard shortcuts and an icon
  switch for camera ↔ screen.

## Requirements

- Windows 10 or 11
- Python 3.11 or newer (developed and tested on 3.14)
- [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- A webcam (optional — screen mode works fine without one)

## Installation

### 1. Clone and install Python dependencies

```powershell
git clone https://github.com/0x4thur/textvision.git
cd textvision
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

Download and run the official Windows installer:
<https://github.com/UB-Mannheim/tesseract/wiki>

When prompted for "Additional language data", check **Portuguese** if you need
Portuguese OCR. The installer places Tesseract at
`C:\Program Files\Tesseract-OCR\` by default and the app will find it automatically.

### 3. (Optional) Add language packs without admin rights

If the installer doesn't include the language pack you need and you can't write to
`Program Files`, place the files in a local `tessdata/` folder at the project root:

```powershell
mkdir tessdata
Invoke-WebRequest "https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata" -OutFile tessdata/por.traineddata
Copy-Item "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" tessdata/
Copy-Item "C:\Program Files\Tesseract-OCR\tessdata\osd.traineddata" tessdata/
```

The app auto-detects a local `tessdata/` folder and uses it via `TESSDATA_PREFIX`. This
works around a pytesseract bug on Windows that mishandles paths with spaces in
`--tessdata-dir`.

## Running

```powershell
python main.py
```

## Usage

1. Point the webcam at any text (book, label, screen) — or press **S** to switch to
   screen mode and **R** to draw a region of interest on the desktop.
2. Adjust **zoom**, **brightness**, **contrast**, the **color filter** and **CLAHE** to
   make the text comfortable to read.
3. Press **O** to run OCR on the current frame, then **L** to read it aloud.
4. Press **M** to open a topmost magnifier window that follows your cursor.

### Keyboard shortcuts

| Key       | Action                            |
|-----------|-----------------------------------|
| `+` `-`   | Zoom in / out                     |
| `[` `]`   | Brightness                        |
| `,` `.`   | Contrast                          |
| `F`       | Cycle color filter                |
| `C`       | Toggle CLAHE                      |
| `O`       | Run OCR                           |
| `L`       | Read aloud                        |
| `S`       | Switch source (camera ↔ screen)   |
| `R`       | Select screen region              |
| `M`       | Toggle cursor magnifier           |
| `P`       | Pause source                      |
| `H`       | Freeze / unfreeze frame           |
| `Esc`     | Quit                              |

## Project structure

```
main.py             Tkinter UI, threading, top-level orchestration
vision.py           Image processing pipeline (zoom, CLAHE, filters)
ocr_engine.py       Tesseract wrapper, preprocessing, multi-PSM, TTS
screen_capture.py   mss-based screen stream, region selector, magnifier
requirements.txt    Python dependencies
tessdata/           Local language packs (gitignored)
```

## Architecture notes

- **Threaded capture.** Camera (OpenCV) and screen (mss) streams run on their own
  threads with a single-frame buffer guarded by a lock. The Tk main thread polls at a
  30 FPS cap and forces redraws via `update_idletasks()` to avoid the classic "video
  only updates while the window moves" Tk pitfall.
- **PhotoImage reuse.** The video `Label` keeps one `ImageTk.PhotoImage` and pastes
  pixels into it across frames; allocation only happens when the display size changes.
- **OCR off the UI thread.** Each recognize call runs in a worker thread and posts
  results back via `root.after`. Tesseract is invoked with multiple PSM modes; the
  best score (confidence × word density) wins, with early exit at high confidence.
- **Smart preprocessing for OCR** — grayscale → ×2 upscale (when small) →
  bilateral filter → Otsu *or* adaptive threshold (chosen by luminance std-dev) →
  optional deskew via `minAreaRect`.

## Dependencies

- [opencv-python](https://github.com/opencv/opencv-python) — camera capture and image
  processing
- [Pillow](https://python-pillow.org/) — Tk image bridge
- [pytesseract](https://github.com/madmaze/pytesseract) — Tesseract Python bindings
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — offline text-to-speech (SAPI5 on
  Windows)
- [mss](https://github.com/BoboTiG/python-mss) — multi-monitor screen capture
- [numpy](https://numpy.org/)

## Troubleshooting

- **OCR returns nothing or "language pack missing"** — install the Tesseract language
  pack or follow the instructions in the *Optional* installation step above.
- **"Could not open camera"** — close other apps that may hold the webcam (Zoom,
  Teams, browser tabs) and relaunch. The screen-capture mode works without a camera.
- **Choppy video** — the app caps at 30 FPS by design. If yours is much lower, check
  whether OneDrive is syncing the project folder on every keystroke (move it out of
  OneDrive if so).
