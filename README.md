# howdyS — Optimized Face Authentication for Linux

A lightweight, modern fork of [Howdy](https://github.com/boltgolt/howdy) that replaces dlib with ONNX Runtime for fast, dependency-friendly face recognition.

Using PAM (Pluggable Authentication Modules), this works everywhere you would otherwise need your password: login, lock screen, sudo, su, etc.

## What's Different

| Feature | Original Howdy | howdyS |
|---------|---------------|--------|
| Face detection | dlib HOG/CNN (300MB models) | OpenCV Haar cascade (built-in) |
| Face recognition | dlib ResNet (150MB) | MobileFaceNet via ONNX Runtime (4MB) |
| Matching | Euclidean distance | Cosine similarity |
| Model download | 450MB at install time | Bundled in repo (4.2MB total) |
| Video backends | OpenCV + ffmpeg + pyv4l2 | OpenCV-only (clean, fast) |
| Build deps | boost, cmake, dlib compilation | meson + ninja only |
| Camera fallback | Single device | Multi-device (IR → RGB) |
| Camera init | Default settings | Config-driven MJPEG/FPS/resolution |

## Installation

### Build from source

```bash
sudo apt-get install -y python3 python3-pip python3-opencv python3-onnxruntime \
    libpam0g-dev libinih-dev libevdev-dev meson ninja-build

git clone https://github.com/hcjaat/howdyS.git
cd howdyS
meson setup build --prefix=/usr -Dconfig_dir=/etc/howdy \
    -Duser_models_dir=/etc/howdy/models -Dlog_path=/var/log/howdy \
    -Dpython_path=/usr/bin/python3 -Dinstall_pam_config=true
meson compile -C build
sudo meson install -C build
sudo pam-auth-update --package
```

### Install .deb

Download the latest `.deb` from the [Releases page](https://github.com/hcjaat/howdyS/releases) and install:

```bash
sudo dpkg -i howdy_*.deb
sudo apt-get install -f
```

## Quick Start

```bash
# Add your face
sudo howdy add

# Test it (look at the camera)
sudo ls /root
```

If you have an IR camera, it's used by default. Falls back to the RGB camera automatically.

## Configuration

Edit `/etc/howdy/config.ini`:

```ini
[video]
device_path = /dev/video2, /dev/video0   ; IR preferred, falls back to RGB
certainty = 7.5                           ; 0.0-1.0, higher = stricter (recommended: 7.5)
timeout = 4                               ; seconds before falling back to password
force_mjpeg = false                       ; MJPEG format (faster init)
dark_threshold = 60                       ; ignore frames darker than this (%)
```

## CLI

```bash
howdy [-U user] [-y] command [argument]
```

| Command    | Description                        |
|------------|------------------------------------|
| `add`      | Add a new face model               |
| `list`     | List saved face models             |
| `remove`   | Remove a specific model            |
| `clear`    | Remove all face models             |
| `test`     | Test camera and recognition        |
| `config`   | Edit configuration                 |
| `disable`  | Enable/disable howdy               |
| `snapshot` | Capture a camera snapshot          |

## Security

Face recognition is convenient, not a replacement for strong passwords. The PAM config is set to `[success=3 default=ignore]` — on failure it falls through to your password prompt, so there's no lockout risk.
