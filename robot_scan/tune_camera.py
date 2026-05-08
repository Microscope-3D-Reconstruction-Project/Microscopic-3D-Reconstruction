"""
robot_scan/tune_camera.py

Interactive tool for tuning microscope camera settings via OpenCV.
Shows a live view with trackbars for every adjustable property.

Not all properties are supported by every camera driver — the script
probes which ones actually take effect and reports them at startup.

Usage:
    python robot_scan/tune_camera.py
    python robot_scan/tune_camera.py --camera_source 4

Keys:
    s  — save current settings to robot_scan/camera_settings.json
    r  — reset all properties to their auto/default values
    q  — quit
"""

import argparse
import json

from pathlib import Path

import cv2
import numpy as np

from termcolor import colored

SETTINGS_PATH = Path(__file__).parent / "camera_settings.json"

# (display name, CAP_PROP constant, min, max, default)
# default=None means read the current value from the camera
PROPS = [
    ("Brightness", cv2.CAP_PROP_BRIGHTNESS, 0, 255, None),
    ("Contrast", cv2.CAP_PROP_CONTRAST, 0, 255, None),
    ("Saturation", cv2.CAP_PROP_SATURATION, 0, 255, None),
    ("Hue", cv2.CAP_PROP_HUE, -180, 180, None),
    ("Gain", cv2.CAP_PROP_GAIN, 0, 255, None),
    ("Exposure", cv2.CAP_PROP_EXPOSURE, -13, 0, None),
    ("Auto Exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 0, 3, 1),
    ("Auto WB", cv2.CAP_PROP_AUTO_WB, 0, 1, 0),
    ("WB Blue", cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 2000, 7500, None),
    ("WB Red", cv2.CAP_PROP_WHITE_BALANCE_RED_V, 2000, 7500, None),
    ("Sharpness", cv2.CAP_PROP_SHARPNESS, 0, 255, None),
    ("Backlight", cv2.CAP_PROP_BACKLIGHT, 0, 3, 0),
]

WIN = "tune_camera"


def _offset_to_cv(name: str, lo: int, hi: int, val: float) -> int:
    """Map a float property value to a trackbar int (0-based)."""
    return int(np.clip(val - lo, 0, hi - lo))


def _cv_to_offset(lo: int, val_bar: int) -> float:
    return float(lo + val_bar)


def probe_properties(cap: cv2.VideoCapture) -> list:
    """Return which properties the camera actually accepts."""
    supported = []
    for name, prop, lo, hi, _default in PROPS:
        cur = cap.get(prop)
        # -1 means the driver doesn't support this property at all
        if cur < 0:
            supported.append(False)
            print(f"  {colored('✗', 'red')}  {name:<14} (not supported)")
            continue
        mid = (lo + hi) / 2
        cap.set(prop, mid)
        after = cap.get(prop)
        writable = after >= 0 and abs(after - cur) > 0.5
        cap.set(prop, cur)  # restore
        supported.append(writable)
        status = colored("✓", "green") if writable else colored("~", "yellow")
        label = "writable" if writable else "read-only"
        print(f"  {status}  {name:<14} current={cur:.1f}  ({label})")
    return supported


def main(camera_source: int = 4) -> None:
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(colored(f"❌ Could not open camera {camera_source}", "red"))
        return

    print(colored(f"\nCamera {camera_source} opened. Probing properties...", "cyan"))
    supported = probe_properties(cap)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    # Pump the Qt event loop until the window handle is confirmed non-null.
    # Qt defers window creation to the next event cycle, so we must wait.
    ret, _frame = cap.read()
    if ret:
        cv2.imshow(
            WIN, cv2.resize(_frame, (_frame.shape[1] // 2, _frame.shape[0] // 2))
        )
    for _ in range(10):
        cv2.waitKey(50)

    # Build trackbars only for supported properties
    active = []
    for (name, prop, lo, hi, default), ok in zip(PROPS, supported):
        if not ok:
            continue
        cur = cap.get(prop)
        if default is not None and cur == 0:
            cur = default
        bar_val = _offset_to_cv(name, lo, hi, cur)
        cv2.createTrackbar(name, WIN, bar_val, hi - lo, lambda _: None)
        active.append((name, prop, lo, hi))

    if not active:
        print(colored("⚠ No writable properties found for this camera.", "yellow"))

    print(
        colored(
            f"\n{len(active)} writable properties — adjust sliders in the window.",
            "cyan",
        )
    )
    print(colored("  s=save  r=reset  q=quit", "cyan"))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Apply trackbar values
        for name, prop, lo, hi in active:
            bar = cv2.getTrackbarPos(name, WIN)
            cap.set(prop, _cv_to_offset(lo, bar))

        disp = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Overlay current values
        y = 20
        for name, prop, lo, hi in active:
            val = cap.get(prop)
            cv2.putText(
                disp,
                f"{name}: {val:.0f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            y += 16

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            settings = {}
            for name, prop, lo, hi in active:
                settings[name] = cap.get(prop)
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
            print(colored(f"✓ Settings saved → {SETTINGS_PATH}", "green"))
            print(json.dumps(settings, indent=2))

        elif key == ord("r"):
            # Re-enable auto modes and let camera pick defaults
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            print(colored("✓ Reset to auto exposure + auto white balance", "cyan"))
            # Refresh trackbars
            for name, prop, lo, hi in active:
                cur = cap.get(prop)
                cv2.setTrackbarPos(name, WIN, _offset_to_cv(name, lo, hi, cur))

    cv2.destroyAllWindows()
    cap.release()
    print(colored("Done.", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune microscope camera settings interactively."
    )
    parser.add_argument(
        "--camera_source", type=int, default=4, help="Camera device index."
    )
    args = parser.parse_args()
    main(camera_source=args.camera_source)
