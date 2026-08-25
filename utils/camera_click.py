"""
utils/camera_click.py

Attaches a mouse callback to an existing OpenCV window, lets the user click
a point (marked with a crosshair), and returns the (u, v) pixel coordinates
in the original frame resolution after the user presses Enter.
"""

import cv2
import numpy as np


def wait_for_click(
    frame: np.ndarray,
    win_name: str = "Live View",
) -> tuple[int, int] | None:
    """
    Show `frame` in `win_name` (must already exist) and wait for a click.
    Returns (u, v) in full-resolution pixels after Enter, or None on Escape.
    """
    h, w = frame.shape[:2]
    scale = 0.5
    dw, dh = int(w * scale), int(h * scale)

    state = {"click": None, "disp": cv2.resize(frame, (dw, dh))}

    def _on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["click"] = (x, y)
        img = cv2.resize(frame, (dw, dh))
        r = max(10, min(dw, dh) // 60)
        cv2.line(img, (x - r, y), (x + r, y), (0, 0, 255), 2)
        cv2.line(img, (x, y - r), (x, y + r), (0, 0, 255), 2)
        cv2.circle(img, (x, y), r, (0, 0, 255), 2)
        state["disp"] = img
        cv2.imshow(win_name, img)

    cv2.setMouseCallback(win_name, _on_mouse)
    cv2.imshow(win_name, state["disp"])

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # Escape
            cv2.setMouseCallback(win_name, lambda *_: None)
            return None
        cv2.imshow(win_name, state["disp"])

    cv2.setMouseCallback(win_name, lambda *_: None)

    if state["click"] is None:
        return None

    u = int(round(state["click"][0] / scale))
    v = int(round(state["click"][1] / scale))
    print(f"  clicked: u={u}, v={v}")
    return (u, v)
