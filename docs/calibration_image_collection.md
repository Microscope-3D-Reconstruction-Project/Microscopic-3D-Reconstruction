# Calibration Image Collection — Best Practices

---

## Target Setup

- Use a **flat, rigid calibration target** (checkerboard or ChArUco board) — avoid foam boards that warp
- Ensure the **target is firmly fixed** and doesn't shift between shots
- The target should be **large enough** to fill at least 1/3 of the image at the closest pose
- Print on **matte paper** — glossy surfaces cause glare
- Use a **known, accurate square size** (measure physically, don't trust the printer)

---

## Lighting

- Use **diffuse, even lighting** — avoid direct spotlights or windows that cast shadows
- **No specular reflections** on the target; if present, reposition lights
- Keep lighting **consistent across all images** — don't move lights mid-session
- For microscope cameras: set **exposure manually** (no auto-exposure), same value for every frame

---

## Image Variety — The Most Important Part

Collect images that cover:

| What to Vary | Why |
|---|---|
| **Distance** (close, mid, far) | Constrains focal length and distortion |
| **Tilt around X-axis** (up/down) | Constrains tangential distortion |
| **Tilt around Y-axis** (left/right) | Same |
| **Rotation around Z-axis** (in-plane roll) | Decouples skew and aspect ratio |
| **Board position in the image** (see below) | Distortion varies across the image — you must sample all regions |

### Board Position in the Image — Deliberately Move It Around

Do not keep the board centered. The distortion model is fit per-pixel, so if all your boards land in the middle of the frame, the edges and corners of the image are unconstrained.

Aim to have the board appear in each of these regions across your image set:

```
┌─────────┬─────────┬─────────┐
│ top-    │  top    │  top-   │
│  left   │ center  │  right  │
├─────────┼─────────┼─────────┤
│  mid-   │ center  │  mid-   │
│  left   │ (least  │  right  │
│         │important│         │
├─────────┼─────────┼─────────┤
│ bot-    │ bottom  │  bot-   │
│  left   │ center  │  right  │
└─────────┴─────────┴─────────┘
```

- **Corners and edges are the most important** — that's where radial distortion is largest
- Center-only datasets consistently underestimate distortion at the image periphery
- A simple check: mentally divide your images into the 9 zones above and make sure you have coverage in all of them

**Minimum:** ~20–30 images  
**Ideal:** 50–80 images with good coverage of the above

---

## Image Quality Checks (Before Saving)

- All checkerboard corners are **fully visible** — no partial boards
- Image is **in focus** — blur kills corner detection accuracy
- No motion blur — robot must be **fully stopped** before capturing
- Board fills a **reasonable fraction of the frame** — not too small, not cropped
- Corner detection succeeds on the image before saving it

---

## What to Avoid

- **Planar degeneracy**: don't take all images from the same distance/angle — you need tilt
- **Too few images**: fewer than ~15 leads to poorly constrained intrinsics
- **Blurry images**: single blurry image can bias the entire calibration
- **Auto-exposure / auto-focus**: these change between shots and ruin consistency
- **Moving the target mid-session**: restart collection if target shifts

---

## Storage

- Save images at **full resolution** — don't downsample before calibrating
- Name images sequentially (e.g., `calib_000.png`, `calib_001.png`)
- Record the **actual physical square size in mm** alongside the images
- Keep a copy of the **raw images** before any preprocessing

---

## Validation

After calibration, always check:

- **Reprojection error** should be < 0.5 pixels (ideally < 0.3 px) for a microscope/machine vision camera
- Visually inspect reprojection overlays on a few images — errors should look random, not systematic
- If reprojection error is high on specific images, remove those images and recalibrate
- Test on **held-out images** not used in calibration
