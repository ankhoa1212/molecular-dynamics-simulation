"""Manual crop tool — browse images in a folder and draw/save crops.

Usage:
    python crop_tool.py [folder]

Controls:
    Left-click-drag   Draw a crop rectangle
    Right-click-drag  Pan the image
    Scroll wheel      Zoom in / out (centered on cursor)
    Left / Right      Previous / next image
    R                 Reset zoom and pan
    Ctrl+Z            Undo last action (draw / move / resize / delete)
    + / -             Zoom in / out (keyboard)
    E                 Toggle Edit Mode
    Delete/Backspace  Delete selected crop (in Edit Mode)

Edit Mode:
    Left-click crop   Select crop (yellow outline + corner handles)
    Drag crop body    Move crop
    Drag corner       Resize crop
    Click empty area  Deselect
"""

from __future__ import annotations

import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CROPS_SUBDIR = "crops"
# Matches: crop_<stem>_y<YYYY>_x<XXXX>.png  (optional _N suffix for duplicates)
CROP_PATTERN = re.compile(r"^crop_(.+)_y(\d+)_x(\d+)(?:_\d+)?\.png$")
MIN_CROP_PX = 5
ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 30.0
HANDLE_RADIUS = 6    # display half-size of corner handle squares (canvas px)
HANDLE_HIT    = 12   # hit-detection half-size for corner handles (canvas px)


class CropTool:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Manual Crop Tool")
        self.root.geometry("1280x800")
        self.root.minsize(640, 480)

        # ── Data state ───────────────────────────────────────────────────────
        self.folder: Optional[Path] = None
        self.image_paths: list[Path] = []
        self.current_index: int = -1
        self.pil_image: Optional[Image.Image] = None

        # ── Transform state ──────────────────────────────────────────────────
        # display_coord = image_coord * display_scale + offset + pan
        self.zoom_level: float = 1.0     # relative to fit-to-canvas scale
        self.pan_x: float = 0.0          # canvas-space additional offset
        self.pan_y: float = 0.0
        self.fit_scale: float = 1.0      # scale that fits image into canvas
        self.img_offset_x: float = 0.0   # letterbox margin
        self.img_offset_y: float = 0.0

        # ── Internal rendering ───────────────────────────────────────────────
        self._tk_img: Optional[ImageTk.PhotoImage] = None  # must hold reference
        self._resize_after_id: Optional[str] = None

        # ── Interaction state ────────────────────────────────────────────────
        self._drag_start: Optional[tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self._pan_start_canvas: Optional[tuple[int, int]] = None
        self._pan_start_offset: tuple[float, float] = (0.0, 0.0)

        # ── Crop tracking ────────────────────────────────────────────────────
        # Each entry: (x1, y1, x2, y2, Path)  — image-space pixel coordinates
        self._saved_crops: list[tuple[int, int, int, int, Path]] = []
        self._undo_stack: list[dict] = []  # {"type": "create"|"edit"|"delete", ...}

        # ── Edit mode state ──────────────────────────────────────────────────
        self._edit_mode: bool = False
        self._selected_idx: Optional[int] = None
        self._edit_drag_mode: Optional[str] = None   # 'move' | 'resize_nw' | etc.
        self._edit_drag_start_canvas: Optional[tuple[int, int]] = None
        self._edit_drag_crop_orig: Optional[tuple[int, int, int, int]] = None

        self._build_ui()
        self._bind_events()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Toolbar
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Open Folder", command=self.open_folder).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        _sep(toolbar)

        self._btn_prev = tk.Button(
            toolbar, text="← Prev", command=self.prev_image, state=tk.DISABLED
        )
        self._btn_prev.pack(side=tk.LEFT, padx=2, pady=3)

        self._lbl_counter = tk.Label(toolbar, text="No images", width=14)
        self._lbl_counter.pack(side=tk.LEFT, padx=4)

        self._btn_next = tk.Button(
            toolbar, text="Next →", command=self.next_image, state=tk.DISABLED
        )
        self._btn_next.pack(side=tk.LEFT, padx=2, pady=3)
        _sep(toolbar)

        tk.Button(toolbar, text="Zoom In (+)", command=self.zoom_in).pack(
            side=tk.LEFT, padx=2, pady=3
        )
        tk.Button(toolbar, text="Zoom Out (−)", command=self.zoom_out).pack(
            side=tk.LEFT, padx=2, pady=3
        )
        tk.Button(toolbar, text="Reset Zoom (R)", command=self.reset_transform).pack(
            side=tk.LEFT, padx=2, pady=3
        )
        _sep(toolbar)

        tk.Button(toolbar, text="Undo (Ctrl+Z)", command=self.undo_last_crop).pack(
            side=tk.LEFT, padx=2, pady=3
        )
        _sep(toolbar)

        self._btn_edit_mode = tk.Button(
            toolbar, text="Edit Mode (E)", command=self.toggle_edit_mode,
            relief=tk.RAISED,
        )
        self._btn_edit_mode.pack(side=tk.LEFT, padx=2, pady=3)

        self._btn_delete = tk.Button(
            toolbar, text="Delete Selected (Del)", command=self.delete_selected_crop,
            state=tk.DISABLED,
        )
        self._btn_delete.pack(side=tk.LEFT, padx=2, pady=3)

        # Canvas
        self.canvas = tk.Canvas(
            self.root, bg="#1e1e1e", cursor="crosshair", highlightthickness=0
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        self._status = tk.StringVar(value="Open a folder to begin.")
        tk.Label(
            self.root,
            textvariable=self._status,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Helvetica", 9),
            padx=4,
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self) -> None:
        root, canvas = self.root, self.canvas

        # Navigation
        root.bind("<Left>",  lambda _: self.prev_image())
        root.bind("<Right>", lambda _: self.next_image())

        # Zoom / reset
        root.bind("<r>",     lambda _: self.reset_transform())
        root.bind("<R>",     lambda _: self.reset_transform())
        root.bind("<plus>",  lambda _: self.zoom_in())
        root.bind("<equal>", lambda _: self.zoom_in())
        root.bind("<minus>", lambda _: self.zoom_out())

        # Undo
        root.bind("<Control-z>", lambda _: self.undo_last_crop())

        # Edit mode toggle / delete
        root.bind("<e>",         lambda _: self.toggle_edit_mode())
        root.bind("<E>",         lambda _: self.toggle_edit_mode())
        root.bind("<Delete>",    lambda _: self.delete_selected_crop())
        root.bind("<BackSpace>", lambda _: self.delete_selected_crop())

        # Scroll-to-zoom: Linux Button-4/5, Windows/Mac MouseWheel
        canvas.bind("<MouseWheel>", self._on_mousewheel)
        canvas.bind("<Button-4>",   self._on_mousewheel)
        canvas.bind("<Button-5>",   self._on_mousewheel)

        # Right-click pan
        canvas.bind("<ButtonPress-3>",   self._on_pan_start)
        canvas.bind("<B3-Motion>",       self._on_pan_drag)
        canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        # Left-click rubber-band crop
        canvas.bind("<ButtonPress-1>",   self._on_drag_start)
        canvas.bind("<B1-Motion>",       self._on_drag_motion)
        canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        # Window resize
        canvas.bind("<Configure>", self._on_canvas_configure)

    # ── Edit mode ────────────────────────────────────────────────────────────

    def toggle_edit_mode(self) -> None:
        self._edit_mode = not self._edit_mode
        if not self._edit_mode:
            self._selected_idx = None
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crop_orig = None
            self._btn_delete.config(state=tk.DISABLED)
        self._btn_edit_mode.config(relief=tk.SUNKEN if self._edit_mode else tk.RAISED)
        self.canvas.config(cursor="arrow" if self._edit_mode else "crosshair")
        self._redraw()
        self._update_status()

    def _find_crop_at(self, cx: float, cy: float) -> Optional[int]:
        """Return the index of the topmost saved crop whose canvas rect contains (cx, cy)."""
        for i in range(len(self._saved_crops) - 1, -1, -1):
            x1, y1, x2, y2, _ = self._saved_crops[i]
            rx1, ry1 = self._image_to_canvas(x1, y1)
            rx2, ry2 = self._image_to_canvas(x2, y2)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return i
        return None

    def _get_handle_at(self, cx: float, cy: float, idx: int) -> Optional[str]:
        """Return 'nw'|'ne'|'sw'|'se' if (cx, cy) is near a corner handle of crop[idx]."""
        x1, y1, x2, y2, _ = self._saved_crops[idx]
        rx1, ry1 = self._image_to_canvas(x1, y1)
        rx2, ry2 = self._image_to_canvas(x2, y2)
        for name, hx, hy in (("nw", rx1, ry1), ("ne", rx2, ry1),
                               ("sw", rx1, ry2), ("se", rx2, ry2)):
            if abs(cx - hx) <= HANDLE_HIT and abs(cy - hy) <= HANDLE_HIT:
                return name
        return None

    def _on_edit_start(self, event: tk.Event) -> None:
        cx, cy = float(event.x), float(event.y)
        # If a crop is already selected, check its handles first
        if self._selected_idx is not None:
            handle = self._get_handle_at(cx, cy, self._selected_idx)
            if handle:
                self._edit_drag_mode = f"resize_{handle}"
                self._edit_drag_start_canvas = (event.x, event.y)
                x1, y1, x2, y2, _ = self._saved_crops[self._selected_idx]
                self._edit_drag_crop_orig = (x1, y1, x2, y2)
                return
        # Hit-test all crops (topmost first)
        idx = self._find_crop_at(cx, cy)
        if idx is not None:
            self._selected_idx = idx
            self._btn_delete.config(state=tk.NORMAL)
            handle = self._get_handle_at(cx, cy, idx)
            self._edit_drag_mode = f"resize_{handle}" if handle else "move"
            self._edit_drag_start_canvas = (event.x, event.y)
            x1, y1, x2, y2, _ = self._saved_crops[idx]
            self._edit_drag_crop_orig = (x1, y1, x2, y2)
        else:
            # Click on empty area — deselect
            self._selected_idx = None
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crop_orig = None
            self._btn_delete.config(state=tk.DISABLED)
        self._redraw()

    def _on_edit_drag(self, event: tk.Event) -> None:
        if (self._edit_drag_mode is None
                or self._edit_drag_start_canvas is None
                or self._selected_idx is None
                or self._edit_drag_crop_orig is None
                or self.pil_image is None):
            return
        dcx = event.x - self._edit_drag_start_canvas[0]
        dcy = event.y - self._edit_drag_start_canvas[1]
        ds = self.display_scale
        dix = dcx / ds
        diy = dcy / ds
        ox1, oy1, ox2, oy2 = self._edit_drag_crop_orig
        iw, ih = self.pil_image.size
        _, _, _, _, path = self._saved_crops[self._selected_idx]
        if self._edit_drag_mode == "move":
            w, h = ox2 - ox1, oy2 - oy1
            nx1 = int(max(0, min(iw - w, ox1 + dix)))
            ny1 = int(max(0, min(ih - h, oy1 + diy)))
            nx2 = nx1 + w
            ny2 = ny1 + h
        else:
            corner = self._edit_drag_mode[len("resize_"):]
            nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2
            if "w" in corner:
                nx1 = int(max(0, min(ox2 - MIN_CROP_PX, ox1 + dix)))
            if "e" in corner:
                nx2 = int(max(ox1 + MIN_CROP_PX, min(iw, ox2 + dix)))
            if "n" in corner:
                ny1 = int(max(0, min(oy2 - MIN_CROP_PX, oy1 + diy)))
            if "s" in corner:
                ny2 = int(max(oy1 + MIN_CROP_PX, min(ih, oy2 + diy)))
        self._saved_crops[self._selected_idx] = (nx1, ny1, nx2, ny2, path)
        self._redraw()

    def _on_edit_end(self, _event: tk.Event) -> None:
        if (self._edit_drag_mode is None
                or self._edit_drag_crop_orig is None
                or self._selected_idx is None
                or self.pil_image is None):
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crop_orig = None
            return
        nx1, ny1, nx2, ny2, old_path = self._saved_crops[self._selected_idx]
        ox1, oy1, ox2, oy2 = self._edit_drag_crop_orig
        self._edit_drag_mode = None
        self._edit_drag_start_canvas = None
        self._edit_drag_crop_orig = None
        if (nx1, ny1, nx2, ny2) == (ox1, oy1, ox2, oy2):
            return  # no change — nothing to persist
        if self.crops_dir is None:
            return
        stem = self._current_stem()
        base = f"crop_{stem}_y{ny1:04d}_x{nx1:04d}"
        new_path = self.crops_dir / f"{base}.png"
        count = 1
        while new_path.exists() and new_path != old_path:
            new_path = self.crops_dir / f"{base}_{count}.png"
            count += 1
        try:
            crop_img = self.pil_image.crop((nx1, ny1, nx2, ny2))
            crop_img.save(new_path)
            if old_path != new_path:
                old_path.unlink(missing_ok=True)
        except Exception as exc:
            self._status.set(f"Save failed: {exc}")
            self._saved_crops[self._selected_idx] = (ox1, oy1, ox2, oy2, old_path)
            self._redraw()
            return
        self._saved_crops[self._selected_idx] = (nx1, ny1, nx2, ny2, new_path)
        self._undo_stack.append({
            "type": "edit",
            "old_path": old_path,
            "old_coords": (ox1, oy1, ox2, oy2),
            "new_path": new_path,
            "new_coords": (nx1, ny1, nx2, ny2),
        })
        self._redraw()
        self._update_status()

    # ── Folder / image management ─────────────────────────────────────────────

    def open_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select image folder",
            initialdir=str(self.folder or Path.home()),
        )
        if path:
            self._open_folder_path(Path(path))

    def _open_folder_path(self, folder: Path) -> None:
        self.folder = folder
        self.image_paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            messagebox.showinfo("No images", f"No supported images found in:\n{folder}")
            return
        self._ensure_crops_dir()
        self.load_image(0)

    def _ensure_crops_dir(self) -> Optional[Path]:
        if self.folder is None:
            return None
        d = self.folder / CROPS_SUBDIR
        d.mkdir(exist_ok=True)
        return d

    @property
    def crops_dir(self) -> Optional[Path]:
        return (self.folder / CROPS_SUBDIR) if self.folder else None

    def prev_image(self) -> None:
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self) -> None:
        if self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)

    def load_image(self, index: int) -> None:
        self.current_index = index
        path = self.image_paths[index]

        img = Image.open(path)

        # Handle multi-frame images (TIF stacks): always use frame 0
        try:
            img.seek(0)
        except (EOFError, AttributeError):
            pass
        img = img.copy()  # detach from the file handle

        # Normalize exotic bit-depths to uint8
        if img.mode not in ("RGB", "RGBA", "L", "P"):
            arr = np.array(img).astype(float)
            arr_ptp = arr.ptp()
            arr = ((arr - arr.min()) / (arr_ptp if arr_ptp else 1) * 255).astype("uint8")
            img = Image.fromarray(arr)

        # Unify to RGB
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (30, 30, 30))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode == "L":
            img = img.convert("RGB")

        self.pil_image = img

        # Load crops before the first redraw so overlays appear immediately
        self._load_existing_crops()
        self._undo_stack.clear()

        # Reset edit selection on navigation (keep edit mode active for convenience)
        self._selected_idx = None
        self._edit_drag_mode = None
        self._edit_drag_start_canvas = None
        self._edit_drag_crop_orig = None
        self._btn_delete.config(state=tk.DISABLED)

        # Reset pan when navigating (intentionally preserve zoom level so
        # the user can flip through frames at the same magnification)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._fit_to_canvas()

        # Update toolbar
        n = len(self.image_paths)
        self._lbl_counter.config(text=f"{index + 1} / {n}")
        self._btn_prev.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self._btn_next.config(state=tk.NORMAL if index < n - 1 else tk.DISABLED)
        self.root.title(f"Crop Tool — {path.name}")

        self._update_status()

    # ── Transform helpers ─────────────────────────────────────────────────────

    @property
    def display_scale(self) -> float:
        """Pixels-per-image-pixel on the canvas."""
        return self.fit_scale * self.zoom_level

    def _fit_to_canvas(self) -> None:
        """Compute fit_scale and letterbox offsets; defer if canvas not yet sized."""
        if self.pil_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.root.after(50, self._fit_to_canvas)
            return
        iw, ih = self.pil_image.size
        self.fit_scale = min(cw / iw, ch / ih)
        self.img_offset_x = (cw - iw * self.fit_scale) / 2
        self.img_offset_y = (ch - ih * self.fit_scale) / 2
        self._redraw()

    def _canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        ds = self.display_scale
        return (
            (cx - self.img_offset_x - self.pan_x) / ds,
            (cy - self.img_offset_y - self.pan_y) / ds,
        )

    def _image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        ds = self.display_scale
        return (
            ix * ds + self.img_offset_x + self.pan_x,
            iy * ds + self.img_offset_y + self.pan_y,
        )

    def _clamp_to_image(self, ix: float, iy: float) -> tuple[int, int]:
        if self.pil_image is None:
            return 0, 0
        iw, ih = self.pil_image.size
        return int(max(0, min(iw, ix))), int(max(0, min(ih, iy)))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self.pil_image is None:
            return
        ds = self.display_scale
        iw, ih = self.pil_image.size
        new_w = max(1, int(iw * ds))
        new_h = max(1, int(ih * ds))
        resample = Image.NEAREST if ds > 2 else Image.LANCZOS
        resized = self.pil_image.resize((new_w, new_h), resample)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            int(self.img_offset_x + self.pan_x),
            int(self.img_offset_y + self.pan_y),
            anchor=tk.NW,
            image=self._tk_img,
            tags="image",
        )
        self._draw_crop_overlays()

    def _draw_crop_overlays(self) -> None:
        """Draw a rectangle on the canvas for each saved crop."""
        for i, (x1, y1, x2, y2, _) in enumerate(self._saved_crops):
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            selected = self._edit_mode and i == self._selected_idx
            color = "#FFD700" if selected else "#00FF88"
            self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=2, tags="overlay",
            )
            if selected:
                for hx, hy in ((cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)):
                    r = HANDLE_RADIUS
                    self.canvas.create_rectangle(
                        hx - r, hy - r, hx + r, hy + r,
                        fill="#FFD700", outline="#FF8C00", width=1, tags="overlay",
                    )

    # ── Zoom & pan ────────────────────────────────────────────────────────────

    def zoom_in(self) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._zoom_at(cw / 2, ch / 2, ZOOM_STEP)

    def zoom_out(self) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._zoom_at(cw / 2, ch / 2, 1 / ZOOM_STEP)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        """Zoom by factor, keeping the canvas point (cx, cy) fixed over the same image pixel."""
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_level * factor))
        if abs(new_zoom - self.zoom_level) < 1e-9:
            return
        # image point under cursor must stay at the same canvas position after zoom:
        # canvas_x = ix * new_ds + offset_x + new_pan_x  =>  new_pan_x = cx - offset_x - ix * new_ds
        ix, iy = self._canvas_to_image(cx, cy)
        self.zoom_level = new_zoom
        new_ds = self.display_scale
        self.pan_x = cx - self.img_offset_x - ix * new_ds
        self.pan_y = cy - self.img_offset_y - iy * new_ds
        self._redraw()
        self._update_status()

    def reset_transform(self) -> None:
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._fit_to_canvas()
        self._update_status()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.pil_image is None:
            return
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._zoom_at(event.x, event.y, ZOOM_STEP)
        else:
            self._zoom_at(event.x, event.y, 1 / ZOOM_STEP)

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start_canvas = (event.x, event.y)
        self._pan_start_offset = (self.pan_x, self.pan_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_start_canvas is None:
            return
        dx = event.x - self._pan_start_canvas[0]
        dy = event.y - self._pan_start_canvas[1]
        self.pan_x = self._pan_start_offset[0] + dx
        self.pan_y = self._pan_start_offset[1] + dy
        self._redraw()

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start_canvas = None
        self.canvas.config(cursor="crosshair")

    # ── Rubber-band crop ──────────────────────────────────────────────────────

    def _on_drag_start(self, event: tk.Event) -> None:
        if self.pil_image is None:
            return
        if self._edit_mode:
            self._on_edit_start(event)
            return
        self._drag_start = (event.x, event.y)
        self._rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2, dash=(6, 4), tags="rubberband",
        )

    def _on_drag_motion(self, event: tk.Event) -> None:
        if self._edit_mode:
            self._on_edit_drag(event)
            return
        if self._drag_start is None or self._rect_id is None:
            return
        x0, y0 = self._drag_start
        self.canvas.coords(self._rect_id, x0, y0, event.x, event.y)

    def _on_drag_end(self, event: tk.Event) -> None:
        if self._edit_mode:
            self._on_edit_end(event)
            return
        if self._drag_start is None or self.pil_image is None:
            return
        x0c, y0c = self._drag_start
        self._drag_start = None
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

        # Convert canvas corners → image coordinates
        ix0, iy0 = self._canvas_to_image(x0c, y0c)
        ix1, iy1 = self._canvas_to_image(event.x, event.y)

        # Clamp to image bounds and normalize (ensure top-left < bottom-right)
        ix0, iy0 = self._clamp_to_image(ix0, iy0)
        ix1, iy1 = self._clamp_to_image(ix1, iy1)
        x1, x2 = (ix0, ix1) if ix0 <= ix1 else (ix1, ix0)
        y1, y2 = (iy0, iy1) if iy0 <= iy1 else (iy1, iy0)

        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            return  # too small — ignore accidental clicks

        self._do_save_crop(x1, y1, x2, y2)

    # ── Crop persistence ──────────────────────────────────────────────────────

    def _current_stem(self) -> str:
        return self.image_paths[self.current_index].stem if self.current_index >= 0 else ""

    def _do_save_crop(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self.crops_dir is None:
            return
        crop_img = self.pil_image.crop((x1, y1, x2, y2))
        stem = self._current_stem()
        base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
        save_path = self.crops_dir / f"{base}.png"

        # Avoid overwriting an existing crop at the same position
        count = 1
        while save_path.exists():
            save_path = self.crops_dir / f"{base}_{count}.png"
            count += 1

        crop_img.save(save_path)
        self._saved_crops.append((x1, y1, x2, y2, save_path))
        self._undo_stack.append({"type": "create", "path": save_path})
        self._redraw()
        self._update_status()

    def _load_existing_crops(self) -> None:
        """Scan the crops/ subfolder for crops belonging to the current image."""
        self._saved_crops = []
        crops_dir = self.crops_dir
        if crops_dir is None or not crops_dir.exists():
            return
        stem = self._current_stem()
        for f in sorted(crops_dir.iterdir()):
            m = CROP_PATTERN.match(f.name)
            if m and m.group(1) == stem:
                y1 = int(m.group(2))
                x1 = int(m.group(3))
                try:
                    with Image.open(f) as ci:
                        w, h = ci.size
                    self._saved_crops.append((x1, y1, x1 + w, y1 + h, f))
                except Exception:
                    pass  # skip unreadable files

    def undo_last_crop(self) -> None:
        if not self._undo_stack:
            self._status.set("Nothing to undo.")
            return
        entry = self._undo_stack.pop()
        kind = entry["type"]

        if kind == "create":
            path: Path = entry["path"]
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                self._status.set(f"Undo failed: {exc}")
                self._undo_stack.append(entry)
                return
            self._saved_crops = [c for c in self._saved_crops if c[4] != path]

        elif kind == "edit":
            new_path: Path = entry["new_path"]
            old_path: Path = entry["old_path"]
            ox1, oy1, ox2, oy2 = entry["old_coords"]
            try:
                new_path.unlink(missing_ok=True)
                crop_img = self.pil_image.crop((ox1, oy1, ox2, oy2))
                crop_img.save(old_path)
            except Exception as exc:
                self._status.set(f"Undo failed: {exc}")
                self._undo_stack.append(entry)
                return
            self._saved_crops = [
                (ox1, oy1, ox2, oy2, old_path) if c[4] == new_path else c
                for c in self._saved_crops
            ]
            # Keep selection pointing at the restored crop
            if self._selected_idx is not None:
                self._selected_idx = next(
                    (i for i, c in enumerate(self._saved_crops) if c[4] == old_path),
                    None,
                )

        elif kind == "delete":
            path = entry["path"]
            x1, y1, x2, y2 = entry["coords"]
            try:
                crop_img = self.pil_image.crop((x1, y1, x2, y2))
                crop_img.save(path)
            except Exception as exc:
                self._status.set(f"Undo failed: {exc}")
                self._undo_stack.append(entry)
                return
            self._saved_crops.append((x1, y1, x2, y2, path))

        self._redraw()
        self._update_status()

    def delete_selected_crop(self) -> None:
        if self._selected_idx is None:
            return
        x1, y1, x2, y2, path = self._saved_crops[self._selected_idx]
        self._undo_stack.append({
            "type": "delete",
            "path": path,
            "coords": (x1, y1, x2, y2),
        })
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            self._status.set(f"Delete failed: {exc}")
            self._undo_stack.pop()
            return
        self._saved_crops.pop(self._selected_idx)
        self._selected_idx = None
        self._btn_delete.config(state=tk.DISABLED)
        self._redraw()
        self._update_status()

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        if self.pil_image is None or self.current_index < 0:
            self._status.set("Open a folder to begin.")
            return
        name = self.image_paths[self.current_index].name
        iw, ih = self.pil_image.size
        zoom_pct = int(self.display_scale * 100)
        n = len(self._saved_crops)
        if self._edit_mode:
            sel = (f"  |  Selected: #{self._selected_idx + 1}"
                   if self._selected_idx is not None else "")
            self._status.set(
                f"{name}  ({iw}×{ih})  |  Zoom: {zoom_pct}%  |  Crops: {n}"
                f"  |  EDIT MODE{sel}"
                "  |  Click: select  ·  Drag: move  ·  Drag corner: resize"
                "  ·  Del: delete  ·  Ctrl+Z: undo  ·  E: draw mode"
            )
        else:
            self._status.set(
                f"{name}  ({iw}×{ih})  |  Zoom: {zoom_pct}%  |  Crops saved: {n}"
                "  |  Left-drag: crop  ·  Right-drag: pan  ·  Scroll: zoom"
                "  ·  Ctrl+Z: undo  ·  E: edit mode"
            )

    # ── Canvas resize ─────────────────────────────────────────────────────────

    def _on_canvas_configure(self, event: tk.Event) -> None:
        if self.pil_image is None or event.width <= 1 or event.height <= 1:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            40, lambda: self._do_resize(event.width, event.height)
        )

    def _do_resize(self, cw: int, ch: int) -> None:
        self._resize_after_id = None
        if self.pil_image is None:
            return
        iw, ih = self.pil_image.size
        new_fit = min(cw / iw, ch / ih)
        old_fit = self.fit_scale
        self.fit_scale = new_fit
        self.img_offset_x = (cw - iw * new_fit) / 2
        self.img_offset_y = (ch - ih * new_fit) / 2
        # Scale pan proportionally so the visible image region doesn't jump
        if old_fit > 0:
            ratio = new_fit / old_fit
            self.pan_x *= ratio
            self.pan_y *= ratio
        self._redraw()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        # If a folder path is given as a CLI argument, open it automatically
        if len(sys.argv) > 1:
            folder = Path(sys.argv[1])
            if folder.is_dir():
                self.root.after(100, lambda: self._open_folder_path(folder))
        self.root.mainloop()


def _sep(parent: tk.Frame) -> None:
    """Insert a visual separator into a toolbar frame."""
    tk.Frame(parent, width=2, bd=1, relief=tk.SUNKEN).pack(
        side=tk.LEFT, fill=tk.Y, padx=4, pady=2
    )


if __name__ == "__main__":
    CropTool().run()
