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

# pylint: disable=too-many-lines
from __future__ import annotations

import math
import os
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
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
HANDLE_RADIUS = 6  # display half-size of corner handle squares (canvas px)
HANDLE_HIT = 12  # hit-detection half-size for corner handles (canvas px)


class CropTool:  # pylint: disable=too-many-instance-attributes
    """Main application window and logic for the crop tool."""

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
        self.cv_image: Optional[np.ndarray] = None
        self._image_cache: dict[Path, Image.Image] = {}
        self._cache_size: int = 10

        # ── Transform state ──────────────────────────────────────────────────
        # display_coord = image_coord * display_scale + offset + pan
        self.zoom_level: float = 1.0  # relative to fit-to-canvas scale
        self.pan_x: float = 0.0  # canvas-space additional offset
        self.pan_y: float = 0.0
        self.fit_scale: float = 1.0  # scale that fits image into canvas
        self.img_offset_x: float = 0.0  # letterbox margin
        self.img_offset_y: float = 0.0

        # ── Internal rendering ───────────────────────────────────────────────
        self._tk_img: Optional[ImageTk.PhotoImage] = None  # must hold reference
        self._img_id: Optional[int] = None
        self._resize_after_id: Optional[str] = None

        # ── Interaction state ────────────────────────────────────────────────
        self._drag_start: Optional[tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self._pan_start_canvas: Optional[tuple[int, int]] = None
        self._pan_start_offset: tuple[float, float] = (0.0, 0.0)

        # ── Annotation tracking ──────────────────────────────────────────────
        # Each entry: (x1, y1, x2, y2, Optional[Path])
        # Coordinates are image-space pixels. Path points to the PNG crop if it exists.
        self._manual_annots: list[tuple[int, int, int, int, Optional[Path]]] = []
        self._yolo_labels: list[tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, class_id)
        self._yolo_view_mode: str = "box"  # 'box' | 'point'
        self._save_crops_mode: bool = False  # If True, generate PNG snippets
        self._undo_stack: list[dict] = []

        # ── Edit mode state ──────────────────────────────────────────────────
        self._edit_mode: bool = False
        self._selected_type: str = "manual"  # 'manual' | 'yolo'
        self._selected_idx: Optional[int] = None
        self._edit_drag_mode: Optional[str] = None  # 'move' | 'resize_nw' | etc.
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
            toolbar, text="Edit Mode (E)", command=self.toggle_edit_mode, relief=tk.RAISED
        )
        self._btn_edit_mode.pack(side=tk.LEFT, padx=2, pady=3)

        self._btn_delete = tk.Button(
            toolbar,
            text="Delete Selected (Del)",
            command=self.delete_selected_crop,
            state=tk.DISABLED,
        )
        self._btn_delete.pack(side=tk.LEFT, padx=2, pady=3)
        _sep(toolbar)

        self._btn_yolo_mode = tk.Button(
            toolbar, text="YOLO: Box (Y)", command=self.toggle_yolo_view_mode
        )
        self._btn_yolo_mode.pack(side=tk.LEFT, padx=2, pady=3)

        self._btn_save_crops = tk.Button(
            toolbar, text="Crops: Off (C)", command=self.toggle_save_crops_mode
        )
        self._btn_save_crops.pack(side=tk.LEFT, padx=2, pady=3)
        _sep(toolbar)

        self._btn_convert = tk.Button(
            toolbar, text="Convert to Crop (T)", command=self.convert_selected, state=tk.DISABLED
        )
        self._btn_convert.pack(side=tk.LEFT, padx=2, pady=3)

        tk.Button(toolbar, text="Convert All YOLO", command=self.convert_all_yolo).pack(
            side=tk.LEFT, padx=2, pady=3
        )

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        self._status = tk.StringVar(value="Open a folder to begin.")
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")
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
        root.bind("<Left>", lambda _: self.prev_image())
        root.bind("<Right>", lambda _: self.next_image())

        # Zoom / reset
        root.bind("<r>", lambda _: self.reset_transform())
        root.bind("<R>", lambda _: self.reset_transform())
        root.bind("<plus>", lambda _: self.zoom_in())
        root.bind("<equal>", lambda _: self.zoom_in())
        root.bind("<minus>", lambda _: self.zoom_out())

        # Undo
        root.bind("<Control-z>", lambda _: self.undo_last_crop())

        # Edit mode toggle / delete
        root.bind("<e>", lambda _: self.toggle_edit_mode())
        root.bind("<E>", lambda _: self.toggle_edit_mode())
        root.bind("<Delete>", lambda _: self.delete_selected_crop())
        root.bind("<BackSpace>", lambda _: self.delete_selected_crop())
        root.bind("<y>", lambda _: self.toggle_yolo_view_mode())
        root.bind("<Y>", lambda _: self.toggle_yolo_view_mode())
        root.bind("<c>", lambda _: self.toggle_save_crops_mode())
        root.bind("<C>", lambda _: self.toggle_save_crops_mode())
        root.bind("<t>", lambda _: self.convert_selected())
        root.bind("<T>", lambda _: self.convert_selected())

        # Scroll-to-zoom: Linux Button-4/5, Windows/Mac MouseWheel
        canvas.bind("<MouseWheel>", self._on_mousewheel)
        canvas.bind("<Button-4>", self._on_mousewheel)
        canvas.bind("<Button-5>", self._on_mousewheel)

        # Right-click pan
        canvas.bind("<ButtonPress-3>", self._on_pan_start)
        canvas.bind("<B3-Motion>", self._on_pan_drag)
        canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        # Left-click rubber-band crop
        canvas.bind("<ButtonPress-1>", self._on_drag_start)
        canvas.bind("<B1-Motion>", self._on_drag_motion)
        canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        # Window resize
        canvas.bind("<Configure>", self._on_canvas_configure)

    # ── Edit mode ────────────────────────────────────────────────────────────

    def toggle_edit_mode(self) -> None:
        """Toggle between crop drawing and edit modes."""
        self._edit_mode = not self._edit_mode
        if not self._edit_mode:
            self._selected_idx = None
            self._selected_type = "manual"
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crop_orig = None
            self._btn_delete.config(state=tk.DISABLED)
            self._btn_convert.config(state=tk.DISABLED)
        else:
            # Entering edit mode — ensure buttons start disabled
            self._btn_delete.config(state=tk.DISABLED)
            self._btn_convert.config(state=tk.DISABLED)
        self._btn_edit_mode.config(relief=tk.SUNKEN if self._edit_mode else tk.RAISED)
        self.canvas.config(cursor="arrow" if self._edit_mode else "crosshair")
        self._redraw()
        self._update_status()

    def toggle_yolo_view_mode(self) -> None:
        """Toggle the display format of YOLO labels (box or point)."""
        self._yolo_view_mode = "point" if self._yolo_view_mode == "box" else "box"
        self._btn_yolo_mode.config(text=f"YOLO: {self._yolo_view_mode.capitalize()} (Y)")
        self._redraw()
        self._update_status()

        self._redraw()
        self._update_status()

    def toggle_save_crops_mode(self) -> None:
        """Toggle whether PNG crops are saved to disk."""
        self._save_crops_mode = not self._save_crops_mode
        state = "On" if self._save_crops_mode else "Off"
        self._btn_save_crops.config(
            text=f"Crops: {state} (C)", relief=tk.SUNKEN if self._save_crops_mode else tk.RAISED
        )
        self._update_status()

    def _find_crop_at(self, cx: float, cy: float) -> tuple[Optional[str], Optional[int]]:
        """Return (type, index) of the topmost annotation whose canvas rect contains (cx, cy)."""
        # Check manual annotations first (they are on top)
        for i in range(len(self._manual_annots) - 1, -1, -1):
            x1, y1, x2, y2, _ = self._manual_annots[i]
            rx1, ry1 = self._image_to_canvas(x1, y1)
            rx2, ry2 = self._image_to_canvas(x2, y2)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return "manual", i

        # Check YOLO reference labels
        for i in range(len(self._yolo_labels) - 1, -1, -1):
            x1, y1, x2, y2, _ = self._yolo_labels[i]
            rx1, ry1 = self._image_to_canvas(x1, y1)
            rx2, ry2 = self._image_to_canvas(x2, y2)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return "yolo", i

        return None, None

    def _get_handle_at(self, cx: float, cy: float, idx: int) -> Optional[str]:
        """Return 'nw'|'ne'|'sw'|'se' if (cx, cy) is near a corner handle of crop[idx]."""
        x1, y1, x2, y2, _ = self._manual_annots[idx]
        rx1, ry1 = self._image_to_canvas(x1, y1)
        rx2, ry2 = self._image_to_canvas(x2, y2)
        handles = [("nw", rx1, ry1), ("ne", rx2, ry1), ("sw", rx1, ry2), ("se", rx2, ry2)]
        for name, hx, hy in handles:
            if abs(cx - hx) <= HANDLE_HIT and abs(cy - hy) <= HANDLE_HIT:
                return name
        return None

    def _on_edit_start(self, event: tk.Event) -> None:
        cx, cy = float(event.x), float(event.y)
        # If a manual crop is already selected, check its handles first
        if self._selected_idx is not None and self._selected_type == "manual":
            handle = self._get_handle_at(cx, cy, self._selected_idx)
            if handle:
                self._edit_drag_mode = f"resize_{handle}"
                self._edit_drag_start_canvas = (event.x, event.y)
                x1, y1, x2, y2, _ = self._manual_annots[self._selected_idx]
                self._edit_drag_crop_orig = (x1, y1, x2, y2)
                return

        # Hit-test all crops (topmost first)
        kind, idx = self._find_crop_at(cx, cy)
        if idx is not None:
            self._selected_idx = idx
            self._selected_type = kind
            if kind == "manual":
                self._btn_delete.config(state=tk.NORMAL)
                # Allow conversion if it doesn't have a PNG yet
                _, _, _, _, path = self._manual_annots[idx]
                if path is None:
                    self._btn_convert.config(state=tk.NORMAL)
                else:
                    self._btn_convert.config(state=tk.DISABLED)

                handle = self._get_handle_at(cx, cy, idx)
                self._edit_drag_mode = f"resize_{handle}" if handle else "move"
                self._edit_drag_start_canvas = (event.x, event.y)
                x1, y1, x2, y2, _ = self._manual_annots[idx]
                self._edit_drag_crop_orig = (x1, y1, x2, y2)
            else:  # YOLO selection
                self._btn_delete.config(state=tk.DISABLED)
                self._btn_convert.config(state=tk.NORMAL)
                self._edit_drag_mode = None
        else:
            # Click on empty area — deselect
            self._selected_idx = None
            self._selected_type = "manual"
            self._edit_drag_mode = None
            self._edit_drag_start_canvas = None
            self._edit_drag_crop_orig = None
            self._btn_delete.config(state=tk.DISABLED)
            self._btn_convert.config(state=tk.DISABLED)
        self._redraw()

    def _on_edit_drag(self, event: tk.Event) -> None:  # pylint: disable=too-many-locals
        missing_state = (
            self._edit_drag_mode is None
            or self._edit_drag_start_canvas is None
            or self._selected_idx is None
            or self._edit_drag_crop_orig is None
            or self.pil_image is None
        )
        if missing_state:
            return
        dcx = event.x - self._edit_drag_start_canvas[0]
        dcy = event.y - self._edit_drag_start_canvas[1]
        ds = self.display_scale
        dix = dcx / ds
        diy = dcy / ds
        ox1, oy1, ox2, oy2 = self._edit_drag_crop_orig
        iw, ih = self.pil_image.size
        _, _, _, _, path = self._manual_annots[self._selected_idx]
        if self._edit_drag_mode == "move":
            w, h = ox2 - ox1, oy2 - oy1
            nx1 = int(max(0, min(iw - w, ox1 + dix)))
            ny1 = int(max(0, min(ih - h, oy1 + diy)))
            nx2 = nx1 + w
            ny2 = ny1 + h
        else:
            corner = self._edit_drag_mode[len("resize_") :]
            nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2
            if "w" in corner:
                nx1 = int(max(0, min(ox2 - MIN_CROP_PX, ox1 + dix)))
            if "e" in corner:
                nx2 = int(max(ox1 + MIN_CROP_PX, min(iw, ox2 + dix)))
            if "n" in corner:
                ny1 = int(max(0, min(oy2 - MIN_CROP_PX, oy1 + diy)))
            if "s" in corner:
                ny2 = int(max(oy1 + MIN_CROP_PX, min(ih, oy2 + diy)))
        self._manual_annots[self._selected_idx] = (nx1, ny1, nx2, ny2, path)
        self._redraw()

    def _on_edit_end(self, _event: tk.Event) -> None:  # pylint: disable=too-many-locals
        missing_state = (
            self._edit_drag_mode is None
            or self._edit_drag_crop_orig is None
            or self._selected_idx is None
            or self.pil_image is None
        )
        if missing_state:
            self._edit_drag_mode = None
            self._edit_drag_crop_orig = None
            return
        nx1, ny1, nx2, ny2, old_path = self._manual_annots[self._selected_idx]
        ox1, oy1, ox2, oy2 = self._edit_drag_crop_orig
        self._edit_drag_mode = None
        self._edit_drag_start_canvas = None
        self._edit_drag_crop_orig = None
        if (nx1, ny1, nx2, ny2) == (ox1, oy1, ox2, oy2):
            return  # no change — nothing to persist
        try:
            new_path = None
            if self._save_crops_mode:
                if self.crops_dir is None:
                    self._status.set("No crops directory")
                else:
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
                        if old_path != new_path and old_path:
                            old_path.unlink(missing_ok=True)
                    except Exception as exc:  # pylint: disable=broad-except
                        self._status.set(f"Crop save failed: {exc}")
                        # keep going to save the label at least
            else:
                # Delete old path if we switched modes
                if old_path:
                    old_path.unlink(missing_ok=True)

            self._manual_annots[self._selected_idx] = (nx1, ny1, nx2, ny2, new_path)
            self._sync_yolo_labels_file()
            self._undo_stack.append(
                {
                    "type": "edit",
                    "old_path": old_path,
                    "old_coords": (ox1, oy1, ox2, oy2),
                    "new_path": new_path,
                    "new_coords": (nx1, ny1, nx2, ny2),
                }
            )
            self._redraw()
            self._update_status()
        except Exception as exc:  # pylint: disable=broad-except
            self._status.set(f"Save failed: {exc}")
            self._manual_annots[self._selected_idx] = (ox1, oy1, ox2, oy2, old_path)
            self._redraw()
            return

    # ── Folder / image management ─────────────────────────────────────────────

    def open_folder(self) -> None:
        """Prompt the user to select an image folder."""
        path = filedialog.askdirectory(
            title="Select image folder", initialdir=str(self.folder or Path.home())
        )
        if path:
            self._open_folder_path(Path(path))

    def _open_folder_path(self, folder: Path) -> None:
        self.folder = folder
        self._status.set(f"Scanning directory {folder.name}...")
        self.progress_bar.config(mode="determinate", value=0, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.update_idletasks()
        self.root.update()

        def scan_job() -> None:
            paths = []
            try:
                names = os.listdir(folder)
                total = len(names)
                step = max(1, total // 100)
                self.root.after(0, lambda: self.progress_bar.config(maximum=total))

                for i, name in enumerate(names):
                    if i % step == 0:
                        self.root.after(0, lambda pos=i: self.progress_bar.config(value=pos))

                    _, ext = os.path.splitext(name)
                    if ext.lower() in IMAGE_EXTS:
                        p = folder / name
                        if p.is_file():
                            paths.append(p)
            except Exception as e:  # pylint: disable=broad-except
                self.root.after(0, lambda: _finish_scan(None, e))
                return
            self.root.after(0, lambda: _finish_scan(sorted(paths), None))

        def _finish_scan(paths: Optional[list[Path]], error: Optional[Exception]) -> None:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

            if error:
                messagebox.showerror("Error", f"Failed to open folder:\n{error}")
                self._status.set("Error opening folder.")
                return

            if paths is not None:
                self.image_paths = paths
            else:
                self.image_paths = []

            if not self.image_paths:
                messagebox.showinfo("No images", f"No supported images found in:\n{folder}")
                self._status.set("No images loaded.")
                return

            self._ensure_crops_dir()
            self.load_image(0)

        threading.Thread(target=scan_job, daemon=True).start()

    @property
    def crops_dir(self) -> Optional[Path]:
        """Return the path to the crops subdirectory if a folder is loaded."""
        return (self.folder / CROPS_SUBDIR) if self.folder else None

    def _ensure_crops_dir(self) -> Optional[Path]:
        if self.folder is None:
            return None
        d = self.folder / CROPS_SUBDIR
        d.mkdir(exist_ok=True)
        return d

    @property
    def labels_dir(self) -> Optional[Path]:
        """Path to labels/ folder, prefering adjacent to images/ then within it."""
        if self.folder is None:
            return None
        adj = self.folder.parent / "labels"
        if adj.exists() and adj.is_dir():
            return adj
        inner = self.folder / "labels"
        return inner

    def _ensure_labels_dir(self) -> Optional[Path]:
        if self.folder is None:
            return None
        d = self.labels_dir
        if d is None or not d.exists():
            # If folder is called 'images', create 'labels' next to it
            if self.folder.name == "images":
                d = self.folder.parent / "labels"
            else:
                d = self.folder / "labels"
            d.mkdir(parents=True, exist_ok=True)
        return d

    def prev_image(self) -> None:
        """Navigate to the previous image in the directory."""
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self) -> None:
        """Navigate to the next image in the directory."""
        if self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)

    def _get_image(self, path: Path) -> Image.Image:
        """Load and normalize image, using cache if available."""
        if path in self._image_cache:
            return self._image_cache[path]

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

        # Manage cache size (LRU-ish)
        if len(self._image_cache) >= self._cache_size:
            # Pop the first key (oldest)
            self._image_cache.pop(next(iter(self._image_cache)))

        self._image_cache[path] = img
        return img

    def _preload_neighbors(self) -> None:
        """Load next/prev images in a background thread."""

        def _target():
            indices = []
            if self.current_index + 1 < len(self.image_paths):
                indices.append(self.current_index + 1)
            if self.current_index - 1 >= 0:
                indices.append(self.current_index - 1)

            for idx in indices:
                path = self.image_paths[idx]
                if path not in self._image_cache:
                    try:
                        self._get_image(path)
                    except Exception:  # pylint: disable=broad-except
                        pass

        threading.Thread(target=_target, daemon=True).start()

    def load_image(self, index: int) -> None:
        """Load the image at the given index and update UI state."""
        self.current_index = index
        path = self.image_paths[index]

        try:
            self.pil_image = self._get_image(path)
            self.cv_image = np.array(self.pil_image)
        except Exception as e:  # pylint: disable=broad-except
            self._status.set(f"Error loading {path.name}: {e}")
            return

        # Load annotations before the first redraw so overlays appear immediately
        self._load_manual_annotations()
        self._load_yolo_labels()
        self._undo_stack.clear()

        # Reset edit selection on navigation (keep edit mode active for convenience)
        self._selected_idx = None
        self._selected_type = "manual"
        self._edit_drag_mode = None
        self._edit_drag_start_canvas = None
        self._edit_drag_crop_orig = None
        self._btn_delete.config(state=tk.DISABLED)
        self._btn_convert.config(state=tk.DISABLED)

        # Reset pan when navigating (intentionally preserve zoom level so
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Deduplicate YOLO labels from manual annotations to avoid overlapping boxes
        # if they both point to the same labels/ directory.
        if self._manual_annots and self._yolo_labels:
            manual_boxes = set((x1, y1, x2, y2) for (x1, y1, x2, y2, _) in self._manual_annots)
            self._yolo_labels = [
                y for y in self._yolo_labels if (y[0], y[1], y[2], y[3]) not in manual_boxes
            ]

        self._fit_to_canvas()

        # Update toolbar
        n = len(self.image_paths)
        self._lbl_counter.config(text=f"{index + 1} / {n}")
        self._btn_prev.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self._btn_next.config(state=tk.NORMAL if index < n - 1 else tk.DISABLED)
        self.root.title(f"Crop Tool — {path.name}")

        self._update_status()
        self._preload_neighbors()

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
        return (ix * ds + self.img_offset_x + self.pan_x, iy * ds + self.img_offset_y + self.pan_y)

    def _clamp_to_image(self, ix: float, iy: float) -> tuple[int, int]:
        if self.pil_image is None:
            return 0, 0
        iw, ih = self.pil_image.size
        return int(max(0, min(iw, ix))), int(max(0, min(ih, iy)))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _redraw(self) -> None:  # pylint: disable=too-many-locals
        if self.pil_image is None or self.cv_image is None:
            return
        ds = self.display_scale
        iw, ih = self.pil_image.size
        new_w = max(1, int(iw * ds))
        new_h = max(1, int(ih * ds))

        self.canvas.delete("all")

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600

        img_x = int(self.img_offset_x + self.pan_x)
        img_y = int(self.img_offset_y + self.pan_y)

        scaled_x_start = max(0, -img_x)
        scaled_y_start = max(0, -img_y)
        scaled_x_end = min(new_w, cw - img_x)
        scaled_y_end = min(new_h, ch - img_y)

        o_x_start = int(scaled_x_start / ds)
        o_y_start = int(scaled_y_start / ds)
        o_x_end = int(math.ceil(scaled_x_end / ds))
        o_y_end = int(math.ceil(scaled_y_end / ds))

        o_x_start = max(0, min(iw - 1, o_x_start))
        o_y_start = max(0, min(ih - 1, o_y_start))
        o_x_end = max(0, min(iw, o_x_end))
        o_y_end = max(0, min(ih, o_y_end))

        if o_x_end <= o_x_start or o_y_end <= o_y_start:
            self._draw_crop_overlays()
            return

        crop = self.cv_image[o_y_start:o_y_end, o_x_start:o_x_end]

        crop_new_w = max(1, int((o_x_end - o_x_start) * ds))
        crop_new_h = max(1, int((o_y_end - o_y_start) * ds))

        # pylint: disable=no-member
        interp = cv2.INTER_NEAREST if ds > 2 else cv2.INTER_LINEAR
        resized_crop = cv2.resize(crop, (crop_new_w, crop_new_h), interpolation=interp)
        # pylint: enable=no-member
        self._tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized_crop))

        anchor_x = img_x + int(o_x_start * ds)
        anchor_y = img_y + int(o_y_start * ds)

        self._img_id = self.canvas.create_image(
            anchor_x, anchor_y, anchor=tk.NW, image=self._tk_img, tags="image"
        )
        self._draw_crop_overlays()

    def _draw_crop_overlays(self) -> None:  # pylint: disable=too-many-locals
        """Draw a rectangle on the canvas for each manual annotation."""
        for i, (x1, y1, x2, y2, _) in enumerate(self._manual_annots):
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            selected = self._edit_mode and i == self._selected_idx
            color = "#FFD700" if selected else "#00FF88"
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, tags="overlay")
            if selected:
                for hx, hy in ((cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)):
                    r = HANDLE_RADIUS
                    self.canvas.create_rectangle(
                        hx - r,
                        hy - r,
                        hx + r,
                        hy + r,
                        fill="#FFD700",
                        outline="#FF8C00",
                        width=1,
                        tags="overlay",
                    )

        # Draw YOLO labels (fixed detections)
        for i, (x1, y1, x2, y2, cls_id) in enumerate(self._yolo_labels):
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)

            selected = self._edit_mode and self._selected_type == "yolo" and i == self._selected_idx
            tag_color = "#00FFFF" if selected else "#FF00FF"  # Cyan if selected, else Magenta

            if self._yolo_view_mode == "box":
                self.canvas.create_rectangle(
                    cx1,
                    cy1,
                    cx2,
                    cy2,
                    fill="",
                    outline=tag_color,
                    width=2 if selected else 1,
                    dash=None if selected else (2, 2),
                    tags="overlay",
                )
            else:  # point mode
                mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                r = 4 if selected else 3
                self.canvas.create_oval(
                    mx - r,
                    my - r,
                    mx + r,
                    my + r,
                    fill=tag_color,
                    outline="white",
                    width=1,
                    tags="overlay",
                )

            # Label the class if we want
            self.canvas.create_text(
                cx1 if self._yolo_view_mode == "box" else (cx1 + cx2) / 2,
                (cy1 - 2) if self._yolo_view_mode == "box" else (cy1 + cy2) / 2 - 5,
                text=f"cls:{cls_id}",
                fill=tag_color,
                anchor=tk.SW if self._yolo_view_mode == "box" else tk.S,
                font=("Helvetica", 8, "bold"),
                tags="overlay",
            )

    # ── Zoom & pan ────────────────────────────────────────────────────────────

    def zoom_in(self) -> None:
        """Zoom into the image."""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._zoom_at(cw / 2, ch / 2, ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out of the image."""
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
        """Reset zoom and pan to fit the image to the canvas."""
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

        # Real-time shift of all canvas items (don't redraw yet)
        shift_x = dx - (self.pan_x - self._pan_start_offset[0])
        shift_y = dy - (self.pan_y - self._pan_start_offset[1])

        self.pan_x = self._pan_start_offset[0] + dx
        self.pan_y = self._pan_start_offset[1] + dy

        self.canvas.move("all", shift_x, shift_y)

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
            event.x,
            event.y,
            event.x,
            event.y,
            outline="red",
            width=2,
            dash=(6, 4),
            tags="rubberband",
        )

    def _on_drag_motion(self, event: tk.Event) -> None:
        if self._edit_mode:
            self._on_edit_drag(event)
            return
        if self._drag_start is None or self._rect_id is None:
            return
        x0, y0 = self._drag_start
        self.canvas.coords(self._rect_id, x0, y0, event.x, event.y)

    def _on_drag_end(self, event: tk.Event) -> None:  # pylint: disable=too-many-locals
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

        if self.pil_image is None or self.crops_dir is None:
            return

        new_path = None
        if self._save_crops_mode:
            stem = self._current_stem()
            base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
            save_path = self.crops_dir / f"{base}.png"
            count = 1
            while save_path.exists():
                save_path = self.crops_dir / f"{base}_{count}.png"
                count += 1

            try:
                crop_img = self.pil_image.crop((x1, y1, x2, y2))
                crop_img.save(save_path)
                new_path = save_path
            except Exception as e:  # pylint: disable=broad-except
                self._status.set(f"Crop save failed: {e}")

        self._manual_annots.append((x1, y1, x2, y2, new_path))
        self._sync_yolo_labels_file()
        self._undo_stack.append({"type": "create", "path": new_path})
        self._redraw()
        self._update_status()

    # ── Crop persistence ──────────────────────────────────────────────────────

    def _current_stem(self) -> str:
        return self.image_paths[self.current_index].stem if self.current_index >= 0 else ""

    def _load_manual_annotations(self) -> None:  # pylint: disable=too-many-locals
        """Load annotations from YOLO labels file and link to existing PNG crops."""
        self._manual_annots = []
        if self.folder is None or self.pil_image is None:
            return

        lbl_dir = self.labels_dir
        if not lbl_dir:
            return

        stem = self._current_stem()
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            return

        iw, ih = self.pil_image.size
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    px_w, px_h = w * iw, h * ih
                    px_xc, px_yc = xc * iw, yc * ih
                    x1, y1 = int(px_xc - px_w / 2), int(px_yc - px_h / 2)
                    x2, y2 = int(px_xc + px_w / 2), int(px_yc + px_h / 2)

                    # Try to find a matching PNG crop
                    match_path = None
                    if self.crops_dir and self.crops_dir.exists():
                        base_glob = f"crop_{stem}_y{y1:04d}_x{x1:04d}*.png"
                        matches = list(self.crops_dir.glob(base_glob))
                        if matches:
                            match_path = matches[0]

                    self._manual_annots.append((x1, y1, x2, y2, match_path))
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading manual annotations: {e}")

    def _load_existing_crops(self) -> None:
        """DEPRECATED: Replaced by _load_manual_annotations which syncs from labels."""

    def _load_yolo_labels(self) -> None:  # pylint: disable=too-many-locals
        """Scan for YOLO .txt labels in adjacent folders."""
        self._yolo_labels = []
        if self.folder is None or self.pil_image is None:
            return

        # 1. Try ../labels/ (if current is .../images/)
        # 2. Try ./labels/
        candidates = [self.folder.parent / "labels", self.folder / "labels"]

        lbl_dir = None
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                lbl_dir = cand
                break

        if lbl_dir is None:
            return

        stem = self._current_stem()
        lbl_path = lbl_dir / f"{stem}.txt"

        if not lbl_path.exists():
            return

        iw, ih = self.pil_image.size
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    # Convert normalized to pixel coordinates
                    px_w = w * iw
                    px_h = h * ih
                    px_xc = xc * iw
                    px_yc = yc * ih

                    x1 = int(px_xc - px_w / 2)
                    y1 = int(px_yc - px_h / 2)
                    x2 = int(px_xc + px_w / 2)
                    y2 = int(px_yc + px_h / 2)

                    self._yolo_labels.append((x1, y1, x2, y2, cls_id))
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading YOLO labels from {lbl_path}: {e}")

    def undo_last_crop(self) -> None:
        """Revert the last create, edit, or delete action."""
        if not self._undo_stack:
            self._status.set("Nothing to undo.")
            return
        entry = self._undo_stack.pop()
        kind = entry["type"]

        if kind == "create":
            if self._manual_annots:
                _, _, _, _, path = self._manual_annots.pop()
                if path:
                    path.unlink(missing_ok=True)
        elif kind == "delete":
            self._manual_annots.append((*entry["coords"], entry["path"]))
            # We don't restore the PNG file, just the logical annotation
        elif kind == "edit":
            # Search for the newly edited entry to revert it
            curr = (*entry["new_coords"], entry["new_path"])
            try:
                idx = self._manual_annots.index(curr)
                self._manual_annots[idx] = (*entry["old_coords"], entry["old_path"])
                if entry["new_path"]:
                    entry["new_path"].unlink(missing_ok=True)
            except ValueError:
                pass

        self._sync_yolo_labels_file()
        self._redraw()
        self._update_status()

    def _sync_yolo_labels_file(self) -> None:  # pylint: disable=too-many-locals
        """Write all current image's manual crops to the YOLO labels file."""
        lbl_dir = self._ensure_labels_dir()
        if not lbl_dir or not self.pil_image:
            return

        stem = self._current_stem()
        lbl_path = lbl_dir / f"{stem}.txt"

        if not self._manual_annots:
            if lbl_path.exists():
                lbl_path.unlink()
            return

        iw, ih = self.pil_image.size
        try:
            with open(lbl_path, "w", encoding="utf-8") as f:
                for x1, y1, x2, y2, _ in self._manual_annots:
                    # YOLO format: class x_center y_center width height (normalized)
                    w = x2 - x1
                    h = y2 - y1
                    xc = x1 + w / 2
                    yc = y1 + h / 2

                    n_xc = xc / iw
                    n_yc = yc / ih
                    n_w = w / iw
                    n_h = h / ih
                    f.write(f"0 {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}\n")
        except Exception as e:  # pylint: disable=broad-except
            self._status.set(f"YOLO sync failed: {e}")

    def convert_yolo_to_manual(self, y_idx: int) -> bool:  # pylint: disable=too-many-locals
        """Convert a YOLO label to a manual annotation + PNG crop. Returns True if successful."""
        if self.pil_image is None or self.crops_dir is None:
            return False

        x1, y1, x2, y2, _ = self._yolo_labels[y_idx]

        # Avoid duplicate manual labels at the same coordinates
        for mx1, my1, mx2, my2, _ in self._manual_annots:
            if abs(mx1 - x1) < 2 and abs(my1 - y1) < 2 and abs(mx2 - x2) < 2 and abs(my2 - y2) < 2:
                return False

        stem = self._current_stem()
        base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
        save_path = self.crops_dir / f"{base}.png"
        count = 1
        while save_path.exists():
            save_path = self.crops_dir / f"{base}_{count}.png"
            count += 1

        try:
            crop_img = self.pil_image.crop((x1, y1, x2, y2))
            crop_img.save(save_path)
            self._manual_annots.append((x1, y1, x2, y2, save_path))
            return True
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to convert YOLO to crop: {e}")
            return False

    def convert_selected(self) -> None:
        """Convert the currently selected YOLO label to a manual label."""
        if self._selected_idx is None:
            self._status.set("Select a YOLO label (magenta) or a manual label (green) first.")
            return

        success = False
        if self._selected_type == "yolo":
            success = self.convert_yolo_to_manual(self._selected_idx)
        elif self._selected_type == "manual":
            # Just generate the PNG for an existing logical annotation
            x1, y1, x2, y2, path = self._manual_annots[self._selected_idx]
            if path is not None:
                self._status.set("Crop PNG already exists.")
                return

            if self.crops_dir is None or self.pil_image is None:
                return

            stem = self._current_stem()
            base = f"crop_{stem}_y{y1:04d}_x{x1:04d}"
            save_path = self.crops_dir / f"{base}.png"
            count = 1
            while save_path.exists():
                save_path = self.crops_dir / f"{base}_{count}.png"
                count += 1

            try:
                crop_img = self.pil_image.crop((x1, y1, x2, y2))
                crop_img.save(save_path)
                self._manual_annots[self._selected_idx] = (x1, y1, x2, y2, save_path)
                success = True
            except Exception as e:  # pylint: disable=broad-except
                self._status.set(f"Manual conversion failed: {e}")
                return

        if success:
            self._sync_yolo_labels_file()
            # If it was yolo, it's now at the end of manual_annots
            if self._selected_type == "yolo":
                self._selected_type = "manual"
                self._selected_idx = len(self._manual_annots) - 1

            self._btn_convert.config(state=tk.DISABLED)
            self._btn_delete.config(state=tk.NORMAL)
            self._redraw()
            self._update_status()
            self._status.set("Crop PNG created successfully.")
        else:
            self._status.set("Conversion failed or crop already exists.")

    def convert_all_yolo(self) -> None:
        """Convert all YOLO labels in the current image to manual labels."""
        if not self._yolo_labels:
            self._status.set("No YOLO labels to convert.")
            return

        count = 0
        for i in range(len(self._yolo_labels)):
            if self.convert_yolo_to_manual(i):
                count += 1

        if count > 0:
            self._sync_yolo_labels_file()
            self._redraw()
            self._update_status()
            self._status.set(f"Converted {count} YOLO labels to LodeSTAR crops.")
        else:
            self._status.set("No new YOLO labels were converted.")

    def delete_selected_crop(self) -> None:
        """Delete the currently selected manual crop."""
        if self._selected_idx is None or self._selected_type != "manual":
            return
        x1, y1, x2, y2, path = self._manual_annots[self._selected_idx]
        self._undo_stack.append({"type": "delete", "path": path, "coords": (x1, y1, x2, y2)})
        if path:
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:  # pylint: disable=broad-except
                self._status.set(f"Delete failed: {exc}")
                self._undo_stack.pop()
                return
        self._manual_annots.pop(self._selected_idx)
        self._selected_idx = None
        self._btn_delete.config(state=tk.DISABLED)
        self._sync_yolo_labels_file()
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
        m_count = len(self._manual_annots)
        y_count = len(self._yolo_labels)
        mode = "EDIT" if self._edit_mode else "CROP"
        crops_gen = "Crops:ON" if self._save_crops_mode else "Crops:OFF"

        self._status.set(
            f"{name} ({iw}x{ih}) | Zoom:{zoom_pct}% | {mode} | "
            f"{crops_gen} | {m_count} labels | {y_count} ref"
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
        """Start the application main loop."""
        # If a folder path is given as a CLI argument, open it automatically
        if len(sys.argv) > 1:
            folder = Path(sys.argv[1])
            if folder.is_dir():
                self.root.after(500, lambda: self._open_folder_path(folder))
        self.root.mainloop()


def _sep(parent: tk.Frame) -> None:
    """Insert a visual separator into a toolbar frame."""
    tk.Frame(parent, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)


if __name__ == "__main__":
    CropTool().run()
