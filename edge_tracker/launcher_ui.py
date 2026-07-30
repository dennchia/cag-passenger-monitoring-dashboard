import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from constants import DEFAULT_RTSP_URL, DEFAULT_TRACKER_CONFIG_PATH, DEFAULT_YOLO_NMS_IOU


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "main_tracker.py"
CORNER_OPTIONS = ["None", "top_left", "top_right", "bottom_right", "bottom_left"]


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECDS Camera Launcher")
        self.geometry("760x620")
        self.minsize(500, 400) # Reduced minimum size so it fits on smaller laptop screens

        self.camera_mode = tk.StringVar(value="Both cameras")
        self.setup = tk.BooleanVar(value=False)
        self.use_mediapipe = tk.BooleanVar(value=True)
        self.use_reid = tk.BooleanVar(value=False)
        self.use_mqtt = tk.BooleanVar(value=False)

        self.yolo_device = tk.StringVar(value="GPU (0)")
        self.reid_device = tk.StringVar(value="GPU (1)")

        self.source_1 = tk.StringVar(value=DEFAULT_RTSP_URL)
        self.source_2 = tk.StringVar(value="rtsp://admin:P@ssword1@192.168.50.76:554/Streaming/Channels/101")
        self.camera_id_1 = tk.StringVar(value="cam_1")
        self.camera_id_2 = tk.StringVar(value="cam_2")
        self.matrix_1 = tk.StringVar(value="homography_matrix.json")
        self.matrix_2 = tk.StringVar(value="homography_matrix_2.json")
        self.corner_1 = tk.StringVar(value="None")
        self.corner_2 = tk.StringVar(value="None")

        self.model = tk.StringVar(value="yolo11n.pt")
        self.yolo_nms_iou = tk.StringVar(value=str(DEFAULT_YOLO_NMS_IOU))
        self.tracker_config = tk.StringVar(value=DEFAULT_TRACKER_CONFIG_PATH)
        self.run_id = tk.StringVar(value="field_test_001")
        self.fusion_distance = tk.StringVar(value="50")
        self.mqtt_broker = tk.StringVar(value="192.168.50.45")
        self.mqtt_port = tk.StringVar(value="1883")
        self.reid_api_url = tk.StringVar(value="http://localhost:8000")

        self.command_preview = tk.StringVar(value="")
        self._build_ui()
        self._refresh_preview()

    def _build_ui(self):
        # --- SCROLLING SETUP ---
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        container = ttk.Frame(canvas, padding=16)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Configure canvas resizing
        container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Mousewheel binding for Windows
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.bind_all("<MouseWheel>", _on_mousewheel)
        # --- END SCROLLING SETUP ---

        title = ttk.Label(container, text="Camera Run Setup", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w")

        mode_frame = ttk.LabelFrame(container, text="Run Mode", padding=12)
        mode_frame.pack(fill="x", pady=(12, 8))
        ttk.Label(mode_frame, text="Camera selection").grid(row=0, column=0, sticky="w")
        mode = ttk.Combobox(
            mode_frame,
            textvariable=self.camera_mode,
            values=["Camera 1 only", "Camera 2 only", "Both cameras"],
            state="readonly",
            width=22,
        )
        mode.grid(row=0, column=1, sticky="w", padx=(12, 0))
        mode.bind("<<ComboboxSelected>>", self._on_change)

        options_frame = ttk.Frame(mode_frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(12, 0))
        for label, variable in [
            ("Setup calibration", self.setup),
            ("MediaPipe feet", self.use_mediapipe),
            ("Appearance ReID", self.use_reid),
            ("MQTT publish", self.use_mqtt),
        ]:
            checkbox = ttk.Checkbutton(options_frame, text=label, variable=variable, command=self._refresh_preview)
            checkbox.pack(side="left", padx=(0, 18))

        # 🚨 YOLO/LOCATION DEVICE DROPDOWN 🚨
        ttk.Label(mode_frame, text="YOLO Device").grid(row=2, column=0, sticky="w", pady=(12, 0))
        yolo_box = ttk.Combobox(mode_frame, textvariable=self.yolo_device, values=["GPU (0)", "GPU (1)", "CPU"], state="readonly", width=22)
        yolo_box.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(12, 0))
        yolo_box.bind("<<ComboboxSelected>>", self._on_change)

        # 🚨 REID DEVICE DROPDOWN 🚨
        ttk.Label(mode_frame, text="ReID Device").grid(row=3, column=0, sticky="w", pady=(4, 0))
        reid_box = ttk.Combobox(mode_frame, textvariable=self.reid_device, values=["GPU (0)", "GPU (1)", "CPU"], state="readonly", width=22)
        reid_box.grid(row=3, column=1, sticky="w", padx=(12, 0), pady=(4, 0))
        reid_box.bind("<<ComboboxSelected>>", self._on_change)

        camera_frame = ttk.LabelFrame(container, text="Cameras", padding=12)
        camera_frame.pack(fill="x", pady=8)
        camera_frame.columnconfigure(1, weight=1)

        self._add_camera_rows(camera_frame, row_offset=0, label="Camera 1", source=self.source_1, camera_id=self.camera_id_1, matrix=self.matrix_1, corner=self.corner_1)
        ttk.Separator(camera_frame).grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        self._add_camera_rows(camera_frame, row_offset=5, label="Camera 2", source=self.source_2, camera_id=self.camera_id_2, matrix=self.matrix_2, corner=self.corner_2)

        advanced = ttk.LabelFrame(container, text="Common Settings", padding=12)
        advanced.pack(fill="x", pady=8)
        advanced.columnconfigure(1, weight=1)
        self._entry(advanced, "YOLO model", self.model, 0)
        self._entry(advanced, "YOLO NMS IoU", self.yolo_nms_iou, 1, width=12)
        self._entry(advanced, "ByteTrack config", self.tracker_config, 2)
        self._entry(advanced, "Run ID", self.run_id, 3)
        self._entry(advanced, "Fusion distance cm", self.fusion_distance, 4, width=12)
        self._entry(advanced, "MQTT broker", self.mqtt_broker, 5)
        self._entry(advanced, "MQTT port", self.mqtt_port, 6, width=12)
        self._entry(advanced, "ReID backend URL", self.reid_api_url, 7)

        preview_frame = ttk.LabelFrame(container, text="Command Preview", padding=12)
        preview_frame.pack(fill="both", expand=True, pady=8)
        preview = tk.Text(preview_frame, height=4, wrap="word")
        preview.pack(fill="both", expand=True)
        preview.configure(state="disabled")
        self.preview_widget = preview

        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(8, 0))
        ttk.Button(button_frame, text="Refresh", command=self._refresh_preview).pack(side="left")
        ttk.Button(button_frame, text="Start", command=self._start).pack(side="right")

        for variable in [
            self.yolo_device,
            self.reid_device,
            self.source_1,
            self.source_2,
            self.camera_id_1,
            self.camera_id_2,
            self.matrix_1,
            self.matrix_2,
            self.corner_1,
            self.corner_2,
            self.model,
            self.yolo_nms_iou,
            self.tracker_config,
            self.run_id,
            self.fusion_distance,
            self.mqtt_broker,
            self.mqtt_port,
            self.reid_api_url,
        ]:
            variable.trace_add("write", lambda *_args: self._refresh_preview())

    def _add_camera_rows(self, parent, row_offset, label, source, camera_id, matrix, corner):
        ttk.Label(parent, text=label, font=("Segoe UI", 10, "bold")).grid(row=row_offset, column=0, sticky="w", pady=(0, 4))
        self._entry(parent, "Source", source, row_offset + 1)
        self._entry(parent, "Camera ID", camera_id, row_offset + 2, width=18)
        self._entry(parent, "Matrix", matrix, row_offset + 3, width=28)
        ttk.Label(parent, text="Missing corner").grid(row=row_offset + 2, column=2, sticky="e", padx=(16, 8))
        corner_box = ttk.Combobox(parent, textvariable=corner, values=CORNER_OPTIONS, state="readonly", width=18)
        corner_box.grid(row=row_offset + 3, column=2, sticky="e")
        corner_box.bind("<<ComboboxSelected>>", self._on_change)

    def _entry(self, parent, label, variable, row, width=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=3)
        return entry

    def _on_change(self, _event=None):
        self._refresh_preview()

    def _selected_cameras(self):
        mode = self.camera_mode.get()
        if mode == "Camera 1 only":
            return ["1"]
        if mode == "Camera 2 only":
            return ["2"]
        return ["1", "2"]

    def _add_corner(self, command, option, value):
        if value and value != "None":
            command.extend([option, value])

    def _build_command(self):
        selected = self._selected_cameras()
        command = [sys.executable, str(SCRIPT_PATH)]

        # 1. SET YOLO DEVICE
        if self.yolo_device.get() == "GPU (0)":
            command.extend(["--device", "0"])
        elif self.yolo_device.get() == "GPU (1)":
            command.extend(["--device", "1"])
        else:
            command.extend(["--device", "cpu"])

        if self.setup.get():
            command.append("--setup")

        command.extend(["--model", self.model.get().strip() or "yolo11n.pt"])
        command.extend(["--iou", self.yolo_nms_iou.get().strip() or str(DEFAULT_YOLO_NMS_IOU)])
        command.extend(["--tracker-config", self.tracker_config.get().strip() or DEFAULT_TRACKER_CONFIG_PATH])

        if self.use_mediapipe.get():
            command.append("--use-mediapipe-feet")

        if self.use_reid.get():
            command.extend([
                "--use-appearance-reid",
                "--reid-checkpoint",
                "transreid_msmt17.pth",
                "--fastreid-root",
                "fast-reid",
            ])
            # 2. SET REID DEVICE
            if self.reid_device.get() == "GPU (0)":
                command.extend(["--reid-device", "cuda:0"])
            elif self.reid_device.get() == "GPU (1)":
                command.extend(["--reid-device", "cuda:1"])
            else:
                command.extend(["--reid-device", "cpu"])
            if self.reid_api_url.get().strip():
                command.extend(["--reid-api-url", self.reid_api_url.get().strip()])

        if selected == ["1"]:
            command.extend(["--source", self.source_1.get().strip()])
            command.extend(["--matrix", self.matrix_1.get().strip()])
            command.extend(["--camera-id", self.camera_id_1.get().strip()])
            self._add_corner(command, "--missing-corner", self.corner_1.get())
        elif selected == ["2"]:
            command.extend(["--source", self.source_2.get().strip()])
            command.extend(["--matrix", self.matrix_2.get().strip()])
            command.extend(["--camera-id", self.camera_id_2.get().strip()])
            self._add_corner(command, "--missing-corner", self.corner_2.get())
        else:
            command.extend(["--source", self.source_1.get().strip()])
            command.extend(["--source-2", self.source_2.get().strip()])
            command.extend(["--matrix", self.matrix_1.get().strip()])
            command.extend(["--matrix-2", self.matrix_2.get().strip()])
            command.extend(["--camera-id", self.camera_id_1.get().strip()])
            command.extend(["--camera-id-2", self.camera_id_2.get().strip()])
            self._add_corner(command, "--missing-corner", self.corner_1.get())
            self._add_corner(command, "--missing-corner-2", self.corner_2.get())
            if self.fusion_distance.get().strip():
                command.extend(["--fusion-distance-cm", self.fusion_distance.get().strip()])

        if self.run_id.get().strip():
            command.extend(["--run-id", self.run_id.get().strip()])

        if self.use_mqtt.get():
            command.extend(["--mqtt-broker", self.mqtt_broker.get().strip()])
            command.extend(["--mqtt-port", self.mqtt_port.get().strip()])
            command.extend(["--mqtt-publish-interval", "0.5"])

        return command

    def _refresh_preview(self):
        command = self._build_command()
        pretty = subprocess.list2cmdline(command)
        self.preview_widget.configure(state="normal")
        self.preview_widget.delete("1.0", "end")
        self.preview_widget.insert("1.0", pretty)
        self.preview_widget.configure(state="disabled")

    def _validate(self):
        selected = self._selected_cameras()
        checks = []
        if "1" in selected:
            checks.extend([
                ("Camera 1 source", self.source_1.get()),
                ("Camera 1 ID", self.camera_id_1.get()),
                ("Camera 1 matrix", self.matrix_1.get()),
            ])
        if "2" in selected:
            checks.extend([
                ("Camera 2 source", self.source_2.get()),
                ("Camera 2 ID", self.camera_id_2.get()),
                ("Camera 2 matrix", self.matrix_2.get()),
            ])
        for label, value in checks:
            if not value.strip():
                messagebox.showerror("Missing value", f"{label} cannot be empty.")
                return False
        try:
            nms_iou = float(self.yolo_nms_iou.get())
        except ValueError:
            messagebox.showerror("Invalid value", "YOLO NMS IoU must be a number greater than 0 and at most 1.")
            return False
        if not 0.0 < nms_iou <= 1.0:
            messagebox.showerror("Invalid value", "YOLO NMS IoU must be greater than 0 and at most 1.")
            return False
        return True

    def _start(self):
        if not self._validate():
            return

        command = self._build_command()
        try:
            subprocess.Popen(command, cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE)
        except AttributeError:
            subprocess.Popen(command, cwd=str(ROOT))
        except OSError as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        messagebox.showinfo("Started", "Camera detection has started in a new console window.")


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
