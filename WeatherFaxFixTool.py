import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

img = None
original_img = None
height = 0
width = 0
segments = []
brightness = 0
sharpness = 1.0
contrast = 1.0
saturation = 1.0
slant_points = []
current_filename = ""

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.ANTIALIAS

def cv_imread(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img

def apply_fix():
    if img is None:
        return None

    corrected = img.copy()
    for seg in segments:
        s, h, sh = seg['start'], seg['height'], seg['shift']
        end = min(s + h, height)
        corrected[s:end] = np.roll(corrected[s:end], sh, axis=1)

    if len(slant_points) == 2:
        x1, y1 = slant_points[0]
        x2, y2 = slant_points[1]
        dx = x2 - x1
        dy = y2 - y1
        if dy != 0:
            slope = dx / dy
            for row in range(height):
                shift = int(-slope * (row - y1))
                corrected[row] = np.roll(corrected[row], shift, axis=0)

    if len(corrected.shape) == 2 or corrected.shape[2] == 1:
        corrected = cv2.convertScaleAbs(corrected, alpha=contrast, beta=brightness)
    else:
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        corrected = cv2.convertScaleAbs(corrected, alpha=contrast, beta=brightness)

    if sharpness != 1.0:
        kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
        sharpened = cv2.filter2D(corrected, -1, kernel)
        corrected = cv2.addWeighted(corrected, sharpness, sharpened, 1 - sharpness, 0)
    return corrected

def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def fix_bad_lines(img, std_threshold=5):
    fixed = img.copy()
    h, w = img.shape
    for y in range(1, h - 1):
        line_std = np.std(img[y])
        if line_std < std_threshold:
            fixed[y] = ((img[y - 1].astype(np.uint16) + img[y + 1].astype(np.uint16)) // 2).astype(np.uint8)
    return fixed

class SegmentEditor(tk.Tk):
    CONTROL_WIDTH = 350

    def __init__(self):
        super().__init__()
        self.title("WeatherFax 图像处理")
        self.geometry("1600x900")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        right_frame = tk.Frame(self, width=self.CONTROL_WIDTH)
        right_frame.grid(row=0, column=1, sticky='ns')
        right_frame.grid_propagate(False)

        button_frame = tk.Frame(right_frame)
        button_frame.pack(side='top', fill='x', pady=5)
        tk.Button(button_frame, text="打 开 图 像", command=self.open_image).pack(side='left', padx=15)
        tk.Button(button_frame, text="保 存 图 像", command=self.save_image_as).pack(side='left', padx=15)
        tk.Button(button_frame, text="新 增 分 段", command=self.add_segment).pack(side='left', padx=15)
        tk.Button(button_frame, text="取 消 倾 斜", command=self.clear_slant).pack(side='left', padx=15)

        button_frame = tk.Frame(right_frame)
        button_frame.pack(side='top', fill='x', pady=5)
        tk.Button(button_frame, text="修 正 条 纹", command=self.remove_striping).pack(side='left', padx=15)
        tk.Button(button_frame, text="取 消 修 正", command=self.restore_original).pack(side='left', padx=15)
        tk.Button(button_frame, text="向 左 旋 转", command=self.rotate_left).pack(side='left', padx=15)
        tk.Button(button_frame, text="向 右 旋 转", command=self.rotate_right).pack(side='left', padx=15)

        self.status = tk.Label(right_frame, text="行号: --", anchor='w')
        self.status.pack(fill='x')

        tk.Label(right_frame, text="亮度").pack(anchor='w', padx=5)
        self.brightness_var = tk.DoubleVar(value=0)
        tk.Scale(right_frame, from_=-100, to=100, orient='horizontal',
                 variable=self.brightness_var, command=self.update_brightness).pack(fill='x', padx=5)

        tk.Label(right_frame, text="对比度").pack(anchor='w', padx=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        tk.Scale(right_frame, from_=0.5, to=3.0, resolution=0.05, orient='horizontal',
                 variable=self.contrast_var, command=self.update_contrast).pack(fill='x', padx=5)

        tk.Label(right_frame, text="锐度").pack(anchor='w', padx=5)
        self.sharpness_var = tk.DoubleVar(value=1.0)
        tk.Scale(right_frame, from_=0.0, to=3.0, resolution=0.05, orient='horizontal',
                 variable=self.sharpness_var, command=self.update_sharpness).pack(fill='x', padx=5)

        tk.Label(right_frame, text="饱和度").pack(anchor='w', padx=5)
        self.saturation_var = tk.DoubleVar(value=1.0)
        tk.Scale(right_frame, from_=0.0, to=2.0, resolution=0.05, orient='horizontal',
                 variable=self.saturation_var, command=self.update_saturation).pack(fill='x', padx=5)

        container = tk.Frame(right_frame)
        container.pack(side='top', fill='both', expand=True)
        self.control_canvas = tk.Canvas(container, width=self.CONTROL_WIDTH)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_frame = tk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor='nw')
        self.control_frame.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.bind_all("<MouseWheel>", lambda e: self.control_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.seg_controls = []

        self.bind("<Configure>", self.on_window_resize)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self._scale = 1.0
        self._image_offset_y = 0
        self._update_preview_after_id = None

    def schedule_preview_update(self):
        """限频刷新预览，避免过于频繁"""
        if self._update_preview_after_id:
           self.after_cancel(self._update_preview_after_id)
        self._update_preview_after_id = self.after(150, self.update_preview)

    def update_brightness(self, val):
        global brightness
        brightness = float(val)
        self.schedule_preview_update()

    def update_contrast(self, val):
        global contrast
        contrast = float(val)
        self.schedule_preview_update()

    def update_sharpness(self, val):
        global sharpness
        sharpness = -1 * float(val)
        self.schedule_preview_update()

    def update_saturation(self, val):
        global saturation
        saturation = float(val)
        self.schedule_preview_update()

    def remove_striping(self):
        global img
        if img is None:
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        median = median_filter(gray, ksize=3)
        fixed = fix_bad_lines(median, std_threshold=5)
        if len(img.shape) == 3:
            img[..., 0] = fixed
            img[..., 1] = fixed
            img[..., 2] = fixed
        else:
            img = fixed
        self.schedule_preview_update()

    def restore_original(self):
        global img
        if original_img is not None:
            img = original_img.copy()
            self.schedule_preview_update()

    def display_image(self, image_array):
        image = Image.fromarray(image_array)
        max_w = self.canvas.winfo_width()
        max_h = self.canvas.winfo_height()
        iw, ih = image.size
        scale = min(max_w / iw, max_h / ih)
        self._scale = scale
        self._image_offset_y = (max_h - int(ih * scale)) // 2
        image_resized = image.resize((int(iw * scale), int(ih * scale)), RESAMPLE)
        self.tk_image = ImageTk.PhotoImage(image_resized)

        self.canvas.delete("all")
        x_offset = (max_w - image_resized.width) // 2
        y_offset = self._image_offset_y

        self.canvas.create_image(x_offset, y_offset, anchor='nw', image=self.tk_image)

        # 画红色斜率线（如果有两点）
        if len(slant_points) == 2:
            (x1, y1), (x2, y2) = slant_points
            sx1 = x1 * scale + x_offset
            sy1 = y1 * scale + y_offset
            sx2 = x2 * scale + x_offset
            sy2 = y2 * scale + y_offset
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill='red', width=2)

    def rotate_left(self):
        global img, original_img, height, width
        if img is None:
            return
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.reset_all()
        if original_img is not None:
            original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width = img.shape[:2]
        self.reset_all()
        self.schedule_preview_update()

    def rotate_right(self):
        global img, original_img, height, width
        if img is None:
            return
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if original_img is not None:
            original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
        height, width = img.shape[:2]
        self.reset_all()
        self.schedule_preview_update()

    def reset_all(self):
            global height, width, segments, slant_points
            height, width = img.shape[:2]
            segments.clear()
            segments.append({'start': 0, 'height': min(100, height), 'shift': 0})
            for seg in list(self.seg_controls):
                if '_frame' in seg:
                    seg['_frame'].destroy()
            self.seg_controls.clear()
            for seg in segments:
                self.create_segment_controls(seg)
            slant_points.clear()
            self.schedule_preview_update()
        
    def on_mouse_move(self, event):
        if img is None:
            return
        y = event.y - self._image_offset_y
        if y < 0:
            self.status.config(text="行号: --")
            return
        img_y = int(y / self._scale)
        if 0 <= img_y < height:
            self.status.config(text=f"行号: {img_y}")
        else:
            self.status.config(text="行号: --")

    def on_canvas_click(self, event):
        if img is None:
            return
        x = int((event.x - (self.canvas.winfo_width() - width*self._scale)//2) / self._scale)
        y = int((event.y - self._image_offset_y) / self._scale)
        if 0 <= x < width and 0 <= y < height:
            slant_points.append((x, y))
            if len(slant_points) > 2:
                slant_points.pop(0)
            self.schedule_preview_update()

    def clear_slant(self):
        slant_points.clear()
        self.schedule_preview_update()

    def save_image_as(self):
        corrected = apply_fix()
        if corrected is None:
            messagebox.showwarning("提示", "当前无图像可保存")
            return
        base, ext = os.path.splitext(current_filename)
        default_name = base + "_fixed.jpg"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=os.path.basename(default_name),
            filetypes=[("JPEG文件", "*.jpg *.jpeg"), ("PNG文件", "*.png"), ("所有文件", "*.*")]
        )
        if not file_path:
            return
        ext = file_path.split('.')[-1].lower()
        params = []
        if ext in ('jpg', 'jpeg'):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        elif ext == 'png':
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        try:
            cv2.imwrite(file_path, corrected, params)
            messagebox.showinfo("保存成功", f"图像已保存为 {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{e}")

    def open_image(self):
        global img, original_img, height, width, segments, brightness, contrast, saturation, sharpness, slant_points, current_filename
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if not file_path:
            return
        img_cv = cv_imread(file_path)
        if img_cv is None:
            messagebox.showerror("错误", "无法打开图像文件")
            return
        img = img_cv
        original_img = img.copy()
        current_filename = file_path
        height, width = img.shape[:2]
        segments.clear()
        segments.append({'start': 0, 'height': min(100, height), 'shift': 0})
        for seg in list(self.seg_controls):
            if '_frame' in seg:
                seg['_frame'].destroy()
        self.seg_controls.clear()
        for seg in segments:
            self.create_segment_controls(seg)
        brightness = 0
        contrast = 1.0
        saturation = 1.0
        sharpness = 1.0
        slant_points.clear()
        self.brightness_var.set(brightness)
        self.contrast_var.set(contrast)
        self.saturation_var.set(saturation)
        self.sharpness_var.set(sharpness)
        self.update_preview()

    def create_segment_controls(self, seg):
        frame = tk.Frame(self.control_frame, bd=1, relief='solid', padx=5, pady=5)
        frame.pack(fill='x', pady=3)
        def on_value_change(key, val_str):
            try:
                val = int(val_str)
            except:
                return
            if key == 'start':
                val = max(0, min(val, height))
            elif key == 'height':
                val = max(1, min(val, height))
            elif key == 'shift':
                val = max(-width//2, min(val, width//2))
            seg[key] = val
            if key in ('start', 'height'):
                self.update_following_starts()
            self.update_preview()
            self.after(1, self.sync_controls)
        vars_dict = {
            'start': tk.StringVar(value=str(seg['start'])),
            'height': tk.StringVar(value=str(seg['height'])),
            'shift': tk.StringVar(value=str(seg['shift']))
        }
        sliders = {}
        def trace_var(key):
            return lambda *args: on_value_change(key, vars_dict[key].get())
        tk.Label(frame, text="起始行").grid(row=0, column=0)
        sliders['start'] = tk.Scale(frame, from_=0, to=height, orient='horizontal', length=200,
                                   command=lambda v: on_value_change('start', v))
        sliders['start'].set(seg['start'])
        sliders['start'].grid(row=0, column=1)
        ent_start = tk.Spinbox(frame, from_=0, to=height, textvariable=vars_dict['start'], width=6)
        ent_start.grid(row=0, column=2, padx=5)
        vars_dict['start'].trace_add('write', trace_var('start'))

        tk.Label(frame, text="高度").grid(row=1, column=0)
        sliders['height'] = tk.Scale(frame, from_=1, to=height, orient='horizontal', length=200,
                                    command=lambda v: on_value_change('height', v))
        sliders['height'].set(seg['height'])
        sliders['height'].grid(row=1, column=1)
        ent_height = tk.Spinbox(frame, from_=1, to=height, textvariable=vars_dict['height'], width=6)
        ent_height.grid(row=1, column=2, padx=5)
        vars_dict['height'].trace_add('write', trace_var('height'))

        tk.Label(frame, text="位移").grid(row=2, column=0)
        sliders['shift'] = tk.Scale(frame, from_=-width//2, to=width//2, orient='horizontal', length=200,
                                   command=lambda v: on_value_change('shift', v))
        sliders['shift'].set(seg['shift'])
        sliders['shift'].grid(row=2, column=1)
        ent_shift = tk.Spinbox(frame, from_=-width//2, to=width//2, textvariable=vars_dict['shift'], width=6)
        ent_shift.grid(row=2, column=2, padx=5)
        vars_dict['shift'].trace_add('write', trace_var('shift'))

        tk.Button(frame, text="删除", command=lambda: self.remove_segment(frame, seg)).grid(row=0, column=3, rowspan=3, padx=(5, 25))
        seg['_frame'] = frame
        seg['_vars'] = vars_dict
        seg['_sliders'] = sliders
        self.seg_controls.append(seg)

    def sync_controls(self):
        for seg in list(self.seg_controls):
            if seg not in segments:
                self.seg_controls.remove(seg)
                continue
            for key in ('start', 'height', 'shift'):
                val = seg[key]
                var = seg['_vars'][key]
                slider = seg['_sliders'][key]
                if var.get() != str(val):
                    var.set(str(val))
                if slider.get() != val:
                    slider.set(val)

    def update_following_starts(self):
        for i in range(1, len(segments)):
            prev = segments[i-1]
            curr = segments[i]
            new_start = prev['start'] + prev['height']
            if curr['start'] != new_start:
                curr['start'] = new_start
        self.sync_controls()

    def add_segment(self):
        global img, height
        if img is None:
            messagebox.showwarning("提示", "请先打开图像文件！")
            return
        if segments:
            last = segments[-1]
            new_start = last['start'] + last['height']
            if new_start > height:
                new_start = height
        else:
            new_start = 0
        seg = {'start': new_start, 'height': min(100, height), 'shift': 0}
        segments.append(seg)
        self.create_segment_controls(seg)
        self.update_preview()

    def remove_segment(self, frame, seg):
        if seg in segments:
            segments.remove(seg)
            frame.destroy()
            if seg in self.seg_controls:
                self.seg_controls.remove(seg)
            self.update_following_starts()
            self.update_preview()

    def update_preview(self):
        corrected = apply_fix()
        if corrected is None:
            return
        image = Image.fromarray(corrected if len(corrected.shape)==2 else cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        max_w = self.canvas.winfo_width()
        max_h = self.canvas.winfo_height()
        iw, ih = image.size
        scale = min(max_w / iw, max_h / ih)
        self._scale = scale
        self._image_offset_y = (max_h - int(ih * scale)) // 2
        image_resized = image.resize((int(iw * scale), int(ih * scale)), RESAMPLE)
        self.tk_image = ImageTk.PhotoImage(image_resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            (max_w - image_resized.width) // 2,
            self._image_offset_y,
            anchor='nw',
            image=self.tk_image
        )
        self.display_image(corrected)

    def on_window_resize(self, event):
        self.update_preview()

if __name__ == "__main__":
    app = SegmentEditor()
    app.mainloop()

