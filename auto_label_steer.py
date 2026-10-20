"""
auto_label_steer.py — YOLO shape detection with auto-labelling and Arduino steering.

Runs a YOLO model on images, video or a live camera and:
  1. draws detections with an FPS / object-count overlay,
  2. auto-labels: saves every frame that has a high-confidence detection
     (image + YOLO-format .txt label) so it can be added to the training set,
  3. steers: sends "Left" / "Right" over serial to an Arduino depending on
     which half of the frame the detected object is in.

The detection / display loop is based on yolo_detect.py from
EdjeElectronics/Train-and-Deploy-YOLO-Models (Apache-2.0). See NOTICE.

Examples:
  python auto_label_steer.py --model my_model.pt --source usb0 --resolution 1280x720
  python auto_label_steer.py --model my_model.pt --source usb0 --port COM10
  python auto_label_steer.py --model my_model.pt --source /dev/video0 --port /dev/ttyACM0
  python auto_label_steer.py --model my_model.pt --source test_images/ --no-autolabel

Keys: q = quit, s = pause, p = save a screenshot (capture.png)
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', required=True,
                    help='Path to YOLO model file (e.g. "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True,
                    help='Image file, image folder, video file, USB camera ("usb0") or Picamera ("picamera0")')
parser.add_argument('--thresh', type=float, default=0.5,
                    help='Minimum confidence to draw / use a detection (default 0.5)')
parser.add_argument('--resolution', default=None,
                    help='Resize frames to WxH (e.g. "1280x720"); otherwise keep source resolution')
parser.add_argument('--record', action='store_true',
                    help='Record annotated output to demo1.avi (needs --resolution)')
parser.add_argument('--port', default=None,
                    help='Arduino serial port (e.g. COM10 or /dev/ttyACM0). Omit to run without steering.')
parser.add_argument('--baud', type=int, default=9600, help='Serial baud rate (default 9600)')
parser.add_argument('--send-interval', type=float, default=0.1,
                    help='Minimum seconds between steering commands (default 0.1)')
parser.add_argument('--label-dir', default='auto_labels',
                    help='Output folder for auto-labelled frames (default ./auto_labels)')
parser.add_argument('--label-thresh', type=float, default=0.8,
                    help='Minimum confidence for a detection to be auto-labelled (default 0.8)')
parser.add_argument('--no-autolabel', action='store_true', help='Disable auto-labelling')
args = parser.parse_args()

if not os.path.exists(args.model):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(1)

# ── Model ────────────────────────────────────────────────────────────────────
model = YOLO(args.model, task='detect')
labels = model.names

# ── Source type ──────────────────────────────────────────────────────────────
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
src = args.source

if os.path.isdir(src):
    source_type = 'folder'
elif os.path.isfile(src):
    ext = os.path.splitext(src)[1].lower()
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(1)
elif src.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(src[3:])
elif src.startswith('picamera'):
    source_type = 'picamera'
elif src.startswith('/dev/video'):
    source_type = 'usb'
    usb_idx = src
else:
    print(f'Input {src} is invalid.')
    sys.exit(1)

is_stream = source_type in ('video', 'usb', 'picamera')

resize = args.resolution is not None
if resize:
    resW, resH = map(int, args.resolution.lower().split('x'))

if args.record:
    if source_type not in ('video', 'usb'):
        print('Recording only works for video and camera sources.')
        sys.exit(1)
    if not resize:
        print('Please specify --resolution to record video.')
        sys.exit(1)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# ── Open source ──────────────────────────────────────────────────────────────
if source_type == 'image':
    imgs_list = [src]
elif source_type == 'folder':
    imgs_list = sorted(f for f in glob.glob(os.path.join(src, '*'))
                       if os.path.splitext(f)[1].lower() in img_ext_list)
elif source_type in ('video', 'usb'):
    cap = cv2.VideoCapture(src if source_type == 'video' else usb_idx)
    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# ── Serial link to Arduino (optional) ────────────────────────────────────────
arduino = None
if args.port:
    import serial
    arduino = serial.Serial(args.port, args.baud)
    time.sleep(2)  # Arduino resets when the port opens
    print(f'Steering commands -> {args.port} @ {args.baud}')
last_send = 0.0
last_cmd = None

# ── Auto-labelling ───────────────────────────────────────────────────────────
autolabel = not args.no_autolabel
if autolabel:
    os.makedirs(os.path.join(args.label_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.label_dir, 'labels'), exist_ok=True)
saved_count = 0

bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

avg_frame_rate = 0.0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# ── Main loop ────────────────────────────────────────────────────────────────
while True:
    t_start = time.perf_counter()

    if source_type in ('image', 'folder'):
        if img_count >= len(imgs_list):
            print('All images have been processed.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type in ('video', 'usb'):
        ret, frame = cap.read()
        if not ret or frame is None:
            print('No more frames (end of video or camera disconnected).')
            break
    else:  # picamera
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from the Picamera.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    raw_frame = frame.copy()          # unannotated copy for auto-labelling
    img_h, img_w = frame.shape[:2]

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    best = None                        # (conf, center_x) of the most confident object
    frame_labels = []                  # YOLO-format lines for this frame

    for i in range(len(detections)):
        xmin, ymin, xmax, ymax = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf < args.thresh:
            continue

        color = bbox_colors[classidx % 10]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f'{classname}: {int(conf * 100)}%'
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        object_count += 1

        center_x = (xmin + xmax) // 2
        if best is None or conf > best[0]:
            best = (conf, center_x)

        if autolabel and conf >= args.label_thresh:
            x_c = (xmin + xmax) / 2 / img_w
            y_c = (ymin + ymax) / 2 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            frame_labels.append(f'{classidx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}')

    # Save image + label together so the pair can go straight into a YOLO dataset
    if frame_labels:
        if source_type in ('image', 'folder'):
            base = os.path.splitext(os.path.basename(img_filename))[0]
        else:
            base = f'frame_{int(time.time() * 1000)}'
        cv2.imwrite(os.path.join(args.label_dir, 'images', base + '.jpg'), raw_frame)
        with open(os.path.join(args.label_dir, 'labels', base + '.txt'), 'w') as f:
            f.write('\n'.join(frame_labels) + '\n')
        saved_count += 1

    # Steering: which half of the frame is the most confident object in?
    if arduino is not None and best is not None:
        now = time.perf_counter()
        if now - last_send >= args.send_interval:
            cmd = 'Left' if best[1] < img_w // 2 else 'Right'
            arduino.write(f'{cmd}\n'.encode())
            last_send, last_cmd = now, cmd

    # Overlay + display
    if is_stream:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    if last_cmd:
        cv2.putText(frame, f'Steer: {last_cmd}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.imshow('YOLO detection results', frame)
    if args.record:
        recorder.write(frame)

    key = cv2.waitKey() if source_type in ('image', 'folder') else cv2.waitKey(5)
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('s'), ord('S')):
        cv2.waitKey()
    elif key in (ord('p'), ord('P')):
        cv2.imwrite('capture.png', frame)

    # FPS (rolling average)
    frame_rate_buffer.append(1.0 / (time.perf_counter() - t_start))
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

# ── Clean up ─────────────────────────────────────────────────────────────────
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if autolabel:
    print(f'Auto-labelled frames saved: {saved_count} -> {os.path.abspath(args.label_dir)}')
if source_type in ('video', 'usb'):
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if args.record:
    recorder.release()
if arduino is not None:
    arduino.close()
cv2.destroyAllWindows()
