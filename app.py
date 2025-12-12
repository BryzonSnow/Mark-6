import os
import time
import threading
import queue

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

#   CONFIGURACIÓN GENERAL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "best-fp16.tflite")

app = Flask(__name__)

# Detectar solo en 1 de cada N frames (reduce carga de CPU)
DETECT_EVERY_N = 1  

#   HARDWARE: PiCar-X

try:
    from picarx import Picarx
except ImportError:
    Picarx = None
    print("[WARN] No se encontró la librería picarx. Modo simulación.")

px = None
if Picarx is not None:
    try:
        px = Picarx()
        print("[INFO] Picar-X inicializado correctamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar Picar-X: {e}")
        px = None

#   TFLITE / DETECCIÓN

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print("[WARN] Usando TensorFlow completo para TFLite.")
    except ImportError:
        Interpreter = None
        print("[ERROR] No se pudo importar ningún intérprete TFLite.")

interpreter = None
input_details = None
output_details = None
INPUT_W, INPUT_H = 320, 320

# Umbrales de confianza
CONF_THRESH = 0.45         # umbral general
CONF_THRESH_LEFT = 0.30    # umbral más bajo solo para GIRO IZQUIERDA

# Clases 0 a 8:
CLASS_NAMES = [
    "STOP",                       # 0
    "SIGA",                       # 1
    "GIRAR DERECHA",              # 2
    "GIRAR IZQUIERDA",            # 3
    "FONDO",                      # 4
    "CEDA EL PASO",               # 5
    "RETONO (GIRO EN U)",         # 6
    "VEL MÁX 10 KM/H",            # 7
    "VEL MÁX 30 KM/H",            # 8
]

BACKGROUND_CLASS_ID = 4  # FONDO

CLASS_STOP = 0
CLASS_GO_STRAIGHT = 1
CLASS_TURN_RIGHT = 2
CLASS_TURN_LEFT = 3
CLASS_BACKGROUND = 4
CLASS_YIELD = 5
CLASS_UTURN = 6
CLASS_VMAX_10 = 7
CLASS_VMAX_30 = 8

#   CONFIG MOVIMIENTO

SPEED_STOP   = 0
SPEED_SLOW   = 10
SPEED_30     = 20
SPEED_NORMAL = 30
SPEED_MAX    = 40

# --- Calibración de giros 
TURN_SPEED       = 20   # velocidad mientras gira (más lento = giro más preciso)
TURN_ANGLE_DEG   = 30   # ángulo del servo para giros de 90°
TURN_TIME_90_I     = 1.8  # duración del giro para 90°
TURN_TIME_90_D     = 1.2

UTURN_ANGLE_DEG  = 30   # ángulo para la vuelta en U
UTURN_TIME_180   = 2.5  # duración para 180° 

current_speed = SPEED_STOP
control_status = "Inicializando..."

COMMAND_COOLDOWN = 3.0  # segundos: evita encadenar giros mientras aún gira
last_command_time = 0.0
last_class_executed = None

EMERGENCY_STOP = False
AUTO_DRIVE_ENABLED = False

#   CONFIG STREAM (MÓVIL)

TARGET_FPS   = 12.0 # Se puede aumentar
STREAM_WIDTH = 320
STREAM_HEIGHT = 240
JPEG_QUALITY = 30


#   ESTADO + LOCKS


state_lock = threading.Lock()
hardware_lock = threading.Lock()
frame_lock = threading.Lock()

last_detection = {
    "cls_id": None,
    "label": None,
    "confidence": 0.0,
    "timestamp": None,
    "obstacle_distance_cm": None,
    "emergency_stop": False,
    "control_status": control_status,
    "auto_drive_enabled": AUTO_DRIVE_ENABLED,
    "target_fps": TARGET_FPS,
    "jpeg_quality": JPEG_QUALITY,
    "width": STREAM_WIDTH,
    "height": STREAM_HEIGHT,
}

last_raw_frame = None
last_processed_frame = None


#   ULTRASONIDO


OBSTACLE_STOP_DISTANCE = 25.0
CLEAR_DISTANCE = 45.0

def read_distance():
    if px is None:
        return None
    try:
        with hardware_lock:
            dist = px.get_distance()
        if dist is None or dist <= 0 or dist > 500:
            return None
        return float(dist)
    except Exception as e:
        print(f"[WARN] Error leyendo distancia: {e}")
        return None


#   HILO DE CONTROL


control_queue = queue.Queue(maxsize=100)

def _set_speed_and_direction_hw(speed, angle=0):
    global current_speed

    speed = max(0, min(speed, SPEED_MAX))
    current_speed = speed

    if px is None:
        print(f"[SIM] speed={speed}, angle={angle}")
        return

    try:
        with hardware_lock:
            if speed == 0:
                px.stop()
            else:
                px.forward(speed)
            px.set_dir_servo_angle(angle)
    except Exception as e:
        print(f"[WARN] Error controlando PiCar-X: {e}")

class ControlThread(threading.Thread):
    def __init__(self, cmd_queue):
        super().__init__(daemon=True)
        self.cmd_queue = cmd_queue
        self.running = True

    def stop(self):
        self.running = False
        try:
            self.cmd_queue.put_nowait({"type": "STOP"})
        except queue.Full:
            pass

    def run(self):
        global EMERGENCY_STOP, control_status
        print("[CONTROL] Hilo de control iniciado.")
        while self.running:
            try:
                cmd = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if cmd is None:
                self.cmd_queue.task_done()
                continue

            cmd_type = cmd.get("type")

            if EMERGENCY_STOP and cmd_type not in ("STOP", "EMERGENCY_STOP"):
                self.cmd_queue.task_done()
                continue

            if cmd_type == "STOP":
                _set_speed_and_direction_hw(SPEED_STOP, 0)
                with state_lock:
                    control_status = cmd.get("reason", "Detenido")
                    last_detection["control_status"] = control_status

            elif cmd_type == "SET_SPEED":
                speed = cmd.get("speed", SPEED_STOP)
                angle = cmd.get("angle", 0)
                _set_speed_and_direction_hw(speed, angle)
                with state_lock:
                    control_status = cmd.get("status", "Actualizando velocidad/dirección")
                    last_detection["control_status"] = control_status

            elif cmd_type == "TURN":
                speed = cmd.get("speed", SPEED_NORMAL)
                angle = cmd.get("angle", 0)
                duration = cmd.get("duration", 0.5)
                after_speed = cmd.get("after_speed", SPEED_NORMAL)
                after_angle = cmd.get("after_angle", 0)

                # 1) Frenar un poco para que el giro siempre parta igual
                _set_speed_and_direction_hw(SPEED_STOP, 0)
                time.sleep(0.1)

                # 2) Giro
                _set_speed_and_direction_hw(speed, angle)
                with state_lock:
                    control_status = cmd.get("status", "Girando...")
                    last_detection["control_status"] = control_status

                if duration > 0:
                    time.sleep(duration)

                # 3) Estado final del movimiento (seguir avanzando)
                _set_speed_and_direction_hw(after_speed, after_angle)

            elif cmd_type == "UTURN":
                speed = cmd.get("speed", SPEED_NORMAL)
                angle = cmd.get("angle", -UTURN_ANGLE_DEG)
                duration = cmd.get("duration", UTURN_TIME_180)
                after_speed = cmd.get("after_speed", SPEED_NORMAL)
                after_angle = cmd.get("after_angle", 0)

                # 1) Frenar antes de la vuelta en U
                _set_speed_and_direction_hw(SPEED_STOP, 0)
                time.sleep(0.1)

                # 2) Giro en U
                _set_speed_and_direction_hw(speed, angle)
                with state_lock:
                    control_status = cmd.get("status", "Retorno en U")
                    last_detection["control_status"] = control_status

                if duration > 0:
                    time.sleep(duration)

                # 3) Estado final después de la maniobra (seguir avanzando)
                _set_speed_and_direction_hw(after_speed, after_angle)

            elif cmd_type == "EMERGENCY_STOP":
                _set_speed_and_direction_hw(SPEED_STOP, 0)
                with state_lock:
                    EMERGENCY_STOP = True
                    control_status = cmd.get("reason", "Detenido por emergencia")
                    last_detection["control_status"] = control_status
                    last_detection["emergency_stop"] = True

            self.cmd_queue.task_done()

        _set_speed_and_direction_hw(SPEED_STOP, 0)
        print("[CONTROL] Hilo de control finalizado.")

def enqueue_command(cmd_type, **kwargs):
    cmd = {"type": cmd_type}
    cmd.update(kwargs)
    try:
        control_queue.put_nowait(cmd)
    except queue.Full:
        print(f"[WARN] Cola de control llena, descartando {cmd_type}")

#   LÓGICA DE ACCIONES

def decide_and_enqueue_action(cls_id, conf):
    global last_command_time, last_class_executed

    now = time.time()
    with state_lock:
        auto_enabled = AUTO_DRIVE_ENABLED
        emergency = EMERGENCY_STOP

    if px is None:
        print(f"[ACCION-SIM] Clase {cls_id}, conf={conf:.2f}")
        return

    if emergency:
        return

    if not auto_enabled:
        with state_lock:
            last_detection["control_status"] = "Autoconducción desactivada"
        return

    if cls_id == last_class_executed and (now - last_command_time) < COMMAND_COOLDOWN:
        return

    last_command_time = now
    last_class_executed = cls_id

    if cls_id == CLASS_STOP:
        enqueue_command("STOP", reason="Detenido por STOP")
        with state_lock:
            last_detection["control_status"] = "Detenido por STOP"

    elif cls_id == CLASS_GO_STRAIGHT:
        enqueue_command(
            "SET_SPEED",
            speed=SPEED_NORMAL,
            angle=0,
            status="Avanzando recto",
        )

    elif cls_id == CLASS_TURN_RIGHT:
        # Detectar -> parar -> girar 90° -> seguir
        enqueue_command(
            "TURN",
            speed=TURN_SPEED,
            angle=+TURN_ANGLE_DEG,
            duration=TURN_TIME_90_D,
            after_speed=SPEED_NORMAL,
            after_angle=0,
            status="Giro ~90° a la derecha",
        )

    elif cls_id == CLASS_TURN_LEFT:
        # Detectar -> parar -> girar 90° -> seguir
        enqueue_command(
            "TURN",
            speed=TURN_SPEED,
            angle=-TURN_ANGLE_DEG,
            duration=TURN_TIME_90_I,
            after_speed=SPEED_NORMAL,
            after_angle=0,
            status="Giro ~90° a la izquierda",
        )

    elif cls_id == CLASS_YIELD:
        enqueue_command(
            "SET_SPEED",
            speed=SPEED_SLOW,
            angle=0,
            status="Cediendo el paso (simple)",
        )

    elif cls_id == CLASS_UTURN:
        # Detectar -> parar -> girar 180° -> seguir
        enqueue_command(
            "UTURN",
            speed=TURN_SPEED,
            angle=-UTURN_ANGLE_DEG,
            duration=UTURN_TIME_180,
            after_speed=SPEED_NORMAL,
            after_angle=0,
            status="Giro ~180° (vuelta en U)",
        )

    elif cls_id == CLASS_VMAX_10:
        enqueue_command(
            "SET_SPEED",
            speed=SPEED_SLOW,
            angle=0,
            status="Velocidad ~10 km/h",
        )

    elif cls_id == CLASS_VMAX_30:
        enqueue_command(
            "SET_SPEED",
            speed=SPEED_30,
            angle=0,
            status="Velocidad ~30 km/h",
        )
    else:
        with state_lock:
            last_detection["control_status"] = f"Clase {cls_id} sin acción"


#   TFLITE HELPERS


def init_tflite():
    global interpreter, input_details, output_details, INPUT_W, INPUT_H

    if Interpreter is None:
        print("[ERROR] No hay intérprete TFLite disponible.")
        return

    print("[INFO] Cargando modelo TFLite:", TFLITE_MODEL_PATH)
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, INPUT_H, INPUT_W, _ = input_details[0]["shape"]
    print(f"[OK] Modelo TFLite cargado. Input: {INPUT_W}x{INPUT_H}")

def detect_tflite(frame):
    """
    Corre TFLite sobre el frame y dibuja SOLO la mejor caja (no FONDO).
    Actualiza last_detection y encola acción.
    """
    global interpreter, input_details, output_details, control_status

    if interpreter is None:
        return frame

    h, w, _ = frame.shape

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    boxes, scores, classes = [], [], []

    for det in preds:
        x, y, bw, bh, obj_conf = det[:5]
        cls_scores = det[5:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id] * obj_conf)

        # --- UMBRAL POR CLASE ---
        if cls_id == CLASS_TURN_LEFT:
            min_conf = CONF_THRESH_LEFT
        else:
            min_conf = CONF_THRESH

        if score < min_conf:
            continue
      

        x *= w
        y *= h
        bw *= w
        bh *= h

        x1 = int(x - bw / 2)
        y1 = int(y - bh / 2)
        x2 = int(x + bw / 2)
        y2 = int(y + bh / 2)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls_id)

    if not boxes:
        return frame

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    # Elegir la mejor detección NO FONDO
    best_idx = None
    best_score = -1.0

    for i in range(len(scores)):
        cid = int(classes[i])
        if cid == BACKGROUND_CLASS_ID:
            continue

        score_i = scores[i]

        # Pequeño bonus a giro izquierda para que se active antes
        if cid == CLASS_TURN_LEFT:
            score_i += 0.02  # ajustable

        if score_i > best_score:
            best_score = score_i
            best_idx = i

    if best_idx is None:
        return frame

    x1, y1, x2, y2 = boxes[best_idx]
    best_cls = int(classes[best_idx])
    best_conf = float(scores[best_idx])
    best_label = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else f"cls{best_cls}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{best_label} {best_conf:.2f}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    with state_lock:
        last_detection["cls_id"] = best_cls
        last_detection["label"] = best_label
        last_detection["confidence"] = best_conf
        last_detection["timestamp"] = time.time()
        last_detection["control_status"] = control_status
        last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED

    decide_and_enqueue_action(best_cls, best_conf)

    return frame

#   CÁMARA: PICAMERA2 / CV2

picam2 = None
cap = None

try:
    from picamera2 import Picamera2

    try:
        picam2 = Picamera2()

        video_config = picam2.create_video_configuration(
            main={
                "size": (320, 240),
                "format": "BGR888",
            },
            buffer_count=2,
        )
        picam2.configure(video_config)
        picam2.set_controls({"FrameRate": 20})

        picam2.start()
        print("[INFO] Picamera2 modo video, buffer_count=2, FrameRate=20.")
    except Exception as e:
        print(f"[ERROR] Falló Picamera2: {e}")
        picam2 = None
except ImportError:
    print("[WARN] picamera2 no disponible. Usando OpenCV VideoCapture.")
    picam2 = None

if picam2 is None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara con OpenCV.")
    else:
        print("[INFO] OpenCV VideoCapture(0) inicializado.")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass


#   HILOS DE CÁMARA Y DETECCIÓN


class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        global last_raw_frame

        print("[CAMERA] Hilo de cámara iniciado.")
        while self.running:
            if picam2 is not None:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                if cap is None:
                    time.sleep(0.05)
                    continue
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                for _ in range(2):
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    frame = f2

            with frame_lock:
                last_raw_frame = frame

            time.sleep(0.005)

        print("[CAMERA] Hilo de cámara finalizado.")

class DetectionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.counter = 0

    def stop(self):
        self.running = False

    def run(self):
        global last_raw_frame, last_processed_frame, EMERGENCY_STOP

        print("[DETECT] Hilo de detección iniciado.")
        while self.running:
            with frame_lock:
                frame = None if last_raw_frame is None else last_raw_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            self.counter += 1

            dist = read_distance()
            with state_lock:
                last_detection["obstacle_distance_cm"] = dist

            if dist is not None and dist < OBSTACLE_STOP_DISTANCE and not EMERGENCY_STOP:
                with state_lock:
                    EMERGENCY_STOP = True
                    last_detection["emergency_stop"] = True
                    last_detection["control_status"] = f"Detenido por obstáculo a {dist:.1f} cm"
                enqueue_command("EMERGENCY_STOP", reason=f"Detenido por obstáculo a {dist:.1f} cm")

            if dist is not None and dist > CLEAR_DISTANCE and EMERGENCY_STOP:
                with state_lock:
                    EMERGENCY_STOP = False
                    last_detection["emergency_stop"] = False
                    last_detection["control_status"] = "Emergencia liberada (vía despejada)"

            if self.counter % DETECT_EVERY_N != 0:
                with frame_lock:
                    last_processed_frame = frame
                time.sleep(0.003)
                continue

            processed = detect_tflite(frame)

            with frame_lock:
                last_processed_frame = processed

        print("[DETECT] Hilo de detección finalizado.")

camera_thread = None
detection_thread = None
control_thread = None


#   GENERADOR DE FRAMES


def generate_frames():
    global last_raw_frame, last_processed_frame

    while True:
        loop_start = time.time()

        with frame_lock:
            frame = last_processed_frame if last_processed_frame is not None else last_raw_frame

        if frame is None:
            time.sleep(0.01)
            continue

        with state_lock:
            width = STREAM_WIDTH
            height = STREAM_HEIGHT
            quality = JPEG_QUALITY
            fps = TARGET_FPS

        if width and height:
            frame_resized = cv2.resize(frame, (width, height))
        else:
            frame_resized = frame

        ok, buffer = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        frame_period = 1.0 / fps if fps > 0 else 0
        elapsed = time.time() - loop_start
        if frame_period > 0 and elapsed < frame_period:
            time.sleep(frame_period - elapsed)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


#   RUTAS FLASK


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/snapshot")
def snapshot():
    global last_raw_frame, last_processed_frame

    with frame_lock:
        frame = last_raw_frame  # mínimo delay visual

    if frame is None:
        tmp = np.zeros((240, 320, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", tmp, [cv2.IMWRITE_JPEG_QUALITY, 40])
    else:
        with state_lock:
            width = STREAM_WIDTH
            height = STREAM_HEIGHT
            quality = JPEG_QUALITY

        if width and height:
            frame_resized = cv2.resize(frame, (width, height))
        else:
            frame_resized = frame

        ok, buffer = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])

    if not ok:
        return ("", 500)

    resp = Response(buffer.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.route("/last_detection")
def get_last_detection():
    with state_lock:
        return jsonify(last_detection)

@app.route("/control/start", methods=["POST"])
def control_start():
    global AUTO_DRIVE_ENABLED, control_status
    with state_lock:
        AUTO_DRIVE_ENABLED = True
        control_status = "Autoconducción activada"
        last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED
        last_detection["control_status"] = control_status
    print("[CONTROL] Autoconducción ON")
    return jsonify({"ok": True, "auto_drive_enabled": True})

@app.route("/control/stop", methods=["POST"])
def control_stop():
    global AUTO_DRIVE_ENABLED, control_status
    with state_lock:
        AUTO_DRIVE_ENABLED = False
        control_status = "Detenido manualmente"
        last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED
        last_detection["control_status"] = control_status
    enqueue_command("STOP", reason="Detenido manualmente")
    print("[CONTROL] Autoconducción OFF + STOP")
    return jsonify({"ok": True, "auto_drive_enabled": False})

@app.route("/config", methods=["GET", "POST"])
def config_stream():
    global TARGET_FPS, JPEG_QUALITY, STREAM_WIDTH, STREAM_HEIGHT

    if request.method == "GET":
        with state_lock:
            cfg = {
                "target_fps": TARGET_FPS,
                "jpeg_quality": JPEG_QUALITY,
                "width": STREAM_WIDTH,
                "height": STREAM_HEIGHT,
            }
        return jsonify(cfg)

    data = request.get_json(silent=True) or {}

    with state_lock:
        fps = data.get("target_fps")
        if isinstance(fps, (int, float)) and 1 <= fps <= 30:
            TARGET_FPS = float(fps)

        q = data.get("jpeg_quality")
        if isinstance(q, int) and 10 <= q <= 95:
            JPEG_QUALITY = int(q)

        w = data.get("width")
        h = data.get("height")
        if isinstance(w, int) and isinstance(h, int):
            if 160 <= w <= 640 and 120 <= h <= 480:
                STREAM_WIDTH = w
                STREAM_HEIGHT = h

        last_detection["target_fps"] = TARGET_FPS
        last_detection["jpeg_quality"] = JPEG_QUALITY
        last_detection["width"] = STREAM_WIDTH
        last_detection["height"] = STREAM_HEIGHT

        cfg = {
            "target_fps": TARGET_FPS,
            "jpeg_quality": JPEG_QUALITY,
            "width": STREAM_WIDTH,
            "height": STREAM_HEIGHT,
        }

    print(f"[CONFIG] Actualizado: {cfg}")
    return jsonify({"ok": True, "config": cfg})


#   MAIN


if __name__ == "__main__":
    try:
        init_tflite()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar TFLite: {e}")

    control_thread = ControlThread(control_queue)
    control_thread.start()

    camera_thread = CameraThread()
    camera_thread.start()

    detection_thread = DetectionThread()
    detection_thread.start()

    print("[INFO] Servidor Flask en http://0.0.0.0:8000")

    try:
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
    finally:
        print("[INFO] Cerrando recursos...")
        if camera_thread is not None:
            camera_thread.stop()
        if detection_thread is not None:
            detection_thread.stop()
        if control_thread is not None:
            control_thread.stop()

        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
        if px is not None:
            try:
                with hardware_lock:
                    px.stop()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
