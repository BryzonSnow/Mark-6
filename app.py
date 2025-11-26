import os
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

# ==========================
#   CONFIGURACIÓN GENERAL
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "best-fp16.tflite")

app = Flask(__name__)

# ==========================
#   HARDWARE: PiCar-X
# ==========================

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

# ==========================
#   TFLITE / DETECCIÓN
# ==========================

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
CONF_THRESH = 0.45
IOU_THRESH = 0.45

# Clases 0 a 8:
# 0=STOP
# 1=SIGA
# 2=GIRAR DERECHA
# 3=GIRAR IZQUIERDA
# 4=FONDO
# 5=CEDA EL PASO
# 6=RETONO (GIRO EN U)
# 7=VEL. MÁX 10 KM/H
# 8=VEL. MÁX 30 KM/H
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

# ==========================
#   CONFIG MOVIMIENTO + PROXIMIDAD
# ==========================

SPEED_STOP = 0
SPEED_SLOW = 10      # aprox. "10 km/h"
SPEED_30 = 20        # aprox. "30 km/h"
SPEED_NORMAL = 30    # velocidad "crucero"
SPEED_MAX = 40       # límite superior

TURN_ANGLE = 25      # grados giro normal
UTURN_ANGLE = 35     # grados giro fuerte
UTURN_TIME = 0.9     # segundos de giro para U

CLASS_STOP = 0
CLASS_GO_STRAIGHT = 1
CLASS_TURN_RIGHT = 2
CLASS_TURN_LEFT = 3
CLASS_BACKGROUND = 4
CLASS_YIELD = 5
CLASS_UTURN = 6
CLASS_VMAX_10 = 7
CLASS_VMAX_30 = 8

# Distancias (cm) para lógica de seguridad
YIELD_SAFE_DISTANCE = 25.0          # para CEDA EL PASO
OBSTACLE_SLOW_DISTANCE = 40.0       # bajar velocidad
OBSTACLE_STOP_DISTANCE = 25.0       # frenar
CLEAR_DISTANCE = 45.0               # salir de emergencia

current_speed = SPEED_STOP
control_status = "Inicializando..."

COMMAND_COOLDOWN = 1.0  # segundos
last_command_time = 0.0
last_class_executed = None

EMERGENCY_STOP = False

# 👉 NUEVO: bandera para activar/desactivar autoconducción
AUTO_DRIVE_ENABLED = False

# Estado expuesto a la interfaz
last_detection = {
    "cls_id": None,
    "label": None,
    "confidence": 0.0,
    "timestamp": None,
    "obstacle_distance_cm": None,
    "emergency_stop": False,
    "control_status": control_status,
    "auto_drive_enabled": AUTO_DRIVE_ENABLED,
}

# ==========================
#   PROXIMIDAD: get_distance() DEL PiCar-X
# ==========================

def read_distance():
    """
    Lee la distancia desde el sensor ultrasónico integrado del PiCar-X.
    Usa px.get_distance(), que maneja internamente los pines de la placa.
    """
    if px is None:
        return None
    try:
        dist = px.get_distance()
        if dist is None or dist <= 0 or dist > 500:
            return None
        return float(dist)
    except Exception as e:
        print(f"[WARN] Error leyendo distancia: {e}")
        return None


# ==========================
#   CONTROL DE MOVIMIENTO
# ==========================

def set_speed_and_direction(speed, angle=0):
    """Control básico del PiCar-X (o simulación si no hay hardware)."""
    global px, current_speed

    speed = max(0, min(speed, SPEED_MAX))
    current_speed = speed

    if px is None:
        # Modo simulación: solo mostrar
        print(f"[SIM] speed={speed}, angle={angle}")
        return

    try:
        if speed == 0:
            px.stop()
        else:
            px.forward(speed)

        px.set_dir_servo_angle(angle)
    except Exception as e:
        print(f"[WARN] Error controlando PiCar-X: {e}")


def handle_detection_action(cls_id, conf):
    """
    Decide qué hacer con el robot según la mejor clase detectada.
    Respeta:
      - EMERGENCY_STOP (proximidad)
      - AUTO_DRIVE_ENABLED (botón interfaz)
    """
    global last_command_time, last_class_executed, control_status
    global EMERGENCY_STOP, AUTO_DRIVE_ENABLED, last_detection

    now = time.time()

    # Si no hay hardware, solo log
    if px is None:
        print(f"[ACCION-SIM] Clase {cls_id}, conf={conf:.2f}")
        return

    # 🚨 Si estamos en modo emergencia, NO hacemos nada más
    if EMERGENCY_STOP:
        set_speed_and_direction(SPEED_STOP, 0)
        control_status = "Detenido (emergencia proximidad)"
        print("[ACCION] EMERGENCY_STOP activo -> stop()")
        last_detection["control_status"] = control_status
        return

    # 👉 Si autoconducción está desactivada, no mover el robot
    if not AUTO_DRIVE_ENABLED:
        control_status = "Autoconducción desactivada (sin movimiento)"
        last_detection["control_status"] = control_status
        # Podemos seguir detectando, pero sin acciones
        return

    # Evitar comandos repetidos muy seguidos de la misma clase
    if cls_id == last_class_executed and (now - last_command_time) < COMMAND_COOLDOWN:
        return

    last_command_time = now
    last_class_executed = cls_id

    if cls_id == CLASS_STOP:
        # Detener completamente
        set_speed_and_direction(SPEED_STOP, 0)
        control_status = "Detenido por STOP"
        print("[ACCION] STOP -> px.stop()")

    elif cls_id == CLASS_GO_STRAIGHT:
        # Avanzar recto
        set_speed_and_direction(SPEED_NORMAL, 0)
        control_status = "Avanzando recto"
        print("[ACCION] SIGA -> forward(NORMAL)")

    elif cls_id == CLASS_TURN_RIGHT:
        # Giro suave derecha + seguir recto
        control_status = "Girando a la derecha"
        print("[ACCION] GIRO DERECHA -> giro + luego recto")
        set_speed_and_direction(SPEED_NORMAL, TURN_ANGLE)
        time.sleep(0.5)
        set_speed_and_direction(SPEED_NORMAL, 0)

    elif cls_id == CLASS_TURN_LEFT:
        # Giro suave izquierda + seguir recto
        control_status = "Girando a la izquierda"
        print("[ACCION] GIRO IZQUIERDA -> giro - luego recto")
        set_speed_and_direction(SPEED_NORMAL, -TURN_ANGLE)
        time.sleep(0.5)
        set_speed_and_direction(SPEED_NORMAL, 0)

    elif cls_id == CLASS_YIELD:
        # Ceda el paso: usar ultrasonido si existe
        dist = read_distance()
        if dist is not None:
            print(f"[ULTRA] Distancia (ceda): {dist:.1f} cm")
            if dist < YIELD_SAFE_DISTANCE:
                set_speed_and_direction(SPEED_STOP, 0)
                control_status = f"Cediendo el paso - obstáculo a {dist:.1f} cm"
                print("[ACCION] CEDA: OBJETO CERCA -> stop()")
            else:
                set_speed_and_direction(SPEED_SLOW, 0)
                time.sleep(0.5)
                set_speed_and_direction(SPEED_NORMAL, 0)
                control_status = "Cediendo el paso - vía libre"
        else:
            # Sin lectura, al menos reducimos la velocidad
            set_speed_and_direction(SPEED_SLOW, 0)
            control_status = "Cediendo el paso - sin lectura ultrasónico"

    elif cls_id == CLASS_UTURN:
        # Retorno en U
        control_status = "Retorno en U"
        print("[ACCION] RETORNO -> giro fuerte para U")
        set_speed_and_direction(SPEED_NORMAL, -UTURN_ANGLE)
        time.sleep(UTURN_TIME)
        set_speed_and_direction(SPEED_NORMAL, 0)

    elif cls_id == CLASS_VMAX_10:
        # Ajustar velocidad a ~10 km/h
        control_status = "Velocidad ajustada a ~10 km/h"
        print("[ACCION] VEL MÁX 10 -> SPEED_SLOW")
        set_speed_and_direction(SPEED_SLOW, 0)

    elif cls_id == CLASS_VMAX_30:
        # Ajustar velocidad a ~30 km/h
        control_status = "Velocidad ajustada a ~30 km/h"
        print("[ACCION] VEL MÁX 30 -> SPEED_30")
        set_speed_and_direction(SPEED_30, 0)

    else:
        control_status = f"Clase {cls_id} sin acción específica"
        print(f"[ACCION] Clase {cls_id} sin acción definida. conf={conf:.2f}")

    # Actualizar estado en last_detection
    last_detection["control_status"] = control_status


# ==========================
#   TFLITE HELPERS
# ==========================

def init_tflite():
    """Carga el modelo TFLite y prepara detalles de entrada/salida."""
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


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression básica."""
    idxs = np.argsort(scores)[::-1]
    selected = []

    while len(idxs) > 0:
        current = idxs[0]
        selected.append(current)

        rest = idxs[1:]
        if len(rest) == 0:
            break

        ious = []
        for i in rest:
            x1 = max(boxes[current][0], boxes[i][0])
            y1 = max(boxes[current][1], boxes[i][1])
            x2 = min(boxes[current][2], boxes[i][2])
            y2 = min(boxes[current][3], boxes[i][3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[current][2] - boxes[current][0]) * (boxes[current][3] - boxes[current][1])
            area2 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            ious.append(iou)

        idxs = rest[np.array(ious) < iou_threshold]

    return selected


def detect_tflite(frame):
    """
    Corre el modelo TFLite sobre un frame BGR.
    Se queda SOLO con la mejor detección (mayor confianza) distinta de FONDO,
    dibuja una sola caja y dispara la acción correspondiente en el PiCar-X.
    """
    global interpreter, input_details, output_details, last_detection, AUTO_DRIVE_ENABLED

    if interpreter is None:
        return frame

    h, w, _ = frame.shape

    # Preprocesamiento
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    # Asumimos salida tipo YOLO: [N, 5 + num_classes]
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    boxes = []
    scores = []
    classes = []

    num_classes = preds.shape[1] - 5

    for det in preds:
        x, y, bw, bh, obj_conf = det[:5]
        cls_scores = det[5:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id] * obj_conf)

        if score < CONF_THRESH:
            continue

        # Escalar al tamaño original
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

    idxs = nms(boxes, scores, IOU_THRESH)

    # Buscar SOLO la mejor detección que NO sea FONDO
    best_idx = None
    best_score = -1.0

    for i in idxs:
        cls_id = int(classes[i])

        if cls_id == BACKGROUND_CLASS_ID:
            continue

        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    if best_idx is None:
        # Solo había FONDO o detecciones descartadas
        return frame

    x1, y1, x2, y2 = boxes[best_idx]
    best_cls = int(classes[best_idx])
    best_conf = float(scores[best_idx])
    best_label = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else f"cls{best_cls}"

    # Dibujar SOLO una caja
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

    # Actualizar última detección (para la interfaz web)
    last_detection["cls_id"] = best_cls
    last_detection["label"] = best_label
    last_detection["confidence"] = best_conf
    last_detection["timestamp"] = time.time()
    last_detection["control_status"] = control_status
    last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED

    # Ejecutar acción de movimiento
    handle_detection_action(best_cls, best_conf)

    return frame

# ==========================
#   CÁMARA: PICAMERA2 / CV2
# ==========================

picam2 = None
cap = None

try:
    from picamera2 import Picamera2

    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        print("[INFO] Picamera2 inicializada.")
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


def generate_frames():
    """
    Generador de frames MJPEG para la ruta /video_feed.
    Incluye lógica de ultrasonido para EMERGENCY_STOP y reducción de velocidad.
    """
    global EMERGENCY_STOP, control_status, AUTO_DRIVE_ENABLED

    while True:
        # Leer cámara
        if picam2 is not None:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            if cap is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = cap.read()
                if not ret:
                    continue

        # --------- SEGURIDAD POR ULTRASONIDO ---------
        dist = read_distance()
        if dist is not None:
            # Actualizar info para la interfaz SIEMPRE
            last_detection["obstacle_distance_cm"] = dist

            if dist < OBSTACLE_STOP_DISTANCE:
                # Activar emergencia y detener
                if not EMERGENCY_STOP:
                    print(f"[SEGURIDAD] Obstáculo a {dist:.1f} cm -> EMERGENCY STOP")
                EMERGENCY_STOP = True
                set_speed_and_direction(SPEED_STOP, 0)
                control_status = f"Detenido por obstáculo a {dist:.1f} cm"

            elif dist < OBSTACLE_SLOW_DISTANCE and not EMERGENCY_STOP and current_speed > SPEED_SLOW:
                set_speed_and_direction(SPEED_SLOW, 0)
                control_status = f"Reduciendo velocidad por obstáculo a {dist:.1f} cm"
                print(f"[SEGURIDAD] Obstáculo a {dist:.1f} cm -> SLOW")

            elif dist > CLEAR_DISTANCE and EMERGENCY_STOP:
                EMERGENCY_STOP = False
                control_status = "Emergencia liberada (vía despejada)"
                print(f"[SEGURIDAD] Distancia {dist:.1f} cm -> EMERGENCIA OFF")
        else:
            # Sin lectura, dejamos distancia en None
            last_detection["obstacle_distance_cm"] = None

        # Guardar flags de emergencia/estado para la interfaz
        last_detection["emergency_stop"] = EMERGENCY_STOP
        last_detection["control_status"] = control_status
        last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED

        # --------- DETECCIÓN DE SEÑALES ---------
        frame = detect_tflite(frame)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

# ==========================
#          RUTAS FLASK
# ==========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/last_detection")
def get_last_detection():
    """Devuelve la última detección + estado de proximidad en JSON."""
    return jsonify(last_detection)


# 👉 NUEVAS RUTAS: control desde la interfaz

@app.route("/control/start", methods=["POST"])
def control_start():
    """
    Activa autoconducción.
    El robot empezará a reaccionar a las señales (si no hay emergencia).
    """
    global AUTO_DRIVE_ENABLED, control_status
    AUTO_DRIVE_ENABLED = True
    control_status = "Autoconducción activada desde interfaz"
    last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED
    last_detection["control_status"] = control_status
    print("[CONTROL] Autoconducción ON (interfaz)")
    return jsonify({"ok": True, "auto_drive_enabled": AUTO_DRIVE_ENABLED})


@app.route("/control/stop", methods=["POST"])
def control_stop():
    """
    Detiene el robot manualmente y desactiva autoconducción.
    """
    global AUTO_DRIVE_ENABLED, control_status
    AUTO_DRIVE_ENABLED = False
    set_speed_and_direction(SPEED_STOP, 0)
    control_status = "Detenido manualmente desde interfaz"
    last_detection["auto_drive_enabled"] = AUTO_DRIVE_ENABLED
    last_detection["control_status"] = control_status
    print("[CONTROL] Autoconducción OFF + stop() (interfaz)")
    return jsonify({"ok": True, "auto_drive_enabled": AUTO_DRIVE_ENABLED})


if __name__ == "__main__":
    try:
        init_tflite()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar TFLite: {e}")

    print("[INFO] Servidor Flask iniciado en http://0.0.0.0:8000")

    try:
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
    finally:
        print("[INFO] Cerrando recursos...")
        if picam2 is not None:
            picam2.stop()
        if px is not None:
            px.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
