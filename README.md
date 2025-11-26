# 🚗 PiCar-X: Sistema de Autoconducción con Detección de Señales de Tráfico

Sistema de autoconducción autónoma para el robot PiCar-X que utiliza visión por computadora y aprendizaje profundo para detectar y responder a señales de tráfico en tiempo real.

## 📋 Descripción

Este proyecto implementa un sistema completo de autoconducción que combina:
- **Detección de señales de tráfico** mediante un modelo TensorFlow Lite (TFLite)
- **Control autónomo del robot** basado en las señales detectadas
- **Sistema de seguridad** con sensor ultrasónico para detección de obstáculos
- **Interfaz web en tiempo real** para monitoreo y control

El sistema procesa el video de la cámara del robot, detecta señales de tráfico, y ejecuta acciones de movimiento correspondientes (avanzar, detenerse, girar, ajustar velocidad) mientras monitorea la proximidad de obstáculos para garantizar la seguridad.

## ✨ Características Principales

### 🎯 Detección de Señales
- **9 clases de señales detectadas:**
  - `STOP` - Detención completa
  - `SIGA` - Avanzar recto
  - `GIRAR DERECHA` - Giro a la derecha
  - `GIRAR IZQUIERDA` - Giro a la izquierda
  - `CEDA EL PASO` - Reducción de velocidad con verificación de proximidad
  - `RETONO (GIRO EN U)` - Giro en U
  - `VEL MÁX 10 KM/H` - Limitación de velocidad a 10 km/h
  - `VEL MÁX 30 KM/H` - Limitación de velocidad a 30 km/h
  - `FONDO` - Ignorado (no genera acción)

### 🤖 Control Autónomo
- **Sistema de decisión inteligente:** Selecciona la señal con mayor confianza (excluyendo fondo)
- **Control de velocidad adaptativo:** Ajusta la velocidad según la señal detectada
- **Gestión de giros:** Manejo automático de giros suaves y retornos en U
- **Cooldown de comandos:** Evita comandos repetidos muy seguidos

### 🛡️ Sistema de Seguridad
- **Detección de obstáculos:** Sensor ultrasónico integrado del PiCar-X
- **Frenado de emergencia:** Detención automática cuando detecta obstáculos cercanos (< 25 cm)
- **Reducción preventiva de velocidad:** Disminuye velocidad cuando detecta obstáculos a distancia media (40 cm)
- **Liberación automática:** Reanuda el movimiento cuando el camino está despejado (> 45 cm)

### 🌐 Interfaz Web
- **Streaming de video en vivo:** Visualización en tiempo real de la cámara con detecciones superpuestas
- **Panel de información:** Muestra la señal actual detectada, confianza y timestamp
- **Historial de detecciones:** Registro de las últimas 6 señales detectadas
- **Control manual:** Botones para iniciar/detener la autoconducción
- **Indicadores de estado:** Banner de emergencia y estado del control en tiempo real
- **Diseño responsive:** Interfaz moderna y adaptable a diferentes tamaños de pantalla

## 🛠️ Tecnologías Utilizadas

- **Backend:**
  - Python 3.x
  - Flask - Framework web
  - TensorFlow Lite / tflite_runtime - Inferencia del modelo de detección
  - OpenCV (cv2) - Procesamiento de imágenes
  - NumPy - Operaciones numéricas
  - picarx - Control del hardware PiCar-X
  - picamera2 - Captura de video (Raspberry Pi)

- **Frontend:**
  - HTML5 / CSS3
  - JavaScript (Vanilla)
  - Bootstrap 5.3.2 - Framework CSS

- **Hardware:**
  - PiCar-X (SunFounder)
  - Cámara Raspberry Pi (Picamera2) o cámara USB (OpenCV)
  - Sensor ultrasónico integrado

## 📦 Requisitos

### Hardware
- Raspberry Pi (recomendado) o sistema compatible
- PiCar-X de SunFounder
- Cámara Raspberry Pi o cámara USB compatible

### Software
- Python 3.7 o superior
- Sistema operativo Linux (Raspberry Pi OS recomendado)

### Dependencias Python
```bash
flask
opencv-python
numpy
tflite-runtime  # Para Raspberry Pi
# O alternativamente:
tensorflow      # Para sistemas sin tflite-runtime
picarx          # Librería de control PiCar-X
picamera2       # Para cámara Raspberry Pi (opcional)
```

## 🚀 Instalación

1. **Clonar o descargar el repositorio:**
```bash
cd /ruta/al/proyecto
```

2. **Instalar dependencias:**
```bash
pip install flask opencv-python numpy
```

Para Raspberry Pi:
```bash
pip install tflite-runtime picarx picamera2
```

Para otros sistemas:
```bash
pip install tensorflow
```

3. **Colocar el modelo TFLite:**
   - Asegúrate de tener el archivo `best-fp16.tflite` en el directorio raíz del proyecto
   - El modelo debe estar entrenado para detectar las 9 clases mencionadas

4. **Verificar hardware:**
   - Conecta el PiCar-X según las instrucciones del fabricante
   - Verifica que la cámara esté funcionando correctamente

## 📖 Uso

### Iniciar el servidor

```bash
python app.py
```

El servidor se iniciará en `http://0.0.0.0:8000`

### Acceder a la interfaz web

Abre tu navegador y visita:
```
http://localhost:8000
```
O desde otro dispositivo en la misma red:
```
http://[IP_DE_LA_RASPBERRY_PI]:8000
```

### Control del robot

1. **Iniciar autoconducción:**
   - Haz clic en el botón "Iniciar autoconducción"
   - El robot comenzará a reaccionar a las señales detectadas

2. **Detener manualmente:**
   - Haz clic en el botón "STOP inmediato"
   - El robot se detendrá y desactivará la autoconducción

3. **Monitoreo:**
   - Observa el streaming de video en tiempo real
   - Revisa la señal actual detectada y su confianza
   - Consulta el historial de detecciones recientes
   - Monitorea el estado de emergencia y proximidad de obstáculos

## 🏗️ Estructura del Proyecto

```
TESIS/
├── app.py                 # Aplicación Flask principal
├── best-fp16.tflite      # Modelo TensorFlow Lite (debe estar presente)
├── templates/
│   └── index.html        # Interfaz web
└── README.md             # Este archivo
```

## ⚙️ Configuración

### Parámetros de Detección
```python
CONF_THRESH = 0.45        # Umbral de confianza mínimo
IOU_THRESH = 0.45         # Umbral para Non-Maximum Suppression
INPUT_W, INPUT_H = 320    # Tamaño de entrada del modelo
```

### Parámetros de Velocidad
```python
SPEED_STOP = 0            # Detención
SPEED_SLOW = 10           # ~10 km/h
SPEED_30 = 20             # ~30 km/h
SPEED_NORMAL = 30         # Velocidad normal
SPEED_MAX = 40            # Límite máximo
```

### Parámetros de Seguridad
```python
OBSTACLE_STOP_DISTANCE = 25.0    # Distancia para frenado de emergencia (cm)
OBSTACLE_SLOW_DISTANCE = 40.0    # Distancia para reducir velocidad (cm)
CLEAR_DISTANCE = 45.0            # Distancia para liberar emergencia (cm)
YIELD_SAFE_DISTANCE = 25.0       # Distancia segura para ceda el paso (cm)
```

### Parámetros de Giro
```python
TURN_ANGLE = 25          # Ángulo de giro normal (grados)
UTURN_ANGLE = 35         # Ángulo de giro para retorno en U (grados)
UTURN_TIME = 0.9         # Tiempo de giro para retorno (segundos)
```

## 🔧 Modo Simulación

El sistema puede funcionar en modo simulación si no se detecta el hardware PiCar-X:
- Las acciones se registran en consola pero no se ejecutan físicamente
- La detección de señales sigue funcionando normalmente
- Útil para desarrollo y pruebas sin hardware

## 📡 API Endpoints

- `GET /` - Interfaz web principal
- `GET /video_feed` - Stream de video MJPEG en tiempo real
- `GET /last_detection` - Obtiene la última detección en formato JSON
- `POST /control/start` - Activa la autoconducción
- `POST /control/stop` - Detiene el robot y desactiva autoconducción

## 🔍 Funcionamiento Técnico

1. **Captura de video:** El sistema captura frames de la cámara (Picamera2 o OpenCV)
2. **Preprocesamiento:** Cada frame se redimensiona a 320x320 y normaliza
3. **Inferencia:** El modelo TFLite procesa el frame y genera detecciones
4. **Post-procesamiento:** Se aplica Non-Maximum Suppression (NMS) para filtrar detecciones
5. **Selección:** Se elige la detección con mayor confianza (excluyendo fondo)
6. **Acción:** Se ejecuta la acción correspondiente en el robot
7. **Seguridad:** Se verifica la distancia de obstáculos en cada frame
8. **Actualización:** La interfaz web se actualiza con la información más reciente

## ⚠️ Notas de Seguridad

- **Siempre supervisa el robot** durante la operación
- El sistema tiene protecciones de seguridad, pero no reemplaza la supervisión humana
- Asegúrate de tener espacio suficiente para las maniobras del robot
- Verifica que el sensor ultrasónico esté funcionando correctamente antes de usar la autoconducción

## 🐛 Solución de Problemas

### El modelo no se carga
- Verifica que `best-fp16.tflite` esté en el directorio raíz
- Asegúrate de tener instalado `tflite-runtime` o `tensorflow`

### La cámara no funciona
- Verifica los permisos de la cámara
- Prueba con `cv2.VideoCapture(0)` manualmente
- En Raspberry Pi, asegúrate de que `picamera2` esté instalado

### El PiCar-X no responde
- Verifica las conexiones del hardware
- Asegúrate de que la librería `picarx` esté instalada
- Revisa los logs en consola para mensajes de error

### La detección no funciona
- Verifica que el modelo esté entrenado para las clases correctas
- Ajusta `CONF_THRESH` si las detecciones son demasiado estrictas o permisivas
- Asegúrate de que la iluminación sea adecuada

## 📝 Licencia

Este proyecto es parte de una tesis de investigación. Consulta con el autor para más información sobre el uso y distribución.

## 👤 Autor

Proyecto desarrollado como parte de una tesis de investigación.

## 🙏 Agradecimientos

- SunFounder por el hardware PiCar-X
- TensorFlow por el framework de aprendizaje profundo
- Comunidad de código abierto por las librerías utilizadas

---

**⚠️ Importante:** Este sistema está diseñado para uso educativo y de investigación. Úsalo con precaución y siempre bajo supervisión.

