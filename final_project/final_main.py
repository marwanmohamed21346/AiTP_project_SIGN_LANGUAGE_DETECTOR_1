# authors: marwan mohamed / mariam ibrahim

# لسماع صوت ترجمة الاشارة "s"
# لايقاف البرنامج "q"

import numpy as np
import cv2
import torch
import pyttsx3
import queue
import threading
from model import Net

# تهيئة مكتبة pyttsx3 وقائمة الانتظار
engine = pyttsx3.init()
q = queue.Queue()
stop_event = threading.Event()

# وظيفة لتحويل النص إلى صوت
def speak():
    while not stop_event.is_set():
        try:
            text = q.get(timeout=1)
            engine.say(text)
            engine.runAndWait()
            q.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in speak function: {e}")

# بدء خيط معالجة النصوص إلى صوت
speak_thread = threading.Thread(target=speak, daemon=True)
speak_thread.start()

cap = cv2.VideoCapture(0)

cap.set(3, 700)
cap.set(4, 480)

modelo = torch.load('model_trained.pt')
modelo.eval()

signs = {'0': 'A i t p', '1': 'Hello!', '2': 'hurting a lot', '3': 'excuse me', '4': 'A i t p', '5': 'i am fine', '6': 'A i t p', '7': 'What is your name?', '8': 'I am not fine',
         '10': 'that is a lie!', '11': 'i Love you', '12': 'A i t p', '13': 'A i t p', '14': 'hurting a bit', '15': 'A i t P', '16': 'let us go!', '17': 'i respect you',
         '18': 'A i t p', '19': 'A i t p', '20': 'bathroom', '21': 'see you later', '22': 'i am thirsty', '23': 'change it', '24': 'call me'}

last_text = ""

# معلمات تتبع اليد
tracker = cv2.TrackerCSRT_create()
initBB = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read from webcam")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            img = frame[y:y+h, x:x+w]
        else:
            initBB = None
            img = frame[20:250, 20:250]
    else:
        img = frame[20:250, 20:250]

    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res1 = np.reshape(res, (1, 1, 28, 28)) / 255.0
    res1 = torch.from_numpy(res1)
    res1 = res1.type(torch.FloatTensor)

    out = modelo(res1)
    probs, label = torch.topk(out, 25)
    probs = torch.nn.functional.softmax(probs, 1)

    pred = out.max(1, keepdim=True)[1]

    if float(probs[0, 0]) < 0.4:
        texto_mostrar = 'Sign not detected'
    else:
        texto_mostrar = signs.get(str(int(pred)), 'Unknown') + ': {:.2f}%'.format(float(probs[0, 0]) * 100)

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, texto_mostrar, (60, 285), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if initBB is not None:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    else:
        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

    cv2.imshow('Cam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        q.put(texto_mostrar.split(':')[0])  # استخراج النص قبل النسبة المئوية
    elif key == ord('t') and initBB is None:
        initBB = cv2.selectROI('Cam', frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

# تنظيف الخيط وقائمة الانتظار
stop_event.set()
speak_thread.join()

cap.release()
cv2.destroyAllWindows()
