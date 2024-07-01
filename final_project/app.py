from imutils.video import VideoStream
from flask import Response, request, render_template, Flask
import threading
import argparse
import time
import cv2
import numpy as np
import torch
from model import Net

# Load the model
model = torch.load('model_trained.pt')
model.eval()

# Sign dictionary
signs = {'0': 'A i t p', '1': 'Hello!', '2': 'hurting a lot', '3': 'excuse me', '4': 'A i t p', 
         '5': 'i am fine', '6': 'A i t p', '7': 'What is your name?', '8': 'I am not fine',
         '10': 'that is a lie!', '11': 'i Love you', '12': 'A i t p', '13': 'A i t p', 
         '14': 'hurting a bit', '15': 'A i t P', '16': 'let us go!', '17': 'i respect you',
         '18': 'A i t p', '19': 'A i t p', '20': 'bathroom', '21': 'see you later', 
         '22': 'i am thirsty', '23': 'change it', '24': 'call me'} 

outputFrame = None
lock = threading.Lock()
trigger_flag = False
full_sentence = ''

app = Flask(__name__)

vc = VideoStream(src=0).start()
time.sleep(2.0)
            
def detect_gesture(frameCount):
    global vc, outputFrame, lock, trigger_flag, full_sentence, text_suggestion

    while True:
        try:
            frame = vc.read()
            width, height = 700, 480
            frame = cv2.resize(frame, (width, height))
            img = frame[20:250, 20:250]

            res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            res1 = np.reshape(res, (1, 1, 28, 28)) / 255.0
            res1 = torch.from_numpy(res1).type(torch.FloatTensor)

            out = model(res1)
            probs, label = torch.topk(out, 25)
            probs = torch.nn.functional.softmax(probs, 1)

            pred = out.max(1, keepdim=True)[1]

            if float(probs[0, 0]) < 0.4:
                detected = 'Nothing detected'
            else:
                detected = signs[str(int(pred))] + ': {:.2f}%'.format(float(probs[0, 0]) * 100)

            if trigger_flag:
                full_sentence += signs[str(int(pred))].lower()
                trigger_flag = False

            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, detected, (60, 285), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

            with lock:
                outputFrame = frame.copy()

        except Exception as e:
            print(f"Error in detect_gesture: {e}")

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/char') 
def char():
    global text_suggestion
    option = request.args.get('character')
    if option == 'space':
        text_suggestion = " "
    print(text_suggestion)
    return Response("done")

@app.route('/trigger')
def trigger():
    global trigger_flag
    trigger_flag = True
    return Response('done')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/sentence')
def sentence():
    global full_sentence
    return jsonify(full_sentence)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="IP address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="Port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32, help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_gesture, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

    vc.stop()
