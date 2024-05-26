from imutils.video import VideoStream
from flask import Response, request, Flask, render_template, jsonify, send_from_directory
import threading
import argparse
import time
import cv2
import numpy as np
import torch
from model import Net

# تحميل النموذج المدرب
model = torch.load('model_trained.pt')
model.eval()

# قاموس لتخزين الإشارات ومعانيها
signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'}

# متغيرات عالمية
outputFrame = None
lock = threading.Lock()
trigger_flag = False
full_sentence = ''
text_suggestion = ''

# إنشاء تطبيق Flask
app = Flask(__name__)

# تعريف مسار المجلد الرئيسي للملفات الثابتة
app.static_folder = 'static'

# تعريف مسار مجلد القوالب
app.template_folder = 'templates'

# تعريف مسارات المجلدات الفرعية
app.config['STYLE_FOLDER'] = 'static/style'
app.config['CSS_FOLDER'] = 'static/style/css'
app.config['IMG_FOLDER'] = 'static/style/img'
app.config['JS_FOLDER'] = 'static/style/js'
app.config['SASS_FOLDER'] = 'static/style/sass'

# بدء بث الفيديو من الكاميرا
vc = VideoStream(src=0).start()
time.sleep(2.0)

# دالة لاكتشاف الإشارات
def detect_gesture(frameCount):
    global vc, outputFrame, lock, trigger_flag, full_sentence, text_suggestion

    while True:
        frame = vc.read()

        # تعيين أبعاد الإطار
        width = 700
        height = 480
        frame = cv2.resize(frame, (width, height))

        # اقتطاع جزء من الإطار للاستخدام في التعرف
        img = frame[20:250, 20:250]

        # إعادة تحجيم الصورة وتحويلها إلى الرمادي
        res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # تحويل الصورة إلى مصفوفة مناسبة للنموذج
        res1 = np.reshape(res, (1, 1, 28, 28)) / 255
        res1 = torch.from_numpy(res1)
        res1 = res1.type(torch.FloatTensor)

        # تمرير الصورة عبر النموذج للحصول على التوقعات
        out = model(res1)
        probs, label = torch.topk(out, 25)
        probs = torch.nn.functional.softmax(probs, 1)

        # الحصول على التوقع النهائي
        pred = out.max(1, keepdim=True)[1]

        if float(probs[0, 0]) < 0.4:
            detected = 'Nothing detected'
        else:
            detected = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0, 0])) + '%'

        if trigger_flag:
            full_sentence += signs[str(int(pred))].lower()
            trigger_flag = False

        if text_suggestion != '':
            if text_suggestion == ' ':
                full_sentence += ' '
                text_suggestion = ''
            else:
                full_sentence_list = full_sentence.strip().split()
                if len(full_sentence_list) != 0:
                    full_sentence_list.pop()
                full_sentence_list.append(text_suggestion)
                full_sentence = ' '.join(full_sentence_list)
                full_sentence += ' '
                text_suggestion = ''

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, detected, (60, 285), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

        with lock:
            outputFrame = frame.copy()

# دالة لتوليد الفيديو المستمر
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

# المسار الأساسي لعرض الصفحة الرئيسية
@app.route("/")
def index():
    return render_template("index.html")

# مسار لعرض صفحة "تعرف على المزيد"
@app.route("/index_learn_more.html")
def learn_more():
    return render_template("index_learn_more.html")

# مسار لعرض صفحة "بدء تشغيل الكود"
@app.route("/start_code_run")
def start_code_run():
    return render_template("index_start_code_run.html")

# مسار لاستقبال الأحرف المقترحة
@app.route('/char')
def char():
    global text_suggestion
    option = request.args.get('character')
    text_suggestion = option
    print(text_suggestion)
    return Response("done")

# مسار لتفعيل الإضافة إلى الجملة
@app.route('/trigger')
def trigger():
    global trigger_flag
    trigger_flag = True
    return Response('done')

# مسار لتوليد بث الفيديو
@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# مسار لإرسال الجملة الكاملة
@app.route('/sentence')
def sentence():
    global full_sentence
    return jsonify(full_sentence)

# مسار لتحميل ملفات CSS
@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory(app.config['CSS_FOLDER'], filename)

# مسار لتحميل الصور
@app.route("/img/<path:filename>")
def img(filename):
    return send_from_directory(app.config['IMG_FOLDER'], filename)

# مسار لتحميل ملفات JavaScript
@app.route("/js/<path:filename>")
def js(filename):
    return send_from_directory(app.config['JS_FOLDER'], filename)

# مسار لتحميل ملفات SASS
@app.route("/sass/<path:filename>")
def sass(filename):
    return send_from_directory(app.config['SASS_FOLDER'], filename)

# بدء التطبيق
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=False, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=False, help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32, help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_gesture, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host='192.168.1.7', port=8080, debug=True, threaded=True, use_reloader=False)

vc.stop()