# يلتقط صورة من الكاميرا الافتراضية (أو أي كاميرا أخرى حسب رقم الكاميرا المحدد) كل 0.5 ثانية.
# يحفظ الصورة في مجلد "captured_images" مع تسمية الصور بترقيم تسلسلي.
# بعد كل 1000 صورة، ينتظر لمدة 3 دقائق قبل أن يبدأ في الدورة التالية.
# يستمر حتى يتم التقاط 7410 صورة في المجموع.
# يتم اغلاق الكاميرا والبرنامج في اي وقت بامان عند الضغط على "q"

import cv2
import time
import os

# التأكد من وجود المجلد لحفظ الصور
output_folder = "captured_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# إعداد الكاميرا
cap = cv2.VideoCapture(0)  # استخدم 0 للكاميرا الافتراضية، أو رقم آخر لكاميرا مختلفة

# طباعة الأرقام من 1 إلى 10 قبل البدء
for i in range(1, 11):
    print(i)
    time.sleep(1)

images_per_cycle = 1000
total_images = 7410
cycles = total_images // images_per_cycle

try:
    for cycle in range(cycles + 1):
        for i in range(images_per_cycle):
            if cycle * images_per_cycle + i >= total_images:
                break
            # التقاط الصورة
            ret, frame = cap.read()
            if not ret:
                print("فشل في التقاط الصورة")
                break
            # عرض الصورة على الشاشة
            cv2.imshow('Camera Feed', frame)
            # حفظ الصورة
            image_name = f"{output_folder}/image_{cycle * images_per_cycle + i + 1}.png"
            cv2.imwrite(image_name, frame)
            print(f"تم حفظ الصورة: {image_name}")
            # الانتظار لمدة 0.5 ثانية
            if cv2.waitKey(500) & 0xFF == ord('q'):
                raise KeyboardInterrupt
        # الانتظار لمدة 3 دقائق بعد كل دورة
        if cycle < cycles:
            print("الانتظار لمدة 3 دقائق...")
            for _ in range(3 * 60):  # التكرار لمدة 3 دقائق
                time.sleep(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
except KeyboardInterrupt:
    print("تم إيقاف البرنامج بواسطة المستخدم.")
finally:
    # إغلاق الكاميرا والنوافذ
    cap.release()
    cv2.destroyAllWindows()
