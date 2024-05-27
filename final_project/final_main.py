# authors: marwan mohamed / mariam ibrahim

# importing packages
try:
    import numpy as np
    import cv2
    import torch
    import pyttsx3
    import pygame
    import os
    import asyncio
    from model import Net
    print("packages imprted..........")
except Exception as e:
    print("error loading packages!!!",e)

# initilize pygame
pygame.mixer.init()

# initilize pyttsx3
engine = pyttsx3.init(driverName='sapi5')
engine.setProperty('rate', 150)



# import and intilize trainned model
try:
    model = torch.load('model_trained.pt')
    model.eval()
    print("model imported sucessfully.......")
except:
    print("error loading model",e)



# signs lapels
signs = { '1': 'stop',
          '2': 'it hurts me so much',
          '3': 'HELLO',
          '5': "i'm fine",
          '7': 'please',
          '8':'i am sick',
          '11': 'i love you', 
          '14': 'it hurts me a little',
          '21': 'see you later',
          '22': 'thirsty',
          '23': 'pain',
          '24': 'call me please'}

last_text = ""

# convert signs to speach
async def speak(text):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.save_to_file, text, "temp.mp3")
    await loop.run_in_executor(None, engine.runAndWait)
    pygame.mixer.music.load("temp.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

# main process function
async def main():
    global last_text
    
    # setting cam
    cam = cv2.VideoCapture(0)
    cam.set(3, 700) # setting wedth
    cam.set(4, 480) #setting heigh

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read from webcam")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Region of interest for capturing the hand
        img = frame[20:250, 20:250]

        res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        res1 = np.reshape(res, (1, 1, 28, 28)) / 255.0
        res1 = torch.from_numpy(res1)
        res1 = res1.type(torch.FloatTensor)

        out = model(res1)
        probs, label = torch.topk(out, 25)
        probs = torch.nn.functional.softmax(probs, 1)

        pred = out.max(1, keepdim=True)[1]

        if float(probs[0, 0]) < 0.4:
            text_mostrar = 'Sign not detected'
        else:
            text_mostrar = signs.get(str(int(pred)), 'Unknown') + ': {:.2f}%'.format(float(probs[0, 0]) * 100)

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, text_mostrar, (60, 285), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

        # Convert text to audio only if the text changes
        if text_mostrar != last_text:
            asyncio.create_task(speak(text_mostrar.split(':')[0]))  # Extract text before percentage
            last_text = text_mostrar

        cv2.imshow('Cam', frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Delete the temporary audio file
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")

# start process
asyncio.run(main())