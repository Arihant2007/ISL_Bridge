import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2, min_detection_confidence = 0.5)
mp_draw = mp.solutions.drawing_utils

def main():
    choice = input("Camera input or Video file : ").lower().strip()

    if choice == "camera input" or choice == "camera":

        label = input("Enter the sign you want to record (press enter to stop recording) : ").upper().strip()

        while True:
            if (len(label) == 1 and label.isalnum()) or label == "SPACE" :
                cap = cv2.VideoCapture(0)
                record_sign(label,cap)

                label = input("Enter next sign (press enter to stop recording) : ").upper().strip()

            elif (len(label) > 1 and label != "SPACE") or (not label.isalnum() and label != "")  :
                label = input("Enter a valid sign (press enter to stop recording) : ").upper().strip()

            elif label == "" :
                break

    elif choice == "video file" or choice == "video":
        label = input("Enter the sign the video is showing : ")
        video = input("Enter video file path : ").lower().strip()
        cap = cv2.VideoCapture(video)
        record_sign(label,cap)

def record_sign(label,cap):

    count = 0
    while True:  
        success , frame = cap.read()
        if success:
            frame = cv2.flip(frame,1)
            rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

            #video and live-cam
            count += 1
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                if count % 5 == 0 :
                    write_to_csv(results, label)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame,hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Test_cam",frame)
        elif not success:
            break
        
        #live-cam
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

def write_to_csv (results, sign_label):
    header = []
    for i in ('L','R'):
        for _ in range(21):
            a = f"h{i}x{_}"
            b = f"h{i}y{_}"
            c = f"h{i}z{_}"
            header.append(a)
            header.append(b)
            header.append(c)
    header.append("sign")   

    dataL = []
    dataR = [] 
    for i,hand in enumerate(results.multi_hand_landmarks):
        label = results.multi_handedness[i].classification[0].label
        if label.lower() == "left":
            for point in hand.landmark:
                a = point.x
                b = point.y
                c =  point.z
                dataL.extend([a,b,c])   
        if label.lower() == "right" :
            for point in hand.landmark:
                a = point.x
                b = point.y
                c =  point.z
                dataR.extend([a,b,c])     
    if dataL == []:
        data = [0] * 63 + dataR + [sign_label] 
    elif dataR == []:
        data = dataL + [0] * 63 + [sign_label] 
    else :
        data = dataL + dataR + [sign_label]

                 
    if not os.path.exists("../data/landmarks_own.csv"):
        with open("../data/landmarks_own.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(data)
    else :
        with open("../data/landmarks_own.csv","a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)      
         

if __name__ == "__main__" :
     main()