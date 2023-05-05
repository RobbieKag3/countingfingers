import cv2
import mediapipe as mp
camera=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)
tipid=[8,12,16,20]
def countfingers(image,hand_landmarks):
    if hand_landmarks:
        landmarks=hand_landmarks[0].landmark
        fingers=[]
        for lm_index in tipid:
            fingertip_y=landmarks[lm_index].y
            fingerbottom_y=landmarks[lm_index-2].y
            if fingertip_y<fingerbottom_y:
                fingers.append(1)
            if fingertip_y>fingerbottom_y:
                fingers.append(0)
        totalfingers=fingers.count(1)
        text=f'fingers :{totalfingers}'
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
def drawhandlandmarks(image,hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image,landmarks,mp_hands.HAND_CONNECTIONS)
         
while True:
    success,image=camera.read()
    image=cv2.flip(image,1)
    results=hands.process(image)
    hand_landmarks=results.multi_hand_landmarks
    drawhandlandmarks(image,hand_landmarks)
    countfingers(image,hand_landmarks)
    cv2.imshow("counting fingers",image)
    if cv2.waitKey(1)==32:
        break
cv2.destroyAllWindows()