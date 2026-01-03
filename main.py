import cv2
from ultralytics import YOLO

logo=cv2.imread("logo.jpg")
cv2.imshow("Birdwatcher",logo)

# laddar modellen
model = YOLO("yolov8m.pt")


# COCO är datasettet som modellen tränas på. 
# Varje siffra representerar en sak som går att hitta.
ALL_CLASSES = {
    0, #person
    1, #bicycle
    2, #car
    3, #motorcycle
    4, #airplane
    5, #bus
    6, #train
    7, #truck
    8, #boat
    9, #traffic light
    10, #fire hydrant
    11, #stop sign
    12, #parking meter
    13, #bench
    14, #bird
    15, #cat
    16, #dog
    17, #horse
    18, #sheep
    19, #cow
    20, #elephant
    21, #bear
    22, #zebra
    23, #giraffe
    24, #backpack
    25, #umbrella
    26, #handbag
    27, #tie
    28, #suitcase
    29, #frisbee
    30, #skis
    31, #snowboard
    32, #sports ball
    33, #kite
    34, #baseball bat
    35, #baseball glove
    36, #skateboard
    37, #surfboard
    38, #tennis racket
    39, #bottle
    40, #wine glass
    41, #cup
    42, #fork
    43, #knife
    44, #spoon
    45, #bowl
    46, #banana
    47, #apple
    48, #sandwich
    49, #orange
    50, #broccoli
    51, #carrot
    52, #hot dog
    53, #pizza
    54, #donut
    55, #cake
    56, #chair
    57, #couch
    58, #potted plant
    59, #bed
    60, #dining table
    61, #toilet
    62, #tv
    63, #laptop
    64, #mouse
    65, #remote
    66, #keyboard
    67, #cell phone
    68, #microwave
    69, #oven
    70, #toaster
    71, #sink
    72, #refrigerator
    73, #book
    74, #clock
    75, #vase
    76, #scissors
    77, #teddy bear
    78, #hair drier
    79, #toothbrush
}

SPECIFIED_CLASSES = {  #Här definieras vilka objekt vi letar efter
    0, #person
    1, #bicycle
    2, #car
    14, #bird
    15, #cat
    16, #dog
    17, #horse
}

#Skapar kamerobjektet
capture = cv2.VideoCapture(0)  # 0 = default webcam

cv2.destroyAllWindows

while True:
    ret, frame = capture.read()
    if not ret:  #Dödar programmet om det inte existerar en kamera
        break

    results = model(frame, stream=True) #här kör modellen

    for r in results: 
        for box in r.boxes: 
            class_id = int(box.cls[0]) 
            if class_id in SPECIFIED_CLASSES: #Kollar om det modellen hittar är det vi vill se
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #objektets position i kameravyn
                label = model.names[class_id] #vad är det som modellen hittat         
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #Skapar rutan
                cv2.putText(        #Skapar texten
                frame,      
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
                )

    cv2.imshow("Birdwatcher", frame) #Visar bilden i ett fönster
    
    if cv2.waitKey(1) & 0xFF == ord("q"): #Stänger av programmet på knapptryck
        break

capture.release()
cv2.destroyAllWindows()