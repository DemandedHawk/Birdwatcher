import cv2
from collections import defaultdict
from ultralytics import YOLO

cv2.imshow("Birdwatcher", cv2.imread("logo.jpg"))

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

doTracking = True

# Store past positions per object ID
track_history = defaultdict(list)


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

SPECIFIED_CLASSES=[
    0
]

cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()  #cpu to only run tensors on cpu
        track_ids = results[0].boxes.id.cpu().tolist()  #tolist turns tensors into python lists
        class_ids = results[0].boxes.cls.cpu().tolist() 

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            if cls_id in SPECIFIED_CLASSES: #Checks if we are intresteted in the track
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Save center point
                track_history[track_id].append((cx, cy))

                # Limit history length
                if len(track_history[track_id]) > 50:
                    track_history[track_id].pop(0)

                # Draw bounding box
                label = f"{model.names[int(cls_id)]} #{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 
                            label, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 255, 0), 
                            2)

                # Draw path
                points = track_history[track_id]
                if doTracking:
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

    cv2.imshow("Birdwatcher", frame)
    if cv2.waitKey(1) & 0xFF == ord("t"):
        doTracking = not doTracking
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()