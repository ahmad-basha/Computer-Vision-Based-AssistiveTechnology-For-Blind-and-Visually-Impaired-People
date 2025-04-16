import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # sets the speaking speed


# Loads the YOLOv3 model weights (yolov3.weights)
#  and configuration file (yolov3.cfg)
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


# LReads 80 class labels from coco.names 
# (e.g., person, car, dog, etc.).
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#Extracts the output layers of the YOLO model.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

#Capturing Video from Webcam
cap = cv2.VideoCapture(0)

# Setting Confidence Threshold and 
# Non-Maximum Suppression (NMS)
confidence_threshold = 0.5
nms_threshold = 0.4


# Track labels from the previous frame
prev_frame_labels = set()

# Loop for real-time object detection
while True:
    ret, frame = cap.read()

    # If frame is not captured properly
    if not ret:
        print("Error: Failed to capture the frame")
        break

    height, width = frame.shape[:2]

    # Preprocess the input image for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform forward pass to get output from output layers
    detections = net.forward(output_layers)

    # Store information about detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maxima suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    current_frame_labels = set()

    # Draw the bounding boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add label to current frame's set
            current_frame_labels.add(label)

    # Speak only new labels that appear in the current frame
    new_labels = current_frame_labels - prev_frame_labels
    for label in new_labels:
        engine.say(label)
        engine.runAndWait()

    # Update the previous frame labels set
    prev_frame_labels = current_frame_labels.copy()

    # Show the resulting frame
    cv2.imshow('Webcam Object Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()