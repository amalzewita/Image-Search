# Import required libraries
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import random
import matplotlib.pyplot as plt

# Load the YOLO object detection model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# Load the COCO names for classes
classes = open('coco.names').read().strip().split('\n')

# Function to open a file dialog and let user select an image
def take_image():
    global image_path
    image_path = filedialog.askopenfilename()

# Function to open a directory dialog and let user select a dataset
def take_dataset():
    global data_path
    data_path = filedialog.askdirectory()

# Function to detect objects in an image using YOLO
def detect_objects(image):
    # Convert image to blob for processing
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)
    # Lists to store detection boxes and their class IDs
    boxes = []
    class_ids = []
    # Iterate over detections and filter ones with confidence > 0.5
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Convert detection coordinates to standard x,y,w,h format
                center_x, center_y, width, height = map(int, detection[:4] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                x, y = center_x - width // 2, center_y - height // 2
                boxes.append([x, y, width, height])
                class_ids.append(class_id)
    return boxes, class_ids

# Function to draw bounding boxes and class names on the image
def draw_boxes(image, boxes, class_ids):
    # Iterate over each detection
    for i, box in enumerate(boxes):
        x, y, w, h = box
        class_id = class_ids[i]
        # Set color for bounding boxes
        color = (0, 255, 0)
        # Get class name
        class_name = classes[class_id]
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # Draw class name
        cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Function to display images for a set of labels
def show_images(unique_labels, dataset_folder, image_display_frame, n_lim=1):
    # Clear existing widgets
    for widget in image_display_frame.winfo_children():
        widget.destroy()
    # Iterate over labels and display images
    for label in unique_labels:
        label_folder = os.path.join(dataset_folder, label)
        image_files = [f for f in os.listdir(label_folder) if f.endswith(".jpg")]
        plt_arr = random.sample(image_files, n_lim)
        fig, axes = plt.subplots(1, n_lim, figsize=(12, 4))
        for idx, img_file in enumerate(plt_arr):
            img = Image.open(os.path.join(label_folder, img_file))
            img = img.resize((100, 100))
            axes[idx].imshow(img)
            axes[idx].axis('off')
        plt.tight_layout()
        plt.show()

# Function to initiate the search using YOLO
def Search():
    # Read image from path
    image = cv2.imread(image_path)
    # Detect objects in the image
    boxes, class_ids = detect_objects(image)
    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image, boxes, class_ids)
    # Show image with bounding boxes
    cv2.imshow("Detected Objects", image_with_boxes)
    # Extract unique class labels
    unique_labels = [classes[i] for i in set(class_ids)]
    result_label.config(text=", ".join(unique_labels))
    show_images(unique_labels, data_path, image_display_frame, int(image_num_in.get()))

# Main GUI
root = tk.Tk()
root.title("Image search engine")
root.geometry("400x400")
root.configure(bg='pink')
root.resizable(False, False)

# GUI components for user interaction
upload_button = tk.Button(root, text="select Image", command=take_image)
upload_button.pack(pady=10)
label = tk.Label(root, text="", bg='pink').pack()
browse_button = tk.Button(root, text="dataset", bg='white', command=take_dataset).pack()
label = tk.Label(root, text="", bg='pink').pack()
label = tk.Label(root, text="Number of Images:", bg='pink').pack()
image_num_in = tk.Entry(root)
image_num_in.pack()
label = tk.Label(root, text="", bg='pink').pack()
button = tk.Button(root, text="    search      ", bg='white', command=Search).pack()
result_label = tk.Label(root, text="", justify="left")
result_label.pack(pady=10)
image_display_frame = tk.Frame(root)
image_display_frame.pack()
# Main event loop
root.mainloop()
