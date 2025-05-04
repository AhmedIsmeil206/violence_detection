from ultralytics import YOLO
import cv2
import os

try:
    # Load the trained YOLOv8 model
    print("Loading model...")
    model = YOLO("./Hcrowded_project.pt")  # Path to your trained model
    print("Model loaded successfully!")

    # Run inference on a test image
    print("Running inference on image...")
    results = model.predict("./test3.jpg", conf=0.25)  # Lower confidence threshold
    print("Inference completed!")

    # Check if results are available
    if len(results) > 0:
        result = results[0]
        print(f"Number of detections: {len(result.boxes)}")

        # Define the output directory and ensure it exists
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Define the output image path for saving the result
        output_image_path = os.path.join(output_dir, "result3.jpg")

        # Save the result image with bounding boxes
        result.save(output_image_path)
        print(f"Results saved at {output_image_path}!")

        # Load the saved image with bounding boxes
        print(f"Loading image from {output_image_path}...")
        image = cv2.imread(output_image_path)

        if image is None:
            print(f"Error: Image not found at {output_image_path}")
        else:
            print(f"Image loaded successfully!")

        # Display the image with bounding boxes using OpenCV
        print("Displaying image... Press 'Esc' to close.")
        cv2.imshow("Detection Results", image)

        # Wait for the Esc key to be pressed
        while True:
            key = cv2.waitKey(1)
            if key == 27:  # ASCII code for 'Esc'
                print("Esc key pressed. Closing...")
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()
    else:
        print("No detections found.")

except Exception as e:
    print(f"An error occurred: {e}")