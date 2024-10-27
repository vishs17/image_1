import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import streamlit as st

def load_and_display_image(image):
    if image is not None:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return img
    else:
        st.error("Error: No image uploaded. Please upload an image.")
        return None

def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    st.write("Threshold limit: " + str(ret))

    # Create a 2x2 layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(thresh, channels="GRAY", caption="Otsu's Binarization", use_column_width=True)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.imshow(sure_bg, cmap='gray')
    plt.title('Sure Background (Dilated)')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('Sure Foreground (Eroded)')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(unknown, cmap='gray')
    plt.title('Unknown Region')
    plt.axis('off')

    # Use Streamlit's method to display matplotlib figures
    with col2:
        st.pyplot(fig)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    # Add green border for the segmentation
    output_image = image.copy()
    output_image[markers == -1] = [0, 255, 0]  # Boundary color (Green)

    return output_image  # Return the output image with borders

def enhance_red_orange_contrast(image):
    # Convert image to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define HSV ranges for red and orange
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Create masks for red and orange colors
    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    orange_mask = cv2.inRange(hsv_img, lower_orange, upper_orange)

    # Combine masks for red and orange
    mask = cv2.bitwise_or(red_mask, orange_mask)

    # Enhance contrast and saturation in the selected areas
    hsv_img[:, :, 1] = np.where(mask > 0, np.clip(hsv_img[:, :, 1] * 1.2, 0, 255), hsv_img[:, :, 1])  # Increase saturation
    hsv_img[:, :, 2] = np.where(mask > 0, np.clip(hsv_img[:, :, 2] * 1.1, 0, 255), hsv_img[:, :, 2])  # Increase brightness

    # Convert back to RGB
    enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return enhanced_img

def enhance_image_colors(original_image):
    enhanced_image = enhance_red_orange_contrast(original_image)
    return enhanced_image

# Streamlit interface
def main():
    st.title("Image Processing with Watershed Segmentation and Color Enhancement")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decode the image to OpenCV format

        # Load and display the original image
        original_img = load_and_display_image(img)

        if original_img is not None:
            # Perform watershed segmentation
            segmented_image = watershed_segmentation(original_img)

            # Enhance image colors
            enhanced_image = enhance_image_colors(original_img)

            # Display all images in a 2x2 grid
            st.write("### Processed Images")
            col1, col2 = st.columns(2)

            with col1:
               st.image(segmented_image, caption='Segmented Image with Watershed', use_column_width=True)

            with col2:
                 st.image(enhanced_image, caption='Enhanced Colors Image', use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
