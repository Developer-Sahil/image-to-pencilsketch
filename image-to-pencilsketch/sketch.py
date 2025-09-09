import cv2
import numpy as np
import os

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def convert_to_grayscale(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

def invert_image(image_gray):
    return 255 - image_gray

def blur_image(image, kernel_size=(21, 21)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def dodge_blend(greyscale ,inverted_blur):
    blend = cv2.divide(greyscale, 255-inverted_blur, scale=256)
    return np.clip(blend, 0, 255).astype(np.uint8)

def save_image(output_folder, input_image_path, image):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_folder, f"{name}_sketch{ext}")
    cv2.imwrite(output_path, image)
    return output_path

def main():
    input_path = "images/spiderman.jpeg"
    output_folder = "images/output"

    image_rgb = read_image(input_path)
    gray_image = convert_to_grayscale(image_rgb)
    inverted_image = invert_image(gray_image)
    blurred_image = blur_image(inverted_image)

    sketch = dodge_blend(gray_image, blurred_image)

    save_image(output_folder, input_path, sketch)
    print(f"converted image saved at {output_folder}")

    #show image on a window
    cv2.imshow("Pencil Sketch", sketch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


