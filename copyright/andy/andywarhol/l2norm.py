import numpy as np
from skimage import io
from skimage.color import gray2rgb, rgba2rgb
from skimage.transform import resize

def load_and_preprocess_image(image_path, target_size=(178, 218)):
    # Load image
    image = io.imread(image_path)
    
    # Convert grayscale images to RGB (3 channels)
    if len(image.shape) == 2:  # if the image is grayscale
        image = gray2rgb(image)
    elif image.shape[2] == 4:  # if the image has an alpha channel (RGBA)
        image = rgba2rgb(image)
    
    # Resize the image to the target size
    image = resize(image, target_size, anti_aliasing=True)
    
    # Debug: Print the shape before flattening
    print(f"Image shape before flattening: {image.shape}")
    
    # Flatten the image to a 1D vector
    image_vector = image.flatten()
    return image_vector

def compute_l2_norm(image1_vector, image2_vector):
    # Compute the difference vector
    difference_vector = image1_vector - image2_vector
    
    # Calculate the L2 norm
    l2_norm = np.linalg.norm(difference_vector)
    return l2_norm

# Load and preprocess the reference image with a fixed target size
image1_vector = load_and_preprocess_image('prince.jpg', target_size=(178, 218))
image2_vector = load_and_preprocess_image('prince1.jpg',target_size=(178, 218))
l2_norm = compute_l2_norm(image1_vector, image2_vector)
print(l2_norm)
# Compute the L2 norm with other images
# for i in range(1, 7):
#     image2_vector = load_and_preprocess_image(f'celeb{i}.jpg', target_size=(178, 218))
    
#     # Ensure both images are of the same size
#     print(f"Image 1 vector shape: {image1_vector.shape}")
#     print(f"Image 2 vector shape: {image2_vector.shape}")
    
#     assert image1_vector.shape == image2_vector.shape, "Images must be of the same size"
    
#     l2_norm = compute_l2_norm(image1_vector, image2_vector)
#     print(f"L2 norm between the images: {l2_norm}")
