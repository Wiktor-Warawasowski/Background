import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from dis_bg_remover import remove_background

input_path = "./images/"    #path to directory with source images
output_path = "./results/"  #path to directory storing images with changed background
model_path = "./isnet_dis.onnx" #path to a model used by remove_background
image_paths = [f for f in os.listdir(input_path)]   #reading the names of all source images

# Checking if images load properly
# for image_path in image_paths:
#     plt.imshow(cv2.cvtColor(cv2.imread("./images/"+image_path), cv2.COLOR_BGR2RGB))
#     plt.show()

# Removing background and substituting it with solid color
def add_solid_background(name, img):
    # Define a solid black background with full opacity
    background_color = [255, 0, 0, 255]  # Black in BGRA order with full opacity for the alpha channel

    # Create a background image of the same size as our img with 4 channels (including alpha)
    background_image = np.full((img.shape[0], img.shape[1], 4), background_color, dtype=np.uint8)

    # Create a 3 channel alpha mask for blending, normalized to the range [0, 1]
    alpha_mask = (img[:, :, 3] / 255.0).reshape(img.shape[0], img.shape[1], 1)

    foreground = img[:, :, :3].astype(np.float32)  # Use only RGB channels of the foreground
    background = background_image[:, :, :3].astype(np.float32)  # Use only RGB channels of the background
    blended_image = (1 - alpha_mask) * background + alpha_mask * foreground
    blended_image = np.uint8(blended_image)

    # Convert from BGR to RGB for displaying in Matplotlib
    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

    # Display the image with the colored background
    # plt.imshow(blended_image_rgb)
    # plt.axis('off')  # Hide the axes
    # plt.show()

    # Saving file to results directory
    cv2.imwrite(os.path.join(output_path, name), blended_image)

    # return blended_image_rgb
counter = 0
for image_path in image_paths:
    counter += 1
    total = len(image_paths)
    if not os.path.isfile(os.path.join(output_path, image_path)):
        image, mask = remove_background(model_path, os.path.join(input_path, image_path))   #removing background
        add_solid_background(image_path, image)   #adding solid background
    # img_with_bg = add_solid_background(image_path, image)   #adding solid background
        print(f"Image #{counter}/{total} done.")
    else:
        print(f"Image #{counter}/{total} already in results. Skipping {image_path}")


    #Creating a comparison plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.imshow(cv2.cvtColor(cv2.imread(input_path + image_path), cv2.COLOR_RGBA2BGRA))
    # plt.axis('off')
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.imshow(img_with_bg)
    # plt.axis('off')
    # plt.show()