import cv2
import numpy as np
import matplotlib.pyplot as plt

def averaging(temporal_codes,k):
    result = np.zeros_like(temporal_codes[0],dtype=np.float64,shape=(64,64))
    for code in temporal_codes:
        result += code
    result /= k
    print(result.dtype)
    print(np.mean(result))

    blurred = cv2.GaussianBlur(result, (5,5), 1)
    sharpened = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
    cv2.imwrite("result.jpg",sharpened)
    return sharpened

def random_masking_function(input_image,n):
    # Reduce the contrast of the input image
    alpha = 0.5  # contrast reduction factor
    low_contrast_image = alpha * input_image + (1/2 - alpha/2)

    # Decompose the low contrast image into a Laplacian pyramid
    k=3
    pyramid = []
    temp = low_contrast_image
    for i in range(k):
        temp = cv2.pyrDown(temp)
        temp = cv2.resize(temp, (64,64))
        pyramid.append(temp)

    # Create the set of temporal codes
    temporal_codes = []
    for i in range(n):
        # Create a random mask for each scale in the pyramid
        masks = []
        for j in range(k):
            mask = np.random.randint(0, 2, size=pyramid[j].shape, dtype=np.uint8)
            masks.append(mask)

        # Apply the masks to the different scales of the pyramid
        masked_pyramid = []
        for j in range(k):
            masked_scale = pyramid[j] * masks[j]
            masked_pyramid.append(masked_scale)

        # Average the masked scales to create the temporal code
        temporal_code = np.zeros_like(input_image, dtype=np.float64,)
        temporal_code = np.resize(temporal_code,new_shape=(64,64))
        for j in range(k):
            temporal_code += masked_pyramid[j]
        temporal_code /= k

        # Convert the temporal code to an unsigned 8-bit integer
        temporal_code = temporal_code.astype(np.uint8)
        
        _, temporal_code = cv2.threshold(temporal_code, np.mean(temporal_code), 255, cv2.THRESH_BINARY)
        # Add the temporal code to the list of codes
        temporal_codes.append(temporal_code)
        # plt.imshow(temporal_codes[-1],cmap="gray")
        # plt.show()
    return temporal_codes

    
