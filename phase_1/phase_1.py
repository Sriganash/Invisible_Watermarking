import math
import cv2
import random_masking as rm
import laplacian as lp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import numpy as np    


def embed_visible_watermark(image, watermark, jnd_threshold,suitable_location):
      # Convert image to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate intensity mean value of the host region
    section = frame[suitable_location[0]:suitable_location[0]+64,suitable_location[1]:suitable_location[1]+64]
    #print(watermark)
    # print(section)
    mean_intensity = np.mean(section)
    #print(mean_intensity)

    # Embed the watermark
    for i in range(64):
        for j in range(64):
            if(watermark[i][j]==0):
                section[i][j] = max(0, mean_intensity - (jnd_threshold // 2))
                #print("1")
            else:
                section[i][j] = min(255, mean_intensity + (jnd_threshold // 2))
                #print(0)
    #print(section)
    return image

# def compute_saliency_modulated_jnd(frame, saliency_map, block_size=8):
#     # Convert the frame to grayscale if necessary
#     if len(frame.shape) > 2:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Normalize the saliency map to the range [0, 1]
#     saliency_map = saliency_map.astype(np.float32) / np.max(saliency_map)

#     # Compute the JND map by subtracting the frame from a slightly modified version of itself
#     jnd_map = np.abs(frame.astype(np.float32) - (frame + 0.2 * saliency_map).astype(np.float32))

#     # Return the average JND value for the entire frame
#     return np.mean(jnd_map)

def embed_invisible_watermark(block, bit):
    # Convert the block to the YCrCb color space
    block = cv2.cvtColor(block, cv2.COLOR_RGB2YCrCb)

    # Split the block into its Y, Cr, and Cb channels
    y, cr, cb = cv2.split(block)

    # Perform the DCT on the Y channel
    dct_y = cv2.dct(np.float32(y))

    # Modify the LSB of the AC1,2 coefficient to encode the bit
    if bit == '0':
        dct_y[0][0] = np.sign(dct_y[0][0]) * np.floor(np.abs(dct_y[0][0])/2) * 2
    else:
        dct_y[0][0] = np.sign(dct_y[0][0]) * (np.floor(np.abs(dct_y[0][0])/2) * 2) + 1

    # Perform the IDCT on the modified Y channel
    print(dct_y)
    idct_y = cv2.idct(dct_y)
    modified_y = np.rint(idct_y).astype(y.dtype)

    # Merge the modified Y channel with the Cr and Cb channels
    modified_block = cv2.merge((modified_y, cr, cb))

    print(modified_y[0][0],modified_block[0][0][0])
    # Convert the modified block back to the RGB color space
    return cv2.cvtColor(modified_block, cv2.COLOR_YCrCb2RGB)

def retrieve_invisible_watermark(block):
    # Convert the block to the YCrCb color space
    block = cv2.cvtColor(block, cv2.COLOR_RGB2YCrCb)

    # Split the block into its Y, Cr, and Cb channels
    y, cr, cb = cv2.split(block) 

    # Perform the DCT on the Y channel
    dct_y = cv2.dct(np.float32(y))

    print(dct_y)
    
    # Check the LSB of the AC1,2 coefficient
    if dct_y[0][0] >= np.ceil(dct_y[0][0])-0.5:
        if np.ceil(dct_y[0][0]) % 2 == 0:
            return '0'
        else:
            return '1'
    else:
        if np.floor(dct_y[0][0]) % 2 == 0:
            return '0'
        else:
            return '1' 

def find_suitable_location(type,keyframe, binary_saliency_map, watermark_size):
    # Get dimensions of watermark
    A, B = watermark_size

    # Convert keyframe to grayscale
    gray = cv2.cvtColor(keyframe, cv2.COLOR_BGR2GRAY)

    # Compute mean and variance for each candidate region
    if type == 'visible':
        candidate_regions = np.where(binary_saliency_map == 0)
    else:
        candidate_regions = np.where(binary_saliency_map == 255)
    candidate_locations = [(x, y) for x, y in zip(candidate_regions[0], candidate_regions[1])  if x + A < keyframe.shape[0] and y + B < keyframe.shape[1]]
    variances = []
    candidates=[]
    for location in candidate_locations:
        x, y = location
        if x+A<keyframe.shape[0] and y+B<keyframe.shape[1]:
            candidate_region = (gray[x:x+A, y:y+B])
            candidates.append(location)
            mean = np.mean(candidate_region)
            variance = 1.0 / (A * B) * np.sum((candidate_region - mean) ** 2)
            variances.append(variance)
    if type =='invisible': return candidates
    # Find candidate region with lowest variance
    min_variance_index = np.argmin(variances)
    suitable_location = candidate_locations[min_variance_index]
 
    return suitable_location
 
def calculate_binary_saliency_map(image, sigma, T):
      # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate image signature
    ksize = int(sigma * 10 + 1)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getGaussianKernel(ksize, sigma)
    image_signature = cv2.sepFilter2D(gray, -1, kernel, kernel)

    # Calculate saliency map
    saliency_map = np.abs(gray - image_signature)
    

    # Compute binary saliency map
    _, binary_saliency_map = cv2.threshold(saliency_map, T, 255, cv2.THRESH_BINARY)
    # cv2.imshow("b",binary_saliency_map)
    # cv2.waitKey(0)
    return binary_saliency_map


if __name__ == "__main__":
    
    logo = cv2.imread("new.png")
    logo = cv2.cvtColor(logo,cv2.COLOR_RGB2GRAY)
    logo = cv2.resize(logo,(64,64))
    print(logo.shape)
    # cv2.imshow("l",logo)
    # cv2.waitKey(0)

    temporal_codes = rm.random_masking_function(logo)
    # plt.imshow(temporal_codes[0],cmap="gray")
    # plt.show()
    r=rm.averaging(temporal_codes)
    r = (r+0.1)/0.5
    # plt.imshow(r,cmap="gray")
    # plt.show()

    clip = clip = cv2.VideoCapture("ice.mp4")
    w_clip = int(clip.get(cv2. CAP_PROP_FRAME_WIDTH ))
    h_clip = int(clip.get(cv2. CAP_PROP_FRAME_HEIGHT ))

    ret = True
    i=0
    j=0
    while ret:
        ret,frame = clip.read()
        if((i+3)%11==0):
            roi = calculate_binary_saliency_map(frame, sigma=10, T=128)
            location_visible = find_suitable_location('visible',frame,roi,(64,64))
            watermark = cv2.resize(temporal_codes[j],(64,64),interpolation=cv2.INTER_AREA)
            image = embed_visible_watermark(frame,watermark,5,location_visible)
            cv2.imshow('Region of Interest', image)
            cv2.waitKey(0)
            invisible_value = '0'*(8-len(bin(int(np.mean(watermark)))[2:])) + bin(int(np.mean(watermark)))[2:] \
                                + '0'*(11-len(bin(int(location_visible[0]))[2:]))+ bin(int(location_visible[0]))[2:] \
                                + '0'*(11-len(bin(int(location_visible[1]))[2:])) + bin(int(location_visible[1]))[2:]
            
            invisible_location = find_suitable_location('invisible',frame,roi,(8,8))
            # Remove the overlapping blocks
            non_overlapping_coordinates = []

            for coord in invisible_location:
                overlaps = False
                for noc in non_overlapping_coordinates:
                    if coord[0] >= noc[0] and coord[0] < noc[0] + 8 and coord[1] >= noc[1] and coord[1] < noc[1] + 8 or (coord[0]+8>=frame.shape[0] and coord[1]+8>=frame.shape[1]):
                        overlaps = True
                        break
                if not overlaps:
                    non_overlapping_coordinates.append(coord)
            print(len(non_overlapping_coordinates))
            k=l=0
            while(k<30):
                if((l+3)%6) == 0:
                    section = frame[non_overlapping_coordinates[l][0]:non_overlapping_coordinates[l][0]+8,non_overlapping_coordinates[l][1]:non_overlapping_coordinates[l][1]+8]
                    section = embed_invisible_watermark(section,invisible_value[k])
                    k+=1
                l+=1
            print("=====================") 
            k=l=0
            selected = ""
            while(k<30):
                if((l+3)%6) == 0:
                    section = frame[non_overlapping_coordinates[l][0]:non_overlapping_coordinates[l][0]+8,non_overlapping_coordinates[l][1]:non_overlapping_coordinates[l][1]+8]
                    selected+=retrieve_invisible_watermark(section)
                    k+=1
                l+=1
            print(selected)
            print(invisible_value)
            # print(invisible_location)
            # print(len(invisible_value),invisible_value)
            j = (j+1)%30


        i+=1




    