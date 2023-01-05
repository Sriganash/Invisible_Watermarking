import math
import cv2
import random_masking as rm
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
    cv2.imwrite("final.jpg",image)
    _
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
    # Convert the block to grayscale
    block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)

    # Perform the FFT on the block
    fft_block = np.fft.fft2(block)

    # Perform the inverse FFT on the modified block
    modified_block = np.fft.ifft2(fft_block)
    modified_block = np.real(modified_block).astype(block.dtype)

    # Embed the bit using bit-wise shifting
    if bit == '0':
        modified_block = (modified_block >> 1 << 1).astype(block.dtype)  # encode 0
    else:
        modified_block = ((modified_block >> 1 << 1) + 1).astype(block.dtype)  # encode 1

    # Convert the modified block back to the RGB color space
    return cv2.cvtColor(modified_block, cv2.COLOR_GRAY2RGB)

def extract_bit(block):
    # Convert the block to grayscale
    block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)

    # Extract the LSB of the block using bit-wise shifting
    bit = block & 1

    return '0' if np.mean(bit) == 0 else '1'

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
    
    if type =='visible':
        for location in candidate_locations:
            x, y = location
            if x+A<keyframe.shape[0] and y+B<keyframe.shape[1]:
                candidate_region = (gray[x:x+A, y:y+B])
                mean = np.mean(candidate_region)
                variance = 1.0 / (A * B) * np.sum((candidate_region - mean) ** 2)
                variances.append(variance)
        # Find candidate region with lowest variance
        min_variance_index = np.argmin(variances)
        suitable_location = candidate_locations[min_variance_index]
        return suitable_location
    else:
        candidate_region = set()
        for coord in candidate_locations:
            overlaps = any(coord[0] >= noc[0] and coord[0] < noc[0] + 8 and coord[1] >= noc[1] \
                and coord[1] < noc[1] + 8 or (coord[0]+8>=frame.shape[0] \
                    and coord[1]+8>=frame.shape[1]) for noc in candidate_region)
            if not overlaps:
                    candidate_region.add(coord)
        return candidate_region


    
 
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

    clip = cv2.VideoCapture("ice.avi")
    fps = round(clip.get(cv2.CAP_PROP_FPS))
    f = round(clip.get(cv2.CAP_PROP_FRAME_COUNT))

    key = 37

    temporal_codes = rm.random_masking_function(logo,f//key)
    # plt.imshow(temporal_codes[0],cmap="gray")
    # plt.show()
    
    r=rm.averaging(temporal_codes,f//key)
    r = (r+0.1)/0.5
    plt.imshow(r,cmap="gray")
    plt.show()


    w_clip = int(clip.get(cv2. CAP_PROP_FRAME_WIDTH ))
    h_clip = int(clip.get(cv2. CAP_PROP_FRAME_HEIGHT ))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    file = cv2.VideoWriter('result.avi', fourcc, fps, (w_clip, h_clip))
    ret = True
    i=j=0
    while ret:
        ret,frame = clip.read()
        if((i+30)%key == 0):
            roi = calculate_binary_saliency_map(frame, sigma=10, T=128)
            cv2.imwrite("frame1.jpg",frame)
            cv2.imwrite("saliency.jpg",roi)
            location_visible = find_suitable_location('visible',frame,roi,(64,64))
            watermark = cv2.resize(temporal_codes[j],(64,64),interpolation=cv2.INTER_AREA)
            image = embed_visible_watermark(frame,watermark,5,location_visible)

            # Calculate the binary of mean intensity an upper left coordinates
            invisible_value = '0'*(8-len(bin(int(np.mean(watermark)))[2:])) + bin(int(np.mean(watermark)))[2:] \
                                + '0'*(11-len(bin(int(location_visible[0]))[2:]))+ bin(int(location_visible[0]))[2:] \
                                + '0'*(11-len(bin(int(location_visible[1]))[2:])) + bin(int(location_visible[1]))[2:]
            
            # Remove the overlapping blocks
            non_overlapping_coordinates = list(find_suitable_location('invisible',frame,roi,(8,8)))
            
            k=l=0
            while(k<30):
                if((l+3)%6) == 0:
                    section = frame[non_overlapping_coordinates[l][0]:non_overlapping_coordinates[l][0]+8,non_overlapping_coordinates[l][1]:non_overlapping_coordinates[l][1]+8]
                    section[ : : ] = embed_invisible_watermark(section,invisible_value[k])
                    k+=1
                l+=1
            
            print(".......................")
            # print(selected)
            print(invisible_value,np.mean(watermark),location_visible[0],location_visible[1])
            j = (j+1)%30
            # cv2.imshow("f",frame)
            # cv2.waitKey(0)
        file.write(frame)
        i+=1
    file.release()

    print("===========================")
    clip = cv2.VideoCapture("result.avi")
    ret = True
    i=j=0
    temporal_codes = []
    while ret:
        ret,frame = clip.read()
        if((i+30)%key == 0) :
            roi = calculate_binary_saliency_map(frame, sigma=10, T=128)
            non_overlapping_coordinates = list(find_suitable_location('invisible',frame,roi,(8,8))) 
            # print(non_overlapping_coordinates)
            k=l=0
            selected = ""
            while(k<30):
                if((l+3)%6) == 0:
                    print(non_overlapping_coordinates[l][0],non_overlapping_coordinates[l][1])
                    section = frame[non_overlapping_coordinates[l][0]:non_overlapping_coordinates[l][0]+8,non_overlapping_coordinates[l][1]:non_overlapping_coordinates[l][1]+8]
                    selected+=extract_bit(section)
                    k+=1
                l+=1
            
            mean_intensity = int('0b' + selected[:8],2)
            r1 = int('0b' + selected[8:19],2)
            c1 = int('0b' + selected[19:],2)

            print(selected,mean_intensity,r1,c1)

            section = frame[r1:r1+logo.shape[0],c1:c1+logo.shape[1]]

            section[section >= mean_intensity] = 255
            section[section < mean_intensity] = 0

            temporal_codes.append(section)    
        i+=1
    plt.imshow(rm.averaging(temporal_codes,f//key), cmap = 'gray')
    plt.show()
    _





    