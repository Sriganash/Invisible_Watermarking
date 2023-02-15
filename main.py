import math
import cv2
import random_masking as rm
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import numpy as np    


def embed_visible_watermark(section, watermark, jnd_threshold,suitable_location):
    # Convert image to grayscale
    # y = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(section,cv2.COLOR_RGB2HSV)
    y = img[:,:,1]
    # Calculate intensity mean value of the host region
    # section = cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb)

    # y,cr,cb = cv2.split(section)
    #print(watermark)
    # print(section)
    mean_intensity = np.mean(y)
    #print(mean_intensity)

    # Embed the watermark
    for i in range(64):
        for j in range(64):
            if(watermark[i][j]==0):
                y[i][j] = max(0, mean_intensity - (jnd_threshold // 2))
                #print("1")
            else:
                y[i][j] = min(255, mean_intensity + (jnd_threshold // 2))
                #print(0)
    #print(section)
    # modified = cv2.merge((y,cr,cb))
    
    img[:,:,1] = y
    
    return mean_intensity,cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

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
    # Embed the bit using bit-wise shifting
    block = cv2.cvtColor(block,cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(block)
    if bit == '0':
        y = (y >> 1 << 1).astype(y.dtype)  # encode 0
    else:
        y = ((y >> 1 << 1) + 1).astype(y.dtype)  # encode 1

    # Convert the modified block back to the RGB color space
    modified = cv2.merge((y,cr,cb))
    return (cv2.cvtColor(modified,cv2.COLOR_YCrCb2RGB))

def extract_bit(block):
    block = cv2.cvtColor(block,cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(block)

    # Extract the LSB of the block using bit-wise shifting
    bit = y & 1
    mean = np.mean(bit)
    return '0' if np.mean(bit) <= 0.5 else '1'

def find_suitable_location(type,keyframe, binary_saliency_map, watermark_size):
    # Get dimensions of watermark
    A, B = watermark_size

    # Convert keyframe to grayscale
    gray = cv2.cvtColor(keyframe, cv2.COLOR_BGR2GRAY)

    # Compute mean and variance for each candidate region

    variances = []
    
    if type == 'visible':
        candidate_regions = np.where(binary_saliency_map == 0)
        candidate_locations = [(x, y) for x, y in zip(candidate_regions[0], candidate_regions[1])  if x + A < keyframe.shape[0] and y + B < keyframe.shape[1]]
    
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
    # else:
    #     candidate_regions = np.where(binary_saliency_map == 255)
    #     candidate_locations = [(x, y) for x, y in zip(candidate_regions[0], candidate_regions[1])  if x + A < keyframe.shape[0] and y + B < keyframe.shape[1]]
    #     print(frame.shape)
    #     candidate_region = []
    #     k=l=0
    #     while k<30:
    #         if (l+3)%6 == 0:
    #             overlaps = False
    #             coord = candidate_locations[l]
    #             for noc in candidate_region:
    #                 if coord[0]>=noc[0] and coord[0]<=noc[0]+8 and coord[1]>=noc[1] and coord[1]<=noc[1]+8:
    #                     overlaps = True
    #                     break
    #                 elif coord[0]+8>=noc[0] and coord[0]+8<=noc[0]+8 and coord[1]>=noc[1] and coord[1]<=noc[1]+8:
    #                     overlaps = True
    #                     break
    #                 elif coord[0]>=noc[0] and coord[0]<=noc[0]+8 and coord[1]+8>=noc[1] and coord[1]+8<=noc[1]+8:
    #                     overlaps = True
    #                     break
    #                 elif coord[0]+8>=noc[0] and coord[0]+8<=noc[0]+8 and coord[1]+8>=noc[1] and coord[1]+8<=noc[1]+8:
    #                     overlaps = True
    #                     break
    #             if not overlaps:
    #                 x, y = coord[0],coord[1]
    #                 if x+8<keyframe.shape[0] and y+8<keyframe.shape[1]:
    #                     candidate = (binary_saliency_map[x:x+8, y:y+8])
    #                     mean = np.mean(candidate)
    #                     if mean==255:
    #                         candidate_region.append(coord)
    #                         k+=1
    #         l+=1
    #     return candidate_region


def Divide(frame):
    frames = []
    for x in range(0,frame.shape[0],8):
        for y in range(0,frame.shape[1],8):
            frames.append((x,y))
    return frames



    
 
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
    logo = cv2.imread("input/new.png")
    logo = cv2.cvtColor(logo,cv2.COLOR_RGB2GRAY)
    logo = cv2.resize(logo,(64,64))
    #print(logo.shape)
    # cv2.imshow("l",logo)
    # cv2.waitKey(0)

    clip = cv2.VideoCapture("input/dawn_of_the_7.avi")
    fps = round(clip.get(cv2.CAP_PROP_FPS))
    clip_w = clip.get(cv2.CAP_PROP_FRAME_WIDTH)
    clip_h = clip.get(cv2.CAP_PROP_FRAME_HEIGHT)
    f = round(clip.get(cv2.CAP_PROP_FRAME_COUNT))

    k = int(str(f**2)[:3])

    temporal_codes = rm.random_masking_function(logo,k)

    key = f//k  
    print(key)
    #plt.imshow(temporal_codes[0],cmap="gray")
    #plt.show()
    
    # r=rm.averaging(temporal_codes,f//key)
    # r = (r+0.1)/0.5
    #plt.imshow(r,cmap="gray")
    #plt.show()


    w_clip = int(clip.get(cv2. CAP_PROP_FRAME_WIDTH ))
    h_clip = int(clip.get(cv2. CAP_PROP_FRAME_HEIGHT ))
    # fourcc = int(clip.get(cv2. CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    print(fourcc, cv2.VideoWriter_fourcc(*'FFV1'))
    
    file = cv2.VideoWriter('output/result_dawn_of_the_7.avi', fourcc, fps, (w_clip, h_clip),isColor=True)
    ret = True
    i=j=0
    m=1
    while ret:
        ret,frame = clip.read()
        if(i%(key) == 0 and ret and j<len(temporal_codes)):
            roi = calculate_binary_saliency_map(frame, sigma=10, T=128)
            location_visible = find_suitable_location('visible',frame,roi,(64,64))
            watermark = cv2.resize(temporal_codes[j],(64,64),interpolation=cv2.INTER_AREA)
            a,b=location_visible
            section = frame[a:a+64,b:b+64,:]
            # y,cr,cb = cv2.split(cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb))
            # mean_intensity = np.mean(y)
            # section = cv2.cvtColor(cv2.merge((y,cr,cb)),cv2.COLOR_YCrCb2RGB)
            mean_intensity,section[:,:,:] = embed_visible_watermark(section,watermark,32,location_visible)
            # y,cr,cb = cv2.split(cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb))
            # print(y)

            # Calculate the binary of mean intensity an upper left coordinates
            invisible_value = '0'*(8-len(bin(int(mean_intensity))[2:])) + bin(int(mean_intensity))[2:] \
                                + '0'*(11-len(bin(int(location_visible[0]))[2:]))+ bin(int(location_visible[0]))[2:] \
                                + '0'*(11- len(bin(int(location_visible[1]))[2:])) + bin(int(location_visible[1]))[2:]
            
            #Divide the frame into 8x8 blocks
            frames  = Divide(frame)

            #Select 30 invisible 8x8 blocks from frames
            l=k=0
            while(l<len(frames) and k<30):
                if l%key == 0 :
                    x,y = frames[l][0], frames[l][1]
                    section = frame[x:x+8,y:y+8,:]
                    section[:,:,:] = embed_invisible_watermark(section,invisible_value[k])
                    # y,cr,cb = cv2.split(cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb))
                    k+=1
                l+=1
            print(".......................")
            # print(selected)
            print(invisible_value,mean_intensity,location_visible[0],location_visible[1])
            j = j+1
            # cv2.imshow("f",frame)
            # cv2.waitKey(0)
        file.write(frame)
        i+=1
        m+=1
    file.release()


    # print("===========================")
    video = cv2.VideoCapture("output/result_dawn_of_the_7.avi")
    ret = True
    i=j=0
    m=1
    
    r = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    c = int(video.get(cv2. CAP_PROP_FRAME_WIDTH ))
    temporal = []
    n = 0
    while ret:
        ret,frame = video.read()
        if i%(key)== 0 and ret:
            frames  = Divide(frame)
            selected = ""
            #Select 30 invisible 8x8 blocks from frames
            l=k=0
            while(l<len(frames) and k<30):
                if l%key == 0 :
                    x,y = frames[l][0], frames[l][1]
                    section = frame[x:x+8,y:y+8,:]
                    # y,cr,cb = cv2.split(cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb))
                    selected += extract_bit(section)
                    k+=1
                l+=1
                    
            mean_intensity = int('0b' + selected[:8],2)
            r1 = int('0b' + selected[8:19],2)
            c1 = int('0b' + selected[19:],2)

            print(selected,mean_intensity,r1,c1)

            section = frame[r1:r1+logo.shape[0],c1:c1+logo.shape[1],:]

            if r1 > r or c1 > c:
                i+=1
                continue
            

            section = cv2.cvtColor(section,cv2.COLOR_RGB2HSV)
            # y,cr,cb = cv2.split(section)
            y = section[:,:,1]

            # print(y)
            y[y >= mean_intensity] = 255
            y[y < mean_intensity] = 0
 
            temporal.append(y)
            # print(y)
            n+=1
        i+=1
        m+=1
    
    r=rm.averaging(temporal,n)
    r = (r+0.1)/0.5
    cv2.imshow("R",r)
    cv2.waitKey(0)
    plt.imshow(r,cmap="gray")
    plt.show()