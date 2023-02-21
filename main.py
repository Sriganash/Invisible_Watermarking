import subprocess
import cv2
import random_masking as rm
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import numpy as np    
import random


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


def embed(clip,file,key):
    ret = True
    i=j=0
    while ret:
        ret,frame = clip.read()
        if(j<len(temporal_codes) and i==key[j] and ret ):
            roi = calculate_binary_saliency_map(frame, sigma=10, T=128)
            location_visible = find_suitable_location('visible',frame,roi,(64,64))
            watermark = cv2.resize(temporal_codes[j],(64,64),interpolation=cv2.INTER_AREA)
            a,b=location_visible
            section = frame[a:a+64,b:b+64,:]
            # y,cr,cb = cv2.split(cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb))
            # mean_intensity = np.mean(y)
            # section = cv2.cvtColor(cv2.merge((y,cr,cb)),cv2.COLOR_YCrCb2RGB)
            cv2.imshow("original",frame)
            cv2.waitKey(0)
            mean_intensity,section[:,:,:] = embed_visible_watermark(section,watermark,32,location_visible)
            
            cv2.imshow("new",frame)
            cv2.waitKey(0)
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
            key_1 = len(key)
            while(l<len(frames) and k<30):
                if l%key_1 == 0 :
                    x,y = frames[l][0], frames[l][1]
                    section = frame[x:x+8,y:y+8,:]
                    section[:,:,:] = embed_invisible_watermark(section,invisible_value[k])
                    # y,cr,cb = cv2.split(cv2.cvtColor(section,cv2.COLOR_RGB2YCrCb))
                    k+=1
                l+=1
            print(".......................")
            # print(selected)
            print(invisible_value,mean_intensity,location_visible[0],location_visible[1])
            # cv2.imshow("f",frame)
            # cv2.waitKey(0)
            j+=1
        file.write(frame)
        i+=1
    print("===========================")

def extract(video,key):
    ret = True
    i=j=0
    m=1
        
    r = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    c = int(video.get(cv2. CAP_PROP_FRAME_WIDTH ))
    temporal = []
    n = 0
    while ret:
        ret,frame = video.read()
        if j<len(key) and i==key[j] and ret:
            frames  = Divide(frame)
            selected = ""
            #Select 30 invisible 8x8 blocks from frames
            l=k=0
            key_1 = len(key)
            while(l<len(frames) and k<30):
                if l%key_1 == 0 :
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
            j+=1
        i+=1
    return temporal, n
        

def gen_num(num,bound,r):
    step=bound/num
    result=[]
    for i in range(num):
        x=round(random.uniform((step*i+r),(step*(i+1)-r)))
        result.append(int(x))
    return result

if __name__ == "__main__":
    logo = cv2.imread("input/new.png")
    logo = cv2.cvtColor(logo,cv2.COLOR_RGB2GRAY)
    logo = cv2.resize(logo,(64,64))
    #print(logo.shape)
    # cv2.imshow("l",logo)
    # cv2.waitKey(0)


    # Read the video
    input_file = "input/ice.avi"
    clip = cv2.VideoCapture(input_file)
    fps = round(clip.get(cv2.CAP_PROP_FPS))
    clip_w = clip.get(cv2.CAP_PROP_FRAME_WIDTH)
    clip_h = clip.get(cv2.CAP_PROP_FRAME_HEIGHT)
    f = round(clip.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract the audio from the video
    audio_file = input_file+"audio.m4a"
    subprocess.call(["ffmpeg", "-i", input_file, "-vn", "-acodec", "copy", audio_file])

    # Creating the temporal codes
    k = int(str(f**2)[:2])
    
    delay = 0*fps
    d = f//(delay+15)
    temporal_codes = rm.random_masking_function(logo,d)

    # Creating the key
    key = gen_num(d,f,delay)
    print(key)
    #plt.imshow(temporal_codes[0],cmap="gray")
    #plt.show()
    
    # r=rm.averaging(temporal_codes,f//key)
    # r = (r+0.1)/0.5
    #plt.imshow(r,cmap="gray")
    #plt.show()

    # Create the output file
    output_file = "output/result_ice.mkv"
    w_clip = int(clip.get(cv2. CAP_PROP_FRAME_WIDTH ))
    h_clip = int(clip.get(cv2. CAP_PROP_FRAME_HEIGHT ))
    # fourcc = int(clip.get(cv2. CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    print(fourcc, cv2.VideoWriter_fourcc(*'FFV1'))
    file = cv2.VideoWriter(output_file, fourcc, fps, (w_clip, h_clip),isColor=True)
    ret = True
    i=j=0
    m=1

    # E#mbedding the temporal codes with input file
    embed(clip,file,key)
    clip.release()
    file.release()

    # Attaching the audio file with newly created video
    subprocess.call(["ffmpeg", "-i", output_file, "-i", audio_file, "-c:v", \
                        "ffv1", "-c:a", "aac", "-shortest", "output/final_ice.mkv"])

    
    
    # Extracting the temporal codes from the output file
    output_file = "output/result_ice.mkv"
    video = cv2.VideoCapture(output_file)
    temporal,n = extract(video,key)

    # Calculating the watermark from the temporal codes
    r=rm.averaging(temporal,n)
    cv2.imshow("R",r)
    cv2.waitKey(0)
    plt.imshow(r,cmap="gray")
    plt.show()