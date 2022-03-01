import cv2
import mmcv
import pickle
import os
import numpy as np
import argparse

def random_colour():
    colour = [int(x) for x in np.random.choice(range(256), size=3)]
    return tuple(colour)

def get_colour_list(num):
    colours = []
    for i in range(0, num+1):
        colours.append(random_colour())
    return colours

def make_video(results, frames):
    
    with open(results, 'rb') as handle:
        results = pickle.load(handle)

    video_name = results['video']
    results = results['annotations']

    img_array = []

    apes = []
    for x in results:
        for d in x['detections']:
            apes.append(d['ape_id'])
    
    ape_no = max(apes)
    colours = get_colour_list(ape_no)
    
    filelist = [os.path.join(frames, x) for x in os.listdir(f"{frames}")]
    no_of_frames = len(filelist)

    for i in range(0, no_of_frames):
        
        filename = f"{frames}/{i}.jpg"

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        is_detection = False

        for r in results:
            
            if(r['frame_id']==i):

                is_detection = True
                
                detections = r['detections']

                # Get bndbxs for each detection
                for i, det in enumerate(detections):
                    ape_id = det['ape_id']
                    
                    bbox = list(map(float, det['bbox']))
                    xmin, ymin, xmax, ymax = bbox    
                    
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colours[ape_id], 2)
                    cv2.putText(img, f"ape: {ape_id}", (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[ape_id], 2)
                
                img_array.append(img)
        
        if(is_detection==False):
            img_array.append(img)
   
    out = cv2.VideoWriter(
        f"{video_name}_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, size
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', type=str)
    parser.add_argument('--results', type=str)
    parser.add_argument('--video', type=str)
    args = parser.parse_args()
    return args

def main():

    args = argparser()

    frames_path = args.frame_path
    results = args.results

    video = mmcv.VideoReader(args.video)
    video.cvt2frames(frame_dir=args.frame_path, filename_tmpl='{:d}.jpg')

    make_video(results, frames_path)


if __name__ == "__main__":
    main()
