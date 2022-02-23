import os
import cv2
import pickle
import json
import imageio
import torch
import argparse
import tqdm

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from cython_bbox import bbox_overlaps as bbox_ious

def format_results(results_file):

    '''
    Input: results_file - pkl file with detection results
    Output: sorted DataFrame of all results
    '''

    with open(results_file, 'rb') as handle:
        detections = pickle.load(handle)

    df = pd.DataFrame(detections)
    # Create series for video identifier
    df['video'] = df.filename.str.split('_').str[0]

    # Create series for frame index
    df['frame'] = df.filename.str.split('_').str[2].str.strip('.jpg')
    df['frame'] = df.frame.apply(lambda x: int(x))

    # Sort values by video and ascending frame index
    df.sort_values(by=['video', 'frame'], inplace=True)

    return df 

def get_video_results(df, video):

    '''
    Input: df - sorted DataFrame of all results
           video - name of the video for extraction
    Outpur: dict with results for specified video
    '''

    video_df = df[df.video==video]
    video_dict = video_df.to_dict()
    
    return [v[0][0] for k, v in video_dict['result'].items()]

# IoU-based distance cost matrix

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type detlbrs: list[tlbr] | np.ndarray
    :type tracktlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs),len(btlbrs)))
    if ious.size ==0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs,dtype=float),
        np.ascontiguousarray(btlbrs,dtype=float)
    )
    return ious

def ious_distance(atlbrs,btlbrs):
    """
    compute cost based on IoU
    :param atlbrs:
    :param btlbrs:
    :return:
    """
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

# Tracking

def get_confidences(detections, conf_thresh):
    confidence = []
    for i, frame in enumerate(detections):
        conf = []
        for f in frame:
            if(f[-1] > conf_thresh):
                conf.append(f[-1])
        confidence.append(conf)
    return confidence

def track(detections, conf_thresh, distance_matrix, height=404, width=720): 
    
    # Tracking loop
    active_tracklets = []
    finished_tracklets = []
    prev_boxes = []
    
    for i, frame in enumerate(detections):
        
        boxes = []
        conf = []
        
        for f in frame:
            if(f[-1] > conf_thresh):
                boxes.append(f[:-1])
                conf.append(f[-1])
        
        if(len(boxes) > 0):
            boxes = boxes / np.array(
                [width, height, width, height]
            )
        
        prev_indices = []
        boxes_indices = []
                
        if len(boxes) > 0 and len(prev_boxes) > 0:
            
            if(distance_matrix=='euclidean'):
                # Pairwise cost: euclidean distance between boxes
                cost = np.linalg.norm(prev_boxes[:, None] - boxes[None], axis=-1)
            elif(distance_matrix=='iou'):
                # Pairwise cost: IoU-based distance between boxes
                cost = ious_distance(prev_boxes, boxes)
            else:
                raise ValueError('Unspecified distance matrix')
            
            # Bipartite matching
            prev_indices, boxes_indices = linear_sum_assignment(cost)

        # Add matches to active tracklets
        for prev_idx, box_idx in zip(prev_indices, boxes_indices):
            active_tracklets[prev_idx]["boxes"].append(
                np.round(boxes[box_idx], 3).tolist()
            )

        # Finalize lost tracklets
        lost_indices = set(range(len(active_tracklets))) - set(prev_indices)
        for lost_idx in sorted(lost_indices, reverse=True):
            finished_tracklets.append(active_tracklets.pop(lost_idx))
        
        # Activate new tracklets
        new_indices = set(range(len(boxes))) - set(boxes_indices)
        for new_idx in new_indices:
            active_tracklets.append(
                {"start": i, "boxes": [np.round(boxes[new_idx], 3).tolist()]}
            )
        # "Predict" next frame for comparison
        prev_boxes = np.array([tracklet["boxes"][-1] for tracklet in active_tracklets])
    
    tracklets = active_tracklets + finished_tracklets
    
    return tracklets

def id2_tracklets(tracklets, length_thresh):
    '''
    Newest iteration of this function
    '''

    # Threshold tracklets by length
    tracklets = [x for x in tracklets if len(x['boxes']) > length_thresh]
    
    for i, t in enumerate(tracklets):
        t['ape_id'] = i
        t['boxes'] = list(enumerate(t['boxes'], start=t['start']))
    
    # Sort tracklets by start frame
    tracklets = sorted(tracklets, key=lambda x: x['start'], reverse=False) 
    
    dets = {}
    
    for track in tracklets:
        ape_id = track['ape_id']
        for box in track['boxes']:

            entry = [(ape_id, box[1])]

            if(box[0] not in dets.keys()):
                dets[box[0]] = entry
            else:
                dets[box[0]].extend(entry) 
    return dets

def assign_id2tracklets(tracklets, length_thresh):
    
    # Get start index of each tracklet in tracklets
    tracklet_starts = list(set([x['start'] for x in tracklets]))
    tracklet_starts.sort(reverse=False)

    tracklet_len = {}
    
    # For each start index
    for start_idx in tracklet_starts:
        idx_max = 0

        # Find the longest tracklet
        for tracklet in tracklets:
            if(tracklet['start']==start_idx): 
                if((len(tracklet['boxes']) >= length_thresh)):
                    idx_max = len(tracklet['boxes'])
                    if(start_idx not in tracklet_len.keys()):
                        tracklet_len[start_idx] = [{
                                # Store max length
                                'len': idx_max,
                                # Index each bbox with corresponding frame index
                                'bboxes': list(enumerate(tracklet['boxes'], start=start_idx))
                            }]
                    else:
                        tracklet_len[start_idx].append(
                                {'len':idx_max,
                                 'bboxes': list(enumerate(tracklet['boxes'], start=start_idx))
                            })

    dets = {}

    _FRAME_INDEX = 0
    _BBOX_INDEX = 1
    
    # For each tracklet
    for k, v in tracklet_len.items():
        for frame in v:
            for d in frame['bboxes']:
                # Store the bbox in dict with frame index as key
                if(d[_FRAME_INDEX] not in dets.keys()):
                    dets[d[_FRAME_INDEX]] = [d[_BBOX_INDEX]]
                else:
                    dets[d[_FRAME_INDEX]].append(d[_BBOX_INDEX])
    
    dets_id = {}
    
    # Attach ID to frames with multiple bboxes
    for k, v in dets.items():
        dets_id[k] = list(enumerate(v))

    return dets_id

# Post processing

def process_tracklets(tracklets, confidence, video_name):
    
    annotation = {}
    annotation['video'] = video_name
    annotation['annotations'] = []

    for k, v in tracklets.items():

        entry = {}
        entry['frame_id'] = k
        entry['detections'] = []
        
        c = confidence[k]

        for i, det in enumerate(v):
            d = {}
            d['ape_id'] = det[0]
            d['bbox'] = list(normalised_xyxy_to_xyxy(det[1], (720,404)))
            d['score'] = c[i]
            entry['detections'].append(d)

        annotation['annotations'].append(entry)
    
    return annotation
# Bbox conversion 

def normalised_xyxy_to_xyxy(bbox, dims=(720,404)):
    """Normalised [x, y, w, h] from Megadetector to [x1, y1, x2, y2]"""
    width = dims[0]
    height = dims[1]
    
    x1 = bbox[0] * width
    y1 = bbox[1] * height
    
    x2 = (bbox[2]) * width
    y2 = (bbox[3]) * height
        
    return x1, y1, x2, y2

# Video processing 

def random_colour():
    colour = [int(x) for x in np.random.choice(range(256), size=3)]
    return tuple(colour)

def get_colour_list(num):
    colours = []
    for i in range(0, num+1):
        colours.append(random_colour())
    return colours

def get_no_frames(filelist):
    video_name = filelist[0].split('_')[-0]
    frames = len([f for f in filelist if f.endswith('.jpg')])
    labels = len([l for l in filelist if l.endswith('.xml')])
    if(frames == labels):
        return frames
    else:
        print(f"Error: The number of frames and labels in {video_name} do not match!")

def get_ape_no(dets):
    ape_num = 0
    for k, v in dets.items():
        for x in v:
            if(x[0] > ape_num):
                ape_num = x[0]
    return ape_num

def stitch_to_video(video, dets):
    
    img_array = []

    path = '/home/dl18206/Desktop/mres/summer_project/project/data/pan_african_dataset/data'
    rgb = 'rgb'

    ape_no = get_ape_no(dets)
    colours = get_colour_list(ape_no)
    
    filelist = os.listdir(f"{path}/{rgb}/{video}")
    no_of_frames = get_no_frames(filelist)

    for i in range(1, no_of_frames):
        
        filename = f"{path}/{rgb}/{video}/{video}_frame_{i}.jpg"

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        
        if(i-1 in dets.keys()):
            bboxes = dets[i-1]
            # Get bndbxs for each detection
            for i, bbox in enumerate(bboxes):
                ape_id = bbox[0]
                
                bbox = normalised_xyxy_to_xyxy(bbox[1])
                bbox = list(map(float, bbox))
                xmin, ymin, xmax, ymax = bbox
                 
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colours[ape_id], 2)
                cv2.putText(img, f"ape: {ape_id}", (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[ape_id], 2)
                
        img_array.append(img)

    out = cv2.VideoWriter(
        f"{video}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, size
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def parse_args():

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--results_file',type=str, help='pickled results file')
    parser.add_argument('--video', type=str, help='name of video to track')
    parser.add_argument('--confidence', type=float, 
            help='confidence threshold for detections')
    parser.add_argument('--distance_matrix', type=str, 
            help='specify which distance matric to use (i.e. euclidean or iou)')
    parser.add_argument('--length', type=int,
            help='tracklets less than this will be discarded')
    parser.add_argument('--write', type=int, help='Write to video (i.e. True)')
    args = parser.parse_args()
    return args
    
def main():
 
    args = parse_args()

    formatted_results = format_results(args.results_file)
    
    if(os.path.isdir(args.video)):
        videos = [x.rstrip('.mp4') for x in os.listdir(args.video)]
    else:
        videos = [args.video]
    
    for video in tqdm.tqdm(videos):

        video_detections = get_video_results(formatted_results, video)
        tracklets = track(detections=video_detections, 
                conf_thresh=args.confidence,
                distance_matrix=args.distance_matrix
            )
        
        confidence = get_confidences(video_detections, args.confidence)
        
        id_tracklets = id2_tracklets(tracklets=tracklets, length_thresh=args.length)

        # process_tracklets
        final = process_tracklets(id_tracklets, confidence, video)
        
        if(args.write):
            stitch_to_video(video=video, dets=id_tracklets)
        
        with open('result_file.pkl', 'wb') as handle:
            pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
