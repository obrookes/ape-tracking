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

from torchvision.ops import nms
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
    Output: dict with results for specified video
    '''
    try:
        video = video.split('/')[-1].split('.mp4')[0]
    except:
        video = video.split('.mp4')[0]
    
    video_df = df[df.video==video]
    video_dict = video_df.to_dict()
    
    return [v[0][0] for k, v in video_dict['result'].items()]

def apply_nms(detection, thresh):
    dets = torch.tensor(detection)
    scores = dets[:,-1:].squeeze(dim=1)
    bboxes = dets[:,:-1]
    idxs = nms(bboxes, scores, thresh)
    return detection[idxs]

def nms2frames(detections, thresh):
    for i in range(len(detections)):
        d = apply_nms(detections[i], thresh)
        if(d.ndim != 2):
            d = [d]
        detections[i] = d     
    return detections

def to_mot(detection):
    det = {}
    det['bbox'] = (detection[0], detection[1], detection[2], detection[3])
    det['score'] = detection[-1]
    det['class'] = 'ape'
    return det

def mot2frames(detections):
    all_dets = []
    for i in range(len(detections)):
        frame_dets = []
        for j in range(len(detections[i])):
            d = to_mot(detections[i][j])
            frame_dets.append(d)
        all_dets.append(frame_dets)
    return all_dets

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

def _iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

# Tracking

def get_confidences(detections, conf_thresh):
    confidence = []
    for i, frame in enumerate(detections):
        conf = []
        for f in frame:
            if(f['score'] > conf_thresh):
                conf.append(f['score'])
        confidence.append(conf)
    return confidence

def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    Args:
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.
    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: _iou(track['bboxes'][-1], x['bbox']))
                if _iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])
                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished

def id2_tracklets(tracklets, length_thresh):
    '''
    Newest iteration of this function
    '''
    for ape_id, track in enumerate(tracklets):
        track['boxes'] = list(enumerate(track['bboxes'], start=track['start_frame']))
        track['ape_id'] = ape_id
       
    dets = {}
    
    for track in tracklets:
        ape_id = track['ape_id']
        
        for box in track['boxes']:
            entry = [(ape_id, box[1])]

            if(box[0] not in dets.keys()):
                dets[box[0]] = entry
            else:
                dets[box[0]].extend(entry)
    
    # Sort dict by key / frame
    new_dict = {}
    for key in sorted(dets.keys()):
        new_dict[key] = dets[key]
    
    return new_dict

# Post processing

def process_tracklets(tracklets, confidence, video_name):
    
    annotation = {}
    annotation['video'] = video_name
    annotation['annotations'] = []

    for k, v in tracklets.items():
        entry = {}
        entry['frame_id'] = k
        entry['detections'] = []
        
        c = confidence[k-1]

        for i, det in enumerate(v):
            d = {}
            d['ape_id'] = det[0]
            d['bbox'] = list(det[1])
            d['score'] = c[i]
            entry['detections'].append(d)

        annotation['annotations'].append(entry)
    
    return annotation

def parse_args():

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--detection_path',type=str, 
            help='path to pickled detection files')
    parser.add_argument('--video_path', type=str, 
            help='path to videos to track')
    parser.add_argument('--l_confidence', type=float, 
            help='lower confidence threshold for detections')
    parser.add_argument('--h_confidence', type=float, 
            help='higher confidence threshold for detections')
    parser.add_argument('--length', type=int,
            help='tracklets less than this will be discarded')
    parser.add_argument('--outpath', type=str,
            help='specify path to write results to')
    args = parser.parse_args()
    return args
    
def main():
 
    args = parse_args()
    
    if(os.path.isdir(args.video_path)):
        videos = [x.split('.mp4')[0] for x in os.listdir(args.video_path)]
    else:
        videos = [args.video.split('/')[-1].split('.mp4')[0]]
    
    for video in tqdm.tqdm(videos):
        
        results_file = f"{args.detection_path}/{video}.pkl"
        formatted_results = format_results(results_file)
 
        video_detections = get_video_results(formatted_results, f"{args.video_path}/{video}.mp4")
        video_detections = nms2frames(video_detections, 0.5) 
        video_detections = mot2frames(video_detections)
         
        tracklets = track_iou(video_detections, args.l_confidence, args.h_confidence, 0.75, args.length)
        confidence = get_confidences(video_detections, args.l_confidence)
        
        id_tracklets = id2_tracklets(tracklets=tracklets, length_thresh=args.length)
         
        # process_tracklets
        final = process_tracklets(id_tracklets, confidence, video)
        
        outfile = f"{args.outpath}/{video.split('/')[-1].split('.mp4')[0]}_track.pkl"
        
        with open(outfile, 'wb') as handle:
            pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()


