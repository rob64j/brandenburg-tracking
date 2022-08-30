import os
import re

import pickle
import json
import mmcv
import torch
import argparse
import tqdm

import numpy as np
import pandas as pd

from torchvision.ops import nms
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


def format_brand_results(results_file):
    with open(results_file, 'rb') as handle:
        detections = json.load(handle)

    new_dict = {'filename': [], 'result': []}
    for image in detections['images']:
        image_res = []
        for det in image['detections']:
            result = det['bbox']
            result.append(det['conf'])
            image_res.append(result)
        res_arr = np.array(image_res, dtype=np.float32)
        if res_arr.any():
            new_dict['filename'].append(image['file'])
            new_dict['result'].append(res_arr)
    df = pd.DataFrame(new_dict)
    df['video'] = results_file.split('.')[0].split('/')[-1]
    df['frame'] = df.filename.str.split('frame').str[1].str.split('.').str[0]
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
        video = re.split('.mp4|.MP4|.avi|.AVI', video.split('/')[-1])[0]
    except:
        video = re.split('.mp4|.MP4|.avi|.AVI', video)[0]

    video_df = df[df.video == video]
    video_dict = video_df.to_dict()

    return [v for k, v in video_dict['result'].items()]


def apply_nms(detection, thresh):
    dets = torch.tensor(detection)
    scores = dets[:, -1:].squeeze(dim=1)
    bboxes = dets[:, :-1]
    idxs = nms(bboxes, scores, thresh)
    return detection[idxs]


def nms2frames(detections, thresh):
    for i in range(len(detections)):
        d = apply_nms(detections[i], thresh)
        if (d.ndim != 2):
            d = [d]
        detections[i] = d
    return detections


def to_mot(detection, animal_class):
    det = {'bbox': (detection[0], detection[1], detection[2], detection[3]),
           'score': detection[-1],
           'class': animal_class}
    return det


def mot2frames(detections, animal_class):
    all_dets = []
    for i in range(len(detections)):
        frame_dets = []
        for j in range(len(detections[i])):
            d = to_mot(detections[i][j], animal_class)
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
    ious = np.zeros((len(atlbrs), len(btlbrs)))
    if ious.size == 0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )
    return ious


def ious_distance(atlbrs, btlbrs):
    """
    compute cost based on IoU
    :param atlbrs:
    :param btlbrs:
    :return:
    """
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def normalised_xywh_to_xywh(bbox, dims):
    """Normalised [x, y, w, h] from Megadetector to [x1, y1, x2, y2]"""
    width = dims[0]
    height = dims[1]

    x1 = bbox[0] * width
    y1 = bbox[1] * height

    x2 = (bbox[0] + bbox[2]) * width
    y2 = (bbox[1] + bbox[3]) * height

    return x1, y1, x2, y2


def _iou(bbox1, bbox2, vdim):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in normalised_xywh_to_xywh(bbox1, vdim)]
    bbox2 = [float(x) for x in normalised_xywh_to_xywh(bbox2, vdim)]

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
            if (f['score'] > conf_thresh):
                conf.append(f['score'])
        confidence.append(conf)
    return confidence


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min, vdim):
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
                best_match = max(dets, key=lambda x: _iou(track['bboxes'][-1], x['bbox'], vdim))
                if _iou(track['bboxes'][-1], best_match['bbox'], vdim) >= sigma_iou:
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

    sorted_tracks = sorted(tracks_finished, key=lambda d: d['max_score'], reverse=True)
    if not sorted_tracks:
        return []
    else:
        return [sorted_tracks[0]]


def id2_tracklets(tracklets, length_thresh):
    '''
    Newest iteration of this function
    '''
    for id, track in enumerate(tracklets):
        track['boxes'] = list(enumerate(track['bboxes'], start=track['start_frame']))
        track['id'] = id

    dets = {}

    for track in tracklets:
        id = track['id']

        for box in track['boxes']:
            entry = [(id, box[1])]

            if (box[0] not in dets.keys()):
                dets[box[0]] = entry
            else:
                dets[box[0]].extend(entry)

    # Sort dict by key / frame
    new_dict = {}
    for key in sorted(dets.keys()):
        new_dict[key] = dets[key]

    return new_dict


# Post processing

def process_tracklets(tracklets, confidence, video_name, vdim, species):
    annotation = {}
    annotation['video'] = video_name
    annotation['species'] = species
    annotation['annotations'] = []

    for k, v in tracklets.items():
        entry = {}
        entry['frame_id'] = k * 5
        entry['detections'] = []

        c = confidence[k - 1]

        for i, det in enumerate(v):
            d = {}
            d['id'] = det[0]
            d['bbox'] = list(normalised_xywh_to_xywh(det[1], vdim))
            d['score'] = c[i]
            entry['detections'].append(d)

        annotation['annotations'].append(entry)

    return annotation


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--detection_path', type=str,
                        help='path to json detection files')
    parser.add_argument('--video_path', type=str,
                        help='path to videos to track')
    parser.add_argument('--l_confidence', type=float,
                        help='lower confidence threshold for detections')
    parser.add_argument('--h_confidence', type=float,
                        help='higher confidence threshold for detections')
    parser.add_argument('--iou', type=float,
                        help='IoU threshold for NMS and tracking')
    parser.add_argument('--length', type=int,
                        help='tracklets less than this will be discarded')
    parser.add_argument('--outpath', type=str,
                        help='specify path to write results to')
    parser.add_argument('--animal_class', type=str, help='class of animal')
    args = parser.parse_args()
    return args


def check_res(results_file):
    with open(results_file, 'rb') as handle:
        detections = json.load(handle)
    for frame in detections['images']:
        if frame['detections']:
            return True
    return False


def main():
    args = parse_args()
    videos = os.listdir(args.video_path)
    for video in tqdm.tqdm(videos):
        if 'DS_Store' not in video:
            results_file = f"{args.detection_path}/{video}".split('.')[0] + '.json'

            # If MegaDetector has no detections, generate an empty tracking file
            if not check_res(results_file):
                final = {'video': video, 'species': args.animal_class, 'annotations': []}
                write_out(args, final, video)
                continue

            formatted_results = format_brand_results(results_file)

            video_detections = get_video_results(formatted_results, f"{args.video_path}/{video}")
            video_detections = nms2frames(video_detections, args.iou)
            video_detections = mot2frames(video_detections, args.animal_class)

            v = mmcv.VideoReader(f"{args.video_path}/{video}")
            vid_dims = v.width, v.height
            tracklets = track_iou(video_detections, args.l_confidence, args.h_confidence, args.iou, args.length,
                                  vid_dims)
            confidence = get_confidences(video_detections, args.l_confidence)

            id_tracklets = id2_tracklets(tracklets=tracklets, length_thresh=args.length)

            # process_tracklets
            final = process_tracklets(id_tracklets, confidence, video, vid_dims, args.animal_class)

            write_out(args, final, video)


def write_out(args, final, video):
    ## Write to PKL.
    if 'pkl' not in os.listdir(f"{args.outpath}"):
        os.mkdir(f"{args.outpath}/pkl")
        os.mkdir(f"{args.outpath}/json")
    outfile = f"{args.outpath}/pkl/{re.split('.mp4|.MP4|.avi|.AVI', video.split('/')[-1])[0]}_track.pkl"
    with open(outfile, 'wb') as handle:
        pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ## Write to JSON.
    outfile = f"{args.outpath}/json/{re.split('.mp4|.MP4|.avi|.AVI', video.split('/')[-1])[0]}_track.json"
    json_obj = json.loads(json.dumps(final, default=str))
    with open(outfile, 'w', encoding='utf-8') as handle:
        json.dump(json_obj, handle, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
