import argparse
import configparser
import glob
import json
import os
import shutil
import math
from random import shuffle


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json_obj = json.loads(json.dumps(data, default=str))
        json.dump(json_obj, f, ensure_ascii=False, indent=4)


def update_species_dict(species):
    try:
        with open(raw_data_dir + '/species_dict.json', 'rb') as f:
            species_dict = json.load(f)
        if species not in species_dict.keys():
            species_dict[species] = len(species_dict)
            write_json(raw_data_dir + '/species_dict.json', species_dict)

    except:
        species_dict = dict()
        species_dict[species] = 0
        write_json(raw_data_dir + '/species_dict.json', species_dict)


def sort_raw_data():
    for d in os.listdir(raw_data_dir):
        if '.' not in d:
            if ' ' in d:
                os.rename(raw_data_dir + '/' + d, raw_data_dir + '/' + d.replace(' ', '_'))
                d = d.replace(' ', '_')
            conts = os.listdir(raw_data_dir + '/' + d)
            curr_dir = raw_data_dir + '/' + d
            update_species_dict(d.lower())
            organise_raw_data_directory_structure(conts, curr_dir)
            move_annotations_and_videos(conts, curr_dir, d)


def move_annotations_and_videos(conts, curr_dir, d):
    for item in conts:
        if '.json' in item:
            filepath = raw_data_dir + '/' + d + '/' + item
            os.system('mv ' + filepath + ' ' + curr_dir + '/detections')

        video_exts = ['.mp4', '.MP4', '.avi', '.AVI']
        if any(ext in item for ext in video_exts):
            filepath = raw_data_dir + '/' + d + '/' + item
            os.system('mv ' + filepath + ' ' + curr_dir + '/videos')


def organise_raw_data_directory_structure(conts, curr_dir):
    if 'detections' not in conts:
        os.mkdir(curr_dir + '/detections')
    if 'videos' not in conts:
        os.mkdir(curr_dir + '/videos')
    if 'tracks' not in conts:
        os.mkdir(curr_dir + '/tracks')
    if 'frames' not in conts:
        os.mkdir(curr_dir + '/frames')


def make_tracklets(make_vid):
    for d in os.listdir(raw_data_dir):
        if '.' not in d:
            os.chdir(raw_data_dir + '/' + d)
            print("Creating tracking information for " + d)
            os.system("python "
                      + root + "/track.py \
                        --detection_path=detections --video_path=videos " +
                      " --l_confidence=" + cfg.get('tracking', 'l_confidence') +
                      " --h_confidence=" + cfg.get('tracking', 'h_confidence') +
                      " --iou=" + cfg.get('tracking', 'iou') +
                      " --length=" + cfg.get('tracking', 'length') +
                      " --outpath=tracks --animal_class=" + d.lower())

            if make_vid:
                for track in os.listdir(os.getcwd() + '/tracks/pkl'):
                    title = track.split('_track')[0]
                    vid = glob.glob(f"{os.getcwd() + '/videos/' + title}*")[0].split('videos/')[-1]
                    os.system("python "
                              + root + "/brandenburg-tracking/make_video.py \
                                --frame_path=frames \
                                --results=tracks/pkl/" + track + " \
                                --video=videos/" + vid)


def make_subfolders(path):
    os.mkdir(path + '/train')
    os.mkdir(path + '/validation')
    os.mkdir(path + '/test')


def sort_data(train, val, test, output_dir, loo):
    print('------------------------------')
    print("Sorting data to directory: " + output_dir)
    print(f"Train/Validation/Test = {str(train)}/{str(val)}/{str(test)}")
    clean_output_dir(output_dir)

    for d in os.listdir(raw_data_dir):
        if '.' in d:
            continue
        video_path = f"{raw_data_dir}/{d}/videos"
        tracks_path = f"{raw_data_dir}/{d}/tracks/json"
        videos = os.listdir(video_path)

        all_indices = list(range(0, len(videos)))

        test_idx = {'target_dir': 'test', 'indices': []}
        train_idx = {'target_dir': 'train', 'indices': []}
        validation_idx = {'target_dir': 'validation', 'indices': []}

        if loo and d.lower() in output_dir:
            test_idx['indices'] = all_indices

        else:
            num_vals = math.ceil(len(videos) * val)
            num_test = math.ceil(len(videos) * test)
            shuffle(all_indices)
            test_idx['indices'] = all_indices[:num_test]
            validation_idx['indices'] = all_indices[num_test:num_vals + num_test]
            train_idx['indices'] = all_indices[num_test + num_vals:]

            # Test and validation sets should not contain empty annotations.
            test_idx, validation_idx, train_idx = update_with_non_empty(test_idx, tracks_path, validation_idx, train_idx,
                                                                    videos)
        move_files(test_idx, tracks_path, train_idx, validation_idx, video_path, videos, output_dir)


def move_files(test_idx, tracks_path, train_idx, validation_idx, video_path, videos, output_dir):
    for group in [test_idx, validation_idx, train_idx]:
        target_dir = group['target_dir']
        for index in group['indices']:
            if '.DS_Store' in videos[index]:
                continue
            v_path = f"{video_path}/{videos[index]}"
            a_path = f"{tracks_path}/{videos[index].split('.')[0]}_track.json"
            vo_path = f"{output_dir}/data/{target_dir}/{videos[index]}"
            ao_path = f"{output_dir}/annotations/{target_dir}/{videos[index].split('.')[0]}_track.json"
            shutil.copyfile(v_path, vo_path)
            shutil.copyfile(a_path, ao_path)


def find_non_empty_annotation(train_idx, tracks_path, videos):
    for idx in train_idx['indices']:
        with open(f"{tracks_path}/{videos[idx].split('.')[0]}_track.json", 'r') as f:
            annotation = json.load(f)
            if annotation['annotations']:
                return idx


def update_with_non_empty(test_idx, tracks_path, validation_idx, train_idx, videos):
    for group in [test_idx, validation_idx]:
        for idx in group['indices']:
            with open(f"{tracks_path}/{videos[idx].split('.')[0]}_track.json", 'r') as f:
                annotation = json.load(f)
                if not annotation['annotations']:
                    new_idx = find_non_empty_annotation(train_idx, tracks_path, videos)
                    group['indices'] = [x if x != idx else new_idx for x in group['indices']]
                    train_idx['indices'] = [x if x != new_idx else idx for x in train_idx['indices']]
    return test_idx, validation_idx, train_idx


def sort_for_loo(train, val, test):
    with open(raw_data_dir + '/species_dict.json', 'r') as f:
        s_dict = json.load(f)
        species = [x for x in s_dict.keys()]

    if not os.path.isdir(output_data_dir):
        os.mkdir(output_data_dir)

    for s in species:
        dir_path = output_data_dir + '/no_' + s
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        out_dir = dir_path
        sort_data(train, val, test, out_dir, True)


def copy_output_data(train, val, test, loo):
    if loo:
        sort_for_loo(train, val, test)
    else:
        sort_data(train, val, test, output_data_dir, False)


def clean_output_dir(output_dir):
    try:
        shutil.rmtree(output_dir)
    except:
        pass
    os.mkdir(output_dir)
    for directory in ['data', 'annotations']:
        os.mkdir(output_dir + '/' + directory)
        make_subfolders(output_dir + '/' + directory)
    shutil.copyfile(raw_data_dir + '/species_dict.json', output_dir + '/species_dict.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-Processing')
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    root = os.getcwd()
    raw_data_dir = cfg.get('program', 'raw_data_dir')
    output_data_dir = cfg.get('program', 'output_data_dir')
    sort_raw_data()
    if cfg.getboolean('program', 'make_tracklets'):
        make_tracklets(cfg.getboolean('program', 'make_video'))
    copy_output_data(cfg.getfloat('data', 'train'),
                     cfg.getfloat('data', 'val'),
                     cfg.getfloat('data', 'test'),
                     cfg.getboolean('train', 'leave_one_out'))
