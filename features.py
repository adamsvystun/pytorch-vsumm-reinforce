from __future__ import print_function
import os
import numpy as np
import h5py
import caffe
import argparse
import sys
import gc
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser("Code to extract caffe features")
parser.add_argument('-r', '--root-path', type=str, required=True, help="Path to where to find frame folders (required)")
parser.add_argument('-s', '--save-path', type=str, required=True, help="Path to where to save results (required)")
parser.add_argument('-c', '--caffe-root', type=str, required=True, help="Path to caffe root dir (required)")
parser.add_argument('--use-cpu', action='store_true', help="Use CPU, instead of GPU")

args = parser.parse_args()


def get_model(caffe_root, gpu=True):
    model_file = caffe_root + "models/bvlc_googlenet/deploy.prototxt"
    pretrained = caffe_root + "models/bvlc_googlenet/bvlc_googlenet.caffemodel"
    if not os.path.isfile(pretrained):
        print("PRETRAINED Model not found.")
    if not gpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained, caffe.TEST)
    net.blobs["data"].reshape(1, 3, 224, 224)
    return net


def get_transformer(caffe_root, model):
    mu = np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy")
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called "data"
    transformer = caffe.io.Transformer({"data": model.blobs["data"].data.shape})

    transformer.set_transpose("data", (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean("data", mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale("data", 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap("data", (2, 1, 0))  # swap channels from RGB to BGR
    return transformer


def get_features(frames, net, transformer):
    features = []
    for frame in frames:
        transformed_image = transformer.preprocess("data", frame)
        net.blobs["data"].data[0] = transformed_image
        net.forward()
        temp = net.blobs["pool5/7x7_s1"].data[0]
        temp = temp.squeeze().copy()
        features.append(temp)
        print(".", end="")
        sys.stdout.flush()
    features = np.array(features)
    features = normalize(features)
    print("")
    return features


def get_frames(frames_paths):
    frames = []
    for frame_path in frames_paths:
        frame = caffe.io.load_image(frame_path)
        frames.append(frame)
        print(".", end="")
        sys.stdout.flush()
    print("")
    return frames


def save_file(save_path, features):
    data_file = h5py.File(save_path, 'w')
    for i, features_list in enumerate(features):
        data_file.create_dataset(str(i) + "/features", data=features_list)
    data_file.close()


def main(root_path, save_path, caffe_root, gpu=True):
    print("Getting frames from: " + root_path)
    print("Saving to: " + save_path)

    net = get_model(caffe_root=caffe_root, gpu=gpu)
    transformer = get_transformer(caffe_root=caffe_root)

    features_array = []
    video_frame_folders = next(os.walk(root_path))[1]
    video_frame_folders.sort()
    print("Number of videos to process: ", len(video_frame_folders))
    for n, video_folder in enumerate(video_frame_folders):
        _, _, images = next(os.walk(root_path + "/" + video_folder, followlinks=True), (None, None, []))
        images = [i for i in images if i.endswith(".jpg")]
        print("[S] " + str(n + 1) + " | Images: " + str(len(images)))
        images.sort()
        frames_paths = []
        for i, image_filename in enumerate(images):
            if i % 15 == 0 or i == 0:
                frames_paths.append(root_path + "/" + video_folder + "/" + image_filename)
        print("[>] Reading frames: " + str(len(frames_paths)))
        frames = get_frames(frames_paths)
        print("[>] Getting features ...")
        features = get_features(frames, net, transformer)
        print("[>] Features shape: " + str(features.shape))
        features_array.append(features)
        print("[F] " + str(n + 1))
        del frames
        del features
        gc.collect()
    save_file(save_path, features_array)


if __name__ == '__main__':
    main(
        root_path=args.root_path,
        save_path=args.save_path,
        caffe_root=args.caffe_root,
        gpu=not args.use_cpu
    )




