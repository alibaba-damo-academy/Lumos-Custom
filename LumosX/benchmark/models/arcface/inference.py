from PIL import Image
import argparse
import cv2
import numpy as np
import torch

from .backbones import get_model

def load_image(pil_image):
    img = np.array(pil_image)
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img

def compute_cosine_similarity(features):
    norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_features, norm_features.T)
    return similarity_matrix

def build_arcface_model(network, weight):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    return net

@torch.no_grad()
def extract_arcface_embeddings(image_list, model):
    images = torch.stack([load_image(i) for i in image_list])
    if torch.cuda.is_available():
        images = images.to("cuda")
    features = model(images).cpu()
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='weight/backbone.pth')
    parser.add_argument('--image_list', type=list, default=['samples/donald_trump_front.png', 'samples/donald_trump_side.png', 'samples/yann_lecun_front.png', 'samples/yann_lecun_side.png'])
    args = parser.parse_args()

    arcface_model = build_arcface_model(args.network, args.weight)
    features = run_arcface_inference(args.image_list, arcface_model)

    cosine_similarity_matrix = compute_cosine_similarity(features)
    print(cosine_similarity_matrix)
