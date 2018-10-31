from PIL import Image
from torchvision import transforms

"""Pre-computed channel statistics for ImageNet, as prescirbed by the torchvision
pre-trained model dicumentation"""
imagenet_mean = [0.485, 0.456, 0.406] 
imagenet_std = [0.229, 0.224, 0.225]
inv_mean = [-x for x in imagenet_mean]
inv_std = [1/x for x in imagenet_std]

def get_imagenet_labels(label_path='./data/labels.txt'):
    """Load in a list of ImageNet classes, and return functions which can
    convert between class index and human-readable class labels."""
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            class_labels.append(line[:-1])

    def idx_to_label_fn(class_idx):
        return class_labels[class_idx].split(',')[0]

    def label_to_idx_fn(class_label):
        for idx, label in enumerate(class_labels):
            if class_label in label.split(', '):
                return idx
        return -1

    return idx_to_label_fn, label_to_idx_fn

def imagenet_preprocess(img, size=(224, 224)):
    """Convert a single PIL Image to a tensor, normalized according to ImageNet
    statistics."""
    img = transforms.functional.resize(img, size)
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, imagenet_mean, imagenet_std)
    return img

def imagenet_deprocess(img):
    """Convert a single PyTorch Tensor to a PIL Image, de-normalized according 
    to ImageNet statistics."""
    img = transforms.functional.normalize(img, [0, 0, 0], inv_std)
    img = transforms.functional.normalize(img, inv_mean, [1, 1, 1])
    img = transforms.functional.to_pil_image(img)
    return img

def load_image(path):
    """Load an image from disk and convert to a Tensor suitable for an ImageNet
    pre-trained classifier."""
    img = Image.open(path).convert('RGB')
    return imagenet_preprocess(img), img
