from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(is_grayscale: bool = False) -> transforms.Compose:
    transform_list = [
        transforms.ToPILImage(),
    ]

    # If image is grayscale, convert it to 3 channels for pretrained models
    if is_grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    transform_list += [
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(transform_list)
