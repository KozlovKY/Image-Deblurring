import albumentations as A
from albumentations.pytorch import ToTensorV2


# Custom transformation to crop to dimensions divisible by 32
class CropToDivisibleBy32(A.DualTransform):
    """Crop image to make both height and width divisible by 32"""

    def __init__(self, always_apply=False, p=1.0):
        super(CropToDivisibleBy32, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        # Calculate new dimensions divisible by 32
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        # Calculate crop coordinates (center crop)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2

        return img[start_y : start_y + new_h, start_x : start_x + new_w]

    def apply_to_bbox(self, bbox, **params):
        # Handle bounding boxes if needed (not used in our case)
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        # Handle keypoints if needed (not used in our case)
        return keypoint

    def get_transform_init_args_names(self):
        return ()


CNN_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
CNN_NORMALIZATION_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = A.Compose(
    [
        A.RandomCrop(width=256, height=256, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf(
            [
                A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-5, 5), p=1.0),
                A.NoOp(p=1.0),
            ],
            p=0.5,
        ),
        A.Normalize(
            mean=CNN_NORMALIZATION_MEAN, std=CNN_NORMALIZATION_STD, always_apply=True
        ),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

TEST_TRANSFORMS = A.Compose(
    [
        A.CenterCrop(width=256, height=256, always_apply=True),
        A.Normalize(
            mean=CNN_NORMALIZATION_MEAN, std=CNN_NORMALIZATION_STD, always_apply=True
        ),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

EVAL_TRANSFORMS = A.Compose(
    [
        CropToDivisibleBy32(
            always_apply=True
        ),  # Crop to make dimensions divisible by 32
        A.Normalize(
            mean=CNN_NORMALIZATION_MEAN, std=CNN_NORMALIZATION_STD, always_apply=True
        ),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
