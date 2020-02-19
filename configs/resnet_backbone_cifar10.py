import torchvision.transforms as transforms

config = {
    'params': {
        "backbone": {
            "kernel_size": 3,
            "output_dim": 128,
            "input_dim": 3,
            "stride": 2,
            "padding": 1,
            "out_img_size": 16
        },
        "primary_capsules": {
            "kernel_size": 1,
            "stride": 1,
            "input_dim": 128,
            "caps_dim": 16,
            "num_caps": 32,
            "padding": 0,
            "out_img_size": 16
        },
        "capsules": [{
            "type": "CONV",
            "num_caps": 32,
            "caps_dim": 16,
            "kernel_size": 3,
            "stride": 2,
            "matrix_pose": True,
            "out_img_size": 7
        }, {
            "type": "CONV",
            "num_caps": 32,
            "caps_dim": 16,
            "kernel_size": 3,
            "stride": 1,
            "matrix_pose": True,
            "out_img_size": 5
        }],
        "class_capsules": {
            "num_caps": 10,
            "caps_dim": 16,
            "matrix_pose": True
        },
    },
    "transform_train":
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]),
    "transform_test":
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
}
