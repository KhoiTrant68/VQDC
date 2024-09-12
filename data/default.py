from easydict import EasyDict

# Use a base path and format strings for clarity and potential efficiency
base_data_path = "D:"
imagenet_path = f"{base_data_path}/imagenet-mini"

DefaultDataPath = EasyDict()
DefaultDataPath.ImageNet = EasyDict(
    root=imagenet_path,
    train_write_root=f"{imagenet_path}/train",
    val_write_root=f"{imagenet_path}/val",
)
