from torchvision import transforms

#Image Transforms
def get_train_transform():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # mild for chest X-rays
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transform

def get_eval_transform():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], #The mean and standard deviation of each color channel (R, G, B) across all ImageNet training images. -> normalized_pixel = (pixel - mean) / std
                             [0.229, 0.224, 0.225])
    ])
    return transform

def preprocess_image(pil_image):
    transform = get_eval_transform()
    return transform(pil_image)