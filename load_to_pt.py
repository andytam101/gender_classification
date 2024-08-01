import torch
import cv2
import os
import numpy as np


RESIZED_IMAGE_WIDTH  = 60
RESIZED_IMAGE_HEIGHT = 48


def read_images(path):
    images = []
    
    for file in os.listdir(path):
        img = cv2.resize(cv2.imread(os.path.join(path, file)), (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH))        
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten())

    return torch.tensor(np.array(images), dtype=torch.float32)


def normalise(data):
    return (data / 255)


def combine(female, male):
    female_size = female.size()[0]
    male_size   = male.size()[0]

    zeros = torch.zeros((female_size,1))
    ones  = torch.ones((male_size, 1))

    data = torch.concat((torch.concat((zeros, female), dim=1), torch.concat((ones,  male), dim=1)), dim=0)
    return data


def shuffle(data):
    m = data.size()[0]
    idx = torch.randperm(m)
    return data[idx, :]


def main(name="val_data.pt"):
    print("Reading female images now...")
    female = normalise(read_images("./female_validate"))
    print("Reading male images now...")
    male   = normalise(read_images("./male_validate"))

    print("combining data...")
    data = combine(female, male)

    print("shuffling data...")
    data = shuffle(data)

    torch.save(data, name)

if __name__ == "__main__":
    main()
