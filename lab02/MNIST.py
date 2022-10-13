from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = datasets.MNIST("mnist", train=True,
                         transform=transforms.ToTensor(),
                         download=True)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


def img_show(image):
    plt.imshow(image, cmap="gray")
    plt.show()


# show one picture
img, lb = dataset[0]
img = img.reshape(28, 28)
img_show(img)
