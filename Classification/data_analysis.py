import cv2
from matplotlib import pyplot as plt
import numpy as np


def cv_img_visualize(config):

    img = cv2.imread(config['train_dset_path'])

    #########################
    #cv2로 이미지 다루는 것
    #########################
    cv2.imshow('Test Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite('test2.png', img) # 이미지 다른 파일로 저장



def matplot_img_visualize(config):

    #########################
    # matplotlib로 이미지 보기
    #########################
    img = cv2.imread(config['train_dset_path'])
    plt.figure(figsize=(10,30))
    plt.imshow(img)
    plt.show()


def imshow(inp, title=None): # from pytorch tutorial
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()





# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
#
# imshow(out, title=[class_names[x] for x in classes])