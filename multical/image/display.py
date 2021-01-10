import cv2
import numpy as np

def to_color(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        return image
    elif image.shape[2] == 4:
        return image[:, :, 0:3]
    else:
        assert False, "Unknown image shape: " + str(image.shape)

def stack_images(images, resize_height=None, rotate=0):

    def scale_height(image):
        if rotate > 0:
            image = np.rot90(image, k=rotate)

        if resize_height != image.shape[0]:
            scale = resize_height / image.shape[0]
            image = cv2.resize(image, (int(image.shape[1] * scale), resize_height))

        return to_color(image)

    resize_height = resize_height or images[0].shape[0]

    images = [scale_height(image) for image in images]
    return np.hstack(images)


def display_stacked(images, resize_height=None, rotate=0, resizeable=True):
    image = stack_images(images, resize_height, rotate)
    display(image, resizeable=resizeable)



def display(t, name="image", resizeable=True, height=800):
    cv2.namedWindow(name, flags=cv2.WINDOW_NORMAL if resizeable else cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, t)

    scale = height / t.shape[0]
    cv2.resizeWindow(name, int(t.shape[1] * scale), height)

    while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) > 0:
        keyCode = cv2.waitKey(1)
        if keyCode >= 0:
            return keyCode

    return -1


