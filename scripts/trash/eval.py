from models.ggcnn_keras import ggcnn
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt


rgb_img = imread("pcd0100r.png")
h,w = rgb_img.shape[0], rgb_img.shape[1]
rgb_img = rgb_img[int((h-300)/2):int((h+300)/2), int((w-300)/2):int((w+300)/2)]
rgb_img = np.expand_dims(rgb_img, axis=0)

model = ggcnn().model()
model.load_weights('models/ggcnn_weight.h5')

if __name__ == "__main__":
    pred = model.predict(rgb_img)
    pos_out = pred[..., 0]
    plt.imshow(pos_out[0])
    plt.show()
