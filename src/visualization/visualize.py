# Visualize a feature

from PIL import Image
import numpy as np

def visualize_reconstructed(vec, imgpath):
    w, h = 293, 13
    img_data = np.reshape(vec, (w, h))

    img = Image.fromarray(img_data, mode='1')
    img.save(imgpath)
    img.show()

def visualize_feat_vec_sample(sample_id, encoded=False):
    filepath = './data/interim/features/feat_vectors/{}.npy'.format(str(sample_id))
    npy = np.load(filepath)
    w, h = 293, 13
    img_data = np.reshape(npy, (w, h))

    img = Image.fromarray(img_data, mode='1')
    img.save('./data/interim/features/visuals/{}_{}.png'.format(sample_id, 'encoded' if encoded else 'unencoded'))
    img.show()