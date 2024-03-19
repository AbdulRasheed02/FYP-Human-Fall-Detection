import numpy as np
import scipy.signal


def compute_spatial_temporal_loss(input):
    loss_kernal = [
        [[0.0, -1.0, 0.0], [-1.0, -1.0, -1.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [-1.0, 14.0, -1.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [-1.0, -1.0, -1.0], [0.0, -1.0, 0.0]],
    ]
    loss_kernal = np.array(loss_kernal)
    for i in range(input.shape[1]):
        image = input[0, i, :, :, :].data.cpu().numpy()
        convlved = scipy.signal.fftconvolve(image, loss_kernal, mode="same")
        adjactent_loss = sum(sum(sum(convlved)))
    return adjactent_loss
