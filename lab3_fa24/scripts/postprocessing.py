import numpy as np

def post_process(sr_red, sr_green, sr_blue):
    # the hue of each color channel
    red_coeff = 1
    green_coeff = 0.998
    blue_coeff = 1

    # percentile cutoff for brightest and darkest pixels
    # increasing this is similar to increasing the contrast
    contrast_percentile = 5

    # Don't touch these
    sr = np.zeros((32, 32, 3))
    sr[:,:,0] = np.reshape(sr_red, (32,32)) ** red_coeff
    sr[:,:,1] = np.reshape(sr_green, (32,32)) ** green_coeff
    sr[:,:,2] = np.reshape(sr_blue, (32, 32)) ** blue_coeff

    minval = np.percentile(sr, contrast_percentile)
    maxval = np.percentile(sr, 100 - contrast_percentile)
    sr = np.clip(sr, minval, maxval)
    sr = ((sr - minval) / (maxval - minval)) * 255
    return sr
