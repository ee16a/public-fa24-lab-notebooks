import numpy as np

def test1b_H(H):
    H_correct = np.eye(25)
    if np.isfinite(H).all() and np.array_equal(H, H_correct):
        print("H mask matrix is correct")
    else:
        print("H mask matrix is incorrect")

def test1b_H_alt(H_alt):
    if np.isfinite(H_alt).all() and H_alt.shape == (25, 25):
        print("H_alt mask matrix is correct")
    else:
        print("H_alt mask matrix is incorrect")

def test_masks_img2(H, H_Alt):

    errors = False
    if H.shape != (30*40, 30*40):
        errors = True
        print('H shape is incorrect: H.shape = {}'.format(H.shape))
    try:
        np.linalg.inv(H)
    except np.linalg.LinAlgError:
        errors = True
        print('H is not invertible')

    if H_Alt.shape != (30*40, 30*40):
        errors = True
        print('H_Alt shape is incorrect: H_Alt.shape = {}'.format(H_Alt.shape))
    try:
        np.linalg.inv(H_Alt)
    except np.linalg.LinAlgError:
        errors = True
        print('H_Alt is not invertible')
    if (errors):
        print("\nPlease fix any errors before moving on.")
    else:
        print("H and H_Alt are the correct dimension and are both invertible. Proceed to the next step")
