import numpy as np
import matplotlib.pyplot as plt
import os
import serial
import serial.tools.list_ports
import signal
import time
import tkinter as tk
from tqdm import tqdm 
import sys


WINDOW_DIMENSION_PX = 720
MASK_TO_READ_SLEEP = 0.1

root = None
window = None
canvas = None

def _exit():
    destroy_screen()
    print('ERROR: Image capture exited prematurely')
    sys.exit(0)
    


def create_screen():
    global root
    root = tk.Tk()
    root.withdraw()
    
    signal.signal(signal.SIGINT, lambda sig, frame: _exit())

    global window
    window = tk.Toplevel()
    window.geometry('1280x720+1920+0')
    window.overrideredirect(True)
    window.state('zoomed')
    window.configure(background='black')
    
    global canvas
    canvas = tk.Canvas(
        window,
        width=WINDOW_DIMENSION_PX,
        height=WINDOW_DIMENSION_PX,
        bd=0,
        highlightthickness=0,
        relief='ridge',
        background="black"
    )
    canvas.pack()


def destroy_screen():
    global root
    if root is not None:
        root.destroy()
    root = None
    global window
    window = None
    global canvas
    canvas = None


def display_mask(mask, brightness=255):
    mask_width = mask.shape[1]
    mask_height = mask.shape[0]

    canvas_width = int(WINDOW_DIMENSION_PX * mask_width / mask_height)
    canvas_height = WINDOW_DIMENSION_PX

    grid_width = int(canvas_width / mask_width)
    grid_height = int(canvas_height / mask_height)

    canvas.delete('all')

    for x in range(mask_width):
        for y in range(mask_height):
            if mask[y, x] != 0:
                canvas.create_rectangle(
                    grid_width * x,
                    grid_height * y,
                    grid_width * (x + 1),
                    grid_height * (y + 1),
                    fill=rgbtohex(brightness, brightness, brightness)
                )
    window.update()


def find_optimal_brightness(ser, mask):
    print('Searching for optimal brightness')
    low = 0
    high = 255

    while low != high:
        low = max(0, low)
        high = min(255, high)
        optimal_brightness = (low + high) // 2
        reading = read_sensor(ser, mask, optimal_brightness)

        if reading < 1000:
            low = optimal_brightness + 1
        else:
            low -= 1
            high = optimal_brightness - 1
        
    optimal_brightness = max(0, min(255, optimal_brightness)) # Bound to [0,255]
    print(f'Found optimal brightness: {optimal_brightness}/255')
    return optimal_brightness


def rgbtohex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'


def get_arduino_port():
    return next(device.device for device in serial.tools.list_ports.comports() if 'Arduino Leonardo' in device.description)


def connect_arduino():
    try:
        arduino_port = get_arduino_port()
    except:
        print('Arduino is not connected')
        _exit()

    try:
        ser = serial.Serial(arduino_port, 115200)
        ser.write(b'9\r\n')

        try:
            ser.write(b'6\r\n')
            ser.flush()
            ser.readline()
            return ser
        except:
            print('Cannot read from Arduino. Check if code was uploaded successfully.')
            _exit()
    except:
        print('Unable to connect to Arduino')
        _exit()


def read_from_arduino(ser):
    ser.write(b'6\r\n')
    ser.flush()
    return int(ser.readline())


def read_sensor(ser, mask, brightness):
    display_mask(mask, brightness)
    time.sleep(MASK_TO_READ_SLEEP)
    return read_from_arduino(ser)


def scan(H, multi_pixel, width, height, brightness=None):
    shape = (height, width)
    create_screen()
    with connect_arduino() as ser:
        if multi_pixel:
            print('Beginning multipixel scan')
            max_mask = H[np.argmax(np.sum(H, axis=1))]
            brightness = find_optimal_brightness(ser, max_mask.reshape((width, height)))
        else:
            print('Beginning single pixel scan')
            H_shape = (width * height, width * height)
            if H.shape != H_shape:
                print(f'Mask matrix shape is incorrect. Expected {H_shape} got {H.shape}')
                _exit()
        
        # Set to max brightness if not specified
        if brightness is None:
            brightness = 255
            
        print('Scanning...')
        for _ in range(10):
            read_sensor(ser, H[0].reshape(shape), brightness)
        raw_pixels = []
        masks = tqdm(H, position=0, unit='scans', leave=True)
        for mask in masks:
            val = read_sensor(ser, mask.reshape(shape), brightness)
            raw_pixels.append(val)
            masks.set_postfix_str(f'value: {val}')
        pixels = np.reshape(raw_pixels, shape)
    destroy_screen()
    
    plt.imshow(pixels, cmap='gray')
    return raw_pixels
    
def reconstruct_multipixel(H, sr, width, height):
    reconstruction = np.linalg.inv(H)@sr
    reconstruction[0] = np.mean(reconstruction[1:])
    plt.imshow(reconstruction.reshape((width, height)), cmap='gray')