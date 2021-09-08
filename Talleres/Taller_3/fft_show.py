# Clase FFT
import cv2
import numpy as np

""" FFT class to properly visualize fft of gray image

"""

### Clase FFT: encargada de realizar la transformada de fourier de la imagen con n muestras y visualizarla
class FFT:
    # Atributos de la clase
    image_gray = []
    image_gray_fft = []
    image_gray_fft_shift = []
    image_gray_fft_mag = []
    image_fft_view = []
    # Constructor de la clase
    def __init__(self, image, n):
        shape = image.shape
        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        image_with_zeros = np.zeros((max(n, shape[0]), max(n, shape[1])), dtype=np.float)
        image_with_zeros[0:shape[0], 0:shape[1]] = np.copy(image_gray)
        self.image_gray = image_with_zeros
    # Método display
    def display(self):
        self.image_gray_fft = np.fft.fft2(self.image_gray)
        self.image_gray_fft_shift = np.fft.fftshift(self.image_gray_fft)
        self.image_gray_fft_mag = np.absolute(self.image_gray_fft_shift)
        self.image_fft_view = np.log(self.image_gray_fft_mag + 1)
        self.image_fft_view = self.image_fft_view / np.max(self.image_fft_view)
        # cv2.imshow("FFT", self.image_fft_view)
        # cv2.waitKey(0)
