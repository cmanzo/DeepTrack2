import numpy as np
from abc import ABC, abstractmethod
'''
    An optical device that generates the pupil based on input parameters.
    The base device is stateless, but will be extended to allow for aberrations.
'''


class BaseOpticalDevice2D:
    def __init__(self,
                    shape,
                    NA=0.7,
                    wavelength=0.66,
                    pixel_size=0.1,
                    upscale=2):
        self.shape = shape
        self.NA = NA
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.upscale = upscale
    # Calculates the pupil of the optical system using the NA, wavelength and the pixel size.
    def getPupilRadius(self):
        return self.pixel_size * self.NA / self.wavelength

    def getPupil(self, shape=None):
        if shape is None:
            shape = self.shape
        
        upscaled_shape = shape*self.upscale
        R = self.getPupilRadius()
        x_radius = R*upscaled_shape[0]
        y_radius = R*upscaled_shape[1]
        W, H = np.meshgrid(np.arange(0, upscaled_shape[0]), np.arange(0, upscaled_shape[1]))

        pupilMask = ((W - upscaled_shape[0] / 2) / x_radius) ** 2  + ((H - upscaled_shape[1] / 2) / (y_radius) ) **2 <= 1
        if self.upscale > 1:
            pupilMask = np.reshape(pupilMask, (shape[0], self.upscale, shape[1], self.upscale)).mean(axis=(3,1))
        pupil = pupilMask * (1 + 0j)
        return pupil

        

