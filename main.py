import numpy as np
from pyqtgraph.Qt import QtCore, QtGui,QtWidgets
import pyqtgraph as pg
import cv2 as cv

pg.setConfigOptions(imageAxisOrder='row-major')
app = pg.mkQApp()

win = QtGui.QMainWindow()
win.resize(800, 600)

win2 = QtGui.QMainWindow()
win2.resize(800, 600)

timer = pg.QtCore.QTimer()

imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('Camera')

imv2 = pg.ImageView()
pg.setConfigOptions(antialias=True)
win2.setCentralWidget(imv2)
win2.show()
win2.setWindowTitle('FFT')

cap = cv.VideoCapture(0)

#set aspect ratio 1,2,3,4... choose 1 for better SNR.
aspect_ratio=1

def update():
    ret, frame = cap.read()

    frame = cv.resize(frame, (frame.shape[1] // aspect_ratio, frame.shape[0] // aspect_ratio))
    frame = cv.flip(frame, 1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    invert = cv.bitwise_not(gray)
    imv.setImage(invert)

    f = np.fft.fft2(invert)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    ms = magnitude_spectrum
    imv2.setImage(ms, levels=[9,12])

    cmap = pg.colormap.get('CET-L7')
    imv2.setColorMap(cmap)

timer.timeout.connect(update)
timer.start(10)

if __name__ == '__main__':
    pg.mkQApp().exec_()

cap.release()
