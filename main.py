import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFile
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
from skimage import exposure, morphology, transform


class Ui_MainWindow(object):
    original_image = ""
    image = ""
    temp_image = ""
    o_filename = ""

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1121, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(9, 9, 1101, 25))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.imagePath = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.imagePath.setObjectName("imagePath")
        self.horizontalLayout.addWidget(self.imagePath)

        self.chooseImageButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.chooseImageButton.setObjectName("chooseImageButton")
        self.horizontalLayout.addWidget(self.chooseImageButton)
        self.chooseImageButton.clicked.connect(self.openFile)

        self.saveImageButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.saveImageButton.setObjectName("saveImageButton")
        self.horizontalLayout.addWidget(self.saveImageButton)
        self.saveImageButton.clicked.connect(self.saveButtonClicked)

        self.originalImage = QtWidgets.QLabel(self.centralwidget)
        self.originalImage.setGeometry(QtCore.QRect(9, 448, 441, 391))
        self.originalImage.setObjectName("originalImage")

        self.editedImage = QtWidgets.QLabel(self.centralwidget)
        self.editedImage.setGeometry(QtCore.QRect(660, 450, 441, 391))
        self.editedImage.setObjectName("editedImage")

        self.resetImage = QtWidgets.QPushButton(self.centralwidget)
        self.resetImage.setGeometry(QtCore.QRect(523, 450, 75, 23))
        self.resetImage.setObjectName("resetImage")
        self.resetImage.clicked.connect(self.resetImageClicked)

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 50, 371, 221))
        self.groupBox.setObjectName("groupBox")
        self.filterActions = QtWidgets.QComboBox(self.groupBox)
        self.filterActions.setGeometry(QtCore.QRect(10, 40, 351, 22))
        self.filterActions.setObjectName("filterActions")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")
        self.filterActions.addItem("")

        self.applyFilter = QtWidgets.QPushButton(self.groupBox)
        self.applyFilter.setGeometry(QtCore.QRect(150, 90, 75, 23))
        self.applyFilter.setObjectName("applyFilter")
        self.applyFilter.clicked.connect(self.applyFilterClicked)

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(390, 50, 381, 221))
        self.groupBox_2.setObjectName("groupBox_2")
        self.histogramActions = QtWidgets.QComboBox(self.groupBox_2)
        self.histogramActions.setGeometry(QtCore.QRect(10, 40, 351, 22))
        self.histogramActions.setObjectName("histogramActions")
        self.histogramActions.addItem("")
        self.histogramActions.addItem("")

        self.applyHistogram = QtWidgets.QPushButton(self.groupBox_2)
        self.applyHistogram.setGeometry(QtCore.QRect(160, 110, 75, 23))
        self.applyHistogram.setObjectName("applyHistogram")
        self.applyHistogram.clicked.connect(self.applyHistogramClicked)

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(790, 50, 321, 221))
        self.groupBox_3.setObjectName("groupBox_3")
        self.spatialActions = QtWidgets.QComboBox(self.groupBox_3)
        self.spatialActions.setGeometry(QtCore.QRect(10, 40, 291, 22))
        self.spatialActions.setObjectName("spatialActions")
        self.spatialActions.addItem("")
        self.spatialActions.addItem("")
        self.spatialActions.addItem("")
        self.spatialActions.addItem("")
        self.spatialActions.addItem("")
        self.spatialInput = QtWidgets.QLineEdit(self.groupBox_3)
        self.spatialInput.setGeometry(QtCore.QRect(10, 70, 281, 20))
        self.spatialInput.setObjectName("spatialInput")

        self.applySpatial = QtWidgets.QPushButton(self.groupBox_3)
        self.applySpatial.setGeometry(QtCore.QRect(120, 100, 75, 23))
        self.applySpatial.setObjectName("applySpatial")
        self.applySpatial.clicked.connect(self.applySpatialClicked)

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 280, 371, 161))
        self.groupBox_4.setObjectName("groupBox_4")
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox_4)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 30, 331, 80))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 10, 0)
        self.formLayout.setHorizontalSpacing(30)
        self.formLayout.setObjectName("formLayout")
        self.minLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.minLabel.setObjectName("minLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.minLabel)
        self.minLineEdit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.minLineEdit.setObjectName("minLineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.minLineEdit)
        self.maxLabel = QtWidgets.QLabel(self.formLayoutWidget)
        self.maxLabel.setObjectName("maxLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.maxLabel)
        self.maxLineEdit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.maxLineEdit.setObjectName("maxLineEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.maxLineEdit)

        self.applyRescaleIntensity = QtWidgets.QPushButton(self.groupBox_4)
        self.applyRescaleIntensity.setGeometry(QtCore.QRect(140, 120, 75, 23))
        self.applyRescaleIntensity.setObjectName("applyRescaleIntensity")
        self.applyRescaleIntensity.clicked.connect(self.applyRescaleIntensityClicked)

        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(390, 280, 381, 161))
        self.groupBox_5.setObjectName("groupBox_5")
        self.morphologyActions = QtWidgets.QComboBox(self.groupBox_5)
        self.morphologyActions.setGeometry(QtCore.QRect(10, 40, 351, 22))
        self.morphologyActions.setObjectName("morphologyActions")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")
        self.morphologyActions.addItem("")

        self.applyMorphology = QtWidgets.QPushButton(self.groupBox_5)
        self.applyMorphology.setGeometry(QtCore.QRect(160, 110, 75, 23))
        self.applyMorphology.setObjectName("applyMorphology")
        self.applyMorphology.clicked.connect(self.applyMorphologyClicked)

        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(790, 280, 321, 161))
        self.groupBox_6.setObjectName("groupBox_6")
        self.showVideo = QtWidgets.QPushButton(self.groupBox_6)
        self.showVideo.setGeometry(QtCore.QRect(120, 70, 75, 23))
        self.showVideo.setObjectName("showVideo")
        self.showVideo.clicked.connect(self.showVideoClicked)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1121, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.chooseImageButton.setText(_translate("MainWindow", "Choose"))
        self.saveImageButton.setText(_translate("MainWindow", "Save"))
        self.resetImage.setText(_translate("MainWindow", "Reset Image"))
        self.groupBox.setTitle(_translate("MainWindow", "Görüntü İyileştirme İşlemleri, Filtreler"))
        self.filterActions.setItemText(0, _translate("MainWindow", "2D Convolution"))
        self.filterActions.setItemText(1, _translate("MainWindow", "Bilateral"))
        self.filterActions.setItemText(2, _translate("MainWindow", "Gaussian Blur"))
        self.filterActions.setItemText(3, _translate("MainWindow", "Median"))
        self.filterActions.setItemText(4, _translate("MainWindow", "Sobel"))
        self.filterActions.setItemText(5, _translate("MainWindow", "Inverse"))
        self.filterActions.setItemText(6, _translate("MainWindow", "Unsharp"))
        self.filterActions.setItemText(7, _translate("MainWindow", "Canny"))
        self.filterActions.setItemText(8, _translate("MainWindow", "Laplace"))
        self.filterActions.setItemText(9, _translate("MainWindow", "Gabor"))
        self.applyFilter.setText(_translate("MainWindow", "Apply"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Histogram Görüntüleme ve Eşikleme"))
        self.histogramActions.setItemText(0, _translate("MainWindow", "Histogram"))
        self.histogramActions.setItemText(1, _translate("MainWindow", "Histogram Equalization"))
        self.applyHistogram.setText(_translate("MainWindow", "Apply"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Uzaysal Dönüşüm İşlemleri"))
        self.spatialActions.setItemText(0, _translate("MainWindow", "Rotate"))
        self.spatialActions.setItemText(1, _translate("MainWindow", "Resize Height"))
        self.spatialActions.setItemText(2, _translate("MainWindow", "Translate"))
        self.spatialActions.setItemText(3, _translate("MainWindow", "Perspective"))
        self.spatialActions.setItemText(4, _translate("MainWindow", "Affine"))
        self.applySpatial.setText(_translate("MainWindow", "Apply"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Rescale Intensity"))
        self.minLabel.setText(_translate("MainWindow", "Min"))
        self.maxLabel.setText(_translate("MainWindow", "Max"))
        self.applyRescaleIntensity.setText(_translate("MainWindow", "Apply"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Morphology"))
        self.morphologyActions.setItemText(0, _translate("MainWindow", "Erosion"))
        self.morphologyActions.setItemText(1, _translate("MainWindow", "Dilation"))
        self.morphologyActions.setItemText(2, _translate("MainWindow", "Opening"))
        self.morphologyActions.setItemText(3, _translate("MainWindow", "Closing"))
        self.morphologyActions.setItemText(4, _translate("MainWindow", "White Tophat"))
        self.morphologyActions.setItemText(5, _translate("MainWindow", "Black Tophat"))
        self.morphologyActions.setItemText(6, _translate("MainWindow", "Gradient"))
        self.morphologyActions.setItemText(7, _translate("MainWindow", "Skeletonize"))
        self.morphologyActions.setItemText(8, _translate("MainWindow", "Hit And Miss"))
        self.morphologyActions.setItemText(9, _translate("MainWindow", "Cross"))
        self.applyMorphology.setText(_translate("MainWindow", "Apply"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Video Edit"))
        self.showVideo.setText(_translate("MainWindow", "Show"))

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "", "All Files(*);;Python "
                                                                                             "Files (*.py)",
                                                  options=options)
        if filename:
            self.o_filename = os.path.basename(filename)
            pixmap = QtGui.QPixmap(filename)
            self.imagePath.setText(filename)
            self.imagePath.setEnabled(False)
            self.originalImage.setPixmap(pixmap)
            self.original_image = self.image = cv2.imread(filename, 1)
            cv2.imwrite("temp.jpeg", self.image)
            self.showEdited()

    def resetImageClicked(self):
        self.image = self.original_image
        cv2.imwrite("temp.jpeg", self.original_image)
        self.showEdited()
        print("reset")

    def saveButtonClicked(self):
        cv2.imwrite(self.o_filename, self.image)
        print("reset")

    def applyFilterClicked(self):
        current = str(self.filterActions.currentText())
        if current == "2D Convolution":
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            kernel = np.ones((5, 5), np.float32) / 25
            self.image = cv2.filter2D(self.image, -1, kernel)
        elif current == "Bilateral":
            print("wiener selected")
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            self.image = cv2.bilateralFilter(self.image, 9, 75, 75)

        elif current == "Gaussian Blur":
            print("gaussian selected")
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)

        elif current == "Median":
            print("median selected")
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            self.image = cv2.medianBlur(self.image, 5)

        elif current == "Sobel":
            print("sobel selected")
            # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
            self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(image_gray, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(image_gray, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            self.image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        elif current == "Inverse":
            print("inverse selected")
            self.image = cv2.bitwise_not(self.image)

        elif current == "Unsharp":
            print("sharped selected")
            # https://stackoverflow.com/questions/32454613/python-unsharp-mask
            blur = cv2.GaussianBlur(self.image, (9, 9), 10.0)
            self.image = cv2.addWeighted(self.image, 1.5, blur, -0.5, 0, self.image)

        elif current == "Canny":
            print("wiener selected")
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
            self.image = cv2.Canny(self.image, 100, 200)

        elif current == "Laplace":
            print("laplacian selected")
            # https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
            self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=3)

        elif current == "Gabor":
            print("gabor selected")
            # https://gist.github.com/kendricktan/93f0da88d0b25087d751ed2244cf770c
            g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.filter2D(self.image, cv2.CV_8UC3, g_kernel)
        else:
            print("else")

        cv2.imwrite("temp.jpeg", self.image)
        self.showEdited()

    def applyHistogramClicked(self):
        current = str(self.histogramActions.currentText())

        if current == "Histogram":
            # https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
            bgr_planes = cv2.split(self.image)
            histSize = 256
            histRange = (0, 256)
            accumulate = False
            b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
            g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
            r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
            hist_w = 512
            hist_h = 400
            bin_w = int(np.round(hist_w / histSize))
            histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
            cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
            for i in range(1, histSize):
                cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(b_hist[i - 1]))),
                         (bin_w * i, hist_h - int(np.round(b_hist[i]))),
                         (255, 0, 0), thickness=2)
                cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(g_hist[i - 1]))),
                         (bin_w * i, hist_h - int(np.round(g_hist[i]))),
                         (0, 255, 0), thickness=2)
                cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(r_hist[i - 1]))),
                         (bin_w * i, hist_h - int(np.round(r_hist[i]))),
                         (0, 0, 255), thickness=2)

            cv2.imshow('Histogram', histImage)
        elif current == "Histogram Equalization":
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(self.image)
            self.image = equ
            cv2.imwrite("temp.jpeg", self.image)
            self.showEdited()

    def applySpatialClicked(self):
        current = str(self.spatialActions.currentText())
        value = float(self.spatialInput.text())

        if current == "Rotate":
            (h, w) = self.image.shape[:2]
            center = (w / 2, h / 2)
            rotation = cv2.getRotationMatrix2D(center, value, 1.0)

            abs_cos = abs(rotation[0, 0])
            abs_sin = abs(rotation[0, 1])

            bound_w = int(h * abs_sin + w * abs_cos)
            bound_h = int(h * abs_cos + w * abs_sin)

            rotation[0, 2] += bound_w / 2 - center[0]
            rotation[1, 2] += bound_h / 2 - center[1]

            self.image = cv2.warpAffine(self.image, rotation, (bound_w, bound_h))

        elif current == "Resize Height":
            height = int(self.spatialInput.text())
            self.image = cv2.resize(self.image, (self.image.shape[0], height))

        elif current == "Translate":
            height, width = self.image.shape[:2]

            quarter_height, quarter_width = height / value, width / value

            T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

            self.image = cv2.warpAffine(self.image, T, (width, height))

        elif current == "Affine":
            rows, cols, ch = self.image.shape
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

            m = cv2.getAffineTransform(pts1, pts2)

            self.image = cv2.warpAffine(self.image, m, (cols, rows))

        elif current == "Perspective":
            rows, cols, ch = self.image.shape
            pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
            pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

            M = cv2.getPerspectiveTransform(pts1, pts2)

            self.image = cv2.warpPerspective(self.image, M, (cols, rows))

        cv2.imwrite("temp.jpeg", self.image)
        self.showEdited()

    def applyRescaleIntensityClicked(self):
        # https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
        alpha = float(self.minLineEdit.text())
        beta = float(self.maxLineEdit.text())

        self.image = exposure.rescale_intensity(self.image, in_range=(alpha, beta))

        cv2.imwrite("temp.jpeg", self.image)
        self.showEdited()

    def applyMorphologyClicked(self):
        current = self.morphologyActions.currentText()
        kernel = np.ones((5, 5), np.uint8)

        if current == "Erosion":
            self.image = cv2.erode(self.image, kernel, iterations=1)
        elif current == "Dilation":
            self.image = cv2.dilate(self.image, kernel, iterations=1)
        elif current == "Opening":
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        elif current == "Closing":
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        elif current == "White Tophat":
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_TOPHAT, kernel)
        elif current == "Black Tophat":
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
        elif current == "Gradient":
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)
        elif current == "Skeletonize":
            self.image = morphology.skeletonize(self.image == 0)
        elif current == "Hit And Miss":
            self.image = cv2.morphologyEx(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.MORPH_HITMISS, kernel)
        elif current == "Cross":
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_CROSS, kernel)

        cv2.imwrite("temp.jpeg", self.image)
        self.showEdited()
        print("morp")

    def showVideoClicked(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(gray, 60, 120)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        print("show video")

    def showEdited(self):
        pixmap = QtGui.QPixmap("temp.jpeg")
        self.editedImage.setPixmap(pixmap)
        # height, width, bytePerComponent = self.image.shape
        # bytesPerLine = 3 * width
        # image = QtGui.QImage(self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
        # image = QtGui.QPixmap.fromImage(image)
        # self.editedImage.setPixmap(image)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
