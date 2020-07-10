import sys
import os
import qdarkstyle
from PIL import Image
from PyQt5.QtWidgets import QWidget, QMessageBox, QGridLayout, QApplication, QDesktopWidget, QPushButton, QLabel, QRadioButton, QFileDialog, QComboBox, QLineEdit
from PyQt5.QtGui import QIcon, QColor, QPainter, QPen, QPixmap, QFont
from PyQt5.QtCore import Qt
import numpy as np
from letter_iden.utils.inum import num_predict
from letter_iden.utils.liden import pred_letter


class App(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.width = self.screenRect.width()
        self.scale = self.width/1920
        self.setFixedSize(500*self.scale, 500*self.scale)

        self.setWindowTitle('App')
        self.center()
        self.setWindowIcon(QIcon('icon.png'))
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        font = QFont()
        font.setFamily("Yu Mincho Demibold")
        font.setPointSize(12)
        font1=QFont()
        font1.setPointSize(1)

        self.mode1 = QRadioButton('Number Identity', self)
        self.mode2 = QRadioButton('Letter Identity', self)
        self.mode1.move(50*self.scale, 50*self.scale)
        self.mode2.move(275*self.scale, 50*self.scale)
        self.mode1.setFont(font)
        self.mode2.setFont(font)
        self.mode1.setChecked(True)
        self.fileButton = QPushButton('Load', self)
        self.fileButton.move(35*self.scale, 320*self.scale)
        self.fileButton.setFont(font)
        self.processButton = QPushButton('Start', self)
        self.processButton.move(290*self.scale, 440*self.scale)
        self.processButton.setFont(font)
        self.clearButton = QPushButton('Clear', self)
        self.clearButton.move(390*self.scale, 440*self.scale)
        self.clearButton.setFont(font)
        self.label = QLabel('Result:', self)
        self.label.setFont(font)
        self.label.setGeometry(40*self.scale, 440*self.scale, 150*self.scale, 50)
        self.map = Paint(self, self.scale)
        self.map.move(260*self.scale, 190*self.scale)
        self.map2 = QLabel(self)
        self.map2.setGeometry(259*self.scale, 189*self.scale,
                              212*self.scale, 212*self.scale)
        pixmap = QPixmap("label.png")
        self.map2.setPixmap(pixmap)
        self.label2 = QLabel('Mode:', self)
        self.label2.setFont(font)
        self.label2.setGeometry(35*self.scale, 135*self.scale, 150, 50)
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("HandWriting")
        self.comboBox.addItem("Photograph")
        self.comboBox.move(35*self.scale, 180*self.scale)
        self.comboBox.setFont(font)
        self.text = QLineEdit(self)
        self.text.setGeometry(35*self.scale, 250*self.scale, 200*self.scale, 50)
        self.text.setFont(font)
        self.openButton = QPushButton('Open', self)
        self.openButton.move(145*self.scale, 320*self.scale)
        self.openButton.setFont(font)

        self.text.setVisible(False)
        self.openButton.setVisible(False)
        self.fileButton.setVisible(False)
        self.map2.setVisible(False)
        self.processButton.clicked.connect(self.process)
        self.clearButton.clicked.connect(self.clear)
        self.comboBox.currentIndexChanged.connect(self.select)
        self.fileButton.clicked.connect(self.choose)
        self.openButton.clicked.connect(self.open)
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def process(self):
        if self.comboBox.currentIndex() == 0:
            self.map.pixmap.save("tmp.jpg", "JPG")
            img = Image.open("tmp.jpg")
            img = img.resize((28, 28), Image.ANTIALIAS)
            img = img.convert('L')
        else:
            img = Image.open(self.text.text())
        self.array = np.array(img)
        if self.mode1.isChecked():
            self.result = num_predict(self.array)
        else:
            self.result = pred_letter(self.array)
        max = []
        for i in range(3):
            j = self.result.argmax()
            max.append(j)
            self.result[j] = 0
        if self.mode1.isChecked():
            self.label.setText("Result:%d/%d/%d" % (max[0], max[1], max[2]))
        else:
            self.label.setText(
                "Result:%c/%c/%c" % (chr(ord('A')+max[0]), chr(ord('A')+max[1]), chr(ord('A')+max[2])))
        if self.comboBox.currentIndex() == 0:
            os.remove("tmp.jpg")

    def clear(self):
        self.map.pos_xy.clear()
        self.map.update()
        self.text.setText("")
        pixmap = QPixmap("label.png")
        self.map2.setPixmap(pixmap)
        self.label.setText("Result:")

    def select(self):
        if self.comboBox.currentIndex() == 0:
            self.text.setVisible(False)
            self.openButton.setVisible(False)
            self.fileButton.setVisible(False)
            self.map.setVisible(True)
            self.map2.setVisible(False)
            self.label.setText("Result:")
        else:
            self.text.setVisible(True)
            self.openButton.setVisible(True)
            self.fileButton.setVisible(True)
            self.map.setVisible(False)
            self.map2.setVisible(True)
            self.label.setText("Result:")

    def choose(self):
        self.filename = QFileDialog.getOpenFileName(
            self, "Choose the file", "./", "Images (*.png;*.jpg);;All Files (*)")
        if self.filename[0]:
            self.text.setText(self.filename[0])
            self.open()

    def open(self):
        try:
            img = Image.open(self.text.text())
            img = img.resize(
                (int(212*self.scale), int(212*self.scale)), Image.ANTIALIAS)
            img.save("tmp.png")
            pixmap = QPixmap("tmp.png")
            self.map2.setPixmap(pixmap)
            os.remove("tmp.png")
        except:
            QMessageBox.warning(self, 'Warning',
                                     "The file cannot be found.")


class Paint(QWidget):
    def __init__(self, parent=None, scale=1):
        super().__init__(parent)
        self.scale = scale
        self.initUI()

    def initUI(self):
        self.setMouseTracking(False)
        self.pos_xy = []
        self.resize(210*self.scale, 210*self.scale)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        self.pixmap = QPixmap("label.png")
        painter.drawPixmap(self.rect(), self.pixmap)
        painter.end()
        painter.begin(self.pixmap)
        painter2 = QPainter()
        painter2.begin(self)
        pen = QPen(Qt.white, 10*self.scale, Qt.SolidLine)
        painter.setPen(pen)
        painter2.setPen(pen)
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(
                    point_start[0], point_start[1], point_end[0], point_end[1])
                painter2.drawLine(
                    point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()
        painter2.end()

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
