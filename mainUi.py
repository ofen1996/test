from smaSTViewer import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QMainWindow

import sys


class SmaSTViewerMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(SmaSTViewerMain, self).__init__(parent)
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SmaSTViewerMain()
    window.show()
    sys.exit(app.exec_())