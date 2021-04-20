from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
import signal
import multiprocessing
import DQN
import PG_reinforce
import PG_td
import PPO
import os

if __name__ == '__main__':
        
    main_class = uic.loadUiType("main.ui")[0]
    setting1_class = uic.loadUiType("setting1.ui")[0]
    game = 'CartPole-v1'
    n_epi = 10000
    n_play = 100
    schk = 0
    d = {}
    d['CartPole-v1'] = (4,2)
    d['MountainCar-v0'] = (2,3)
    d['Acrobot-v1'] = (6,3)
    manager = multiprocessing.Manager()
    share = manager.dict()
    share['wait'] = 1
    share['pgre'] = 0
    share['pgtd'] = 0
    share['dqn'] = 0
    share['ppo'] = 0
    share['r1'] = 0.0
    share['r2'] = 0.0
    share['r3'] = 0.0
    share['r4'] = 0.0
    n_iter = 0

    class setting1(QDialog, setting1_class):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
            self.setWindowModality(Qt.ApplicationModal)
            if game == 'CartPole-v1':
                self.rbtn1.setChecked(True)
            elif game == 'MountainCar-v0':
                self.rbtn2.setChecked(True)
            else:
                self.rbtn3.setChecked(True)
            self.line1.setText(str(n_epi))
            self.line2.setText(str(n_play))
            self.btn.clicked.connect(self.savedata)

        def savedata(self):
            global game
            global n_epi
            global n_play
            
            if self.rbtn1.isChecked():
                game = 'CartPole-v1'
            elif self.rbtn2.isChecked():
                game = 'MountainCar-v0'
            else:
                game = 'Acrobot-v1'
            n_epi = int(self.line1.text())
            n_play = int(self.line2.text())
            self.close()

    class window(QMainWindow, main_class):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
            self.setup.triggered.connect(self.menu_set1)
            self.end.triggered.connect(QApplication.quit)
            self.btn1.clicked.connect(self.epi_rl)
            self.btn2.clicked.connect(self.load_gif)

        def epi_rl(self):
            global schk
            global n_iter
            if schk == 0:
                self.setup.setEnabled(False)
                self.para.setEnabled(False)
                self.btn1.setEnabled(False)
                n_input, n_output = d[game]
                process1 = multiprocessing.Process(target=PG_reinforce.RL, args=(share, n_epi, game, n_input, n_output, n_play,))
                process2 = multiprocessing.Process(target=PG_td.RL, args=(share, n_epi, game, n_input, n_output, n_play,))
                process3 = multiprocessing.Process(target=DQN.RL, args=(share, n_epi, game, n_input, n_output, n_play,))
                process4 = multiprocessing.Process(target=PPO.RL, args=(share, n_epi, game, n_input, n_output, n_play,))
                process1.start()
                process2.start()
                process3.start()
                process4.start()
                schk = 1
                while not share['pgre']:
                    continue
                while not share['pgtd']:
                    continue
                while not share['dqn']:
                    continue
                while not share['ppo']:
                    continue
                self.btn1.setText('학습진행')
                self.btn1.setEnabled(True)

            else:
                n_iter += 1
                self.iter.setText(str(n_iter))
                self.btn1.setEnabled(False)
                self.btn2.setEnabled(False)
                share['wait'] = 0
                while share['pgre']:
                    continue
                while share['pgtd']:
                    continue
                while share['dqn']:
                    continue
                while share['ppo']:
                    continue
                share['wait'] = 1
                
                while not share['pgre']:
                    continue
                while not share['pgtd']:
                    continue
                while not share['dqn']:
                    continue
                while not share['ppo']:
                    continue
                self.r1.setText(str(share['r1']))
                self.r2.setText(str(share['r2']))
                self.r3.setText(str(share['r3']))
                self.r4.setText(str(share['r4']))
                self.btn1.setEnabled(True)
                self.btn2.setEnabled(True)

        def menu_set1(self):
            self.new = setting1()
            self.new.show()

        def load_gif(self):
            self.data1 = QMovie(os.getcwd() + '\data\pgre'+str(n_iter)+'.gif')
            self.data2 = QMovie(os.getcwd() + '\data\pgtd'+str(n_iter)+'.gif')
            self.data3 = QMovie(os.getcwd() + '\data\dqn'+str(n_iter)+'.gif')
            self.data4 = QMovie(os.getcwd() + '\data\ppo'+str(n_iter)+'.gif')
            self.gif1.setMovie(self.data1)
            self.gif2.setMovie(self.data2)
            self.gif3.setMovie(self.data3)
            self.gif4.setMovie(self.data4)
            self.data1.start()
            self.data2.start()
            self.data3.start()
            self.data4.start()

    app = QApplication(sys.argv)
    mwindow = window()
    mwindow.show()
    sys.exit(app.exec_())