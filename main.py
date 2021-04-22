from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
    setting2_class = uic.loadUiType("setting2.ui")[0]
    chart_class = uic.loadUiType("chart.ui")[0]

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
    share['s'] = 1
    r1 = []
    r2 = []
    r3 = []
    r4 = []

    pgre_hyper = [0.98, 0.001]
    pgtd_hyper = [0.98, 0.001]
    dqn_hyper = [0.98, 0.001, 0.1, 0.01, 0.0001]
    ppo_hyper = [0.98, 0.001, 0.95, 0.1]

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

    class setting2(QDialog, setting2_class):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
            self.setWindowModality(Qt.ApplicationModal)
            self.re1.setText(str(pgre_hyper[0]))
            self.re2.setText(str(pgre_hyper[1]))
            self.td1.setText(str(pgtd_hyper[0]))
            self.td2.setText(str(pgtd_hyper[1]))
            self.dqn1.setText(str(dqn_hyper[0]))
            self.dqn2.setText(str(dqn_hyper[1]))
            self.dqn3.setText(str(dqn_hyper[2]))
            self.dqn4.setText(str(dqn_hyper[3]))
            self.dqn5.setText(str(dqn_hyper[4]))
            self.ppo1.setText(str(ppo_hyper[0]))
            self.ppo2.setText(str(ppo_hyper[1]))
            self.ppo3.setText(str(ppo_hyper[2]))
            self.ppo4.setText(str(ppo_hyper[3]))
            self.btn.clicked.connect(self.savedata)

        def savedata(self):
            global pgre_hyper
            global pgtd_hyper
            global dqn_hyper
            global ppo_hyper
            pgre_hyper = [float(self.re1.text()), float(self.re2.text())]
            pgtd_hyper = [float(self.td1.text()), float(self.td2.text())]
            dqn_hyper = [float(self.dqn1.text()), float(self.dqn2.text()), float(self.dqn3.text()), float(self.dqn4.text()), float(self.dqn5.text())]
            ppo_hyper = [float(self.ppo1.text()), float(self.ppo2.text()), float(self.ppo3.text()), float(self.ppo4.text())]
            self.close()

    class chart(QDialog, chart_class):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
            self.setWindowModality(Qt.ApplicationModal)
            self.fig = plt.Figure()
            self.canvas = FigureCanvas(self.fig)
            self.chart.addWidget(self.canvas)
        
        def draw(self):
            p = self.fig.add_subplot()
            x = [(x+1)*n_play for x in range(n_iter)]
            p.plot(x, r1, label='PG_RE')
            p.plot(x, r2, label='PG_TD')
            p.plot(x, r3, label='DQN')
            p.plot(x, r4, label='PPO')
            p.set_xlabel("n_episode")
            p.set_ylabel("Return")
            p.legend()
            self.canvas.draw()

    class window(QMainWindow, main_class):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
            self.setup.triggered.connect(self.menu_set1)
            self.para.triggered.connect(self.menu_set2)
            self.chart.triggered.connect(self.menu_plot)
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
                process1 = multiprocessing.Process(target=PG_reinforce.RL, args=(share, n_epi, game, n_input, n_output, n_play, pgre_hyper))
                process2 = multiprocessing.Process(target=PG_td.RL, args=(share, n_epi, game, n_input, n_output, n_play, pgtd_hyper))
                process3 = multiprocessing.Process(target=DQN.RL, args=(share, n_epi, game, n_input, n_output, n_play, dqn_hyper))
                process4 = multiprocessing.Process(target=PPO.RL, args=(share, n_epi, game, n_input, n_output, n_play, ppo_hyper))
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
                self.chart.setEnabled(True)
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
                if share['s'] == 1 or share['s'] == 2:
                    self.r1.setText(str(share['r1']))
                    self.r2.setText(str(share['r2']))
                    self.r3.setText(str(share['r3']))
                    self.r4.setText(str(share['r4']))
                    r1.append(share['r1'])
                    r2.append(share['r2'])
                    r3.append(share['r3'])
                    r4.append(share['r4'])
                if share['s'] == 1:
                    self.btn1.setEnabled(True)
                    self.btn2.setEnabled(True)
                else:
                    self.btn1.setText('학습완료')
                    self.btn2.setEnabled(True)

        def menu_set1(self):
            self.new1 = setting1()
            self.new1.show()

        def menu_set2(self):
            self.new2 = setting2()
            self.new2.show()

        def menu_plot(self):
            self.newchart = chart()
            self.newchart.draw()
            self.newchart.show()

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
    try:
        if not os.path.isdir(os.getcwd() + '\data'):
            os.makedirs(os.getcwd() + '\data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("failed to create dir!")

    app = QApplication(sys.argv)
    mwindow = window()
    mwindow.show()
    sys.exit(app.exec_())