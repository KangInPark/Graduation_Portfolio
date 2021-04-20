import DQN
import PG_reinforce
import PG_td
import PPO
import multiprocessing
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import *
from PIL import Image

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    share = manager.dict()
    share['wait'] = 1
    share['pgre'] = 0
    share['pgtd'] = 0
    share['dqn'] = 0
    share['ppo'] = 0
    game = 'CartPole-v1'
    n_epi = 10000
    n_input = 4
    n_output = 2
    n_play = 300
    schk = 0
    d = {}
    d['CartPole-v1'] = (4,2)
    d['MountainCar-v0'] = (2,3)
    d['Acrobot-v1'] = (6,3)

    def epirl():
        global schk
        if schk == 0:
            btn_epoch.configure(state=DISABLED)
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
            btn_epoch.configure(text='학습 진행')
            btn_epoch.configure(state=NORMAL)

        else:
            btn_epoch.configure(state=DISABLED)
            btn_ani.configure(state=DISABLED)

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

            btn_epoch.configure(state=NORMAL)
            btn_ani.configure(state=NORMAL)
    
    def playani():
        btn_ani.configure(state=DISABLED)
        file1 = 'pgre.gif'
        file2 = 'pgtd.gif'
        file3 = 'dqn.gif'
        file4 = 'ppo.gif'
        info1 = Image.open(file1)
        frames1 = info1.n_frames
        im1 = [PhotoImage(file=file1, format= 'gif -index %i' %(i)) for i in range(frames1)]
        info2 = Image.open(file2)
        frames2 = info2.n_frames
        im2 = [PhotoImage(file=file2, format= 'gif -index %i' %(i)) for i in range(frames2)]
        info3 = Image.open(file3)
        frames3 = info3.n_frames
        im3 = [PhotoImage(file=file3, format= 'gif -index %i' %(i)) for i in range(frames3)]
        info4 = Image.open(file4)
        frames4 = info4.n_frames
        im4 = [PhotoImage(file=file4, format= 'gif -index %i' %(i)) for i in range(frames4)]
        maxframes = max(frames1, frames2, frames3, frames4)
        cnt = 0
        def gifani(cnt):
            i1 = im1[min(frames1-1,cnt)]
            i2 = im2[min(frames2-1,cnt)]
            i3 = im3[min(frames3-1,cnt)]
            i4 = im4[min(frames4-1,cnt)]
            gif_label1.configure(image=i1)
            gif_label2.configure(image=i2)
            gif_label3.configure(image=i3)
            gif_label4.configure(image=i4)
            cnt+=1
            if cnt == maxframes:
                btn_ani.configure(state=NORMAL)
                return
            else:
                root.after(20, gifani ,cnt)
        gifani(cnt)

    def setting():
        new = Toplevel(root)
        new.wm_attributes("-topmost", 1)
        l1 = Label(new, text="학습할 게임")
        l1.pack()
        game_var = StringVar()
        gamebtn1 = Radiobutton(new, text='CartPole', value='CartPole-v1', variable=game_var)
        gamebtn1.select()
        gamebtn2 = Radiobutton(new, text='MountainCar', value='MountainCar-v0', variable=game_var)
        gamebtn3 = Radiobutton(new, text='Acrobot', value='Acrobot-v1', variable=game_var)
        gamebtn1.pack()
        gamebtn2.pack()
        gamebtn3.pack()

        l2 = Label(new, text="총 에피소드 횟수")
        l2.pack()
        nepi = Entry(new, width = 10)
        nepi.insert(0, str(n_epi))
        nepi.pack()
        l3 = Label(new, text="Visualization Update 단위 (설정 값 만큼의 에피소드를 진행한 후 Visualization 제공)")
        l3.pack()
        nplay = Entry(new, width = 10)
        nplay.insert(0, str(n_play))
        nplay.pack()

        def get_var():
            global game
            global n_epi
            global n_play
            game = game_var.get()
            n_epi = int(nepi.get())
            n_play = int(nplay.get())
            msgbox.showinfo("알림", "정상적으로 설정이 완료되었습니다.")
            new.destroy()

        btn_get = Button(new, padx=3, pady=3, text="설정", command=get_var)
        btn_get.pack()

    root = Tk()
    root.wm_attributes("-topmost", 1)
    root.title("")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int(screen_width/2)-320
    y = int(screen_height/2)-240
    position = "+" + str(x) + "+" + str(y)
    #root.geometry("640x480" + position)
    #root.resizable(False, False)

    menu = Menu(root)
    menu_menu = Menu(menu, tearoff=0)
    menu_menu.add_command(label="학습 설정",command=setting)
    menu_menu.add_command(label="종료", command=root.quit)
    menu.add_cascade(label="메뉴", menu=menu_menu)
    root.config(menu=menu)
    label1 = Label(root, text="PG_reinforce")
    label1.grid(row=0, column=0)
    gif_label1 = Label(root, image="")
    gif_label1.grid(row=1,column=0)
    label2 = Label(root, text="PG_td")
    label2.grid(row=0, column=1)
    gif_label2 = Label(root, image="")
    gif_label2.grid(row=1,column=1)
    label3 = Label(root, text="DQN")
    label3.grid(row=2, column=0)
    gif_label3 = Label(root, image="")
    gif_label3.grid(row=3,column=0)
    label4 = Label(root, text="PPO")
    label4.grid(row=2, column=1)
    gif_label4 = Label(root, image="")
    gif_label4.grid(row=3,column=1)

    btn_epoch = Button(root, padx=3, pady=3, text="학습 준비", command=epirl)
    btn_epoch.grid(row=4,column=0, sticky=N+E+W+S)
    btn_ani = Button(root, padx=3, pady=3, text="Visualization", command=playani, state=DISABLED)
    btn_ani.grid(row=4,column=1, sticky=N+E+W+S)

    root.mainloop()
    