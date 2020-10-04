# -*- coding:UTF-8 -*-
from src.utils import *
from src.batch_recongize import *
from src.section_detection import *


FONT = "Τ²Με"

def creat_windows():
    win = tk.Tk() 
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 500, 700
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  
    win.title('Invoice Identify')  

    bg1_open = Image.open("pic/timg.png").resize((300, 400))
    bg1 = ImageTk.PhotoImage(bg1_open)
    canvas = tk.Label(win, image = bg1)
    canvas.pack()

    var = tk.StringVar() 
    var.set('')
    tk.Label(win, textvariable = var, bg = '#87CEFA', font = (FONT, 10), width = 50, height = 2).pack()

    tk.Button(win, text = 'Select Image', width = 30, height = 2, bg = '#EBEBEB', command = lambda: showpic(canvas),
              font = (FONT, 10)).pack()
    tk.Button(win, text='Detect a single character', width = 30, height = 2, bg ='#EBEBEB', command = lambda: detect_one_char(var),
              font = (FONT, 10)).pack()
    tk.Button(win, text='Extract invoice number', width = 30, height  = 2, bg = '#EBEBEB', command = lambda: detect_invoice_num(var),
              font = (FONT, 10)).pack()
    tk.Button(win, text='Extract Net Weight', width = 30, height = 2, bg = '#EBEBEB', command = lambda: detect_net(var),
              font = (FONT, 10)).pack()
    tk.Button(win, text='Extract Gross Weight', width = 30, height = 2, bg = '#EBEBEB', command = lambda: detect_gross(var),
              font = (FONT, 10)).pack()
    tk.Button(win, text='Select a floder', width = 30, height = 2, bg = '#EBEBEB', command = lambda: batch_identify(win, var),
              font = (FONT, 10)).pack()

    win.mainloop()


if __name__ == "__main__":
    creat_windows()