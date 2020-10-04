# coding:UTF-8
from tkinter import filedialog
import os
import glob
from src.utils import *
from src.section_detection import *
# from section_detection import *


def batch_identify(win, var):
    """
    This function is batch recognition and save the result in txt file
    """
    var.set("")
    #Get folder directory
    current_directory = filedialog.askdirectory()
    if current_directory:
        file_path = os.path.join(current_directory,"*.*")
        img_path=glob.glob(file_path)
    
    print("Recognizing")
    count = 1
    for img in img_path:
        invoice_number_answer = batch_detect_invoice_num(img)
        net_weight_answer = batch_detect_net(img)
        gross_weight_answer = batch_detect_gross(img)

        with open("result.txt", 'a') as fp:
            fp.write(count + '\t')
            fp.write(img + "\t")
            fp.write(invoice_number_answer+"\t")
            fp.write(net_weight_answer + "\t")
            fp.write(gross_weight_answer + "\n")
        count = count + 1

    var.set("Recognition complete")
    print("Recognition complete")











    