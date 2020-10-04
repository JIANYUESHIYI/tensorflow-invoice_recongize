#coding:UTF-8
from src.utils import *

#Invoices are divided into two types based on the position of the invoice number
#Set categories here, divided into categories A and B
CATEGORY = 'A'
# CATEGORY = 'B'


# def detect_one_char(var):
#     """
#     This function is used to recognize a single character
#     """
#     bg1_open = Image.open(file_path).resize((16, 16), Image.ANTIALIAS) 
#     bg1_open_gray = bg1_open.convert('L') 
#     pic = np.array(bg1_open_gray).reshape(256, )

#     answer = ''
#     answer = startdetect(pic,answer)
#     var.set("The character is" + answer)
#     print("The character is" + answer)


def detect_invoice_num(var):
    """
    This function is used to identify the invoice number
    """
    key = 1
    ref = getKeyWord(key, CATEGORY)
    refCnts,ref2 = getRefCnts(ref, "left-to-right", 1)
    clone = np.dstack([ref2.copy()] * 3)

    index = 0
    answer = ""
    for c in refCnts:
        pic = getOneChar(c,clone,index)
        index = index + 1
        if index == 5 or index == 6:
            answer = startdetect_all(pic, answer)
        else:
            answer = startdetect_only_number(pic,answer)
        # if answer>

    var.set("Invoice number is " + answer )
    print("Invoice number is " + answer)

    
def detect_net(var):
    """
    This function is used to identify the net weight
    """
    key = 2
    ref = getKeyWord(key)
    #Here is sorted from right to left to exclude the interference of commas
    refCnts,ref2 = getRefCnts(ref, "right-to-left") 
    clone = np.dstack([ref2.copy()] * 3)

    index = 0
    answer = ""
    for c in refCnts:
        if index == 3:
            index = index + 1
            continue
        # if index
        pic = getOneChar(c,clone,index)
        index = index + 1
        # print(answer)
        answer = startdetect_only_number(pic,answer)

    var.set("Net weight is " + answer[::-1] + " kg")
    print("Net weight is " + answer[::-1] + " kg")


def detect_gross(var):
    """
    This function is used to identify the gross weight
    file_path: address of the picture to be recognized
    """
    key = 3
    ref = getKeyWord(key)
    refCnts,ref2 = getRefCnts(ref, "right-to-left")

    clone = np.dstack([ref2.copy()] * 3)
    index = 0
    answer = ""
    for c in refCnts:
        if index == 3:
            index = index + 1
            continue
        pic = getOneChar(c,clone,index)
        index = index + 1
        answer = startdetect_only_number(pic,answer)

    var.set("Gross weight is " + answer[::-1] + " kg")
    print("Gross weight is " + answer[::-1] + " kg")


def batch_detect_invoice_num(file_path):
    """
    This function is used to identify invoice number in batch
    file_path: address of the picture to be recognized
    answer: recognition result
    """
    file_path = file_path
    key = 1
    ref_in = batch_getKeyWord(file_path, key, CATEGORY)
    refCnts,ref2 = getRefCnts(ref_in, "left-to-right")
    clone = np.dstack([ref2.copy()] * 3)

    index = 0
    answer = ""
    for c in refCnts:
        pic = getOneChar(c,clone,index)
        index = index + 1
        if index == 5 or index == 6:
            answer = startdetect(pic, answer)
        else:
            answer = startdetect_only_number(pic,answer)

    return answer


def batch_detect_net(file_path):
    """
    This function is used to identify net weight in batch
    file_path: address of the picture to be recognized
    answer: recognition result

    """
    file_path = file_path
    key = 2
    ref_ne = batch_getKeyWord(file_path, key)
    refCnts,ref2 = getRefCnts(ref_ne, "right-to-left")  
    clone = np.dstack([ref2.copy()] * 3)

    index = 0
    answer = ""
    for c in refCnts:
        if index == 3:
            index = index + 1
            continue
        pic = getOneChar(c,clone,index)
        index = index + 1
        answer = startdetect_only_number(pic,answer)

    return answer[::-1]


def batch_detect_gross(file_path):
    """
    This function is used to identify gross weight in batch
    file_path: address of the picture to be recognized
    answer: recognition result
    """
    file_path = file_path
    key = 3
    ref_gr = batch_getKeyWord(file_path, key)
    refCnts, ref2 = getRefCnts(ref_gr, "right-to-left")

    clone = np.dstack([ref2.copy()] * 3)
    index = 0
    answer = ""
    for c in refCnts:
        if index == 3:
            index = index + 1
            continue
        pic = getOneChar(c,clone,index)
        index = index + 1
        answer = startdetect_only_number(pic,answer)

    return answer[::-1]




