# -*- coding: utf-8 -*-


def check_size(size, custom_msg,array):
    if(size!=len(array)):
        error_msg = "{}  shall be an array of {} scalar float values".format(custom_msg,size)
        raise ValueError(error_msg)

def check_scalar(custom_msg,val):
    if not(isinstance(val,float) or isinstance(val,int)):
        error_msg = "{}  shall be of type int or float  but is of type {}  ".format(custom_msg,type(val))
        raise ValueError(error_msg)

def check_non_zero(custom_msg, val):
    if val==0:
        error_msg="{} shall not be equal to zero".format(custom_msg)
        raise ValueError(error_msg)

def checkAcknowledgment(msg):
    ak="done"
    nak="nak"
    print(msg)
    if msg=="":
        return False
    if (msg==nak):
        return False
    if msg==ak:
        return True
    else:
        return False
