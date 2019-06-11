# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:21:02 2018

@author: Mohammad SAFEEA
"""
import socket
import time
from io import  BytesIO


class mySock:
    '''demonstration class only
      - coded for clarity, not efficiency
    '''

    def __init__(self, tup):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("",tup[1]))
        print("connecting to iiwa...." )
        self.server_sock.listen(1)
        (self.sock, self.clientAdress) = self.server_sock.accept()
        print("OK")
        self.buff = BytesIO()

    def send(self, msg):
        totalsent = 0
        try :
            while totalsent < len(msg):
                sent = self.sock.send(msg[totalsent:])
                if sent == 0:
                    raise RuntimeError("socket connection broken")
                totalsent = totalsent + sent
        except:
            self.close()
            return None

    def receive(self):
        returnVal=[]

        try:
            while True:
                data = self.sock.recv(1)
                if (data=='\n'.encode("ascii")):break
                returnVal.append(data.decode("ascii"))
            return "".join(returnVal)
        except:
            self.close()
            raise
            return False

        
    def close(self):
        endCommand=b'end\n'
        self.sock.send(endCommand)
        time.sleep(1) # sleep for one seconds
        self.sock.close()
        self.server_sock.close() 
