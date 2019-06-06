import socket
import random
s = socket.socket()
s.connect(('127.0.0.1', 3425))

i=0
while i<5:
    re = s.recv(1024).decode()
    print(re)
    s.send("ack {}".format(i).encode())
    i+=1
s.close()