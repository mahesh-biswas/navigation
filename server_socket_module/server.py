import socket

s = socket.socket()
s.bind(('',3425))
print("bound")
s.listen(5)
print("listening...")
i=0
c,addr = s.accept()
while i<5:
    print("connected to {}".format(addr))
    string = "You have connected succesfully... at itter: {}".format(i)
    c.send(string.encode())
    re = c.recv(1024).decode()
    print("client said: {} and int: {}".format(re,int(float(re))))
    i+=1
c.close()