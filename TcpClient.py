from socket import *
servername = '192.168.25.25'
serverport = 12000
clientSocket = socket(AF_INET,SOCK_STREAM)
clientSocket.connect((servername,serverport))
sentence = input('Input lowercase sentence:')
clientSocket.send(sentence.encode())
modifiedSentence = clientSocket.recv(1024)
print('From server: ',modifiedSentence.decode())
clientSocket.close()