from socket import * 
servername = '192.168.25.25'
serverport = 13000
clientSocket = socket(AF_INET,SOCK_DGRAM)
message = input('Input lowercase sentence :')
clientSocket.sendto(message.encode(),(servername,serverport))
modifiedMessage,serverAddress = clientSocket.recvfrom(1024)
print(modifiedMessage.decode())
clientSocket.close()