import socket
import threading
import random
import string
import time
import os

HOST = '127.0.0.1'
PORT = 65432

def random_string(length=40):
    """Generate a random string of given length"""
    return ''.join(random.choice(string.ascii_letters) for i in range(length))

def handle_client(client_socket):
    random_str = random_string() + '\n'
    print(f"Sending string: {random_str}")
    client_socket.send("Send me the reversed string!".encode())
    client_socket.send(random_str.encode())
    random_str = random_str[:-1]

    start_time = time.time()
    try:
        client_socket.settimeout(2)
        data = client_socket.recv(1024).decode()
    except socket.timeout:
        print("Client didn't respond in time!")
        client_socket.close()
        return

    elapsed_time = time.time() - start_time
    print(elapsed_time)
    data = data.strip()
    if elapsed_time <= 2 and data == random_str[::-1]:
        print("Client sent the reversed string in time!")
        secret = os.environ.get('FLAG', 'ctf{fake_flag}') + '\n'
        client_socket.send(secret.encode())
    else:
        print("Failed or took too long!")
    
    client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[*] Listening on {HOST}:{PORT}")

    while True:
        client, addr = server.accept()
        print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")
        client_handler = threading.Thread(target=handle_client, args=(client,))
        client_handler.start()

if __name__ == "__main__":
    main()
