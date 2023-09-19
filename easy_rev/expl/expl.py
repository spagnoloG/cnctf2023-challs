from pwn import *

def main():
    r = remote('127.0.0.1', 65432)

    random_str = r.recvline().decode().strip()
    print(f"Received string: {random_str}")

    random_str = random_str.split('!')[1]
    print(random_str)

    reversed_str = random_str[::-1]
    r.sendline(reversed_str)

    flag = r.recvline().decode().strip()
    print(f"Flag: {flag}")

    r.close()

if __name__ == '__main__':
    main()
