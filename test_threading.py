
import threading

index = 0
def hi():
    global index
    print(index)
    index += 1

index2 = 0
def hi2():
    global index2
    print("---------------------------", index2)
    index2 += 1
        # print("-------------------------hoc goi")

print(1)

while True:
    thread2 = threading.Thread(target= hi2)
    thread2.start()
    thread1 = threading.Thread(target= hi)
    thread1.start()