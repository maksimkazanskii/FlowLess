import pyautogui
import random
import time

while True:
    x, y = pyautogui.position()

    dx = random.choice([-1, 1]) * random.randint(1, 7)
    dy = random.choice([-1, 1]) * random.randint(1, 7)

    pyautogui.moveTo(x + dx, y + dy, duration=0.2)

    time.sleep(10)  # 5 minutes