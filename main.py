import sys
import time

from pygame import mixer  # Load the popular external library

from pipeline import run_experimental_setup_new

if __name__ == '__main__':
    try:
        # ================== default arguments ===================
        # available arguments:
        #
        #           "e2000_Acrobot.json"
        #           "e3000_CartPole.json"
        #           "e4000_MountainCar.json"
        #           "e5000_Taxi.json"
        #           "e6000_FrozenLake.json"
        #           "e7000_Breakout.json"
        #

        args = None
        if len(sys.argv) != 2:
            args = [sys.argv[0], "e2000_Acrobot.json"]
        else:
            args = sys.argv

        # ================== experimental setup ==================
        render_mode = "rgb_array"       # "human", "rgb_array"
        debug_print = True             # False, True
        run_experimental_setup_new(arguments=args, render_mode=render_mode, debug_print=debug_print)
        print(f'finisehd gracefully')
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
    except ValueError as e:
        print(f'Value error: {e}')
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)

    print(9)
