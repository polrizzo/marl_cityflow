import os
import datetime

datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
path = "../grid_scenarios/" + datenow
os.mkdir(path)