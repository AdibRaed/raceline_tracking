from sys import argv

# qt5 used for testing 
# comment this before submitting - Adib 
import matplotlib
matplotlib.use("Qt5Agg")

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    simulator = Simulator(racetrack)
    simulator.start()
    plt.show()