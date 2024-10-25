"""Main application entry point.

At least for development purposes.
"""

import matplotlib.pyplot as plt
from time import sleep
from piano_sim.plot import keyboard
from piano_sim.notes import notes_dict, play_note


def start() -> None:
    """Start the application."""
    fig, ax = plt.subplots()
    keyboard(ax, root=notes_dict["C"], halfsteps=[2, 2, 1, 2, 2, 2])
    keyboard(ax, root=notes_dict["D"], halfsteps=[2, 2, 1, 2, 2, 2])
    plt.show()
    while True:
        keyboard(ax, root=notes_dict["C"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("C4")
        sleep(1)
        keyboard(ax, root=notes_dict["D"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("D4")
        sleep(1)
        keyboard(ax, root=notes_dict["E"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("E4")
        sleep(1)
        keyboard(ax, root=notes_dict["F"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("F4")
        sleep(1)
        keyboard(ax, root=notes_dict["G"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("G4")
        sleep(2)
        keyboard(ax, root=notes_dict["A"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("G4")
        sleep(1)
        keyboard(ax, root=notes_dict["A"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("G4")
        sleep(1)
        keyboard(ax, root=notes_dict["G"], halfsteps=[2, 2, 1, 2, 2, 2])
        play_note("G4")
