import matplotlib.pyplot as plt
from piano_sim.notes import notes_dict, Notes
import functools

DIMINISHED_CHORD_COLOR = "#C9C4D9"
MAJOR_CHORD_COLOR = "#CCE7D4"
MINOR_CHORD_COLOR = "#FCD2C2"
MAJOR_ROOT_COLOR = "#5ADD80"
MINOR_ROOT_COLOR = "#FF9A75"
SCALE_MEMBER_COLOR = "#C9C4D9"
DEFAULT_COLOR = "white"
TEXT_DEFAULT_COLOR = "black"


def plotlive(func):
    plt.ion()

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # Clear all axes in the current figure.
        axes = plt.gcf().get_axes()
        for axis in axes:
            axis.cla()

        # Call func to plot something
        result = func(*args, **kwargs)

        # Draw the plot
        plt.draw()
        plt.pause(0.01)

        return result

    return new_func


@plotlive
def keyboard(ax, octaves=1, root=None, halfsteps=None, title="Piano"):
    """Keyboard keys.

    Each octave starts with C.

    """
    _notes = Notes()
    START_X = 0.0
    START_Y = 0.0
    SCALE_X = 1.3
    SCALE_Y = 1.8
    # W_X W_Y are white key dims, B_X B_Y are black key dims.
    W_X = 2.35 * SCALE_X
    W_Y = W_X * SCALE_Y
    B_X = 1.37 * SCALE_X
    B_Y = B_X * SCALE_Y
    # ax = plt.axes()

    _notes.set_cur(notes_dict["C"])
    CUR_X = START_X

    scaleNotes = []
    rootName = ""
    if halfsteps != None and root != None:
        _notes.set_cur(root)
        rootName = _notes.cur()
        for steps in halfsteps:
            scaleNotes.append(_notes.read_then_next(steps))
        scaleNotes.append(_notes.cur())

    _notes.set_cur(notes_dict["C"])
    for o in range(octaves):
        for i in range(7):
            note = _notes.read_then_next(1)
            # white key
            if note == rootName:
                ax.add_patch(
                    plt.Rectangle(
                        (CUR_X, START_Y),
                        W_X,
                        -W_Y,
                        color=MAJOR_ROOT_COLOR,
                        fill=True,
                        zorder=5,
                        ec="black",
                    )
                )
                ax.text(
                    CUR_X + W_X / 2,
                    START_Y - 3 * W_Y / 4,
                    note,
                    c="white",
                    va="center",
                    ha="center",
                    zorder=5,
                )
            elif note in scaleNotes:
                ax.add_patch(
                    plt.Rectangle(
                        (CUR_X, START_Y),
                        W_X,
                        -W_Y,
                        color=SCALE_MEMBER_COLOR,
                        fill=True,
                        zorder=5,
                        ec="black",
                    )
                )
                ax.text(
                    CUR_X + W_X / 2,
                    START_Y - 3 * W_Y / 4,
                    note,
                    c="white",
                    va="center",
                    ha="center",
                    zorder=5,
                )
            else:
                ax.add_patch(
                    plt.Rectangle(
                        (CUR_X, START_Y),
                        W_X,
                        -W_Y,
                        color="white",
                        fill=True,
                        zorder=5,
                        ec="black",
                    )
                )
                ax.text(
                    CUR_X + W_X / 2,
                    START_Y - 3 * W_Y / 4,
                    note,
                    c="black",
                    va="center",
                    ha="center",
                    zorder=5,
                )

            # black key
            if i != 2 and i != 6:
                note = _notes.read_then_next(1)
                if note == rootName:
                    ax.add_patch(
                        plt.Rectangle(
                            (CUR_X + W_X - (B_X / 2), START_Y),
                            B_X,
                            -B_Y,
                            color=MAJOR_ROOT_COLOR,
                            fill=True,
                            zorder=6,
                            ec="black",
                        )
                    )
                elif note in scaleNotes:
                    ax.add_patch(
                        plt.Rectangle(
                            (CUR_X + W_X - (B_X / 2), START_Y),
                            B_X,
                            -B_Y,
                            color=SCALE_MEMBER_COLOR,
                            fill=True,
                            zorder=6,
                            ec="black",
                        )
                    )
                else:
                    ax.add_patch(
                        plt.Rectangle(
                            (CUR_X + W_X - (B_X / 2), START_Y),
                            B_X,
                            -B_Y,
                            color="black",
                            fill=True,
                            zorder=6,
                            ec="black",
                        )
                    )
                ax.text(
                    CUR_X + W_X,
                    START_Y - 3 * B_Y / 4,
                    note,
                    c="white",
                    va="center",
                    ha="center",
                    zorder=6,
                )

            CUR_X += W_X

    ax.set_title("Piano")
    plt.xlim(
        [START_X - W_X, CUR_X + W_X]
    )  # one +1 for the open fret, other +1 for the boundary
    plt.ylim([START_Y - W_Y, START_Y])
    plt.tight_layout()
    plt.axis("off")
    plt.show()
