import enum
from typing import Iterable, Optional, Tuple
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

class Actions(enum.IntEnum):
    STAY = 4
    LEFT = 2
    UP = 0
    RIGHT = 3
    DOWN = 1

ACTION_DELTA = {
Actions.UP: (0, 1),
Actions.DOWN: (0, -1),
Actions.LEFT: (-1, 0),
Actions.RIGHT: (1, 0),
Actions.STAY: (0, 0)}
CORNERS = [(0, 0), (0, 1), (1, 1), (1, 0)]
OFFSETS = {
    direction: np.array(
        [CORNERS[action.value - 1], [0.5, 0.5], CORNERS[action.value % len(CORNERS)]]
    )
    for action, direction in ACTION_DELTA.items()}
OFFSETS[(0, 0)] = np.array([0.5, 0.5])

def reward_draw_spline(
    action: int,
    reward: float,
    mappable: matplotlib.cm.ScalarMappable,
    annot_padding: float,
    from_dest: bool=False,
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    # Compute shape position and color
    pos = np.array([0, 0])
    direction = np.array(ACTION_DELTA[action])
    if from_dest:
        pos = pos + direction
        direction = -direction
    vert = pos + OFFSETS[tuple(direction)]
    color = mappable.to_rgba(reward)
    xy = pos + 0.5
    if tuple(direction) != (0, 0):
        xy = xy + annot_padding * direction
    return vert, color

def _make_triangle(vert, color, **kwargs):
    return mpatches.Polygon(xy=vert, facecolor=color, **kwargs)

def _make_circle(vert, color, radius, **kwargs):
    return mpatches.Circle(xy=vert, radius=radius, facecolor=color, **kwargs)

def _make_rectangle(vert,w,h, color, **kwargs):
    return mpatches.Rectangle(xy=vert,width=w,height=h,facecolor=color, **kwargs)

def ax_draw(
    state_rewards: np.ndarray,
    fig: plt.Figure,
    ax: plt.Axes,
    mappable: matplotlib.cm.ScalarMappable,
) -> None:
    #adjust circle size
    circle_radius=0.2
    annot_padding = 0.25 + 0.5 * circle_radius
    for a,v in enumerate(state_rewards):
        vert, color = reward_draw_spline(a,v,mappable, annot_padding)
        #linewidth=0.15, edgecolor=edgecolor
        if a== Actions.STAY:
            t=_make_circle(vert,tuple(color),radius=circle_radius)
        else:t= _make_triangle(vert,tuple(color))
        #the STAY action (4) is the last one to cover triangles
        ax.add_patch(t)
    ax.set_aspect("equal", adjustable="box")

def color_map(
    rewards: Iterable[np.ndarray],
    vmin: Optional[float]= None,
    vmax: Optional[float]= None,
    normalizer=mcolors.Normalize,
) -> matplotlib.cm.ScalarMappable:
    if vmin is None:vmin = rewards.min()
    if vmax is None:vmax = rewards.max()
    norm = normalizer(vmin=vmin, vmax=vmax)
    return matplotlib.cm.ScalarMappable(norm=norm)

def _set_ticks(n: int, subaxis: matplotlib.axis.Axis) -> None:
    subaxis.set_ticks(np.arange(0, n + 1), minor=True)
    subaxis.set_ticks(np.arange(n) + 0.5)
    # subaxis.set_ticklabels(np.arange(n))
    for tick in subaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

def axis_formatting(ax: plt.Axes, xlen: int, ylen: int) -> None:
    # Axes limits
    ax.set_xlim(0, xlen)
    ax.set_ylim(0, ylen)
    # Make ticks centred in each cell
    _set_ticks(xlen, ax.xaxis)
    _set_ticks(ylen, ax.yaxis)
    # Draw grid along minor ticks, then remove those ticks so they don't protrude
    ax.grid(which="minor", color="k", linewidth=0.25)
    ax.tick_params(which="minor", length=0, width=0)

def plot_grid_rewards(
        h,w,
        rewards,
        colorbar_size=0.25,
        name="rewards",
        style="Q",
        ):
    fig = plt.figure(figsize=(8.5, 8.5))
    ncols = w;nrows = h
    width_ratios = [1] * ncols + [colorbar_size]*2
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols+2, width_ratios=width_ratios)
    base_ax = fig.add_subplot(gs[0,0])
    if style=="Q":mappable = color_map(rewards)
    elif style=="V":mappable=color_map(np.array([max(l) for l in rewards]))
    for idx in range(len(rewards)):
        i = idx // w
        j = idx % w
        if i == 0 and j == 0:ax = base_ax
        else:ax = fig.add_subplot(gs[i, j], sharex=base_ax, sharey=base_ax)
        axis_formatting(ax, 1, 1)
        if not ax.is_last_row():ax.tick_params(axis="x", labelbottom=False)
        if not ax.is_first_col():ax.tick_params(axis="y", labelleft=False)
        if style=="Q":ax_draw(rewards[idx],fig=fig, ax=ax, mappable=mappable)
        elif style=="V":
            r=mappable.to_rgba(max(rewards[idx]))
            ax.add_patch(_make_rectangle((0,0),1,1,tuple(r)))
    plt.subplots_adjust(wspace=0,hspace=0)
    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(mappable=mappable, cax=cax, format="%.1f")
    return fig

if __name__ =="__main__":
    rewards=np.random.rand(64,5)
    fig=plot_grid_rewards(8,8,rewards,style="Q")
    #plt.suptitle("rewards",fontsize = 20, color = 'blue',backgroundcolor='white')
    fig.show()
    
    # fig=plt.figure()
    # gs = fig.add_gridspec(5,5)
    # ax=fig.add_subplot(gs[0,0])
    # ax=fig.add_subplot(gs[0,1])
    # #c=mappable.to_rgba(reward)
    # t= _make_rectangle((0,0),1,1,"black")
    # #the STAY action (4) is the last one to cover triangles
    # ax.add_patch(t)