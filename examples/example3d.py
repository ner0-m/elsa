import pyelsa as elsa
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

res = 600
size = np.array([res, res, res])  # 3D now

start = time.time()
phantom = elsa.phantoms.modifiedSheppLogan(size)
timeNew = time.time() - start

reconstruction = np.array(phantom)


start = time.time()
phantom2 = elsa.phantoms.old.modifiedSheppLogan(size)
timeOld = time.time() - start

reconstruction2 = np.array(phantom2)

print("old ", timeOld, "s -  new ", timeNew, "s   -> factor ", timeNew/timeOld)

reconstruction3 = reconstruction - reconstruction2


class IndexTracker:
    def __init__(self, rec):
        self.axs = []
        self.data = []
        self.slicingPerspective = 2
        self.slicesX, self.slicesY, self.slicesZ = rec.shape
        self.indX = self.slicesX // 2
        self.indY = self.slicesY // 2
        self.indZ = self.slicesZ // 2

    def add(self, ax, rec):
        self.axs.append(ax)
        ax.set_title(
            'scroll or use ←/→ keys to navigate \nd key to change perspective \nesc to close')
        self.data.append((ax.imshow(rec[:, :, self.indZ]), rec))
        self.update()
        return self

    def on_scroll(self, event):
        if event.button == 'up':
            self.move(1)
        else:
            self.move(-1)
        self.update()

    def on_click(self, event):
        if event.key == "escape":
            plt.close()
            sys.exit()
        if event.key == 'right':
            self.move(10)
        if event.key == 'left':
            self.move(-10)
        if event.key == 'd':
            self.slicingPerspective = (self.slicingPerspective + 1) % 3
        self.update()

    def getSplitValue(self):
        if self.slicingPerspective == 0:
            return self.indX
        if self.slicingPerspective == 1:
            return self.indY
        if self.slicingPerspective == 2:
            return self.indZ

    def move(self, steps):
        if self.slicingPerspective == 0:
            self.indX = (self.indX + steps) % self.slicesX
        if self.slicingPerspective == 1:
            self.indY = (self.indY + steps) % self.slicesY
        if self.slicingPerspective == 2:
            self.indZ = (self.indZ + steps) % self.slicesZ

    def getDimensionText(self):
        if self.slicingPerspective == 0:
            return "X"
        if self.slicingPerspective == 1:
            return "Y"
        if self.slicingPerspective == 2:
            return "Z"

    def getSliceData(self, rec):
        if self.slicingPerspective == 0:
            return rec[self.indX, :, :]
        if self.slicingPerspective == 1:
            return rec[:, self.indY, :]
        if self.slicingPerspective == 2:
            return rec[:, :, self.indZ]

    def update(self):
        splitValue = self.getSplitValue()
        dimension = self.getDimensionText()
        for ax in self.axs:
            ax.set_ylabel('slice %s in %s' % (splitValue, dimension))

        for (im, rec) in self.data:
            im.set_data(self.getSliceData(rec))
            im.axes.figure.canvas.draw_idle()
            im.axes.figure.canvas.draw()


fig, ax = plt.subplots(1, 1, num="New approach")
tracker = IndexTracker(reconstruction).add(ax, reconstruction)

fig2, ax2 = plt.subplots(1, 1, num="Old approach")
tracker.add(ax2, reconstruction2)

fig3, ax3 = plt.subplots(1, 1, num="Diff")
tracker.add(ax3, reconstruction3)


fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
fig.canvas.mpl_connect('key_press_event', tracker.on_click)
fig2.canvas.mpl_connect('scroll_event', tracker.on_scroll)
fig2.canvas.mpl_connect('key_press_event', tracker.on_click)
fig3.canvas.mpl_connect('scroll_event', tracker.on_scroll)
fig3.canvas.mpl_connect('key_press_event', tracker.on_click)

plt.show()
