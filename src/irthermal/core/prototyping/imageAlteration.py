import matplotlib.patches
import numpy
import matplotlib.pyplot as plt

data = numpy.load("pi-rack-28478.npy")

print(data)
print("Shape:", data.shape)
print(type(data.shape))
print("Data type:", data.dtype)
print("Dimensions:", data.ndim)

plt.imshow(data, cmap="inferno")
plt.colorbar(label="Temperature")
plt.title("Thermal Image")
plt.show()

highest = numpy.max(data)
lowest = numpy.min(data)

#tolerance = 200
tolerance = (highest - lowest) / 4

validPixels = []
groups = []
squares = []

height, width = data.shape


# for tolerated heat spots
for y in range(0, data.shape[0]):
    for x in range(0, data.shape[1]):
        if data[y, x] > highest - tolerance:
            validPixels.append((y, x))

        else:
            data[y, x] = highest - tolerance
            pass


unvisited = set(validPixels)
groups = []

while unvisited:

    # Start a new group
    start = unvisited.pop()
    group = [start]
    stack = [start]

    while stack:
        y, x = stack.pop()

        # Check all 3x3 neighbouring pixels
        for dy in range(-1, 2):
            for dx in range(-1, 2):

                neighbour = (y + dy, x + dx)

                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    group.append(neighbour)
                    stack.append(neighbour)

    groups.append(group)

print(len(groups))

fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(data, cmap="inferno")
ax.set_title("Thermal Image")

for group in groups:
    checkingGroup = set(group)
    start = checkingGroup.pop()

    square = [start[0], start[1], start[0], start[1]]

    if len(group) > 50:
        while checkingGroup:
            nextPixel = checkingGroup.pop()

            if nextPixel[0] < square[0]:
                square[0] = nextPixel[0]
            if nextPixel[0] > square[2]:
                square[2] = nextPixel[0]
            if nextPixel[1] < square[1]:
                square[1] = nextPixel[1]
            if nextPixel[1] > square[3]:
                square[3] = nextPixel[1]

        rect = matplotlib.patches.Rectangle(
            (square[1], square[0]),
            square[3] - square[1] + 1,
            square[2] - square[0] + 1,
            linewidth=1,
            edgecolor="green",
            facecolor="none"
        )

        ax.add_patch(rect)

plt.colorbar(ax.images[0], ax=ax, label="Temperature")
plt.show()

