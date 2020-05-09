import matplotlib.pyplot as plt

file = open('/home/arjun/Courseworks/PlanningCourse/RRT--Path-Replanning/nodeReplannedPath.txt', 'r')
lines = file.readlines()

x = []
y = []
for line in lines:
	coords = line.rstrip().split(',')
	x.append(float(coords[0]))
	y.append(float(coords[1]))

	plt.plot(x, y, color='r')
	plt.pause(0.1)
plt.show()