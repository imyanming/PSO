import numpy as np
import math
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading
import matplotlib.animation

# this is the top level function
# This is the fitness function 收斂到最小值
def fitness(x, y):
    return x * x + y * y

# to find the best position of the particle
class Partical(object): 
    def __init__(self, downbound, upperbound): 
        # initialize the position of the particle
        self.position = [0.0, 0.0] 
        self.position[0] = np.random.uniform(downbound, upperbound) 
        self.position[1] = np.random.uniform(downbound, upperbound)

        # initialize the velocity of the particle
        self.velocity = [0.0, 0.0]

        
        self.best_position = [self.position[0], self.position[1]]
        self.best_value = fitness(self.position[0], self.position[1])

# The class of the PSO algorithm
# 3 
class PSO_Algorithm(object):
    def __init__(self):
        self.lower_bound = -10
        self.upper_bound = 10
        self.particles = []

        self.global_best_position = None
        self.global_best_value = 999999999.0

    def createParticals(self, num_particles):
        self.particles = []
        for i in range(num_particles):
            new_particle = Partical(self.lower_bound, self.upper_bound)
            self.particles.append(new_particle)
            if new_particle.best_value < self.global_best_value:
                self.global_best_value = new_particle.best_value
                self.global_best_position = [new_particle.best_position[0], new_particle.best_position[1]]

    def search(self):
        w = 0.5   
        c1 = 1.0  
        c2 = 1.0 

        for particle in self.particles:
            for i in range(2):
                r1 = np.random.random()
                r2 = np.random.random()
                cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i])
                social = c2 * r2 * (self.global_best_position[i] - particle.position[i])
                particle.velocity[i] = w * particle.velocity[i] + cognitive + social
                particle.position[i] = particle.position[i] + particle.velocity[i]
                if particle.position[i] < self.lower_bound:
                    particle.position[i] = self.lower_bound
                if particle.position[i] > self.upper_bound:
                    particle.position[i] = self.upper_bound

            current_value = fitness(particle.position[0], particle.position[1])

            if current_value < particle.best_value:
                particle.best_value = current_value
                particle.best_position = [particle.position[0], particle.position[1]]

            if current_value < self.global_best_value:
                self.global_best_value = current_value
                self.global_best_position = [particle.position[0], particle.position[1]]

    def getBestParameter(self):
        return self.global_best_position, self.global_best_value


# This is the top level function after animate_pso()
# 2
def run_pso_with_history(num_particles, total_epoch):
    pso = PSO_Algorithm()
    pso.createParticals(num_particles)

    position_history = []
    best_value_history = []
    gbest_position_history = []

    for epoch in range(total_epoch):
        pso.search()

        positions_this_epoch = [particle.position[:] for particle in pso.particles]
        position_history.append(positions_this_epoch)
        best_value_history.append(pso.global_best_value)
        gbest_position_history.append([pso.global_best_position[0], pso.global_best_position[1]])

    return position_history, best_value_history, gbest_position_history


# This is the top level function
# 1
def animate_pso():
    position_history, best_value_history, gbest_position_history = run_pso_with_history(10, 50)

    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    color_list = []
    for i in range(len(position_history[0])):
        color_list.append(plt.cm.tab10(i / float(len(position_history[0]))))

    zoom_half_width = 1.2

    def update(frame):
        ax.clear()

        gx = gbest_position_history[frame][0]
        gy = gbest_position_history[frame][1]
        if len(best_value_history) <= 1:
            t = 1.0
        else:
            t = frame / float(len(best_value_history) - 1)
        if t > 1.0:
            t = 1.0
        half = 10.0 * (1.0 - t) + zoom_half_width * t
        xlo = gx - half
        xhi = gx + half
        ylo = gy - half
        yhi = gy + half
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        for particle_id in range(len(position_history[0])):
            trace_x = []
            trace_y = []
            for epoch in range(frame + 1):
                trace_x.append(position_history[epoch][particle_id][0])
                trace_y.append(position_history[epoch][particle_id][1])
            ax.plot(trace_x, trace_y, color=color_list[particle_id], linewidth=1.5, alpha=0.8)

        xs = []
        ys = []
        for particle_id in range(len(position_history[0])):
            xs.append(position_history[frame][particle_id][0])
            ys.append(position_history[frame][particle_id][1])
        ax.scatter(xs, ys, s=60, c="blue", zorder=5, edgecolors="white")

        title_str = "Epoch " + str(frame + 1) + "/" + str(len(best_value_history)) + "   best = " + str(round(best_value_history[frame], 6))
        ax.set_title(title_str)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(best_value_history), interval=200, blit=False)
    plt.show()


# The top level function
def PSO_test_run():
    pso = PSO_Algorithm()
    pso.createParticals(10)

    total_epoch = 1000
    start_time = time.time()
    for epoch in range(total_epoch):
        pso.search()
    elapsed = time.time() - start_time

    best_position, best_value = pso.getBestParameter()
    print(best_position, best_value)
    print("Time:", round(elapsed, 4), "seconds")


# this is the entry point of the program
if __name__ == "__main__":
    animate_pso()


"""
cd /Users/imyanming/Documents/Vmware/PSO
source venv/bin/activate
python psotr.py
"""
