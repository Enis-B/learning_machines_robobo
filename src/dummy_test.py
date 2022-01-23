#!/usr/bin/env python2
from __future__ import print_function
## Simulation robobo
import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
import math
import os
import neat
import visualize
import pickle
import matplotlib.pyplot as plt

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    #rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.98")
    rob = robobo.SimulationRobobo("#0").connect(address='172.29.0.1', port=19997)

    rob.play_simulation()

    time.sleep(0.3)
    #print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    '''
    # IR reading
    for i in range(1000000):
        for i in range(1):
            #print("robobo is at {}".format(rob.position()))
            rob.move(5, 5, 1000)
        print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
        time.sleep(0.1)
        sensor_data = np.log(np.array(rob.read_irs()))/10
        cnt_s = 0
        for sensor in sensor_data:
            if not math.isinf(sensor):
                cnt_s += 1

        if cnt_s > 1:
            # Following code moves the robot
            for i in range(1):
                #print("robobo is at {}".format(rob.position()))
                rob.move(0, 5, 5000)
    #print("robobo is at {}".format(rob.position()))
    #rob.sleep(1)
    '''

    # Hyperbolic Tangent (htan) Activation Function
    def htan(x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # htan derivative
    def der_htan(x):
        return 1 - htan(x) * htan(x)

    ## sigmoid activation function
    def sigmoid_activation(x):
        return 1./(1.+np.exp(-x))

    ## function for evaluation of genomes of population
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            eval_time = 120
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = 0
            for i in range(eval_time):
                ## input transformation
                inputs = np.log(np.array(rob.read_irs()))/10
                for i in range(len(inputs)):
                    if math.isinf(inputs[i]):
                        inputs[i] = 0
                    if inputs[i] < 0:
                        inputs[i] = abs(inputs[i] * 2)
                    if inputs[i] > 1:
                        inputs[i] = 1
                print("ROB Irs: {}".format(inputs))
                #print("ROB Irs: {}".format(inputs))

                ## Getting output from input after feeding it to the network
                outputs = net.activate(inputs)
                #print(outputs)

                ## setting baseline speed of motors
                speed = 15
                for i in range(2):
                    outputs[i] = speed * (outputs[i]) ## prev: 10 speed
                #print(outputs)
                #genome.fitness = 1 - max(inputs)
                #print("Proximity sensor: ", inputs)
                ## 1s

                ## moving the robot
                rob.move(int(outputs[0]),int(outputs[1]),500)
                print("LMS: ",int(outputs[0]),"RMS: ",int(outputs[1]))
                ## prev: disable backward penalty

                ## output transformation
                if outputs[0] < 0:
                    outputs[0] = outputs[0]/speed
                if outputs[1] < 0:
                    outputs[1] = outputs[1]/speed
                # prev:disable normalization for sum of motor speeds, next: no abs for sum

                ## fitness function (cumulative over eval. time)
                fitness += (abs(outputs[0]) + abs(outputs[1])) * abs(1-abs(outputs[0]/speed - outputs[1]/speed)) * (1 - max(inputs)) #* (10-sum(inputs))
                print("fitness: ", fitness)
            ## total fitness of genome
            genome.fitness = fitness
            print("Genome fitness: ", fitness)

    ## evaluation of single genome
    def eval_genome(genome, config):
        fitness_list = []
        eval_time = 120
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for i in range(3):
            for i in range(eval_time):
                inputs = np.log(np.array(rob.read_irs()))/10
                for i in range(len(inputs)):
                    if math.isinf(inputs[i]):
                        inputs[i] = 0
                    if inputs[i] < 0:
                        inputs[i] = abs(inputs[i] * 2)
                    if inputs[i] > 1:
                        inputs[i] = 1
                for i in range(len(inputs)):
                    #print(np.round(inputs[i]/3,1))
                    #inputs[i] = np.round(inputs[i]/3,1)

                    ## Input transformation ( Controlling strength of sensor,
                    # tradeoff between turning of robot when obstacle detected and distance to object)
                    if np.round(inputs[i]/3,1) == 0.1:
                        inputs[i] = 0

                print("ROB Irs: {}".format(inputs))
                #print("ROB Irs: {}".format(inputs))
                outputs = net.activate(inputs)
                #print(outputs)
                speed = 15
                for i in range(2):
                    outputs[i] = speed * (outputs[i]) ## prev: 10 speed
                #print(outputs)
                #genome.fitness = 1 - max(inputs)
                #print("Proximity sensor: ", inputs)
                ## 1s
                rob.move(int(outputs[0]),int(outputs[1]),500)
                print("LMS: ",int(outputs[0]),"RMS: ",int(outputs[1]))
                ## prev: disable backward penalty

                if outputs[0] < 0:
                    outputs[0] = outputs[0]/speed
                if outputs[1] < 0:
                    outputs[1] = outputs[1]/speed

                # prev:disable normalization for sum of motor speeds, next: no abs for sum
                fitness += (abs(outputs[0]) + abs(outputs[1])) * abs(1-abs(outputs[0]/speed - outputs[1]/speed)) * (1 - max(inputs)) #* (10-sum(inputs))
                print("fitness: ", fitness)
                fitness_list.append(fitness)
        genome.fitness = fitness
        print("Genome fitness: ", fitness)
        plt.plot(fitness_list,range(3 * eval_time))
        plt.ylabel("Fitness")
        plt.xlabel("Evaluations")
        plt.title("Fitness over time")
        plt.show()

    def run(config_file):
        ## Choose between test or train
        mode = 'test'
        if mode == 'test':
            ## Validation

            #p = neat.Checkpointer.restore_checkpoint('experiments/100pop_100gen_10s_nobackpenalty_0.5s/neat-checkpoint-98')
            #p.run(eval_genomes, 10)
            with open ('experiments/10pop_10gen_15sp_0.5s_1penalty/winner', 'rb') as fp:
                real_winner = pickle.load(fp)
            # Load configuration.
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)
            eval_genome(real_winner,config)
        else:
            ## Training

            # Load configuration.
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(5))

            # Run for up to x generations.
            winner = p.run(eval_genomes, 10)

            # Display the winning genome (in population of last generation)

            print('\nBest genome:\n{!s}'.format(winner))

            ## Overall best genome
            real_winner = stats.best_genome()

            '''
            ## Show output of the most fit genome against training data.
            #print('\nOutput:')
            #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            #for xi, xo in zip(xor_inputs, xor_outputs):
            #    output = winner_net.activate(xi)
            #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

            #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}

            #visualize.draw_net(config, winner, True, node_names=node_names)
            '''

            # Plots for fitness, network and speciation
            visualize.draw_net(config, winner, True)
            visualize.plot_stats(stats, ylog=False, view=True)
            visualize.plot_species(stats, view=True)

            ##
            with open('best_genome', 'wb') as fp:
                pickle.dump(real_winner, fp)
            with open('winner', 'wb') as fp:
                pickle.dump(winner, fp)
            #with open ('best_genome', 'rb') as fp:
            #    real_winner = pickle.load(fp)

            #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
            #p.run(eval_genomes, 10)

    ## run train or validation with neat config. file
    run('src/config_file')

    '''
    # Following code moves the phone stand
    #rob.set_phone_pan(343, 100)
    #rob.set_phone_tilt(109, 100)
    #time.sleep(1)
    #rob.set_phone_pan(11, 100)
    #rob.set_phone_tilt(26, 100)

    # Following code makes the robot talk and be emotional
    #rob.set_emotion('happy')
    #rob.talk('Hi, my name is Robobo')
    #rob.sleep(1)
    #rob.set_emotion('sad')

    # Following code gets an image from the camera
    #image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    #cv2.imwrite("test_pictures.png",image)
    '''

    time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
