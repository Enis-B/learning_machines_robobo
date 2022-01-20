#!/usr/bin/env python2
from __future__ import print_function

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

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    #rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.98")
    rob = robobo.SimulationRobobo("#0").connect(address='172.29.0.1', port=19997)

    rob.play_simulation()

    #time.sleep(0.1)
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

    def sigmoid_activation(x):
        return 1./(1.+np.exp(-x))


    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            eval_time = 600
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = 0
            for i in range(eval_time):
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
                outputs = net.activate(inputs)
                #print(outputs)
                for i in range(2):
                    outputs[i] = 10 * (outputs[i]) ## prev: 10 speed
                #print(outputs)
                #genome.fitness = 1 - max(inputs)
                #print("Proximity sensor: ", inputs)
                ## 1s
                rob.move(outputs[0],outputs[1],500)
                ## prev: disable backward penalty

                #if outputs[0] < 0:
                #    outputs[0] = outputs[0]/2
                #if outputs[1] < 0:
                #    outputs[1] = outputs[1]/2

                # prev:disable normalization for sum of motor speeds, next: no abs for sum
                fitness += ((outputs[0]) + (outputs[1])) * (1-abs(outputs[0]/10 - outputs[1]/10)) * (1 - max(inputs)) #* (10-sum(inputs))
                print("fitness: ", fitness)
                print("LMS: ",outputs[0],"RMS: ",outputs[1])
            genome.fitness = fitness
            print("Genome fitness: ", fitness)



    def run(config_file):
        mode = 'train'
        if mode == 'test':
            p = neat.Checkpointer.restore_checkpoint('experiments/100pop_100gen_10s_nobackpenalty_0.5s/neat-checkpoint-98')
            p.run(eval_genomes, 10)
        else:
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
            winner = p.run(eval_genomes, 50)

            # Display the winning genome.
            print('\nBest genome:\n{!s}'.format(winner))
            real_winner = stats.best_genome()

            ## Show output of the most fit genome against training data.
            #print('\nOutput:')
            #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            #for xi, xo in zip(xor_inputs, xor_outputs):
            #    output = winner_net.activate(xi)
            #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

            #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}

            #visualize.draw_net(config, winner, True, node_names=node_names)
            visualize.draw_net(config, winner, True)
            visualize.plot_stats(stats, ylog=False, view=True)
            visualize.plot_species(stats, view=True)

            with open('best_genome', 'wb') as fp:
                pickle.dump(real_winner, fp)
            #with open ('best_genome', 'rb') as fp:
            #    real_winner = pickle.load(fp)


            #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
            #p.run(eval_genomes, 10)

    run('src/config_file')


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

    time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
