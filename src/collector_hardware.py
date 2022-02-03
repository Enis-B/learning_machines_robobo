#!/usr/bin/env python2
from __future__ import print_function
## hardware robobo
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

    rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.98")
    #rob = robobo.SimulationRobobo().connect(address='172.29.0.1', port=19997)

    #time.sleep(0.3)
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
    # Following code moves the phone stand
    #rob.set_phone_pan(343, 100)
    #time.sleep(0.5)
    #rob.set_phone_pan(11, 100)
    #rob.set_phone_tilt(26, 100)


    '''
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)
    '''

    '''
    # Read image
    im = cv2.imread("test_pictures.png", cv2.IMREAD_GRAYSCALE)
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    # Detect blobs.
    keypoints = detector.detect(im)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
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
            eval_time = 180
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            #fitness = 0
            #rob.play_simulation()
            rob.set_phone_tilt(76, 5)
            for i in range(eval_time):
                # Following code gets an image from the camera
                image = rob.get_image_front()
                # IMPORTANT! `image` returned by the simulator is BGR, not RGB
                cv2.imwrite("test_pictures.png",image)
                img = cv2.imread('test_pictures.png',0)
                # Initiate ORB detector
                orb = cv2.ORB_create()
                # find the keypoints with ORB
                kp = orb.detect(img,None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)
                #print(type(des),len(des),des)
                #print(type(kp),len(kp),kp)
                # draw only keypoints location,not size and orientation
                #img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
                #plt.imshow(img2), plt.show()

                len_kp = [0]

                if isinstance(kp, list):
                    len_kp = [len(kp)]

                inputs = np.array(len_kp) ## next: add sensors make it log 10 (?)

                '''
                inputs = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
                retry = 1
                if isinstance(des, np.ndarray):
                    retry = 0

                if retry == 0:
                    if len(des) > len(inputs):
                        for i in range(len(inputs)):
                            print(inputs[i],des[i])
                            inputs[i] = sum(des[i])
                    else:
                        for i in range(len(des)):
                            inputs[i] = sum(des[i])
                '''

                '''
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
                #new_inputs = np.append(inputs,np.array(len_kp))
                '''

                print("Keypoints: ",inputs)

                ## Getting output from input after feeding it to the network
                outputs = net.activate(inputs)
                #print(outputs)

                ## setting max speed of motors
                speed = 10
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

                '''
                ## output transformation
                if outputs[0] < 0:
                    outputs[0] = outputs[0]/speed
                if outputs[1] < 0:
                    outputs[1] = outputs[1]/speed
                # prev:disable normalization for sum of motor speeds, next: no abs for sum
                '''
                '''
                if max(inputs) == 1:
                    outputs[0] = 0
                    outputs[1] = 0
                '''
                '''
                ## fitness function (cumulative over eval. time)
                fitness += (abs(outputs[0]) + abs(outputs[1])) * abs(1-abs(outputs[0]/speed - outputs[1]/speed)) * max(inputs) #* (10-sum(inputs))
                print("fitness: ", fitness)
                '''

            ## total fitness of genome
            # pause the simulation and read the collected food
            #rob.pause_simulation()
            ## Amound of food collected in the run
            #food = rob.collected_food()
            #genome.fitness = food
            #print("Genome fitness: ", genome.fitness)
            # Stopping the simualtion resets the environment
            #rob.stop_world()
            #rob.wait_for_stop()


    ## evaluation of single genome
    def eval_genome(genome, config):
        genome_list = []
        for i in range(5):
            #fitness_list = []
            eval_time = 180
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            #fitness = 0
            #rob.play_simulation()
            rob.set_phone_tilt(140, 20)
            for i in range(eval_time):
                '''
                inputs = np.log(np.array(rob.read_irs()))/10
                for i in range(len(inputs)):
                    if math.isinf(inputs[i]):
                        inputs[i] = 0
                    if inputs[i] < 0:
                        inputs[i] = abs(inputs[i] * 2)
                    if inputs[i] > 1:
                        inputs[i] = 1
                '''
                '''
                #for i in range(len(inputs)):
                    #print(np.round(inputs[i]/3,1))
                    #inputs[i] = np.round(inputs[i]/3,1)
    
                    ## Input transformation ( Controlling strength of sensor,
                    # tradeoff between turning of robot when obstacle detected and distance to object)
                #    if np.round(inputs[i]/3,1) == 0.1:
                #        inputs[i] = 0
                
                #print("ROB Irs: {}".format(inputs))
                #print("ROB Irs: {}".format(inputs))
                '''
                # Following code gets an image from the camera
                image = rob.get_image_front()
                # IMPORTANT! `image` returned by the simulator is BGR, not RGB
                cv2.imwrite("test_pictures.png",image)
                img = cv2.imread('test_pictures.png',0)
                # Initiate ORB detector
                orb = cv2.ORB_create()
                # find the keypoints with ORB
                kp = orb.detect(img,None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)
                # draw only keypoints location,not size and orientation
                #img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
                #plt.imshow(img2), plt.show()
                #plt.savefig("Keypoints_hardware.png")

                '''
                # Read image
                im = cv2.imread("test_pictures.png", cv2.IMREAD_GRAYSCALE)
                # Set up the detector with default parameters.
                detector = cv2.SimpleBlobDetector_create()
                # Detect blobs.
                keypoints = detector.detect(im)
                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
                im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # Show keypoints
                #cv2.imshow("Blobs", im_with_keypoints)
                #cv2.waitKey(1)
                #cv2.imwrite("test_pictures_blobs.png",im_with_keypoints)
                plt.imshow(im_with_keypoints), plt.show()
                plt.savefig("Blobs_hardware.png")
                '''

                len_kp = [0]

                if isinstance(kp, list):
                    len_kp = [int(len(kp)/10)]## divide by 10, int

                '''
                len_blobs = [0]

                if keypoints:
                    len_blobs = [len(keypoints)]

                inputs = np.array(len_kp + len_blobs)
                '''

                inputs = np.array(len_kp)


                '''
                for i in range(1,len(inputs)):
                    #print(np.round(inputs[i]/4,1))
                    #inputs[i] = np.round(inputs[i]/4,1)
                    inputs[i] = np.round(np.true_divide(inputs,2),1)
                #print("ROB Irs: {}".format(np.round(np.true_divide(inputs,4),1)))
                '''

                #print("Keypoints: ",inputs)
                outputs = net.activate(inputs)
                #print(outputs)
                speed = 60
                for i in range(2):
                    outputs[i] = speed * (outputs[i]) ## prev: 10 speed
                #print(outputs)
                #genome.fitness = 1 - max(inputs)
                #print("Proximity sensor: ", inputs)
                ## 1s
                rob.move(int(outputs[0]),int(outputs[1]),500)
                #print("LMS: ",int(outputs[0]),"RMS: ",int(outputs[1]))
                ## prev: disable backward penalty
                '''
                if outputs[0] < 0:
                    outputs[0] = outputs[0]/speed
                if outputs[1] < 0:
                    outputs[1] = outputs[1]/speed
                '''
                '''
                if max(inputs) == 1:
                    outputs[0] = 0
                    outputs[1] = 0
                '''
                ''' 
                # prev:disable normalization for sum of motor speeds, next: no abs for sum
                fitness += (abs(outputs[0]) + abs(outputs[1])) * abs(1-abs(outputs[0]/speed - outputs[1]/speed)) * max(inputs) #* (10-sum(inputs))
                print("fitness: ", fitness)
                fitness_list.append(fitness)
                '''
            ## total fitness of genome
            # pause the simulation and read the collected food
            #rob.pause_simulation()
            ## Amound of food collected in the run
            #food = rob.collected_food()
            #genome.fitness = food
            #genome_list = genome_list + [genome.fitness]
            #print("Genome fitness: ", genome.fitness)
            # Stopping the simualtion resets the environment
            #rob.stop_world()
            #rob.wait_for_stop()
        #plt.boxplot(genome_list)
        #plt.title("Genome fitness over 5 runs")
        #plt.ylabel("Genome fitness")
        #plt.xlabel("Genome")
        #plt.show()

    def run(config_file):
        ## Choose between test or train
        mode = 'test'
        if mode == 'test':
            ## Validation

            #p = neat.Checkpointer.restore_checkpoint('experiments/100pop_100gen_10s_nobackpenalty_0.5s/neat-checkpoint-98')
            #p.run(eval_genomes, 10)
            with open ('experiments/collect_10sp_10pop_10gen_orbs/winner', 'rb') as fp:
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

            '''    
            #with open ('best_genome', 'rb') as fp:
            #    real_winner = pickle.load(fp)

            #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
            #p.run(eval_genomes, 10)
            '''

    ## run train or validation with neat config. file
    run('src/config_file')

    '''

    # Following code makes the robot talk and be emotional
    #rob.set_emotion('happy')
    #rob.talk('Hi, my name is Robobo')
    #rob.sleep(1)
    #rob.set_emotion('sad')

    '''

    #time.sleep(0.1)



if __name__ == "__main__":
    main()
