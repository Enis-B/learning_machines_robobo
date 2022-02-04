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

    rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.209")
    #rob = robobo.HardwareRobobo(camera=True).connect(address="10.0.0.199")
    #rob = robobo.SimulationRobobo().connect(address='172.28.192.1', port=19997)

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
            rob.play_simulation()
            rob.set_phone_tilt(76.31, 5) ## 76.71
            for i in range(eval_time):
                # Following code gets an image from the camera
                image = rob.get_image_front()
                # IMPORTANT! `image` returned by the simulator is BGR, not RGB
                cv2.imwrite("test_pictures.png",image)

                '''
                ## ORB Detection
                img = cv2.imread('test_pictures.png',-1)
                # Initiate ORB detector
                orb = cv2.ORB_create()
                # find the keypoints with ORB
                kp = orb.detect(img,None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)
                #print(type(des),len(des),des)
                #print(type(kp),len(kp),kp)
                # draw only keypoints location,not size and orientation
                img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
                cv2.imwrite("test_pictures_keypoints.png",img2)
                #plt.imshow(img2), plt.show()
                cv2.imshow("Keypoints", img2)
                cv2.waitKey(1)
                '''

                ## Blob Detection
                # Read image
                im = cv2.imread("test_pictures.png", -1)
                # Set up the detector with default parameters.
                detector = cv2.SimpleBlobDetector_create()

                '''
                # Setup SimpleBlobDetector parameters.
                params = cv2.SimpleBlobDetector_Params()

                # Change thresholds
                params.minThreshold = 10
                params.maxThreshold = 200

                # Filter by Area.
                params.filterByArea = True
                params.minArea = 100

                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = 0.1
                params.maxCircularity = 0.785

                # Filter by Convexity
                params.filterByConvexity = True
                params.minConvexity = 0.87

                # Filter by Inertia
                params.filterByInertia = True
                params.minInertiaRatio = 0.01

                # Filter by Color
                params.filterByColor = True
                params.blobColor = 255

                # Create a detector with the parameters
                ver = (cv2.__version__).split('.')
                if int(ver[0]) < 3 :
                    detector = cv2.SimpleBlobDetector(params)
                else :
                    detector = cv2.SimpleBlobDetector_create(params)
                '''

                # Detect blobs.
                keypoints = detector.detect(im)
                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
                im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # Show keypoints
                cv2.imshow("Blobs", im_with_keypoints)
                cv2.waitKey(1)
                #cv2.imwrite("test_pictures_blobs.png",im_with_keypoints)
                #plt.imshow(im_with_keypoints), plt.show()

                '''
                len_kp = [0]

                if isinstance(kp, list):
                    len_kp = [len(kp)]

                #inputs = np.array(len_kp) ## next: add sensors make it log 10 (?)
                '''

                len_blobs = [0]

                if keypoints:
                    len_blobs = [len(keypoints)]

                inputs = np.array(len_blobs)

                #inputs = np.array(len_kp + len_blobs)

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
                sensor_inputs = np.log(np.array(rob.read_irs()))/10
                for i in range(len(sensor_inputs)):
                    if math.isinf(sensor_inputs[i]):
                        sensor_inputs[i] = 0
                    if sensor_inputs[i] < 0:
                        sensor_inputs[i] = abs(sensor_inputs[i] * 2)
                    if sensor_inputs[i] > 1:
                        sensor_inputs[i] = 1
                #print("ROB Irs: {}".format(sensor_inputs))
                #print("ROB Irs: {}".format(inputs))
                #new_inputs = np.append(inputs,np.array(len_kp))

                inputs = np.concatenate([inputs,sensor_inputs])
                '''

                print("Inputs: ",inputs)

                ## Getting output from input after feeding it to the network
                outputs = net.activate(inputs)
                #print(outputs)

                ## setting max speed of motors
                speed = 40
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

                rob_position = rob.position()
                print("robobo is at {}".format(rob_position))
                base_position = rob.base_position()
                print("Base position: ", base_position)
                food_position = rob.food_position()
                print("Food position: ", food_position)


                pos_diff_food = [0,0,0]
                for i in range(3):
                    pos_diff_food[i] = abs(rob_position[i] - food_position[i])

                sum_diff_food = sum(pos_diff_food)


                if sum_diff_food > 0:
                    sum_diff_food = -sum_diff_food

                pos_diff_base = [0,0,0]
                for i in range(3):
                    pos_diff_base[i] = abs(base_position[i] - food_position[i])

                sum_diff_base = sum(pos_diff_base)

                if sum_diff_base > 0:
                    sum_diff_base = -sum_diff_base

                sum_diff = sum_diff_base + sum_diff_food

                fitness += sum_diff #* len(kp) * len(keypoints) * max(sensor_inputs)
                print("Fitness: ",fitness,"\n")

                foraged = rob.base_detects_food()

                if foraged:
                    fitness = 0
                    break
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
            genome.fitness = fitness
            print("Genome fitness: ", genome.fitness)
            # Stopping the simualtion resets the environment
            rob.stop_world()
            rob.wait_for_stop()


    ## evaluation of single genome
    def eval_genome(genome, config):
        genome_list = []
        for i in range(5):
            #fitness_list = []
            eval_time = 120
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = -1
            #rob.play_simulation()
            rob.set_phone_tilt(180, 100)
            for i in range(eval_time):
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
                time.sleep(0.1)
                image = rob.get_image_front()
                # IMPORTANT! `image` returned by the simulator is BGR, not RGB
                cv2.imwrite("test_pictures.png",image)

                ## HSV green, red mask (image filtering)
                hsv= cv2.cvtColor( image, cv2.COLOR_BGR2HSV)

                lower_green = np.array([66,105, 70])
                upper_green = np.array([112, 255, 255])

                lower_red = np.array([240,70, 70])
                upper_red = np.array([255, 255, 255])

                lower_red_2 = np.array([0,70, 70])
                upper_red_2 = np.array([12, 255, 255])

                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                mask_red = cv2.inRange(hsv, lower_red,upper_red)
                mask_red_2 = cv2.inRange(hsv, lower_red_2,upper_red_2)

                mask = mask_green + mask_red + mask_red_2
                res = cv2.bitwise_and(image, image, mask=mask)

                cv2.imwrite("hsv_res.png",res)
                #cv2.imshow('res1.png', res)
                #cv2.waitKey(0)


                '''
                img = cv2.imread('hsv_res.png',-1)
                # Initiate ORB detector
                orb = cv2.ORB_create()
                # find the keypoints with ORB
                kp = orb.detect(img,None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)
                # draw only keypoints location,not size and orientation
                #img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
                #plt.imshow(img2), plt.show()
                #cv2.imshow("Keypoints", img2)
                #cv2.waitKey(1)
                '''

                # Read image
                im = cv2.imread("test_pictures.png", -1)
                # Set up the detector with default parameters.
                detector = cv2.SimpleBlobDetector_create()
                # Detect blobs.
                keypoints = detector.detect(im)
                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
                #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # Show keypoints
                #cv2.imshow("Blobs", im_with_keypoints)
                #cv2.waitKey(1)
                #cv2.imwrite("test_pictures_blobs.png",im_with_keypoints)
                #plt.imshow(im_with_keypoints), plt.show()

                '''
                len_kp = [0]

                if isinstance(kp, list):
                    len_kp = [int(len(kp)/10)]
                    #len_kp = [len(kp)]

                #inputs = np.array(len_kp)

                len_blobs = [0]

                if keypoints:
                    len_blobs = [len(keypoints)]

                #inputs = np.array(len_blobs)

                inputs = np.array(len_kp + len_blobs)
                '''

                ## Get first keypoint position (x,y) + blob size, divide by factor to make similar to sim.
                if keypoints:
                    key_list = [keypoints[0].pt[0]/3.75,keypoints[0].pt[1]/5,keypoints[0].size]
                else:
                    key_list = [0,0,0]
                key_arr = np.array(key_list)

                inputs = np.array(key_arr)

                '''
                ## input transformation
                sensor_inputs = np.log(np.array(rob.read_irs()))/10
                for i in range(len(sensor_inputs)):
                    if math.isinf(sensor_inputs[i]):
                        sensor_inputs[i] = 0
                    if sensor_inputs[i] < 0:
                        sensor_inputs[i] = abs(sensor_inputs[i] * 2)
                    if sensor_inputs[i] > 1:
                        sensor_inputs[i] = 1
                print("ROB Irs: {}".format(sensor_inputs))
                #print("ROB Irs: {}".format(inputs))
                #new_inputs = np.append(inputs,np.array(len_kp))
                inputs = np.concatenate([inputs,sensor_inputs])
                '''

                print("Inputs: ",inputs)
                outputs = net.activate(inputs)
                #print(outputs)
                speed = 30
                for i in range(2):
                    outputs[i] = speed * (outputs[i]) ## prev: 10 speed
                #print(outputs)
                #genome.fitness = 1 - max(inputs)
                #print("Proximity sensor: ", inputs)
                ## 1s
                rob.move(int(outputs[0]),int(outputs[1]),500)
                print("LMS: ",int(outputs[0]),"RMS: ",int(outputs[1]))
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

                '''
                rob_position = rob.position()
                print("robobo is at {}".format(rob_position))
                base_position = rob.base_position()
                print("Base position: ", base_position)
                food_position = rob.food_position()
                print("Food position: ", food_position)
                '''
                '''
                pos_diff_food = [0,0,0]
                for i in range(3):
                    pos_diff_food[i] = abs(rob_position[i] - food_position[i])

                sum_diff_food = sum(pos_diff_food)


                if sum_diff_food > 0:
                    sum_diff_food = -sum_diff_food

                pos_diff_base = [0,0,0]
                for i in range(3):
                    pos_diff_base[i] = abs(base_position[i] - food_position[i])

                sum_diff_base = sum(pos_diff_base)

                if sum_diff_base > 0:
                    sum_diff_base = -sum_diff_base

                sum_diff = sum_diff_base + sum_diff_food

                fitness += sum_diff #* len(kp) * len(keypoints) * max(sensor_inputs)
                print("Fitness: ",fitness,"\n")

                foraged = rob.base_detects_food()
                print("Foraged: ", foraged)

                if foraged:
                    fitness = 0
                    break
                '''

            '''
            ## total fitness of genome
            # pause the simulation and read the collected food
            #rob.pause_simulation()
            ## Amound of food collected in the run
            #food = rob.collected_food()
            '''
            genome.fitness = fitness
            '''
            genome_list = genome_list + [genome.fitness]
            print("Genome fitness: ", genome.fitness)
            # Stopping the simualtion resets the environment
            rob.stop_world()
            rob.wait_for_stop()
            '''
        '''
        plt.boxplot(genome_list)
        plt.title("Genome fitness over 5 runs")
        plt.ylabel("Genome fitness")
        plt.xlabel("Genome")
        plt.ylim([-500, 50])
        plt.show()
        '''

    def run(config_file):
        ## Choose between test or train
        mode = 'test'
        if mode == 'test':
            ## Validation

            #p = neat.Checkpointer.restore_checkpoint('experiments/100pop_100gen_10s_nobackpenalty_0.5s/neat-checkpoint-98')
            #p.run(eval_genomes, 10)
            with open ('experiments/forager_10pop_10gen_30sp_blob_120eval/winner', 'rb') as fp:
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
    run('src/config_file_hardware')

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
