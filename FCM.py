
from PIL import Image
import random
import numpy
import pdb

from PIL import Image

import array
import logging

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math


class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.data = []  # intialize data into a list
        self.centroid = None  # set the number of centro
        # ids to none

    def addPoint(self, pixel):  # add data to the pixel list
        self.data.append(pixel)


class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=3,min_distance=3.0, size=150, m=2.0, epsilon=.5, max_FCM_iterations = 100):
        self.k = k  # initialize k clusters

        # intialize max_iterations
        self.max_FCM_iterations = max_FCM_iterations

        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = size
        self.size = size  # intialize the size
        self.m = m
        self.epsilon = 0.00001
        self.max_diff = 10.0
        self.data = []
        self.PCA = False
        self.Status = False

    # Takes in an image and performs FCM Clustering.
    def run(self):
        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))

        for i in range(self.s):
            num_1 = random.randint(1, 5) * 0.1
            num_2 = random.randint(1, 5) * 0.1
            num_3 = 1.0 - (num_1 + num_2)
            degreelist = [num_1, num_2, num_3]
            self.degree_of_membership[i] = degreelist

        randomdata = random.sample(self.data, self.k)
        print"INTIALIZE RANDOM data AS CENTROIDS"
        print randomdata
        #    print"================================================================================"
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomdata[idx]
            # if(i ==0):
        for cluster in self.clusters:
            for datapoint in self.data:
                cluster.addPoint(datapoint)

        print "________", self.clusters[0].data[0]
        iterations = 0

        # FCM

        self.PCA = True
        while self.shouldExitFCM(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            self.calculate_centre_vector()
            self.update_degree_of_membershipFCM()
            iterations += 1
        iterations = 0


        for cluster in self.clusters:
            print cluster.centroid
        return [cluster.centroid for cluster in self.clusters]

    def selectSingleSolution(self):
        self.max_FCM_iterations=5


    def shouldExitFCM(self, iterations):

        if self.max_diff < self.epsilon:
            print "--------------------------> max dif ", self.max_diff
            return True

        if iterations <= self.max_FCM_iterations:
            return False
        return True



    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        sum = (pow((a[0]-b[0]),2)) + (pow((a[1]-b[1]),2)) + (pow((a[2]-b[2]),2)) + (pow((a[3]-b[3]),2))
        result = numpy.sqrt(sum)
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        for cluster in range(self.k):
            sum_numerator = [0.0, 0.0, 0.0, 0.0]
            sum_denominator = 0.0
            for i in range(self.s):
                pow_uij= pow(self.degree_of_membership[i][cluster], self.m)
                sum_denominator +=pow_uij

                num=[ pow_uij * self.data[i][0], pow_uij * self.data[i][1], pow_uij * self.data[i][2], pow_uij * self.data[i][3]]


                sum_numerator[0]=sum_numerator[0] + num[0]
                sum_numerator[1]=sum_numerator[1] + num[1]
                sum_numerator[2]=sum_numerator[2] + num[2]
                sum_numerator[3]=sum_numerator[3] + num[3]
                # print "xxxxxxxxxxxx ", sum_numerator

            updatedcluster_center = [sum_numerator[0]/sum_denominator, sum_numerator[1]/sum_denominator, sum_numerator[2]/sum_denominator,
                                    sum_numerator[3]/sum_denominator]

            if self.PCA:
                self.max_diff = self.calcDistance(updatedcluster_center,self.clusters[cluster].centroid)

            self.clusters[cluster].centroid = updatedcluster_center






    # Updates the degree of membership for all of the data points.
    def update_degree_of_membershipFCM(self):
        self.max_diff = 0.0

        for idx in range(self.k):
            for i in range(self.s):
                new_uij = self.get_new_value(self.data[i], self.clusters[idx].centroid)
                if (i == 0):
                    print "This is the Updatedegree centroid number:", idx, self.clusters[idx].centroid
                diff = new_uij - self.degree_of_membership[i][idx]
                if (diff > self.max_diff):
                    self.max_diff = diff
                self.degree_of_membership[i][idx] = new_uij
        return self.max_diff

    def get_new_value(self, i, j):
        sum = 0.0
        val = 0.0
        p = (2 * (1.0) / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in self.clusters:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / denom
            val = pow(val, p)
            sum += val
        return (1.0 / sum)

   

    def defuzzification(self):

        for i in range(self.s):
            max = 0.0
            highest_index = 0
            # Find the index with highest probability
            for j in range(self.k):
                if (self.degree_of_membership[i][j] > max):
                    max = self.degree_of_membership[i][j]
                    highest_index = j
            # Normalize, set highest prob to 1 rest to zero

            # print self.degree_of_membership[i], max
            for j in range(self.k):

                if (j != highest_index):
                    self.degree_of_membership[i][j] = 0
                else:
                    self.degree_of_membership[i][j] = 1
                    # print j

        f = open('result.txt','w')
        for i in range(self.s):
                # Find the index with highest probability
            for j in range(self.k):
                if self.degree_of_membership[i][j] == 1:
                    # print self.data[i], j
                    f.write('%s \t' %self.data[i] )
                    f.write('%d \n' %j )

        f.close()

    def I_index(self):
        result = 0;
        Ek = 0.0
        E1 = 0.0
        for i in range(self.s):
            for j in range(self.k):
                # print "membership ",self.degree_of_membership[i][0][j]
                x = self.degree_of_membership[i][j]
                y = self.calcDistance(self.data[i], self.clusters[j].centroid)
                mul = x * y
                Ek += mul

                if j == 0:
                    E1 += mul

                distance_list = []
        distance_list = []
        for x in range(self.k):
            if x + 1 < self.k:
                distance_list.append(self.calcDistance(self.clusters[x].centroid, self.clusters[x + 1].centroid))
            else:
                distance_list.append(self.calcDistance(self.clusters[x].centroid, self.clusters[0].centroid))

            # print "distance list " , distance_list

        distance_list.sort()

            # sort the list
        print "distance list ", distance_list
        print "E1 ", E1, "Ek", Ek

        result = (1.0 / self.k) * (E1 / Ek) * distance_list[self.k - 1]
        print "result of pb before power", result
        result = pow(result, 2.0)
        print "PB", result
        return result

    def DB_index(self):

        # get the maximum of the Rij
        max_sum = 0.0

        for i in range(self.k):

            r_list = []
            for j in range(self.k):
                if i!=j:
                    var1 = self.getVariance(i)
                    var2 = self.getVariance(j)

                    sum_var = var1+var2
                    dis_cluster_center = self.calcDistance(self.clusters[i].centroid, self.clusters[j].centroid)
                    r_ij = sum_var/dis_cluster_center
                    r_list.append(r_ij)

            print ">>>>>>>> ", r_list

            #get the max Rij from list
            list.sort(r_list)
            print ">>>>>>>> sorted ", r_list, r_list[-1]

            #get the max of Rij, and store it
            max_sum +=r_list[-1]

        db = max_sum/self.k
        print "DB index ", db

        return db

    def analysis(self):

        x=0
        cl1 = 0
        cl2 = 0
        cl3 = 0

        sum = 0

        for i in range(self.s):
            
            x = x +1 

            if x % 50 != 0:
                
                for j in range(self.k):

                    if self.degree_of_membership[i][j] == 1:
                        if j == 0:
                            cl1 = cl1 + 1

                        elif j == 1 :
                            cl2 = cl2 + 1

                        elif j == 2:
                            cl3 = cl3 + 1
            else:

                for j in range(self.k):

                    if self.degree_of_membership[i][j] == 1:
                        if j == 0:
                            cl1 = cl1 + 1

                        elif j == 1 :
                            cl2 = cl2 + 1

                        elif j == 2:
                            cl3 = cl3 + 1
                        
                print "first class ", cl1, i, x
                print "sec class ", cl2
                print "third class ", cl3

                if cl1 < 30:
                    sum += cl1

                if cl2 < 30:
                    sum += cl2

                if cl3 < 30:
                    sum += cl3

                cl1 = 0
                cl2 = 0
                cl3 = 0

                x = 0 


        accuracy =(150.0 - sum) / 150.0 
        accuracy = accuracy * 100.0
        print accuracy


    def openIrishDataset(self):
        #read the file

        file = open("dataset.txt","r")

        for i in range(self.s):
            self.data.append(numpy.random.dirichlet(numpy.ones(5), size=1))

        i = 0
        for line in file:
            fields = line.split(",")
            field1 = fields[0]
            field2 = fields[1]
            field3 = fields[2]
            field4 = fields[3]
            field5 = fields[4]

            print field1, field2, field3, field4, field5

            self.data[i] = [float(field1), float(field2), float(field3), float(field4), field5]
            i+=1

if __name__ == "__main__":
    f = fcm()
    f.openIrishDataset()

    result = f.run()

    f.defuzzification()
    print f.DB_index()
    f.analysis()
