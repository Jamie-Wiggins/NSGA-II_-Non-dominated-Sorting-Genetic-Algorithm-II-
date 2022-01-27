import random
import numpy as np

from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray
from deap import base
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools
import matplotlib.pyplot as plt

numOfBits = 10 #Number of bits in the chromosomes
maxnum = 2**numOfBits #absolute max size of number coded by binary list 1,0,0,1,1,....
genArr = []
hypervolumeArr = []
x1Arr = []

x2Arr = []
x3Arr = []
f1Arr = []
f2Arr = []

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def evaluate(individual):
    x1=individual[0:10]
    x2=individual[10:20]
    x3=individual[20:30]

    x1= ("".join(str(i) for i in x1))
    x2= ("".join(str(i) for i in x2))
    x3= ("".join(str(i) for i in x3))

    x1 = bin_to_gray(x1)
    x2 = bin_to_gray(x2)
    x3 = bin_to_gray(x3)

    x1 = chrom2real(x1)
    x2 = chrom2real(x2)
    x3 = chrom2real(x3)

    f1=(((x1-0.6)/1.6)**2+(x2/3.4)**2+(x3-1.3)**2)/2

    f2=((x1/1.9-2.3)**2+(x2/3.3-7.1)**2+(x3+1.3)**2/3.0)

    x1Arr.append(x1)
    x2Arr.append(x2)
    x3Arr.append(x3)
    f1Arr.append(f1)
    f2Arr.append(f2)

    return f1,f2

# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: real value
def chrom2real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange=-4+8*numasint/maxnum
    return numinrange

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
flipProb=1.0/30 # set to 1/30
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)

# Compare ind to see if dominated or not
def compare(xPop, f, n, pop):
    isDomd = False

    x = len(f) # number of fronts found
    k = 0 # the front now checked

    while k < x:
        for a in range(len(f[k]), 0, -1):# for each value in array - check in reverse for efficeient sorting
            if xPop[n][1][1] < pop[f[k][a-1]].fitness.values[1]: # does not dominate joins front
                isDomd = False
            else:
                isDomd = True
                break
        if isDomd:# if dominated search the next front working left to right
            k += 1
        else:
            break
    return k 

# scatter comparison plot
def plotCompare(plot1, plot2, title, label1, label2, color1, color2):
    tmpF1plt1 = []
    tmpF2plt1 = []
    tmpF1plt2 = []
    tmpF2Plt2 = []

    for j in range(len(plot1)):
        tmpF1plt1.append(plot1[j].fitness.values[0])
        tmpF2plt1.append(plot1[j].fitness.values[1])
    for j in range(len(plot2)):
        tmpF1plt2.append(plot2[j].fitness.values[0])
        tmpF2Plt2.append(plot2[j].fitness.values[1])

    plt.scatter(tmpF1plt1, tmpF2plt1, color=color1, label=label1)
    plt.scatter(tmpF1plt2, tmpF2Plt2, color=color2, label=label2)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(title)
    plt.legend(loc="upper right")

# Assign fronts
def assignFronts(pop):
    f = {} # use dictionary for fronts {front 1:[ind1,ind2,ind n]}
    tempPop = {}

    for i in range(len(pop)):
        tempPop.update({i:pop[i].fitness.values})
    xPop = sorted(tempPop.items(), key=lambda x: x[1])

    for n in range(len(pop)):
        if n == 0:# first value
            f.update({0:[xPop[0][0]]}) # store 
        else:# exsits
            j = compare(xPop,f,n,pop)
            if (j) in f:
                tmpArr = f[j]
                tmpArr.append(xPop[n][0])
                f.update({j:tmpArr})
            else:# new
                f.update({compare(xPop,f,n, pop):[xPop[n][0]]})
    return f

# Calculate Crowding distance
def crowdingDistance(pop, f):
    for i in f:
        for j in range(len(f[i])):
        # crowding calcs
            if len(f[i]) <= 2 or j == 0 or j == len(f[i])-1:
                pop[f[i][j]].fitness.crowding_dist = float('inf')
            else:
                # work out the crowding for remaining individuals
                prevPoint = pop[f[i][j-1]].fitness.values
                nextPoint = pop[f[i][j+1]].fitness.values

                func1 = abs((prevPoint[0] - nextPoint[0]))
                func2 = abs((prevPoint[1] - nextPoint[1]))

                func3 = func1 / (pop[f[i][len(f[i])-1]].fitness.values[0] -
                pop[f[i][0]].fitness.values[0])

                func4 = func2 / (pop[f[i][0]].fitness.values[1] - pop[f[i][len(f[i])-1]].fitness.values[1])

                func5 = func3 + func4

                pop[f[i][j]].fitness.crowding_dist = func5

    for i in f:
        for j in f[i]:
            pop[j].fitness.front = i   

    return pop

# Select best individuals from population
def bestSelection(pop, f):
    selection = []
    remaining = []

    # sort each front in terms of crowding distance
    for i in f:
        f[i].sort(key=lambda x: pop[x].fitness.crowding_dist, reverse=True)

    count = 0
    for i in f:
        for j in f[i]:
            count +=1
            if count < (len(pop)/2)+1:
                selection.append(pop[j])
            else:
                remaining.append(pop[j])

    return selection, remaining

def main(seed=None):
    random.seed(seed)

    NGEN = 30 # number of generations
    MU = 24 # pop size 
    CXPB = 0.9

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Assign fronts
    f = assignFronts(pop)

    # print table for 3.1
    for i in range(len(pop)):
        print(x1Arr[i], x2Arr[i], x3Arr[i], f1Arr[i], f2Arr[i])

    # plot fronts on graph
    # create colour array
    colorArr = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
    colorIdx = 0
    tmpF1 = []
    tmpF2 = []

    # Loop through fronts
    for i in f:
        if colorIdx == len(colorArr):
            colorIdx = 0

        color = colorArr[colorIdx]

        for j in f[i]:
            tmpF1.append(pop[j].fitness.values[0])
            tmpF2.append(pop[j].fitness.values[1])

    # plot front
    plt.scatter(tmpF1, tmpF2, c=color, label='front {}'.format(i+1))

    tmpF1 = []
    tmpF2 = []
    colorIdx+=1

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Solutions in Objective Space Indicating Fronts by Colour")
    plt.legend(loc="upper right")
    plt.show()

    # Assign Crowding distance
    pop = crowdingDistance(pop, f)

    # print table for 3.3
    for i in f:
        for j in f[i]:
            print(pop[j].fitness.values[0], pop[j].fitness.values[1], i+1,pop[j].fitness.crowding_dist)

    # Clone pop
    tmpPop = pop
    tmpPop.sort(key=lambda x: x.fitness.values[0], reverse=True)
    worstF1Val = tmpPop[0].fitness.values[0]
    tmpPop.sort(key=lambda x: x.fitness.values[1], reverse=True)
    worstF2Val = tmpPop[0].fitness.values[1]

    # Begin the generational process: may need to call other functions in here
    for gen in range(1, NGEN):
        # Tournament selection
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # make pairs of all (even,odd) in offspring
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if gen == 1:
        # Plot of offspring and parents
            plotCompare(pop, offspring, "Parents and Offspring in Objective Space", "Parent",
            "Offspring", 'red', 'green')
            plt.show()

        # Combine parents and offspring
        pop = pop + offspring

        # Assign fronts
        f = assignFronts(pop)

        # Assign crowding distance
        pop = crowdingDistance(pop, f)

        # select best indviduals
        pop, remaining = bestSelection(pop, f)


        if gen == 1:
            # plot best selection against remainder
            plotCompare(pop, remaining, "Selected Best and Remaining Indviduals in Objective Space", "Best", "Remainder", 'cyan', 'purple')
            plt.show()
        if gen == NGEN-1:
            plotCompare(pop, [], "Generation 30 Selected Best in Objective Space", "Best", "",'red', '')
            plt.show()

        hypervolumeArr.append(hypervolume(pop, [worstF1Val,worstF2Val]))
        genArr.append(gen)
    return pop

if __name__ == "__main__":
    pop = main()
    pop.sort(key=lambda x: x.fitness.values)

    # plot hypervolume
    plt.plot(genArr, hypervolumeArr, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over the Generations")
    plt.show()