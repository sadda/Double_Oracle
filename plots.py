#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import matplotlib.pyplot as plt

from util import *


def save_figure(file_name):
    if not (file_name is None):
        plt.savefig(file_name)
    else:
        plt.show()

def plot_optimal_distributions(game, actions, distribution, player, file_name=None):
    plt.figure()
    plt.title("Player " + str(player) + " distribution for " + "c = " + str(game.c))
    plt.xlabel("q")
    plt.ylabel("p[%]")
    plt.bar(actions, distribution, width=0.007, bottom=None,  align='center')
    save_figure(file_name)
    
def plot_response_evolution(game, actions, player, file_name=None):
    plt.figure()
    plt.title("Player " + str(player) + " responses for " + "c = " + str(game.c))
    plt.xlabel("iteration")
    plt.ylabel("q")
    plt.plot(actions)
    save_figure(file_name)

def plot_value_evolution(game, lower_bounds, upper_bounds, file_name=None):
    plt.figure()
    plt.title("Upper and lower bounds for " + "c = " + str(game.c))
    plt.xlabel("iteration")
    plt.xlabel("iteration")
    plt.ylabel("value of the game")
    plt.plot(lower_bounds, color="red")
    plt.plot(upper_bounds, color="blue")
    save_figure(file_name)

def plot_x_util_function(y, x1, x2, u, file_name=None):
    utils = discretize_x_utility_newaxis(u, np.array([y]), np.array([1]), np.array([x1, x2]), 1e3)
    plt.figure()
    plt.title("Alice's utility function for y = "+ str(y))
    plt.xlabel("X")
    plt.ylabel("utility")
    plt.plot(np.linspace(x1, x2, 1e3), utils)
    save_figure(file_name)

def plot_x_mixed_util_function(ys, distribution, x1, x2, u, file_name=None):
    utils = discretize_x_utility_newaxis(u, ys, distribution, np.array([x1, x2]), 1e3)
    plt.figure()
    plt.title("Alice's utility function, y's;distribution = "+ str(ys)+";"+str(distribution))
    plt.xlabel("X")
    plt.ylabel("utility")
    plt.plot(np.linspace(x1, x2, 1e3), utils)
    save_figure(file_name)

def plot_y_util_function(x, y1, y2, u, file_name=None):
    utils = discretize_y_utility_newaxis(u, np.array([x]), np.array([1]), np.array([y1, y2]), 1e3)
    plt.figure()
    plt.title("Bob's utility function for x = "+ str(x))
    plt.xlabel("Y")
    plt.ylabel("utility")
    plt.plot(np.linspace(y1, y2, 1e3), utils)
    save_figure(file_name)

def plot_y_mixed_util_function(xs, distribution, y1, y2, u, file_name=None):
    utils = discretize_y_utility_newaxis(u, xs, distribution, np.array([y1, y2]), 1e3)
    plt.figure()
    plt.title("Bobs's utility function for x's;distribution = "+ str(xs)+";"+str(distribution))
    plt.xlabel("Y")
    plt.ylabel("utility")
    plt.plot(np.linspace(y1, y2, 1e3), utils)
    save_figure(file_name)

def plot_util_function(u, x1, x2, y1, y2, file_name=None):
    plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(x1, x2, 200)
    y = np.linspace(y1, y2, 200)

    X, Y = np.meshgrid(x, y)
    Z = u(X, Y)
    ax.plot_surface(X, Y, Z)
    ax.set_title('surface')
    save_figure(file_name)

def plot_game(game, xs, x_responses, xs_distribution, ys, y_responses, ys_distribution, lower_bounds, upper_bounds, label="", run=0, strat="", file_name=None):
    xs_red, x_dist_red = compact_strategies(xs, xs_distribution, 2)
    ys_red, y_dist_red = compact_strategies(ys, ys_distribution, 2)

    fig, axs = plt.subplots(2, 2, figsize=(12,8) )
    
    axs[0, 0].set_title("Alice's optimal distribution")
    axs[0, 0].bar(xs_red, x_dist_red, width=( max(xs_red) - min(xs_red))/80, bottom=None,  align='center')
    axs[0, 0].set_xlim( game.X.getCube()[0][0]-0.1, game.X.getCube()[0][1]+0.1 )

    axs[0, 1].set_title("Bob's optimal distribution")
    axs[0, 1].bar(ys_red, y_dist_red, width=( max(ys_red) - min(ys_red))/80, bottom=None,  align='center')
    axs[0, 1].set_xlim( game.Y.getCube()[0][0]-0.1, game.Y.getCube()[0][1]+0.1 )

    axs[1, 0].set_title("best responses / iter")
    axs[1, 0].plot(x_responses, color="blue")
    axs[1, 0].plot(y_responses, color="red")
    axs[1, 0].set_ylim( game.X.getCube()[0][0]-0.1, game.X.getCube()[0][1]+0.1 )

    axs[1, 1].set_title("iterative utility bounds")
    axs[1, 1].plot(lower_bounds, color="red")
    axs[1, 1].plot(upper_bounds, color="blue")
    axs[1, 1].set_ylim( -2, 2 )
    
    fig.suptitle("Game"+label+" | "+strat+" | < "+str( lower_bounds[-1] )+" , "+str( upper_bounds[-1] )+" >" )  
    save_figure(file_name)





def compute_uniform_a():
    
    return np.sqrt(2)/2


def convert_p3_p2(p3, a):
    
    p2      = np.zeros((len(p3),2))
    p2[:,0] = 0.5*(1-p3[:,0]+p3[:,1])
    p2[:,1] = a*(1-p3[:,0]-p3[:,1])
    return p2


def convert_p2_p3(p2, a):
    
    p3      = np.zeros((len(p2),3))
    p3[:,0] = a-a*p2[:,0]-0.5*p2[:,1]
    p3[:,1] = a*p2[:,0]-0.5*p2[:,1]
    p3[:,2] = p2[:,1]
    return p3/a

def plot_blotto(game, xs, p, player, a=compute_uniform_a(), file_name=None):
    
    x2 = convert_p3_p2(xs, a)

    plt.figure()
    plt.title("Optimal distribution of player " + str(player) + " for " + "c = " + str(game.c))
    plt.plot([0,1,0.5,0], [0,0,a,0])
    if max(p) - min(p) > 1e-8:
        plt.scatter(x2[:,0], x2[:,1], c=p, cmap='jet', alpha=0.75)
        plt.colorbar()
        #plt.colorbar(boundaries=np.linspace(0,0.2,25))    
    else:
        plt.scatter(x2[:,0], x2[:,1], alpha=0.75)        
    save_figure(file_name)
    
    
    
    
    
def csv_strategy(label, qs, distribution, file):
    fieldnames = [label, 'probability']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for i,q in enumerate(qs):
        writer.writerow({label: str(q), 'probability': str(distribution[i])} )
    file.write("\n")

def csv_bounds(lower_bounds, upper_bounds, file):
    fieldnames = ['iteration', 'lower_bound', 'upper_bound']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(upper_bounds)):
        writer.writerow({'iteration': str(i), 'lower_bound': str(lower_bounds[i]), 'upper_bound': str(upper_bounds[i])})
    file.write("\n")
        
def csv_print(xs, xs_distribution, ys, ys_distribution, lower_bounds, upper_bounds, filename):
    xs_red, x_dist_red = compact_strategies(xs, xs_distribution, 2)
    ys_red, y_dist_red = compact_strategies(ys, ys_distribution, 2)
    with open(filename, 'w', newline='') as file:
        csv_strategy("x", xs, xs_distribution, file)
        csv_strategy("y", ys, ys_distribution, file)
        csv_strategy("rounded_x", xs_red, x_dist_red, file)
        csv_strategy("rounded_y", ys_red, y_dist_red, file)
        csv_bounds(lower_bounds, upper_bounds, file)




def prep_optimal_distribution_FP(qs, bounds, player):
    plt.clf()
    plt.xlim( bounds[0], bounds[1] )
    plt.title("Optimal "+player+"'s distribution")
    plt.xlabel("q")
    plt.ylabel("p density")
    X_plot = np.linspace(bounds[0], bounds[1], len(qs)).reshape(-1, 1)
    #kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(qs.reshape(-1, 1))
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.05).fit(qs.reshape(-1, 1))
    log_dens = kde.score_samples(X_plot)
    dens = np.exp(log_dens)
    dens /= 5
    plt.plot(X_plot[:, 0], dens)
    with open("output/game_DOvsFP_continuous_"+player+"_FP.csv", 'w', newline='') as file:
        fieldnames = ['q', 'density']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(dens)):
            writer.writerow({'q': str(X_plot[i][0]), 'density': str(dens[i])})
    return dens, np.linspace(bounds[0], bounds[1], len(qs) )

def prep_game_value(lower_bounds, upper_bounds, up, low):
    plt.clf()
    plt.ylim( up, low )
    plt.title("Iterative bounds")
    plt.xlabel("iteration")
    plt.ylabel("value of the game")
    plt.plot(lower_bounds, color="red")
    plt.plot(upper_bounds, color="blue")
    plt.text(len(upper_bounds)/5, 1.3, "lower bounds = "+str(lower_bounds[-1]), fontsize=10)
    plt.text(len(upper_bounds)/5, 1.1, "upper bounds = "+str(upper_bounds[-1]), fontsize=10)

def prep_mixed_strategy(player, bounds, qs, distribution):
    plt.clf()
    plt.ylim( 0, 1 )
    plt.xlim( bounds[0]-0.1, bounds[1]+0.1 )
    plt.title("Optimal "+player+"'s distribution")
    plt.xlabel("q")
    plt.ylabel("p[%]")
    plt.bar(qs, distribution, width=( max(qs) - min(qs))/80, bottom=None,  align='center')


def prep_util_function(u, x_bounds, y_bounds):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(x_bounds[0], x_bounds[1], 200)
    y = np.linspace(y_bounds[0], y_bounds[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = u(X, Y)
    ax.plot_surface(X, Y, Z)
    ax.set_title('Utility function')
    return fig


def plot_output(game, xs, ys, x_responses, y_responses, xs_distribution, ys_distribution, lower_bounds, upper_bounds, method, run, text):
    x_bounds = game.X.getCube()[0]
    y_bounds = game.Y.getCube()[0]
        
    xs_red, x_dist_red = compact_strategies(xs, xs_distribution, 2)
    ys_red, y_dist_red = compact_strategies(ys, ys_distribution, 2)

    csv_print(xs, xs_distribution, ys, ys_distribution, lower_bounds, upper_bounds, 'output/game'+game.label+'_'+method+'_'+str(run)+'.csv')
        
    fig = prep_util_function(game.u, x_bounds, y_bounds)
    fig.savefig("output/game_"+game.label)

    prep_game_value(lower_bounds, upper_bounds, -2.5, 1.5)
    plt.savefig("output/game_"+game.label+"_"+method+"_"+text+str(run)+"_value")

    prep_mixed_strategy("Alice", x_bounds, xs_red, x_dist_red)
    plt.savefig("output/game_"+game.label+"_"+method+"_"+text+str(run)+"_alice")

    prep_mixed_strategy("Bob", y_bounds, ys_red, y_dist_red)
    plt.savefig("output/game_"+game.label+"_"+method+"_"+text+str(run)+"_bob")

    if method == "FP":
        x_dens, xs = prep_optimal_distribution_FP(x_responses, x_bounds, "Alice")
        plt.savefig("output/game_"+game.label+"_"+method+"_"+text+str(run)+"_alice_KDE")
        y_dens, ys = prep_optimal_distribution_FP(y_responses, y_bounds, "Bob")
        plt.savefig("output/game_"+game.label+"_"+method+"_"+text+str(run)+"_bob_KDE")


def plot_DO_vs_FP_value(game, lower_bounds_DO, upper_bounds_DO, lower_bounds_FP, upper_bounds_FP, run):
    with open("output/game_"+game.label+"_DOvsFP_value_"+str(run)+".csv", 'w', newline='') as file:
        csv_bounds(lower_bounds_DO, upper_bounds_DO, file)
        csv_bounds(lower_bounds_FP, upper_bounds_FP, file)

    plt.clf()
    plt.ylim( -2.5, 1.5 )
    plt.title("Iterative bounds")
    plt.xlabel("iteration")
    plt.ylabel("value of the game")
    plt.plot(lower_bounds_DO, color="green")
    plt.plot(upper_bounds_DO, color="green")
    plt.plot(lower_bounds_FP, color="red")
    plt.plot(upper_bounds_FP, color="red")
    plt.text(len(upper_bounds_DO)/2, 1.3, "Double Oracle", fontsize=10, color="green")
    plt.text(len(upper_bounds_DO)/2, 1.1, "Fictional Play", fontsize=10, color="red")
    plt.savefig("output/game_"+game.label+"_DOvsFP_"+str(run)+"_value")
    


def plot_DO_vs_FP_strategy(game, x_responses_FP, y_responses_FP, xs_DO, ys_DO, xs_distribution_DO, ys_distribution_DO, run):
    x_bounds = game.X.getCube()[0]
    y_bounds = game.Y.getCube()[0]
        
    xs_red, x_dist_red = compact_strategies(xs_DO, xs_distribution_DO, 2)
    ys_red, y_dist_red = compact_strategies(ys_DO, ys_distribution_DO, 2)
    
    prep_optimal_distribution_FP(x_responses_FP,x_bounds, "Alice")
    plt.bar(xs_red, x_dist_red, width=( max(xs_red) - min(xs_red))/80, bottom=None,  align='center')
    plt.savefig("output/game_"+game.label+"_DOvsFP_"+str(run)+"_alice")
    #plt.show()

    prep_optimal_distribution_FP(y_responses_FP, y_bounds, "Bob")
    plt.bar(ys_red, y_dist_red, width=( max(ys_red) - min(ys_red))/80, bottom=None,  align='center')
    #plt.show()
    plt.savefig("output/game_"+game.label+"_DOvsFP_"+str(run)+"_bob")
    with open("output/game_DOvsFP_continuous_Alice_DO.csv", 'w', newline='') as file:
        csv_strategy("q", xs_DO, xs_distribution_DO, file)
    with open("output/game_DOvsFP_continuous_Bob_DO.csv", 'w', newline='') as file:
        csv_strategy("q", ys_DO, ys_distribution_DO, file)
    ##############################################################################################

def benchmark(games, method="DO", iters=20, runs=10, text=""):
    for game in games:    
        run = 0
        while run < runs:
            print("game"+game.label+", run ", run)
            try:
                xs, x_responses, x_dist, ys, y_responses, y_dist, lower_bounds, upper_bounds, itr = solve(game, method, iters, printout=False)
            except:
                continue
            #plot_game(game, xs, x_responses, x_dist, ys, y_responses, y_dist, lower_bounds, upper_bounds, run=run, strat=optimizationStrategy, save=True, label=game.label+"_"+method)
            plot_output(game, xs, ys, x_responses, y_responses, x_dist, y_dist, lower_bounds, upper_bounds, method, run, text)
            run += 1

