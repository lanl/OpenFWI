import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))

def plot_velocity(output, target, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.08)
    ax[1].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.08)
    
    for axis in ax:
        # axis.set_xticks(range(0, 70, 10))
        # axis.set_xticklabels(range(0, 1050, 150))
        # axis.set_yticks(range(0, 70, 10))
        # axis.set_yticklabels(range(0, 1050, 150))
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))

        axis.set_ylabel('Depth (m)', fontsize=12)
        axis.set_xlabel('Offset (m)', fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')

def plot_single_velocity(label, path):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    vmax, vmin = np.max(label), np.min(label)
    im = ax.matshow(label, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    # im = ax.matshow(label, cmap="gist_rainbow", vmin=vmin, vmax=vmax)

    # nx = label.shape[0]
    # ax.set_aspect(aspect=1)
    # ax.set_xticks(range(0, nx, int(150//(1050/nx)))[:7])
    # ax.set_xticklabels(range(0, 1050, 150))
    # ax.set_yticks(range(0, nx, int(150//(1050/nx)))[:7])
    # ax.set_yticklabels(range(0, 1050, 150))
    # ax.set_title('Offset (m)', y=1.08)
    # ax.set_ylabel('Depth (m)', fontsize=18)

    fig.colorbar(im, ax=ax, shrink=1.0, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')

# def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
#     fig, ax = plt.subplots(1, 3, figsize=(15, 6))
#     im = ax[0].matshow(output, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[0].set_title('Prediction')
#     ax[1].matshow(target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[1].set_title('Ground Truth')
#     ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[2].set_title('Difference')
#     fig.colorbar(im, ax=ax, format='%.1e')
#     plt.savefig(path)
#     plt.close('all')


def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    # fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    aspect = output.shape[1]/output.shape[0]
    im = ax[0].matshow(target, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth')
    ax[1].matshow(output, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('Prediction')
    ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title('Difference')
    
    # for axis in ax:
    #     axis.set_xticks(range(0, 70, 10))
    #     axis.set_xticklabels(range(0, 1050, 150))
    #     axis.set_title('Offset (m)', y=1.1)
    #     axis.set_ylabel('Time (ms)', fontsize=12)
    
    # fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.75, label='Amplitude')
    plt.savefig(path)
    plt.close('all')


def plot_single_seismic(data, path):
    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap='gray', vmin=vmin * 0.01, vmax=vmax * 0.01)
    ax.set_aspect(aspect=nx/nz)
    ax.set_xticks(range(0, nx, int(300//(1050/nx)))[:5])
    ax.set_xticklabels(range(0, 1050, 300))
    ax.set_title('Offset (m)', y=1.08)
    ax.set_yticks(range(0, nz, int(200//(1000/nz)))[:5])
    ax.set_yticklabels(range(0, 1000, 200))
    ax.set_ylabel('Time (ms)', fontsize=18)
    fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    plt.savefig(path)
    plt.close('all')
