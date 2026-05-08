# Simulating Neuronal Spiking Behavior: A Leaky Integrate-and-Fire Approach

This repository contains a computational neuroscience project that analyses neuronal spiking behaviour using real hippocampal spike-train data and Python-based statistical visualization.

## Project Overview

Neurons communicate through electrical impulses known as spikes. This project investigates neuronal firing behaviour using concepts from the leaky integrate-and-fire (LIF) model, a simplified mathematical model that describes how a neuron's membrane potential changes over time in response to input current.

The project also applies data analysis techniques to real neural recordings from the CRCNS hippocampal dataset, examining spike timing, firing rates, inter-spike intervals, and firing regularity across multiple neurons.

## Objectives

- Simulate and analyse neuronal spiking behaviour
- Study how membrane potential and input current affect firing patterns
- Process real hippocampal spike-train recordings
- Calculate firing-rate statistics for multiple neurons
- Analyse inter-spike interval variability
- Visualize neural activity using raster plots and statistical distributions
- Connect biological neuronal behaviour with computational modelling

## Computational Model

The project is based on the leaky integrate-and-fire neuron model:

```text
V'(t) = -((V(t) - V_rest) / τ) + I(t) / C
