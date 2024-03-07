# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering;
# Institute of Scientific Instruments of the CAS, v. v. i., Medical signals -
# Computational neuroscience. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
from math import isclose

# Third pary imports
import scipy.signal as sp
import numpy as np

# Local imports
from epycom.univariate import (SignalStats,
                               PowerSpectralEntropy,
                               LyapunovExponent,
                               HjorthMobility,
                               HjorthComplexity,
                               ModulationIndex,
                               MeanVectorLength,
                               PhaseLockingValue,
                               AutoregressiveResidualModulation,
                               ShannonEntropy,
                               ApproximateEntropy,
                               SampleEntropy, 
                               LowFreqMarker,
                               Multiscale_Entropy
                               )


def test_signal_stats(create_testing_data, benchmark):
    compute_instance = SignalStats()
    stats = benchmark(compute_instance.run_windowed,
                      create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    expected_vals = (6.68954,
                     5.0,
                     2.32213,
                     67.65263,
                     0.0,
                     0.49719,
                     7.05092)

    for exp_stat, stat in zip(expected_vals, list(stats[0])[:-1]):
        assert isclose(stat, exp_stat, abs_tol=10e-6)


def test_pse(create_testing_data, benchmark):
    compute_instance = PowerSpectralEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 4.32193, abs_tol=10e-6)


def test_lyapunov_exponent(create_testing_data, benchmark):
    compute_instance = LyapunovExponent(sample_lag=25)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data[:5000], 5000)
    compute_instance.run_windowed(create_testing_data[:5000],
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 5.794813, abs_tol=10e-6)


def test_hjorth_mobility(create_testing_data, benchmark):
    compute_instance = HjorthMobility(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 3113.283, abs_tol=10e-3)


def test_hjorth_complexity(create_testing_data, benchmark):
    compute_instance = HjorthComplexity(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 2.27728, abs_tol=10e-6)


def test_modulation_index(create_testing_data, benchmark):
    compute_instance = ModulationIndex()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 8.5572385e-05, abs_tol=10e-6)


def test_mean_vector_length(create_testing_data, benchmark):
    compute_instance = MeanVectorLength(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 
                    50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0].real, 0.0005341996847015685, abs_tol=10e-6)
    assert isclose(res[0][0].imag, -0.00143533153230829, abs_tol=10e-6)


def test_phase_locking_value(create_testing_data, benchmark):
    compute_instance = PhaseLockingValue(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0].real, 0.003800756826940237, abs_tol=10e-6)
    assert isclose(res[0][0].imag, -0.0071765065150767565, abs_tol=10e-6)


def test_arr(create_testing_data, benchmark):
    compute_instance = AutoregressiveResidualModulation(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0],  0.03176254325984886, abs_tol=10e-6)


def test_shannon_entropy(create_testing_data, benchmark):
    compute_instance = ShannonEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 2.5231089015779533, abs_tol=10e-6)


def test_approximate_entropy(create_testing_data, benchmark):
    compute_instance = ApproximateEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    5000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 1.9744375, abs_tol=10e-6)


def test_sample_entropy(create_testing_data, benchmark):
    compute_instance = SampleEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    5000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 2.2665529038653913, abs_tol=10e-6)
    

def test_low_f_marker(create_testing_data, benchmark):
    compute_instance = LowFreqMarker()
    compute_instance.params = {'fs': 5000}
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    50000)
    compute_instance.run_windowed(create_testing_data,
                                  50000,
                                  n_cores=2)

    assert isclose(res[0][0],  0.4149408406426963, abs_tol=10e-6)

def test_multiscale_entropy(create_testing_data, benchmark):
    compute_instance = MultiscaleEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data,
                    50000)
    compute_instance.run_windowed(create_testing_data,
                                  50000,
                                  n_cores=2)
    assert isclose(res[0][0], 2.0149070428176254, abs_tol=10e-6)