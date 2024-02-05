Feature extraction
======================
This is the core of the whole library. The algorithms for feature extraction are divided into 3 subgroups:

- Univariate
- Bivariate
- Event detection

All the algorithms accept raw or filtered data and provide pandas dataframes as their output.


Univariate feature extraction
*********************************

- Approximate entropy

..
  TODO

- Arr

..
  TODO

- Hjorth complexity

  - The Hjort mobility (Hc) varies in the interval :math:`<0,inf)` and 
    combines properties of signal with its own first and second derivative.

  - By definition Hc is defined as :math:`Hc = Hm(X)/Hm(dX)`, where dx is 
    derivative of original signal and Hm is Hjort mobility, which is further 
    described later. With algebraic adjustment can be obtained 
    :math:`Hc = sqrt(var(ddX)*var(X))/var(dX)` where the X is signal, dX is is 
    first derivative of signal and ddX is second derivative of signal and sqrt
    is square root.
    Because of derivatives are both in numerator and denominator, using the 
    numerical aproximation of derivative, the steps h (inversion of sampling 
    frequency fs) cancel out.    

  - Example

    .. code-block:: py
      :name: Hc-example1.1.1

      #Example1
      x1 = np.linspace(0*np.pi, 2*np.pi, num=5001)
      sig = x1*0
      compute_hjorth_complexity(sig)
        >> nan
      # var(ddx) = 0, var(dx) = 0, var(x) = 0

    Linear functions have undefined Hc value, due to zero variance of dx, which
    causes undefined 0/0 operation.

    .. code-block:: py
      :name: Hc-example1.1.2

      #Example2
      x1 = np.linspace(0*np.pi, 2*np.pi, num=5001)
      sig = np.sin(x1)
      compute_hjorth_complexity(sig)
        >> 1.000000020000001
      # var(ddx) = 1.2470854541719263e-12, var(dx) = 7.895682481841235e-07, 
      # var(x) = 0.4999000199960008

    .. code-block:: py
      :name: Hc-example1.1.3

      #Example3
      x1 = np.linspace(0*np.pi, 2*np.pi, num=5001)
      sig = 13*np.sin(2*x1) + 11
      compute_hjorth_complexity(sig)
        >> 1.0000000200000008
      # var(ddx) = 3.3721164055632688e-09, var(dx) = 0.0005337479250571773,
      # var(x) = 84.48310337932413

    The Hc value is not affected by scaling, moving on y-axis or change of 
    frequency.

    .. code-block:: py
      :name: Hc-example1.1.3

      #Example3
      x2=np.linspace(0, 10, num=4001)
      sig = x1*x1 - 3*x1 + 1
      compute_hjorth_complexity(sig)
        >> 1.2836054995011157e-09
      # var(ddx) = 1.5407193989507215e-28, var(dx) = 0.0002083333203125,
      # var(x) = 464.1486145781251

    The Hc value of quadratic function is 0, because of second derivative of
    quadratic function is a constant with variance equal 0. The ressult is not 
    precisly 0, because of rounding error.

..
  Problem with Hjort features is rounding error. Because of rouding error
  the second derivative could be non-zero, or variance could be non-zero, even
  if it should be. As an error the feature return non-zero value, even if
  the ressult should be nan.
  Although this seems to be a big problem, this cases are not expected to 
  accure in real signals, if the signal is not corrupted.

- Hjorth mobility

  - The Hjort mobility (Hm) varies in the interval :math:`<0,inf)` and 
    combines properties of signal with its own derivative.

  - The Hm is calculated as :math:`Hm = sqrt(var(dX)/var(X))`, where sqrt 
    stands for square root, var is variance, x is original signal and dx is
    derivative of the original signal.

    Derivative of original signal is calculated as 
    :math:`dX(i) = (X(i+1)-X(i)) * fs`, where fs as a sampling frequency is 
    multiplicative inverse of step size. This approach has clear advantage
    against simplier difference without multiplying, due to compereability with 
    data obtained with different sampling frequency. Variance in all cases is 
    calculated by numpy library as :math:`var(X) = mean(X^2) - (mean(X))^2`, 
    where ^2 operator is meant as an element-wise.

  - Example

    .. code-block:: py
      :name: Hm-example1.2.1

      #Example1
      x1 = np.linspace(0*np.pi, 2*np.pi, num=5001)
      sig = x1
      fs = 5000
      compute_hjorth_mobility(sig, fs)
        >> 7.696928775346762e-13
      # var(dx) = 1.9501766826626976e-24, var(x) = 3.2918423177661214

    The Hjort mobility of linear function is 0, because of the derivative of
    linear function is constant value with variance equal 0. The Hjort mobility
    of constant function is undefined, because of variance of constant function
    and variance of its derivative are 0 and 0/0 is undefined.

    .. code-block:: py
      :name: Hm-example1.2.2

      #Example1
      x1 = np.linspace(0*np.pi, 2*np.pi, num=5001)
      sig = np.sin(x1)
      fs = 5000
      compute_hjorth_mobility(sig, fs)
        >> 6.283813306515432
      # var(dx) = 19.743154835570206, var(x) = 0.5

    .. code-block:: py
      :name: Hm-example1.2.3

      #Example1
      x1 = np.linspace(0*np.pi, 2*np.pi, num=5001)
      sig = 13*np.sin(x1) + 11
      fs = 5000
      compute_hjorth_mobility(sig, fs)
        >> 6.283813306515432
      # var(dx) = 19.743154835570206, var(x) = 0.5

    Hm value is not affected by scaling or moving on y-axis.

- Low frequency marker

  - The Low frequency marker (LFM) varies in the interval :math:`<0,inf)` and 
    reflects power ratio between two signal bands.

  - The LFM is calculated as :math:`LFM = median((infra_sig^2)/(main_sig^2))`, 
    where infra signal is signal in lowband frequencies and main signal is 
    signal in highband frequencies. Both infra and main signals are isolated 
    from the input signal by Butterworth filter. The '/' is division element-wise.
      
  - The infra frequency varies :math:`lowband=<0.02, 0.5>` Hz and the main 
    signal varies in :math:`highband=<2.0, 4.0>` Hz and cannot be changed in 
    input. The interval boundaries was identified based on:

    LUNDSTROM, Brian Nils; BRINKMANN, Benjamin a WORRELL, Gregory. Low frequency 
    interictal EEG biomarker for localizing seizures. Online. MedRxiv. June 7, 
    2021, , pages 1-20. Avaiable from: 
    https://doi.org/https://doi.org/10.1101/2021.06.04.21258382. 
    [cit. 2023-09-26].

    Although, some changes of infra and main frequencies could be reached by 
    changing of sampling fraquency value, it is not recomended.

  - Importance of using median insted of mean is, the main signal often croses
    zero value, so mean would be affected by multiple significantly higher
    values.

  - Example
  
    .. code-block:: py
      :name: LFM-example1.3.1

      #Example1
      x1=np.linspace(0*np.pi, 8*np.pi, num=2001)
      sig=np.sin(x1)
      fs = 5000
      compute_low_f_marker(sig, fs)
        >> 0.0922391697746599

    .. figure:: images/1.3.1aExample.png
      :name: Fig1.3.1a

    .. figure:: images/1.3.1Example.png
      :name: Fig1.3.1

    The median in this example is relativly low, but similar signal obtained 
    with different sampling frequency could lead to very different result.
    As you can see in the next example:

    .. code-block:: py
      :name: LFM-example1.3.2

      #Example2
      x1=np.linspace(0*np.pi, 8*np.pi, num=2001)
      sig=np.sin(x1)
      fs = 500
      # the sampling frequency in this case is 10 times lower than in example
      # above, but the samples stays the same
      compute_low_f_marker(sig, fs)
        >> 49.2645621029126

    .. figure:: images/1.3.2Example.png
      :name: Fig1.3.2

    In the practical case, the result is not much affected by different, but 
    large enough sampling frequency, because a higher sampling frequency only 
    leads to a higher sample density:

    .. code-block:: py
      :name: LFM-example1.3.3

      #Example3
      x1=np.linspace(0*np.pi, 8*np.pi, num=4001)
      sig=np.sin(x1)
      fs = 1000
      # both sampling frequency and sample density are two times bigger than in 
      # example above
      compute_low_f_marker(sig, fs)
        >> 50.077958925986536

    .. figure:: images/1.3.3Example.png
      :name: Fig1.3.3

    However, it is important to choose proper size of analyzed signal window,
    otherwise the ressult could be different:

    .. code-block:: py
      :name: LFM-example1.3.4

      #Example4
      x1=np.linspace(0*np.pi, 2*np.pi, num=1001)
      sig=np.sin(x1)
      fs = 1000
      compute_low_f_marker(sig, fs)
        >> 32.024759481499984

    .. figure:: images/1.3.4Example.png
      :name: Fig1.3.4

    The ressult is not dependent on scaling of signal:

    .. code-block:: py
      :name: LFM-example1.3.5

      #Example5
      x1=np.linspace(0*np.pi, 8*np.pi, num=4001)
      sig= 3 + 7*np.sin(x1)
      fs = 1000
      # scaled signal from Example3
      compute_low_f_marker(sig, fs)
        >> 50.07795892596946

    .. figure:: images/1.3.5Example.png
      :name: Fig1.3.5

- Lyapunov exponent

  - The lyapunov exponent (LE) feature estimates chaos in the system. The 
    rosenstein algorithm is used for the computation the of LE. 
    A good approximation of the Lyapunov exponent (lambda) describes the distance 
    between the trajectories :math:`X_(i+k)` and :math:`X_(j+k)` as 
    :math:`c*exp(lambda*k)`. The vector X_j is the nearest neighbor of the vector 
    X_i (using the Euclidean distance). The method of creating vectors will be 
    described later. Of all the Lyapunov exponents, this algorithm finds only the 
    largest one.  

  - At the start of the calculation, the first important step is to choose 
    right lag. Lag could be set on the input, for example:

    .. code-block:: py
      :name: LE-example1.4.0.1

      compute_lyapunov_exponent(sig, sample_lag=500)
      #compute lyapunov exponent with lag = 500 samples

    Or, lag can be computed inside the feature by autocorelation.
    The lag is calculated as the time delay between the starting point and the 
    point where the autocorrelation drops to :math:`1-1/e (~ 0.6321)` of the 
    initial value. Since the expected input signal is the EEG, the point of 
    such autocorrelation is assumed to exist within the first second of the 
    time series. Thus, it is important to set the sampling frequency into the 
    input, otherwise it would be set to default value :mat:`fs = 5000`. This 
    assumption is used to decrese the computational time.

  - Another important parameter is the dimension. Rosenstein's work uses Takens 
    criterion, :math:`dimension > 2*n`, where n is the number of state 
    variables. However, according to Rosenstein, the algorithm could work in 
    some cases without satisfying the Takens criterion. The default number of 
    dimensions is set to 5.

  - With this data, program starts with computing phase space. This step is 
    done by _compute_phase_space function. This function uses signal, dimension 
    and sample_lag.
    This function returns space matrix, each column represents vector X_i, 
    where i is the nuber of column in range 1 to 
    :math:`length(signal) - (dimension-1)*lag`. Vector X_i is created as 
    :math:`X_i = (x_i, x_{i+lag}, ..., x_{i+(dimension-1)*lag})`. The number of 
    rows is dimension. The x_i stands for i-th value of the input signal.

    So for example:

    .. code-block:: py
      :name: LE-example1.4.0.2

      data = np.arange(100)
      dimensions = 5
      sample_lag = 10
      _compute_phase_space(data, dimensions, sample_lag)
      >> [[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
            18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.
            36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53.
            54. 55. 56. 57. 58. 59.]
          [10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27.
            28. 29. 30. 31. 32. 33. 34. 35. 36. 37. 38. 39. 40. 41. 42. 43. 44. 45.
            46. 47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59. 60. 61. 62. 63.
            64. 65. 66. 67. 68. 69.]
          [20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. 37.
            38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53. 54. 55.
            56. 57. 58. 59. 60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70. 71. 72. 73.
            74. 75. 76. 77. 78. 79.]
          [30. 31. 32. 33. 34. 35. 36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47.
            48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59. 60. 61. 62. 63. 64. 65.
            66. 67. 68. 69. 70. 71. 72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82. 83.
            84. 85. 86. 87. 88. 89.]
          [40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57.
            58. 59. 60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70. 71. 72. 73. 74. 75.
            76. 77. 78. 79. 80. 81. 82. 83. 84. 85. 86. 87. 88. 89. 90. 91. 92. 93.
            94. 95. 96. 97. 98. 99.]]

        #number orows is 5 (=dimension)
        #number of columns is 60 = length(signal) - (dimension-1)*lag = 100-(5-1)*10

    For further use in the computation it is important, to take vectors by 
    columns. So everywhere this output will be used, it will need to be 
    transposed.

  - The function controls itself if the data length is long enough to allow the 
    Lyapunov exponent to be calculated.

  - Next step of the calculation is calculation of the nearest neighbor by 
    calculating cross euclidian distance between all vectors. However, since 
    the direct neighbor would probably have been the nearest, all distences 
    between vectors closer than min_step will be set as a infinity. The 
    min_step in samples is needed to be set in the input of the feature. The 
    default vaule is 500 samples. Rosenstein claims the min_step should be 
    greater, than mean period of the input signal.

  - The main idea of Rosenstein algorithm is averaging the logarithmic values 
    of the distances. This step needs trajectory_len variable, which is by 
    default 20, but can be changed by user.
    
    From the distance matrix, it takes minimal distance value in every row. All 
    chosen values are logaritmed and the mean value is calculated. Next we 
    discard first row and add +1 to index of the nearest neighbor. The whole 
    proces is repeated until trajectory_len number of values are obtained.

    If this step fails to obtain any finite value, the -infinity is returned.

  - When the main step is done, the values are associated with its index 
    (index of the values, where mean value is not finite are skipped) and the 
    polynom of the first order (straight line) is interleaved into by the 
    least square method. The straight line could be mathematicly written as 
    :math:`y = a*x + b`. The :math:`a*fs/lag_step` value is returned. Fs is the 
    sampling fraquency and :math:`a` is a element of the mathematical expresion 
    for a straight line.

  - This description is highly reduced and focused on the aplicaton, for better 
    understanding, it is recomended to read original Rosenstein paper:
    ROSENSTEIN, Michael T.; COLLINS, James J. a DE LUCA, Carlo J. A practical 
    method for calculating largest Lyapunov exponents from small data sets. 
    Online. Physica D: Nonlinear Phenomena. 1993, 117-134. ISSN 0167-2789. 
    doi: https://doi.org/10.1016/0167-2789(93)90009-P.

  - Example

    .. code-block:: py
        :name: LE-example1.4.1
        
        length1 = 5000 + 1
        x1=np.linspace(0*np.pi, n*2*np.pi, num=length1)
        sig = np.sin(x1)
        compute_lyapunov_exponent(sig)

        >> 1.5400895452210233e-05

- Mean vector length

  - The mean vector length (MVL) is phase-amplitude coupling feature and varies 
    in complex numbers. 
    Based on article:

    Quantification of Phase-Amplitude Coupling in Neuronal Oscillations: 
    Comparison of Phase-Locking Value, Mean Vector Length, and Modulation Index
    Mareike J. Hülsemann, Dr. rer. nat, Ewald Naumann, Dr. rer. nat, Björn 
    Rasch
    bioRxiv 290361; doi: https://doi.org/10.1101/290361

    the evaluating absolute value of output is recomended to use: 

    .. code-block:: py
      :name: MVL1.5.0

      np.abs(compute_mvl_count(sig, fs))

    The absolute value of MVL reflects the homogenity of signal and existance 
    of phase coupling. The near zero value mean no phase coupling, the great 
    absolute MVL mean the signal contains some phase coupling.

    The complex number also contains information about the dominant phase, or 
    in other words information about the phase lag between the low and high 
    band signals. This information has not yet been investigated further, so it 
    cannot be considered useful and could potentially have some influence on 
    the model in which it is used.

  - The MLV is calculated as: :math:`MVL = mean(amplitude * np.exp(j*phase))`,
    where amplitude is amplitude of Hilbert signal filtered from high frequency 
    band by Butterworth filter, wheras phase is calculated as phase of Hilbert
    signal filtered from low frequency band by Butterworth filter. Low 
    frequency band is by default :math:`<4, 8>` Hz and high frequency band is
    by default :math:`<80, 150>` Hz and both low and high frequency bands can 
    be changed in input. Both low and high high frequency boundaries are based
    on article:

    R. T. Canolty et al. ,High Gamma Power Is Phase-Locked to Theta 
    Oscillations in Human Neocortex.Science313,1626-1628(2006).
    DOI:10.1126/science.1128115

    Further description of the MVL calculation is given in the example below.

  - Important denote is, to count with appropriate higher frequency boundaries. 
    In general cases, high frequency boundaries should not exceed 
    :math:`fs/5`. 

  - Further description of MVL feature is contained in the article:

    Quantification of Phase-Amplitude Coupling in Neuronal Oscillations: 
    Comparison of Phase-Locking Value, Mean Vector Length, and Modulation Index
    Mareike J. Hülsemann, Dr. rer. nat, Ewald Naumann, Dr. rer. nat, Björn 
    Rasch
    bioRxiv 290361; doi: https://doi.org/10.1101/290361

  - Example

    .. code-block:: py
      :name: MVL-example1.5.1

      #Example1
      x1=np.linspace(6*np.pi, 16*np.pi, num=501)
      sig=np.random.rand(501)*np.sin(x1)
      fs = 5000
      compute_mvl_count(sig, fs, lowband=[8, 12], highband=[250, 600])
        >> 0.006292227798293142+0.00038112301129766796j

    .. figure:: images/1.5.1Example.png
      :name: Fig1.5.1

    In first part of the algorithm signal is filtered in lowband :math:`[4,8]` 
    Hz, and on the ressult the hilbert transforamation is aplied (the first 
    row, graph on left). Then from the complex signal values are taken in 
    euklidian formula as :math:`abs*exp(phi*j)` and the phase phi is saved 
    (first row, graph on right).

    Next the same procedure is taken in highband :math:`[80, 150]` Hz, but now
    the abs value is stored (second row on right). 
    
    From these phase and aplitude values the new complex signal is created and 
    values are writen as :math:`a+b*j`, the a(i) row is the real signal part 
    and b(i) is imaginary signal part (graph in third row), j is imaginary 
    number :math:`j^2 = -1`.

    At the end, coresponding a(i), b(i) are taken as one vector (blue stars in 
    graph in the last row) and the mean value is calculated from them (orange 
    line).

    In this example the complex values of Hilbert transformation does not show
    any dominant phase and no phase coupling could not be seen. As a ressult
    the mean value is relativly low.
    The sensitivity of the MVL to amplitude outliers is also visible as one of 
    the caveats of the MVL.

- Modulation index

  - Modulation index (MI) is phase-amplitude coupling feature varies in 
    interval:math:`(0,1)`.

    Quantification of Phase-Amplitude Coupling in Neuronal Oscillations: 
    Comparison of Phase-Locking Value, Mean Vector Length, and Modulation Index
    Mareike J. Hülsemann, Dr. rer. nat, Ewald Naumann, Dr. rer. nat, Björn 
    Rasch
    bioRxiv 290361; doi: https://doi.org/10.1101/290361

  - From all phase-amplitude features the MI is the least sensitive to different
    sampling frequencies, but is sensitive to length of signal and number of
    bins (nbins) given in the input (default number of bins is based on the 
    paper and is set to 18). Number of bins have to be at least 2.

  - The calculation of MI runs in few steps. At first the bins bounds in range 
    :math:`<-pi pi)`, based on number of bins in input, are calculated. 
    Then, the hilbert transformation from the imput signal is calculated.
    In the third step, from the complex signal the apmlitude and phase is 
    calculated using euclidian formula :math:`a+b*j=amplitude*exp(phase*j)`. 
    For each phase bin the mean of amplitudes is calculated. The next step is 
    normalization of amplitudes by 
    :math:`amp[i] := amp[i]/sum(amp[0:(nbins-1)]) `.From the obtained data the 
    Shannon entropy (H) as :math:`H = -sum(amp*log(amp))`. From Shannon entropy 
    the Kullback-Leibler distance (KL) is calculated as 
    :math:`Kl = log(nbins) - H`.  From Kullback-Leibler distance the final MI 
    calculation is computed as :math:`MI = KL/log(nbins)`.

  - For the constant signal the NaN (not a number) is returned, because it 
    would make some bins empty. The nan could be also returned in other cases, 
    if the phase of the signal is not distributed in all phase bins.
 
    In general the higher MI value, the higher phase-amplitude coupling is. In 
    the real signal, values close to 1 should be almost never obtained.

  - Example

    .. code-block:: py
      :name: MI-example1.6.1

      #Example1
      fs = 5000
      sig=np.ones(10001) #constant value 1
      print(compute_mi_count(sig, nbins=18))
        >> nan

    As was said before, the constant signal would ressult with NaN return.

    .. code-block:: py
      :name: MI-example1.6.2

      #Example2
      fs = 5000
      x1=np.linspace(6*np.pi, 16*np.pi, num=10001)
      sig=np.sin(20*x1)+np.sin(120*x1)*np.exp(-x1)
      print(compute_mi_count(sig, nbins=18))
        >> 0.11739821053370704

    .. figure:: images/1.6.2.1Example.png
      :name: Fig1.6.2.1

    First image represents the input signal in the real part and its hilbert 
    transformation as the imaginary part.

    .. figure:: images/1.6.2.2Example.png
      :name: Fig1.6.2.2

    Second graph shows complex signal from image above, represented as 
    amplitude and phase in radians. We can plot the same data in graph, where 
    the amplitude depends on the phase.

    The ticks represents boundaries for each bin.

    .. figure:: images/1.6.2.3Example.png
      :name: Fig1.6.2.3

    From this data is created image below. The x-axis represents phases in 
    radians and the y-axis represents normalized mean value for each bin. The 
    length of the line shows aproximation of width of each bin. The ticks 
    represents boundaries for each bin.

    .. figure:: images/1.6.2.4Example.png
      :name: Fig1.6.2.4
  
    There are large mean values visible around the 0 radians, and also around 
    0.5 pi radians, which is mainly affected by big amplitudes at the beginning 
    and end of the signal (visible on second graph or as the outliers in third 
    graph).

- Phase locking value

  - The phase locking value (PLV) is phase-amplitude coupling feature and 
    varies inside complex unit circle :math:`0 <= abs(PLV) <= 1`.
    Based on article:

    Quantification of Phase-Amplitude Coupling in Neuronal Oscillations: 
    Comparison of Phase-Locking Value, Mean Vector Length, and Modulation Index
    Mareike J. Hülsemann, Dr. rer. nat, Ewald Naumann, Dr. rer. nat, Björn 
    Rasch
    bioRxiv 290361; doi: https://doi.org/10.1101/290361

    the evaluating absolute value of output is recomended to use:

    .. code-block:: py
      :name: PLV1.7.0

      np.abs(compute_plv_count(sig, fs))

  - First of all, algorithm use Butterworth filter in lowband :math:`[4,8]` and
    highband :math:`[80,150]` Hz. This 2 signals are then transformed by 
    Hilbert transformation to complex signals.

    Using euklidian formula as :math:`a + b*j = abs*exp(phi*j)`,we can 
    extract phase1 from the lowband signal and amplitude from the highband 
    signal. To the amplitude is then aplied hilbert transformation and from this
    complex signal is extracted phase2.

    The phases phase1 and phase2 are then subtracted element-wise as 
    :math:`phase = phase1 - phase2` and are used to create phase locking signal 
    (PLS) by formula :math:`PLS = exp(phase*j)`. All values from PLS lays on 
    complex unit circle holding :math:`angle = phase` with oriented x axis.

  - The PVL is calculated as: :math:`PVL = mean(np.exp(phase*j)) = mean(PLS)`, 
    where phase is mentioned earlier and j is complex constant 
    :math:`j^2 = -1`.

  - Example

    .. code-block:: py
      :name: PLV-example1.7.1

      #Example1
      fs = 5000
      x1=np.linspace(0*np.pi, 4*np.pi, num=10001)
      sig=np.sin(20*x1)+np.sin(120*x1)*np.exp(-x1)
      compute_plv_count(sig, fs=fs, lowband=[4, 8], highband=[80, 150])
        >> 0.45569549961750905+0.01530084091583396j

    .. figure:: images/1.7.1Example.png
      :name: Fig1.7.1

    In the picture, on the top left corner there is the signal filtered from 
    the original signal with Butterworth filter in :math:`<4,8>` Hz band as the 
    real signal and its Hilbert transformation as the imag signal. From this 
    complex signal the phase1 is extracted.
    In the second row of graphs on the left, there is amplitude of the Hilbert 
    transforamation of signal filtere in signal in  :math:`80,150` Hz band as 
    the real signal and its hilbert transformation as the imag sig. On the 
    right side there is phase2 extracted from the signal on left.
    In the third row, there is signal created by phase difference of phase1 and 
    phase2, on left with its complex values and on right as simple 
    :math:`phase1-phase2` difference.
    The values of the third row are inserted into complex plane in the bottom 
    of the picture as the blue stars, the more denser blue is, the more values 
    lie on this part of unit circe. The mean is then calculated and displayed 
    as orange vector.

    The PLV in this example shows some phase locking around zero angle (also 
    visible in the third row in values 6000-10000), but not the absolute phase 
    locking because values 0-6000 does not show this coupling.
    This picture is only for better understanding, the real data should never 
    look like this.

    Special example is the constant zero value. When all phase values are same.

    .. code-block:: py
      :name: PLV-example1.7.2

      #Example1
      fs = 5000
      x1=np.linspace(0*np.pi, 4*np.pi, num=10001)
      sig=x1*0
      compute_plv_count(sig, fs=fs, lowband=[4, 8], highband=[80, 150])
        >> 1+0j

    .. figure:: images/1.7.2Example.png
      :name: Fig1.7.2

- Power spectral entropy

  - The Power spectral entropy (PSE) varies in the interval 
    :math:`(0,log2(length(sig))>`, where the log2 is logarithm with base 2 and 
    length(sig) stands for the length of the input signal.

  - The Power spectral entropy is normalized feature, so multiplication by 
    constant would make no difference to the output.

  - In the first step of the calculation, the Fast Fourier Transform (fft) of 
    the input signal is calculated. This fft signal is squared element-wise as 
    :math:`a_i := a_i^2`, where a_i is i-th element of the signal. Then the 
    signal is normalized using :math:`p_i := a_i/sum(a)`, where sum(a) is sum 
    of the elements of the signal. From normalized signal, the entropy H is 
    calculated, using formula :math:`H = sum(p_i * log2(p_i))`, where log2 is 
    the logarithm with base 2. The entropy H is returned as output of this 
    function.

  - Example

    .. code-block:: py
      :name: PSE-example1.8.1

      #Example1
      sig = np.ones(10001)
      compute_pse(sig)
        >> 2.65551518538626e-29

    .. figure:: images/1.8.1.1Example.png
      :name: Fig1.8.1.1

    The original signal contains constant signal (not visible due to big scale) 
    and its Fourier series which is big at the first element, but zero 
    everywhere else.

    .. figure:: images/1.8.1.2Example.png
      :name: Fig1.8.1.2

    The normalized fft signal differs from the unnormalized fft signal only by 
    a different scale.
    The ressult of the PSE in this case is 0 (with some numerical error).

    .. code-block:: py
      :name: PSE-example1.8.2

      #Example2
      sig = np.real(np.fft.ifft(np.ones(10001)))
      compute_pse(sig)
        >> 13.287856641838337
      # np.log2(10001) = 13.287856641840545

    Input signal is created to have constant Fourier transforamation. This 
    signal should have the biggest PSE value, which is close to logarithm of 
    lenght of input signal. However, this type of signal should not be usual 
    at real signals.

    .. figure:: images/1.8.2.1Example.png
      :name: Fig1.8.2.1

    This original signal has only one non-zero value at the beginning. And you 
    can easily see that the Fourier transform of this signal has a constant 
    value (with some calculation error).

    .. figure:: images/1.8.2.2Example.png
      :name: Fig1.8.2.2

    .. code-block:: py
      :name: PSE-example1.8.3

      #Example3
      length1 = 10001
      x1=np.linspace(0*np.pi, 4*np.pi, num=length1
      sig = np.sin(x1)
      compute_pse(sig)
        >> 1.0000036953163833

    .. code-block:: py
      :name: PSE-example1.8.4

      #Example4
      length1 = 10001
      x1=np.linspace(0*np.pi, 4*np.pi, num=length1)
      sig = 7*np.sin(x1)
      compute_pse(sig)
        >> 1.0000036953163833

    As you can see on example 3 and 4 above, scaling by multiplication does 
    not change output, because the feature is normalized.

    However, shifting on y-axis could cause some change as you can see on 
    examples 5 and 6 below. Shift could increase or decrese  PSE value.

    .. code-block:: py
      :name: PSE-example1.8.5

      #Example5
      length1 = 10001
      x1=np.linspace(0*np.pi, 4*np.pi, num=length1)
      sig = 13+7*np.sin(x1)
      compute_pse(sig)
        >> 0.6746547357194002

    .. code-block:: py
      :name: PSE-example1.8.6

      #Example6
      length1 = 10001
      x1=np.linspace(0*np.pi, 4*np.pi, num=length1)
      sig = 3+7*np.sin(x1)
      compute_pse(sig)
        >> 1.5708852216530274

    The difference in the output after shift is caused by change of first 
    element of Fourier transformed signal. If the shift is much stronger than 
    any other frequency, the output of PSE will be smaller. If the shift is 
    similarly strong as other frequencies (elements of the Fourier transformed 
    signal), the output of PSE should be bigger.

- Sample entropy

  - The sample entropy (SE) feature estimates the entropy of a given signal.
    SE varies in the interval 
    :math:`(0,log((length(sig)-m)*(length(sig)-m-1))>`, where the log is 
    natural logarithm length(sig) stands for the length of the input signal an 
    m is the input parametr.

  - The SE feature is dependent on sampling frequency of the signal and also 
    length of the signal. Combining signals with different sampling frequencies, 
    without careful consideration, is not recomended.

  - The input parameters r and m are at default values set to :math:`r = 0.1` 
    and :math:`m = 2`. R is relative distance constant and m is maximal length 
    of subsequences. Calcualation of the SE begins by calculating the standard 
    deviation of signal, which is multiplied by r. This constant will be the 
    maximal distance parametr R.
    
    The main computation begins by creating all subsequences of m consecutive 
    samples of original signal. For example, if the signal is 5001 samples long 
    and :math:`m=8`, :math:`4994 = 5001-8+1` subsequences of length 8 are 
    created.
    Next step is to calculate Chebyshev distance (the biggest difference in 
    absolute value) between all subsequences. If the distance between two 
    vectors is less than R (calculated before), the 1 is added to B.

    Same steps are used again, but only after adding +1 to m. The summed value 
    is now stored in A. 
    
    A is always smaller than B. The final ressult is obtained by computing 
    :math:`SE = -log(A/B)`, where log is natural logarithm.


- Shannon entropy

  - Shannon entropy (SHE) feature, calculating the shannon entropy of the signal.
    SHE varies in the interval :math:`(0, log2(10)=3.321928094887362>`.

  - Signal is separated to nbins = 10 equidistant bins. Number of bins cannot 
    be changed in the input. Bins are normalized by formula 
    :math:`p(i) = C(i)/sum(C)`, where C(i) is number of elements in the bin and 
    sum(C) is length of the signal (or the sum of the elements in all bins). 
    The shannon entropy is then calculated by 
    :math:` SHE = -sum(p(i)*log(p(i)))`, where log is natural logarithm.

  - Example

    .. code-block:: py
      :name: SHE-example1.10.1

      #Example1
      length1 = 5000 + 1
      x1=np.linspace(0*np.pi, 2*np.pi, num=length1)
      sig = np.sin(x1)
      compute_shanon_entropy(sig)
        >> 3.148995547001215

    .. figure:: images/1.10.1.1Example.png
      :name: Fig1.10.1.1

    .. figure:: images/1.10.1.2Example.png
      :name: Fig1.10.1.2

    Shannon entropy is not dependent on scaling or moving on y-axis. As you can 
    see on next example.

    .. code-block:: py
      :name: SHE-example1.10.2

      #Example2
      length1 = 5000 + 1
      x1=np.linspace(0*np.pi, 2*np.pi, num=length1)
      sig = 11+7*np.sin(x1)
      compute_shanon_entropy(sig)
        >> 3.148995547001215

    .. figure:: images/1.10.2.1Example.png
      :name: Fig1.10.2.1

    .. figure:: images/1.10.2.2Example.png
      :name: Fig1.10.2.2

    The next example shows, that stacionary function have zero shannon entropy, 
    because :math:`p(i)*log(p(i)) = 0`, for :math:`p(i)->0` and also for 
    :math:`p(i) = 1` (:math:`log(1) = 0`).

    .. code-block:: py
      :name: SHE-example1.10.3

      #Example3
      length1 = 5000 + 1
      sig = np.ones(length1)
      compute_shanon_entropy(sig)
        >> 0

    .. figure:: images/1.10.3.1Example.png
      :name: Fig1.10.3.1

    .. figure:: images/1.10.3.2Example.png
      :name: Fig1.10.3.2

    The opposite is true for a signal with a homogeneous distribution, as in 
    the next example. In this case, the Shannon entropy is :math:`log2(10)` 
    with some rounding error.
    
    .. code-block:: py
      :name: SHE-example1.10.4

      #Example4
      length1 = 5000 + 1
      x1=np.linspace(0*np.pi, 2*np.pi, num=length1)
      sig = x1
      compute_shanon_entropy(sig)
        >> 3.3219278354443875

    .. figure:: images/1.10.4.1Example.png
      :name: Fig1.10.4.1

    .. figure:: images/1.10.4.2Example.png
      :name: Fig1.10.4.2

- Signal stats

  - Signal stats are some of the basics functions used in statiscics.
    In this case this feature returns standard deviation,mean, median, maximum, 
    minimum, 25 percentil and 75 percentil. Important note is, all of these 
    statistics are taken after the squared signal (element-wise) has been 
    calculated.

  - The output is dependent on position on y-axis, because of the second power. 
    Using this feature with signals around 0 may not produce the expected 
    results.

  - power_std: standard deviation of power in band

    - The standard deviation of the signal is calculated as 
      :math:`STD = sqrt(sum((x(i)-m)^2)/N)`, where m is the mean of the 
      signal, N is the number of samples (signal length), ^2 is the square, 
      sqrt is the square root and x(i) are squared samples of the signal.
  
  - power_mean: mean of power in band

    - The mean value of the signal is calculated as :math:`m = sum(x(i))/N`, 
      where N is the number of samples (signal length) and x(i) are squared 
      samples of the signal.

  - power_median: median of power in band

    - The power median of the signal is calculated as the value, where half 
      of values of the signal are greater, than median value.

  - power_max: max value of power in band

    - The maximum signal value is the largest value in the signal. The value 
      from which all other values are smaller.

  - power_min: min value of power in band

    - The maximum signal value is the largest value in the signal. The value 
      from which all other values are smaller.

  - power_perc25: 25 percentile of power in band

    - The 25 percentile of the signal is calculated as the value where 25% of 
      the signal values are smaller than the returned value.

  - power_perc75: 75 percentile of power in band

    - The 75 percentile of the signal is calculated as the value where 75% of 
      the signal values are smaller than the returned value.

  - Example

    .. code-block:: py
      :name: SST-example1.11.1

      #Example
      length = 5000 + 1
      x1=np.linspace(0.00, 2*np.pi, num=length)
      sig=np.sin(x1)
      print(compute_signal_stats(sig))   
        >>0.3535887229607282, 0.4999000199960008, 0.5000000000000001, 1.0, 0.0, 
          0.14644660940672616, 0.8535533905932737

      # power_std, power_mean, power_median, power_max, power_min
      # power_perc25, power_perc75

    The mean and median do not represent the expected 0 because the second 
    power changes all negative values to positive.

    .. figure:: images/1.11.1Example.png
      :name: Fig1.11.1

    Power mean and power median have in this case similar values, so they could 
    not be both vissible at the same time.

Bivariate feature extraction
*********************************
Bivariate feature extraction algorithms server for calculating relationships 
between two signals. 
They can be used for example to obtain connectivity between different areas 
of the brain.

- Coherence

  - The coherence (Coh) varies in the interval :math:`<0,1>` and reflects 
    frequency similarities between two signals.
    :math:`Coh=1` indicates, the one signal is directly influenced by the 
    second signal, :math:`Coh=0` indicates no influence by second signal.
    The coherence between two signals can be calculated with a time-lag. 
    Maximum time-lag should not exceed :math:`fmax/2`.

  - Coh is calculated by coherence method in scipy.signal as: 
    :math:`Coh(X,Y)=[|P(X,Y)|/(√(P(X,X)・P(Y,Y)))]`. 
    Where X,Y are the two evaluated signals, |・| stands for absolute value, 
    √ stands for square root, P(X,X) and P(Y,Y) stands for power spectral 
    density estimation and P(X,Y) stands for cross spectral density estimation.
    The P(X,X) is calculated as

  - Lagged coherence is calculated (LagCoh) by coherence method in scipy.signal 
    as: :math:`LagCoh(X',Y)=[|P(X',Y)|/(√(P(X',X')・P(Y,Y)))]`.
    Where X' is signal lagged by lag k and Y is nonlagged signal, |・| stands 
    for absolute value, √ stands for square root, P(X',X') and P(Y,Y) stands 
    for power spectral density estimation and P(X',Y) stands for cross spectral 
    density estimation.

  - From all time-lagged values, only the maximum value with its time-lag 
    koeficient are returned.

  - Example

    .. code-block:: py
      :name: Coh-example2.1.1

      x1=np.linspace(0.0, 8*np.pi, num=1001)
      y1=np.sin(x1)
      sig = np.array([y1,y1])
      fs=250
      fband=[1.0, 4.0]
      lag=0
      lag_step=1
      fft_win=250
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.9999999999999999 0
      # the coherence between the same signals is 1

    .. code-block:: py
      :name: Coh-example2.1.2

      sig = np.array([y1,-y1])
      # other variables stands same as in example Coh-example2.1.1 above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.9999999999999999 0
      # the coherence between the same signals is 1

    .. figure:: images/2.1.2Example.png
      :name: Fig2.1.2

    .. code-block:: py
      :name: Coh-example2.1.3

      sig = np.array([y1,-y1])
      lag = 250
      # other variables stands same as in example above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 1.0 0
      # the coherence between the opposite signals is 1

    .. figure:: images/2.1.3Example.gif
      :name: Fig2.1.3

    This gif shows, how does program go through the data with lag = 250 and 
    compute coherence between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the lag (in samples) in the iteration.

    .. code-block:: py
      :name: Coh-example2.1.4.1

      y2  = np.sin(x1)-np.sin(2*x1)+np.sin(3*x1)-np.sin(4*x1)
      sig = np.array([y1,y2])
      lag = 250
      # other variables stands same as in example above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.6180260559346161 250

    .. figure:: images/2.1.4.1Example.gif
      :name: Fig2.1.4.1

    This gif shows, how does program go through the data with lag = 250 and 
    compute coherence between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the lag (in samples) in the iteration.

    Program shows, the maximal  coherence between the signals is, if the first
    signal is 250 samples ahead.

    .. code-block:: py
      :name: Coh-example2.1.4.2

      y2=-np.sin(2*x1)+np.sin(3*x1)-np.sin(4*x1)
      sig = np.array([y1,y2])
      lag = 250
      # other variables stands same as in example above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.40572228497072715 180

    .. figure:: images/2.1.4.2Example.gif
      :name: Fig2.1.4.2

    This gif shows, how does program go through the data with lag = 250 and 
    compute coherence between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the lag (in samples) in the iteration.

    Program shows, the maximal  coherence between the signals is, if the first
    signal is 180 samples ahead.

    Though neither of two correlations above is significantly large. It may 
    show, how this feature could determine the difference between two signals 
    that the human eye cannot see.

- Linear correlation
  
  - The linear correlation (LC) varies in interval :math:`<-1,1>` and reflects 
    shape similarities between two signals. 
    :math:`LC=1` indicates perfect conformity between two signals, 
    :math:`LC=-1` indicates opposite signals and :math:`LC=0` indicates two 
    different signals.
    The linear correlation between two signals can be calculated with a 
    time-lag. Maximum time-lag should not exceed :math:`fmax/2`.

  - LC is calculated by Pearson’s correlation coefficient as: 
    :math:`LC(X,Y)=[cov(X,Y)/(std(X)・std(Y))]`, 
    where X,Y are the two evaluated signals, cov is the covariance and std is 
    the standard deviation. 

  - Lagged linear correlation (LLC) for each time-lag k was calculated by 
    Pearson’s correlation coefficient as: 
    :math:`LLC(X',Y)=[cov(X',Y)/std(X')・std(Y)]`, where X' is signal lagged by 
    lag k and Y is nonlagged signal, cov is the covariance and std is the 
    standard deviation. 
  
  - From all time-lagged values, the real vaule of the greatest corr value and 
    its lag index is returned. Negative corr values are evaluated in its 
    absolute value, but retuned as negative.

  - Example

    .. code-block:: py
      :name: LinCorr-example2.2.0
      
      lag=8
      lag_step=1

      x1=np.linspace(0.0, 8*np.pi, num=41)
      x2=np.linspace(-np.pi, 7*np.pi, num=41)
      y1=np.sin(x1)
      y2=np.sin(x2)
      sig = np.array([y1,y2])
      print(sig)
        >>[[ 0.00000000e+00  5.87785252e-01  9.51056516e-01  9.51056516e-01
           5.87785252e-01  1.22464680e-16 -5.87785252e-01 -9.51056516e-01
           -9.51056516e-01 -5.87785252e-01 -2.44929360e-16  5.87785252e-01
            9.51056516e-01  9.51056516e-01  5.87785252e-01  3.67394040e-16
            -5.87785252e-01 -9.51056516e-01 -9.51056516e-01 -5.87785252e-01
            -4.89858720e-16  5.87785252e-01  9.51056516e-01  9.51056516e-01
            5.87785252e-01  6.12323400e-16 -5.87785252e-01 -9.51056516e-01
            -9.51056516e-01 -5.87785252e-01 -7.34788079e-16  5.87785252e-01
            9.51056516e-01  9.51056516e-01  5.87785252e-01  8.57252759e-16
            -5.87785252e-01 -9.51056516e-01 -9.51056516e-01 -5.87785252e-01
            -9.79717439e-16]
            [-1.22464680e-16 -5.87785252e-01 -9.51056516e-01 -9.51056516e-01
            -5.87785252e-01  0.00000000e+00  5.87785252e-01  9.51056516e-01
            9.51056516e-01  5.87785252e-01  1.22464680e-16 -5.87785252e-01
            -9.51056516e-01 -9.51056516e-01 -5.87785252e-01 -2.44929360e-16
            5.87785252e-01  9.51056516e-01  9.51056516e-01  5.87785252e-01
            3.67394040e-16 -5.87785252e-01 -9.51056516e-01 -9.51056516e-01
            -5.87785252e-01 -4.89858720e-16  5.87785252e-01  9.51056516e-01
            9.51056516e-01  5.87785252e-01  6.12323400e-16 -5.87785252e-01
            -9.51056516e-01 -9.51056516e-01 -5.87785252e-01 -7.34788079e-16
            5.87785252e-01  9.51056516e-01  9.51056516e-01  5.87785252e-01
            8.57252759e-16]]
      # 2 signals are simulated as 2 sin functions, one of them is delayed by 
      #  'pi' so the lag is 5
      # initial lag was 8, so first and last 8 values of sig[0] were discarded
    
    .. figure:: images/2.2.4Example.png
      :name: Fig2.2.0

    To create this graph, two siganls form Example above were used. 
    On y-axis are values of sig[0] and sig[1], x-axis represents koeficients 
    of the values.

    .. code-block:: py
      :name: LinCorr-example2.2.1

      #Example1
      compute_lincorr(sig, lag, lag_step)         # lag=8, lag_step=1   
        >>-1.0 0
      #In lag = -5: lincorr = 0.9999999999999999 due to rounding error
      #In lag = +5: lincorr = 1, but algorithm choose first biggest correlation

    .. figure:: images/2.2.1Example.gif
      :name: Fig2.2.1

      This gif shows, how does program go through the data from Example1 and 
      compute Pearson’s correlation coefficient between them. 
      The y(n_i) represents n_i_th value of signal, 'i' stands for the number 
      of iterations. 

      If  :math:`i == 0` , signals are not shiftet
        | :math:`i < 0` , signal sig[1] is after sig[0].
        | :math:`i > 0` , signal sig[0] is after sig[1].
      :math:`lag = 0` in this example

      At the end the lag with greatest correlation is returned.
    .. The duration of each image in gif  is 1000ms and loop is set to 1000

    .. code-block:: py
      :name: LinCorr-example2.2.2

      #Example2
      y1=np.sin(x1)+1
      sig = np.array([y1,y2])
      compute_lincorr(sig, lag, lag_step)         # lag=8, lag_step=1  
        >>-1.0 0
      # Linear correlation is independent to scalar adition

    .. figure:: images/2.2.2Example.gif
      :name: Fig2.2.2

      This gif shows, how does program go through the data from Example2 and 
      compute Pearson’s correlation coefficient between them. 
      The y(n_i) represents n_i_th value of signal, 'i' stands for the number 
      of iterations. 

      If  :math:`i == 0` , signals are not shiftet
        | :math:`i < 0` , signal sig[1] is after sig[0].
        | :math:`i > 0` , signal sig[0] is after sig[1].
      :math:`lag = 0` in this example

    .. The duration of each image in gif  is 1000ms and loop is set to 1000

    .. code-block:: py
      :name: LinCorr-example2.2.3

      #Example3
      y1=10*np.sin(x1)+1
      sig = np.array([y1,y2])
      compute_lincorr(sig, lag, lag_step)         # lag=8, lag_step=1  
        >>1.0 5
      # also lincorr[13] = 1, the program returns first highest value

    .. figure:: images/2.2.3Example.gif
      :name: Fig2.2.3

      This gif shows, how does program go through the data from Example2 and 
      compute Pearson’s correlation coefficient between them. 
      The y(n_i) represents n_i_th value of signal, 'i' stands for the number 
      of iterations. 

      If  :math:`i == 0` , signals are not shiftet
        | :math:`i < 0` , signal sig[1] is after sig[0].
        | :math:`i > 0` , signal sig[0] is after sig[1].
      :math:`lag = 5` in this example, so sig[0] is ahead sig[1]

      At the end the lag with first greatest correlation is returned.
    .. The duration of each image in gif  is 1000ms and loop is set to 1000

    .. code-block:: py
      :name: LinCorr-example2.2.4

      #Example4
      lag = 0
      y1 = np.sin(x1)
      sig = np.array([y1,-y1])
      compute_lincorr(sig, lag, lag_step) # lag=0, lag_step=1 
        >>-1.0 0
      # The opposite signals have linear correlation equal -1

    .. figure:: images/2.2.4Example.png
      :name: Fig2.2.4

      To create this graph, two opposite siganls form Example4 were used. 
      On y-axis are values of sin, x-axis represents koeficients of the values.
      The correlation of opposite signals is -1.

    .. code-block:: py
      :name: LinCorr-example2.2.5

      #Example5
      lag = 10
      y1 = np.sin(x1)
      y2 = np.cos(x1)
      sig = np.array([y1,y2])
      compute_lincorr(sig, lag, lag_step) # lag=10, lag_step=1 
        >>-0.946761134320959 -3
      # If corr value is negative, method take its absolute value and if it is 
      # the maximal value, than method return value as negative.

    .. figure:: images/2.2.5Example.gif
      :name: Fig2.2.5

    .. The duration of each image in gif  is 1000ms and loop is set to 1000

      To create this graph, two opposite siganls form Example4 were used. 
      On y-axis are values of sin, x-axis represents koeficients of the values.
      If the signal have negative correlation, method take its absolute value 
      and if it is the maximal value, than method return value is negative.

.. questions
  lag < 0 ? https://stackoverflow.com/questions/509211/how-slicing-in-python-works
  2 signals with different lengths?

- Phase consistency

  - Phase consistency (PC) varies in interval :math:`(0,1>` and reflects 
    conformity in phase between two signals, regardless of any phase shift 
    between them. 

  - First, phase synchrony (PS) is calculated as 
    :math:`PS=√[(<cos(ΦZt)>)^2+(<sin(ΦZt)>)^2]`, where ΦZt is instantaneous 
    phase difference of signal ΦXt and ΦYt :math:`ΦZt=ΦXt-ΦYt`, <> stands for 
    mean and √ for square root. Instantaneous phase ΦXt is calculated as 
    :math:`ΦXt=arctan(xH/xt)`, where xH is the Hilbert transformation of the 
    time signal xt.

  - PC is then calculated as :math:`PC = <PS>・(1-2*std(PS))`, where std is the 
    standard deviation and <・> stands for mean.

  - Although this feature is empirical, it has mathematical background.
    The 3 sigma rule says, for normal distribution 95 % of values are in the 
    interval :math:`<mean(・)-2*std(・), mean(・)+2*std(・)>`, where the std(・)
    stands for standard deviation.

    Because all the values of PS lay in the interval :math:`(0,1>` and we 
    obtain again value from interval :math:`(0,1>`, the 3 sigma rule is 
    modified with multiplication standard deviation by mean. Then only the
    lower bound is used.

    In broad strokes, this feature pinpoint the value of PS above which are 
    95 % of all PS values obtained with inserted phase lag and phase lag step.

    The limitation of this feature is, that data often does not satisfy the 
    normal distribution. Then the ressult does not have to fullfil this 
    interpretation, nontheless the result is still usefull.

  - Example

    .. code-block:: py
      :name: PC-example2.4.1

      #Example1
      x1=np.linspace(6*np.pi, 16*np.pi, num=4001)
      y1=np.sin(x1)
      y2=np.cos(x1)

      sig = np.array([y1,y2])
      lag = 500
      lag_step = 1
      compute_phase_const(sig, lag, lag_step)       

        >> 0.8650275116884527                          

    .. figure:: images/2.3.1Example.png
      :name: Fig2.3.1

    The histogram is devided to 10 bins to show the distribution of lagged PS
    values. The orange line represents PC value calculated by this algorithm.

    In previous example are all phase synchrony values near 1 and although they
    are not normally distributed, PC returns value as they would be naturally 
    distribudet with same mean and standard deviation.

    .. code-block:: py
      :name: PC-example2.4.2

      #Example2
      x1=np.linspace(6*np.pi, 16*np.pi, num=4001)
      y1=np.sin(x1)
      y2=np.cos(10000/(x1*x1)-4)

      sig = np.array([y1,y2])
      lag = 500
      lag_step = 1
      compute_phase_const(sig, lag, lag_step)     

        >> 0.35096503373573645                         

    .. figure:: images/2.3.2Example.png
      :name: Fig2.3.2

    The histogram is devided to 10 bins to show the distribution of lagged PS
    values. The orange line represents PC value calculated by this algorithm.

    In previous example are all phase synchrony values distributed across the 
    whole interval and although they are not normally distributed, PC returns 
    value as they would be naturally distributed with same mean and standard 
    deviation.

- Phase lag index

  - Phase lag index (PLI) varies in interval :math:`<0,1>` and represents evaluation of 
    statistical interdependencies between time series, which is supposed to be 
    less influenced by the common sources (Stam et al. 2007). 

  - PLI calculation is based on the phase synchrony between two signals with
    constant, nonzero phase lag, which is most likely not caused by volume 
    conduction from a single strong source. 
  
    Phase lag index is calculated as :math:`PLI=|<sign[ΔΦ(tk)]>|`, where sign 
    represents signum function, <・> stands for mean, |・| stands for absolute 
    value and ΔΦ is a phase difference between two iEEG signals.

  - PLI could be in general also calculated without absolute value, then the sign 
    represents direction. This feature does not alow calculation of signed value.

  - Maximum time-lag should not exceed fmax/2. The maximum value of PLI is stored 
    with its time-lag value.

  - Example

    .. code-block:: py
      :name: PLI-example2.4.1

      #Example1
      lag = 50
      lag_step = 5
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.cos(x1)
      y2=np.cos(x1) + 0.1*np.sin(0.5-np.random.rand(4001))
      sig = np.array([y1,y2])
      compute_pli(sig, lag, lag_step)      # lag = 50, lag_step = 5

        >> 1.0 50                          # max_PLI, max_PLI_lag

      # Program takes the first biggest value with its time-lag value in samples

    .. figure:: images/2.4.1Example.gif
      :name: Fig2.4.1

    This gif shows, how does program go through the data with lag = 50 and 
    compute signes PLI between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the the lag in iteration. Gif shows signed values of
    PLI for better understanding, but this feature counts only with absolute 
    value of PLI.

    .. code-block:: py
      :name: PLI-example2.4.2

      #Example2
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.cos(x1)
      y2=np.cos(x1) + np.sin(0.5-np.random.rand(4001))
      sig = np.array([y1,y2])
      compute_pli(sig, lag, lag_step)       # lag = 50, lag_step = 5

        >> 1.0 50                           # max_PLI, max_PLI_lag

      # Program takes the first biggest value with its time-lag value in samples

    .. figure:: images/2.4.2Example.gif
      :name: Fig2.4.2

    This gif shows, how does program go through the data with lag = 50 and 
    compute signes PLI between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the the lag in iterations. Gif shows signed values of
    PLI for better understanding, but this feature counts only with absolute 
    value of PLI.

    .. code-block:: py
      :name: PLI-example2.4.3

      #Example3
      x1=np.linspace(6*np.pi, 16*np.pi, num=2001)

      y1=np.cos(x1)
      y2=np.cos(10000/(x1*x1)-4)
      sig = np.array([y1,y2])
      compute_pli(sig, lag, lag_step)       # lag = 50, lag_step = 5

        >> 0.5328774329300369 -15            # max_PLI, max_PLI_lag

      # Program takes the first biggest value with its time-lag value in samples.
      # Program calculates only the absolute value of PLI

    .. figure:: images/2.4.3Example.gif
      :name: Fig2.4.3

    This gif shows, how does program go through the data with lag = 50 and 
    compute signes PLI between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the the lag in iterations. Gif shows signed values of
    PLI for better understanding, but this feature counts only with absolute 
    value of PLI.

- Phase synchrony

  - Phase synchrony (PS) varies in interval :math:`(0,1>` and reflects synchrony 
    in phase between two signals.

  - PS is calculated as :math:`PS=√[(<cos(ΦZt)>)^2+(<sin(ΦZt)>)^2]`, where ΦZt 
    is instantaneous phase difference of signal ΦXt and ΦYt :math:`ΦZt=ΦXt-ΦYt`,
    <> stands for mean and √ for square root. 
    Instantaneous phase ΦXt is calculated as :math:`ΦXt=arctan(xH/xt)`, where 
    xH is the Hilbert transformation of the time signal xt.

  - The :math:`PS = 1` indicates constant phase difference :math:`ΦZt` by 
    famous equation :math:`(cos(ΦZt))^2+(sin(ΦZt))^2 = 1`. With bigger number 
    of miscellaneous phase differences the PS decreses, but usually after big 
    enough number of data starts to have convergence character.

    The :math:`PS -> 0` indicates the big diversity in signal frequency.

  - Examples

    .. code-block:: py
      :name: LinCorr-example2.5.1

      #Example1
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(x1)
      y2=np.cos(x1)
      sig = np.array([y1,y2])
      compute_phase_sync(sig)

        >>0.9999999003538571          #PS value

      # Two signals with same phase have PS value close to 1

    .. code-block:: py
      :name: LinCorr-example2.5.2

      #Example2
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(2*x1)
      y2=np.cos(2*x1)
      sig = np.array([y1,y2])
      compute_phase_sync(sig)

        >>0.9999997868133397         #PS value

      # Two signals with same phase have PS value close to 1

    .. code-block:: py
      :name: LinCorr-example2.5.3

      #Example3
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(1.1*x1)
      y2=np.cos(x1)
      sig = np.array([y1,y2])
      compute_phase_sync(sig)

        >>0.7908266399758462         #PS value

      # Two signals with similar phase have PS high PS value, but not that close
      # to 1, as same signals

    .. code-block:: py
      :name: LinCorr-example2.5.4

      #Example4
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(2*x1)
      y2=np.cos(x1)   
      sig = np.array([y1,y2])
      compute_phase_sync(sig)

        >>0.00025832361592383534     #PS value

      # Two signals with different phase have PS value near 1
  
- Relative entropy

  - To evaluate the randomness and spectral richness between two time-series, 
    the Kullback-Leibler divergence, i.e. relative entropy (REN), is calculated.     
    REN is a measure of how entropy of one signal diverges from a second, 
    expected one. 
    
  - REN of signals X, Y  is calculated as :math:`REN(X,Y)=sum[pX_i・log(pX_i/pY_i)]`,
    where pX is a probability distribution of investigated signal, pY is a 
    probability distributions of expected signal and log is natural logarithm.

  - To calculate propability distribution the each signal is devided to 10
    separete equidistant bins by numpy histogram method.
    For example pX_0 is percentage of values in the lowest :math:`10 %`, band
    of signal X.
    The bands for the 2 signals does not have to be the same.
    For consistency of data the numer of bins is fixed and should not be changed
    as parametr of function.

  - The important note to this is, that relative entropy is not 
    metric, because it is not symetric (REN(X, Y) is not equal to REN(Y, X)) 
    and does not satisfy the triangular inequality.
    The value of REN varies in interval :math:`<0,+Inf)` and :math:`REN=0` 
    indicates the equality of  statistical distributions of two signals, 
    while :math:`REN>0` indicates that the two signals are carrying different 
    information. 

    If the value of entropy equals :math:`REN=inf`, program returns np.nan.
    :math:`REN=inf` indicates, the signal Y have too low sampling frequency or 
    one of the signal is sacionar or signal Y is not satisfyingly continuous or
    signal Y is corrupted. :math:`REN=inf` is caused by signal Y having one of 
    the bins empty (probability of pY_i = 0).
   
  - The directional properties in epileptic signals need to be further explored.

  - Examples

    .. code-block:: py
      :name: LinCorr-example2.6.1

      #Example1
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(x1)
      y2=np.cos(x1)
      sig = np.array([y1,y2])
      compute_relative_entropy(sig)

        >>6.323111682295058e-07           #REN  

      # Two different singals should not have relative entropy equal zero
      # Two similar signals should have relativly low relative entropy value  
      
    .. code-block:: py
      :name: LinCorr-example2.6.2

      #Example2
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(x1)
      y2=np.exp(x1)
      sig = np.array([y1,y2])
      compute_relative_entropy(sig)

        >>1.7129570917945496              #REN

      sig = np.array([y2,y1])
      compute_relative_entropy(sig)

        >>1.182381303654846               #REN
      

      # Relative entropy depends on order of signals as are inserted

    .. code-block:: py
      :name: LinCorr-example2.6.3

      #Example3
      x1=np.linspace(0.0, 8*np.pi, num=4001)

      y1=np.sin(x1)
      y2=np.cos(x1*0))
      sig = np.array([y1,y2])
      # np.histogram(sig[0], 10): 
      #         [820, 360, 296, 264, 261,  260, 264, 296, 360, 820]
      # np.histogram(sig[1], 10): 
      #         [  0,   0,   0,   0,   0, 4001,   0,   0,   0,   0]

      compute_relative_entropy(sig)

        >>nan                           #REN

      # Two different singals should not have relative entropy equal zero
      # if the signal sig[1] have one (or more) of the bin probability equal 0
      # the REL = np.inf

      sig = np.array([y2,y1])
      compute_relative_entropy(sig)

        >>2.7336179778417073            #REN

      # Two different singals should not have relative entropy equal zero
      # if the signal sig[0] have one (or more) of the bin probability equal 0
      # and the sig[1] have all bins with non-zero probability, program returns
      # finite value

- Spectra multiplication

  - Spectra multiplication (convolution) of two signals is calculated as 
    :math:`conv(X,Y) = ifft(fft(X)*fft(Y))`, where fft is Fast Fourier 
    Transform, '*' is element-wise multiplication and ifft is Inverse
    Fast Fourier Transform and X,Y are the evaluated signals.
  
    To convolved signal the Hilbert transformation is aplied and from all
    absolute values the mean and standard deviation is calculated. The mean and
    standard deviation are both calculated by numpy library, the Hilbert 
    transform is calculated by scipy.signal library.

  - The Fast Fourier Transform (fft) approach is used, because on big dataset
    as a neural signals it is proved to be significantly faster, than computing 
    convolution by definition. However, for datasets with :math:`samples < 500` 
    this method is less efective than computing by convolution definition.
  
  - The Spectra multiplication mean (SM_mean) varies in the interval 
    :math:`<0,inf)`.
    :math:`SM_mean=0` indicates, the one signal is constantly zero,
    If method evaluates two signals with the phase similarities, the SM_mean 
    value will be significantly bigger. 

  - Example

  .. code-block:: py
    :name: LinCorr-example2.7.1

    #Example1
    x1=np.linspace(0.00, 8*np.pi, num=1001)

    y1=np.sin(x1*0)
    y2=np.sin(x1)
    sig = np.array([y1,y2])
    compute_spect_multp(sig)

      >>0.0 0.0     #np.mean(sig_sm), np.std(max(sig_sm))
    # The two signals have SM_mean value equal 0 if one of the signals 
    # is constantly 0

  .. code-block:: py
    :name: LinCorr-example2.7.2

    #Example2
    x1=np.linspace(0.00, 8*np.pi, num=1001)

    y1=np.sin(x1)
    y2=np.sin(x1)
    sig = np.array([y1,y2])
    compute_spect_multp(sig)

      >>500.473477696902 0.011583149274828326
                                          #np.mean(sig_sm), np.std(max(sig_sm))

    # The two signals have high SM_mean value and low SM_std value, if singals
    # are non-zero and the same

  .. code-block:: py
    :name: LinCorr-example2.7.3

    #Example3
    x1=np.linspace(0.00, 8*np.pi, num=1001)

    y1=np.sin(x1*1.1) + np.sin(3*x1)
    y2=np.sin(x1)
    sig = np.array([y1,y2])
    compute_spect_multp(sig)

      >>391.40497112474554 1.126140158602267
                                          #np.mean(sig_sm), np.std(max(sig_sm))

    # The two signals have high SM_mean value and low SM_std value, if singals
    # have similar frequency

  .. code-block:: py
    :name: LinCorr-example2.7.4

    #Example4
    x1=np.linspace(0.00, 8*np.pi, num=1001)

    y1=10*np.sin(3*x1)
    y2=11*np.sin(x1)
    sig = np.array([y1,y2])
    compute_spect_multp(sig)

      >>52.526392847268205 25.428527556507547
                                          #np.mean(sig_sm), np.std(max(sig_sm))

    # The two signals should have relativly high SM_mean value even if they are 
    # phase independent. Then they have also significantly higher SM_std values 

  .. code-block:: py
    :name: LinCorr-example2.7.5

    #Example5
    x1=np.linspace(0.00, 8*np.pi, num=1001)

    y1=10*np.sin(3*x1)
    y2=np.sin(x1)
    sig = np.array([y1,y2])
    compute_spect_multp(sig)

      >>4.775126622478946 2.3116843233188766
                                          #np.mean(sig_sm), np.std(max(sig_sm))

    # The main role in the signals takes the frequency, with lower amplitude
    # the SM_mean is smaller, but ratio SM_mean/SM_std does not change much
     

.. convolution?
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html#scipy.signal.fftconvolve

Event detection
*********************************
This subsection provides algorithms for detection of events occurring in the signal. 
All algorithms provide event position or event start/stop and some of them provide additional features of detected events. 
Currently the library contains algorithms for detecting interictal epileptiform discharges (IEDs),i.e. epileptic spikes, and a number of algorithms for detection of high frequency oscillations (HFOs).
