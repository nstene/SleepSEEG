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

..
  TODO

- Hjorth mobility

..
  TODO

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
      :name: Coh-example1.3.1

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
      :name: Coh-example1.3.2

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
      :name: Coh-example1.3.3

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
      :name: Coh-example1.3.4

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
      :name: Coh-example1.3.5

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

..
  TODO

- Mean vector length

..
  TODO

- Modulation index

..
  TODO

- Phase locking value

..
  TODO

- Powe spectral entropy

..
  TODO

- Sample entropy
..
  TODO

- Shannon entropy

..
  TODO

- Signal stats


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

  - Coh is calcualted by coherence method in scipy.signal as: 
    :math:`Coh(X,Y)=[|P(X,Y)|/(√(P(X,X)・P(Y,Y)))]`. 
    Where X,Y are the two evaluated signals, |・| stands for absolute value, 
    √ stands for square root, P(X,X) and P(Y,Y) stands for power spectral 
    density estimation and P(X,Y) stands for cross spectral density estimation.
    The P(X,X) is calcualted as

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
        >>1.0 13 #max(lincorr), lincorr.index(max(lincorr))
      #In this case lincorr[3] = 0.9999999999999999 due to rounding error

    .. figure:: images/2.2.1Example.gif
      :name: Fig2.2.1

      This gif shows, how does program go through the data from Example1 and 
      compute Pearson’s correlation coefficient between them. 
      The y(n_i) represents n_i_th value of signal, 'i' stands for the number 
      of iterations. 

      If  :math:`i == lag` , signals are not shiftet
        | :math:`i < lag` , signal sig[1] is after sig[0].
        | :math:`i > lag` , signal sig[0] is after sig[1]. 
      :math:`lag = 8` in this example

      At the end the lag with greatest correlation is returned.
    .. The duration of each image in gif  is 1000ms and loop is set to 1000

    .. code-block:: py
      :name: LinCorr-example2.2.2

      #Example2
      y1=np.sin(x1)+1
      sig = np.array([y1,y2])
      compute_lincorr(sig, lag, lag_step)         # lag=8, lag_step=1  
        >>1.0 13 #max(lincorr), lincorr.index(max(lincorr))
      # Linear correlation is independent to scalar adition

    .. figure:: images/2.2.2Example.gif
      :name: Fig2.2.2

      This gif shows, how does program go through the data from Example2 and 
      compute Pearson’s correlation coefficient between them. 
      The y(n_i) represents n_i_th value of signal, 'i' stands for the number 
      of iterations. 

      If  :math:`i == lag` , signals are not shiftet
        | :math:`i < lag` , signal sig[1] is after sig[0].
        | :math:`i > lag` , signal sig[0] is after sig[1]. 
      :math:`lag = 8` in this example

      At the end the lag with greatest correlation is returned.
    .. The duration of each image in gif  is 1000ms and loop is set to 1000

    .. code-block:: py
      :name: LinCorr-example2.2.3

      #Example3
      y1=10*np.sin(x1)+1
      sig = np.array([y1,y2])
      compute_lincorr(sig, lag, lag_step)         # lag=8, lag_step=1  
        >>1.0 3 #max(lincorr), lincorr.index(max(lincorr))
      # also lincorr[13] = 1, the program returns first highest value

    .. figure:: images/2.2.3Example.gif
      :name: Fig2.2.3

      This gif shows, how does program go through the data from Example2 and 
      compute Pearson’s correlation coefficient between them. 
      The y(n_i) represents n_i_th value of signal, 'i' stands for the number 
      of iterations. 

      If  :math:`i == lag` , signals are not shiftet
        | :math:`i < lag` , signal sig[1] is after sig[0].
        | :math:`i > lag` , signal sig[0] is after sig[1]. 
      :math:`lag = 8` in this example

      At the end the lag with greatest correlation is returned.
    .. The duration of each image in gif  is 1000ms and loop is set to 1000

    .. code-block:: py
      :name: LinCorr-example2.2.4

      #Example4
      lag = 0
      y1 = np.sin(x1)
      sig = np.array([y1,-y1])
      compute_lincorr(sig, lag, lag_step) # lag=0, lag_step=1 
        >>-1.0 0 #max(lincorr), lincorr.index(max(lincorr))
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
        >>-0.946761134320959 13 #max(lincorr), lincorr.index(max(lincorr))
      # If corr value is negative, method take its absolute value and if it is 
      # the maximal value, than method return value is negative.

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
    stands for standart deviation.

    Because all the values of PS lay in the interval :math:`(0,1>` and we 
    obtain again value from interval :math:`(0,1>`, the 3 sigma rule is 
    modified with multiplication standart deviation by mean. Then only the
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
    distribudet with same mean and standart deviation.

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
    value as they would be naturally distributed with same mean and standart 
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

  - PLI could be in general also calculaced without absolute value, then the sign 
    represents direction. This feature does not alow calculation of signed value.

  - Maximum time-lag should not exceed fmax/2. The maximum value of PLI is stored 
    with its time-lag value.

  - Example

    .. code-block:: py
      :name: LinCorr-example2.4.1

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
    signal, 'i' stands for the number of iteration. Gif shows signed values of
    PLI for better understanding, but this feature counts only with absolute 
    value of PLI.

    .. code-block:: py
      :name: LinCorr-example2.4.2

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
    signal, 'i' stands for the number of iterations. Gif shows signed values of
    PLI for better understanding, but this feature counts only with absolute 
    value of PLI.

    .. code-block:: py
      :name: LinCorr-example2.4.3

      #Example3
      x1=np.linspace(6*np.pi, 16*np.pi, num=2001)

      y1=np.cos(x1)
      y2=np.cos(10000/(x1*x1)-4)
      sig = np.array([y1,y2])
      compute_pli(sig, lag, lag_step)       # lag = 50, lag_step = 5

        >> 0.5328774329300369 -15            # max_PLI, max_PLI_lag

      # Program takes the first biggest value with its time-lag value in samples.
      # Program calculates only the absulute value of PLI

    .. figure:: images/2.4.3Example.gif
      :name: Fig2.4.3

    This gif shows, how does program go through the data with lag = 50 and 
    compute signes PLI between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the number of iterations. Gif shows signed values of
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
      # Two similar signals shoul have relativly low relative entropy value  
      
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
  
    To convolved signal the Hilbert transforamation is aplied and from all
    absolute values the mean and standart deviation is calculated. The mean and
    standart deviation are both calculated by numpy library, the Hilbert 
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
