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

- Low f maker

..
  TODO

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
      # the coherence between the opposite signals is 1

    .. figure:: images/2.1.2Example.png
      :name: Fig2.1.2

    .. code-block:: py
      :name: Coh-example2.1.3

      sig = np.array([y1,-y1])
      lag = 250
      # other variables stands same as in example above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.9999999999999999 0
      # the coherence between the opposite signals is 1

    .. figure:: images/2.1.3Example.gif
      :name: Fig2.1.3

    This gif shows, how does program go through the data with lag = 250 and 
    compute coherence between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the number of iterations.

    .. code-block:: py
      :name: Coh-example2.1.4.1

      y2=-1*np.sin(2*x1)+np.sin(3*x1)-np.sin(4*x1)
      sig = np.array([y1,y2])
      lag = 250
      # other variables stands same as in example above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.6180260559346161 0
      # the coherence between the opposite signals is 1

    .. figure:: images/2.1.4.1Example.gif
      :name: Fig2.1.4.1

    This gif shows, how does program go through the data with lag = 250 and 
    compute coherence between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the number of iterations.

    .. code-block:: py
      :name: Coh-example2.1.4.2

      y2=-1*np.sin(2*x1)+np.sin(3*x1)-np.sin(4*x1)
      sig = np.array([y1,y2])
      lag = 250
      # other variables stands same as in example above
      compute_coherence(sig, fs, fband, lag, lag_step, fft_win)
        >> 0.40572228497072715 70

    .. figure:: images/2.1.4.2Example.gif
      :name: Fig2.1.4.2

    This gif shows, how does program go through the data with lag = 250 and 
    compute coherence between them. The y(n_i) represents n_i_th value of 
    signal, 'i' stands for the number of iterations.

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
  
  - From all time-lagged values, only the maximum value with its time-lag 
    koeficient are returned.

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
        >>1.0 13 #np.max(lincorr), lincorr.index(max(lincorr))
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
        >>1.0 13 #np.max(lincorr), lincorr.index(max(lincorr))
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
        >>1.0 3 #np.max(lincorr), lincorr.index(max(lincorr))
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
        >>-1.0 0 #np.max(lincorr), lincorr.index(max(lincorr))
      # The opposite signals have linear correlation equal -1

    .. figure:: images/2.2.4Example.png
      :name: Fig2.2.4

      To create this graph, two opposite siganls form Example4 were used. 
      On y-axis are values of sin, x-axis represents koeficients of the values.
      The correlation of opposite signals is -1.



.. questions
  lag < 0 ? https://stackoverflow.com/questions/509211/how-slicing-in-python-works
  2 signals with different lengths?

- Phase consistency

  Phase consistency (PC) varies in interval :math:`(0,1>` and reflects conformity in phase between two signals, regardless of any phase shift between them. 
  First, phase synchrony (PS) is calculated for multiple steps of time delay between two signals as PS=√[(<cos(ΦXt)>)2+(<sin(ΦYt)>)2], where ΦXt is instantaneous phase of signal X, ΦYt is instantaneous phase of signal Y, <> stands for mean and √ for square root. 
  PC is then calculated as PC = <PS>・(1-std(PS)/0.5), where std is the standard deviation and <・> stands for mean.  
  Instantaneous phase ΦXt is calculated as ΦXt=arctan(xH/xt), where xH is the Hilbert transformation of the time signal xt.

- Phase lag index

  Phase lag index (PLI) varies in interval <0,1> and represents evaluation of statistical interdependencies between time series, which is supposed to be less influenced by the common sources (Stam et al. 2007). 
  PLI calculation is based on the phase synchrony between two signals with constant, nonzero phase lag, which is most likely not caused by volume conduction from a single strong source. 
  Phase lag index is calculated as PLI=|<sign[dΦ(tk)]>|, where sign represents signum function, <> stands for mean and dΦ is a phase difference between two iEEG signals. 
  Maximum time-lag should not exceed fmax/2. The maximum value of PLI is stored with its time-lag value.

- Phase synchrony

  Phase synchrony (PS) varies in interval <0,1> and reflects synchrony in phase between two signals. 
  PS is calculated as PS=√[(<cos(ΦXt)>)^2+(<sin(ΦYt)>)^2], where ΦXt is instantaneous phase of signal X, ΦYt is instantaneous phase of signal Y, <> stands for mean and √ for square root. 
  Instantaneous phase ΦXt is calculated as ΦXt=arctan(xH/xt), where xH is the Hilbert transformation of the time signal xt.

.. questgion
  why unwrap?

- Relative entropy

  To evaluate the randomness and spectral richness between two time-series, the Kullback-Leibler divergence, i.e. relative entropy (REN), is calculated. 
  REN is a measure of how entropy of one signal diverges from a second, expected one. The value of REN varies in interval <0,+Inf>. 
  REN=0 indicates the equality of  statistical distributions of two signals, while REN>0 indicates that the two signals are carrying different information. 
  REN is calculated between signals X, Y as REN=sum[pX・log(pX/pY)], where pX is a probability distribution of investigated signal and pY is a probability distributions of expected signal. 
  Because of asymmetrical properties of REN, REN(X, Y) is not equal to REN(Y, X). 
  REN is calculated in two steps for both directions (both distributions from channel pair were used as expected distributions). 
  The maximum value of REN is then considered as the final result, regardless of direction.

- Spectra multiplication

  - Spectra multiplication (convolution) of two signals is calculated as 
    :math:`conv(X,Y) = ifft(fft(X)*fft(Y))`, where fft is Fast Fourier 
    Transform, '*' is element-wise multiplication and ifft is Inverse
    Fast Fourier Transform and X,Y are the evaluated signals.
    | To convolved signal the Hilbert transforamation is aplied and from all
    absolute values the mean and standart deviation is calculated. The mean and
    standart deviation are both calculated by numpy library, the Hilbert 
    transform is calculated by scipy.signal library.

  - The Fast Fourier Transform (fft) approach is used, because on big dataset
    it is proved to be significantly faster, than computing convolution by
    definition. However, for datasets with :math:`samples < 500` this method
    is less efective than computing by convolution definition.
  
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
    # phase independent. Than they have also significantly higher SM_std values 

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
