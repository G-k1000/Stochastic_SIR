import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

########## Simulation ##########
def exp_rvs_gen(rate):
    '''
    Generate exponential random variable using scipy stats function

    Input:
    - rate of the exponenetial random variable (float).

    Output:
    - random value of an exponenetial distribution with given rate. Note if
    rate is 0, the exponential rv will have infinite scale.
    '''
    if rate == 0:
        X = st.expon.rvs(scale=np.infty)
    else:
        X = st.expon.rvs(scale=1/rate)
    return X


def stochastic_SIR(S0, I0, beta, gamma, T):
    '''
    Simulates the evolution of a disease within a population.

    Inputs:
    -S0, I0: intial number of suceptible and infected individual (int)
    - beta, gamma: rate of infection and recovery (float)
    - T: time horizon of the simulation (float)

    Outputs: 
    - susceptible evolution
    - infected evolution
    - time of events
    '''
    s = S0
    susceptible = [s]
    i = I0
    infected = [i]
    t = 0
    time = [t]
    while t < T:

        # generate possible jump time.
        a1 = exp_rvs_gen(s*i*beta)
        a2 = exp_rvs_gen(i*gamma)

        # pick minimum out of the two and update chain accordingly
        # a1 is an infection.
        if a1 < a2:
            s -= 1
            susceptible.append(s)
            i += 1
            infected.append(i)
            t += a1
            time.append(t)
        
        # a2 is a recovery
        else:
            susceptible.append(s)
            i -= 1
            infected.append(i)
            t += a2
            time.append(t)
    
    # remove last value if it occurs after T
    if time[-1] > T:
        time[-1] = T
        susceptible[-1] = susceptible[-2]
        infected[-1] = infected[-2]

    return susceptible, infected, time


def SIR_mean_field(S0, I0, beta, gamma, T, num_step):
    '''
    Simulate the SIR model using its mean field approximation.

    Input:
    - S0 and I0 are the initial population parameters (suceptible, infected)
    - beta and gamma are the infection and recovery rate parameter respectively.
    - T is the time amount the simulation should run for.
    - num_step is the number of time step to be used in the discretization.

    Output:
    - a triplet comprised of: the evolution of infected people (list), 
    the evolution of suceptible people (list), the time at which the process was
    eavaluated (list)
    '''
    t_list = np.linspace(0, T, num_step)
    infected = [I0]
    susceptible = [S0]
    for i in range(num_step - 1):
        dS = - beta * infected[i] * susceptible[i]
        dI = beta * infected[i] * susceptible[i] - gamma * infected[i]
        St = susceptible[i] + dS * T/num_step
        It = infected[i] + dI * T/num_step
        if St <= 0:
            susceptible.append(0)
        else:
            susceptible.append(St)
        if It <= 0:
            infected.append(0)
        else:
            infected.append(It)
    return np.array(susceptible), np.array(infected), np.array(t_list)


def diff_approxim(S0, I0, beta, gamma, T, num_step):
    '''
    Simulates SIR model using diffusion approximation and
    Euler-Maruyama method

    Input:
    - S0, I0: initial state of the pop. (int)
    - beta and gamma: infection and recovery rate (float)
    - T: time horizon of the simulation (float)
    - num_step: number of step in the simulation (int)

    Output:
    - evolution of the susceptible pop. (array)
    - evolution of the infected pop. (array)
    - time step (array)
    '''
    t_list = np.linspace(0, T, num_step + 1)
    step_size = T/num_step
    susceptible = np.zeros(num_step + 1)
    infected = np.zeros(num_step + 1)
    infected[0] = I0
    susceptible[0] = S0
    for i in range(num_step):
        dW_s = st.norm.rvs(loc=0, scale=step_size)
        dW_i = st.norm.rvs(loc=0, scale=step_size)

        # generate fluctation at next step size.
        dS = (
            - (beta * infected[i] * susceptible[i]) * step_size
            + np.sqrt(beta * infected[i] * susceptible[i]) * dW_s
        )
        dI = (
            beta * infected[i] * susceptible[i] * step_size
            - gamma * infected[i] * step_size
            + np.sqrt(beta * infected[i] * susceptible[i]) * dW_s
            - np.sqrt(gamma * infected[i]) * dW_i
        )

        # apply fluctuation to previous value.
        # update pop. value.
        St = susceptible[i] + dS
        It = infected[i] + dI
        susceptible[i+1] = St
        infected[i+1] = It
    return susceptible, infected, t_list


### RMSE computation
def RMSE(list1, list2):
    '''
    compute the root mean square error between two arrays of number

    Inputs:
    - two arrays of number with the same length.

    Output:
    - The root mean square error between the two arrays.
    '''
    return np.sqrt(((list1 - list2) ** 2).mean())


def RMSE_SIR_mean_field(S0, I0, beta, gamma, T, ref_step):
    '''
    Compute the RMSE of the mean field with [10e1, 10e5] step and a simulation 
    with ref_step step.

    Input:
    - S0, I0: initial state of the population (int)
    - beta, gamma: rate of infection and recovery (float)
    - T: time horizon of the simulation (float)
    - ref_step: number of step in the reference simulation (int)

    Output:
    - graph of the RMSE for the infected and susceptible
    - RMSE susceptible (list)
    - RMSE infected (list)
    '''
    RMSE_S = []
    RMSE_I = []

    # number of step we should compute the RMSE for
    step_list = [10, 100, 1000, 5000, 10000, 50000, 100000]

    # generate reference simulation
    S_ref, I_ref, time_ref = SIR_mean_field(S0, I0, beta, gamma, T, ref_step)

    # compute RMSE of reference simulation and simulation with 
    # number of steps in step_list
    for t in range(len(step_list)):
        S_1, I_1, time_1 = SIR_mean_field(S0, I0, beta, gamma, T, step_list[t])

        # using interpolation, evaluates lower step size data at smaller step count.
        interpolator_S = interp1d(time_ref, S_ref, kind='linear', fill_value="extrapolate")
        interpolator_I = interp1d(time_ref, I_ref, kind='linear', fill_value="extrapolate")
        interp_S = interpolator_S(time_1)
        interp_I = interpolator_I(time_1)

        # evaluate the RMSE of the larger step size with interpolated data
        RMSE_S.append(RMSE(S_1, interp_S))
        RMSE_I.append(RMSE(I_1, interp_I))

    # plot result.
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # plot RMSE of infected values
    axs[0].plot(step_list, RMSE_I)
    axs[0].set_title('RMSE infected')
    axs[0].set_xlabel("Number of Step")
    axs[0].set_ylabel("RMSE")

    # plot RMSE of susceptible values
    axs[1].plot(step_list, RMSE_S)
    axs[1].set_title('RMSE susceptible')
    axs[1].set_xlabel("Number of Step")
    axs[1].set_ylabel("RMSE")

    plt.tight_layout()
    plt.savefig('RMSE_mean_field.png')
    plt.show()
    return RMSE_S, RMSE_I



##########  Probability of Extinction ##########
def prob_extinct_SIR(S0, I0, beta, gamma, T, rel_err):
    '''
    Return the probability (using crude Monte Carlo) of the disease going 
    extinct. Relative error using formula 2.1.4

    Inputs:
    - S0, I0: initial state of the pop. (int)
    - beta and gamma: infection and recovery rate (float)
    - T: time horizon of the simulation (float)
    - tol_rel_err: desired relative error. program acheives it at 
    the 99% confidence level (float > 0).
    - N: number of iteration in pilot run.

    Outputs:
    - mean_CMC: Probability of the disease going extinct
    - var_sample: variance of our sample.
    - N: final number of iteration needed to achieve desires 
    relative error.
    - conf_int_99: 99% confidence interval
    '''
    is_extinct = []
    c_99 = abs(st.norm.ppf(0.995))

    # pilot run the stochastic SIR alg. 1000 times.
    for i in range(1000):
        I = stochastic_SIR(S0, I0, beta, gamma, T)[1]

        # add 1 to the sample the disease went extinct within time T
        # add 0 otherwise.
        is_extinct.append(int(I[-1] == 0))

    # Compute the mean and variance of the 1000 iteratrion
    mean_CMC = np.mean(is_extinct)
    var_sample = np.var(is_extinct, ddof=1)

    # compute confidence interval value for relative error 
    # condition
    lower_99 = mean_CMC - c_99 * np.sqrt(var_sample / 1000)
    half_int = c_99 * np.sqrt(var_sample / 1000)
    N = 1000

    # iterate the population model until the desired
    # relative error is met.
    while half_int / lower_99 > rel_err:

        # generate one more simulation
        Z = int(stochastic_SIR(S0, I0, beta, gamma, T)[1][-1] == 0)

        # update mean and variance accordingly
        mean_CMC = N / (N + 1) * mean_CMC + 1 / (N + 1) * Z
        var_sample = (N - 1) / N * var_sample + 1 / (N + 1) * (Z - mean_CMC) ** 2

        # update iteration count accordingly
        N += 1

        # update confidence interval
        lower_99 = mean_CMC - c_99 * np.sqrt(var_sample / N)
        half_int = c_99 * np.sqrt(var_sample / N)

    # Compute 1% significance confidence interval
    upper_99 = mean_CMC + c_99 * np.sqrt(var_sample / N)
    lower_99 = mean_CMC - c_99 * np.sqrt(var_sample / N)
    conf_int_99 = (lower_99, upper_99)

    return mean_CMC, var_sample, N, conf_int_99


##########  Variance Reudction ##########
def alpha_calc(beta, gamma, T, rel_err):
    '''
    compute the value alpha minimizing variance in control variates setting

    Input:
    - pop. parameters are omitted as algorithm is only valid
    for (99, 1)
    - beta, gamma: rate of infection and recovery (float)
    - T: time horizon of the simulation (float)

    Output:
    - Covariance between process and control variate
    - value alpha
    '''
 
    # depending on the initial condition the probability of the 
    # first iteration being an extinction is different.
    E_Y = 0.168067226891

    # Simulate the process until relative error is reached.
    prob, _, _, _ = prob_extinct_SIR(99, 1, beta, gamma, T, rel_err, 1000)
    alpha = (1 - prob) / (1 - E_Y)
    return alpha


def prob_extinct_control_err(S0, I0, beta, gamma, T, rel_err):
    '''
    Using Monte Carlo method and control variate compute prob of
    extinction. This program run until desire relative error is met (2.1.4).

    Input:
    - S0, I0 are the initial condition of the population (int)
    - beta and gamma are the infection and recovery rate (float)
    - T is the time horizon the simulation is run for (float)
    - rel_err is the the tolerance for the relative error of our estimation
    (0 < float < 1)
    - N is the number of iteration in the pilot run (int)

    Output:
    - prob of extinction (float)
    - variance of estimate (float)
    - number of iteration (int)
    - 99% confidence interval (tuple)
    '''
    c_99 = abs(st.norm.ppf(0.995))

    # get alpha value
    alpha = alpha_calc(beta, gamma, T, rel_err)
    E_Y = 0.168067226891

    # generate first iteration
    _, I, _, = stochastic_SIR(S0, I0, beta, gamma, T)
    Z_alpha_i = int(I[-1] == 0) - alpha * (int(I[1] == I0 - 1) - E_Y)
    mean_Z_alpha = Z_alpha_i
    var_Z_alpha = 0
    N = 1

    for i in range(1000):

        # generate a new iteration of Z_alpha
        _, I, _, = stochastic_SIR(S0, I0, beta, gamma, T)
        Z_alpha_i = int(I[-1] == 0) - alpha * (int(I[1] == I0 - 1) - E_Y)

        # update the mean and the variance
        mean_Z_alpha = N / (N + 1) * mean_Z_alpha + 1 / (N + 1) * Z_alpha_i
        var_Z_alpha = (N - 1) / N * var_Z_alpha + 1 / (N + 1) * (Z_alpha_i - mean_Z_alpha) ** 2

        # update iteration count
        N += 1

    # compute relative error bound
    half_int = c_99 * np.sqrt(var_Z_alpha / N)
    lower_99 = mean_Z_alpha - c_99 * np.sqrt(var_Z_alpha / N)

    # # run program until relative error is achieved
    while half_int / lower_99 > rel_err:

        # generate a new iteration of Z_alpha
        _, I, _, = stochastic_SIR(S0, I0, beta, gamma, T)
        Z_alpha_i = int(I[-1] == 0) - alpha * (int(I[1] == I0 - 1) - E_Y)

        # update the mean and the variance
        mean_Z_alpha = N / (N + 1) * mean_Z_alpha + 1 / (N + 1) * Z_alpha_i
        var_Z_alpha = (N - 1) / N * var_Z_alpha + 1 / (N + 1) * (Z_alpha_i - mean_Z_alpha) ** 2

        # update iteration count and condition
        N += 1
        half_int = c_99 * np.sqrt(var_Z_alpha / N)
        lower_99 = mean_Z_alpha - c_99 * np.sqrt(var_Z_alpha / N)
        c = half_int / lower_99
   
    # Compute confidence interval 1% significance level
    upper_99 = mean_Z_alpha + c_99 * np.sqrt(var_Z_alpha / N)
    lower_99 = mean_Z_alpha - c_99 * np.sqrt(var_Z_alpha / N)
    conf_int_99 = (lower_99, upper_99)

    return mean_Z_alpha, var_Z_alpha, conf_int_99, N


def prob_extinct_control_weak(S0, I0, beta, gamma, T):
    '''
    Using Monte Carlo method and control variate compute prob of
    extinction. This program does not control for the relative error.

    Input:
    - S0, I0 are the initial condition of the population (int)
    - beta and gamma are the infection and recovery rate (float)
    - T is the time horizon the simulation is run for (float)
    - N is the number of iteration in the pilot run (int)

    Output:
    - prob of extinction (float)
    - variance of estimate (float)
    - number of iteration (int)
    - 99% confidence interval (tuple)
    - relative error achieved.
    '''
    c_99 = abs(st.norm.ppf(0.995))

    # Run the pilot run N_bar times
    Y_bar = []
    Z_bar = []

    for i in range(15000):
        print(i)
        _, I, _, = stochastic_SIR(S0, I0, beta, gamma, T)
        Y_bar.append(int(I[1] == 0))
        Z_bar.append(int(I[-1] == 0))
    
    # Variance of Z in pilot run
    E_Z_bar = np.mean(Z_bar)

    # probability disease goes extinct after 5 iteration.
    E_Y = 0.1739130434782609
    
    # compute covariance of Y and Z using pilot run
    Cov_YZ = np.sum((np.array(Z_bar) - E_Z_bar) * (np.array(Y_bar) - E_Y)) / (10e5 - 1)

    # set alpha value
    alpha = Cov_YZ / (E_Y * (1 - E_Y))

    # generate first iteration
    _, I, _, = stochastic_SIR(S0, I0, beta, gamma, T)
    Z_alpha_i = int(I[-1] == 0) - alpha * (int(I[1] == I0 - 1) - E_Y)
    mean_Z_alpha = Z_alpha_i
    var_Z_alpha = 0
    N = 1

    # run algortihm using alpha value minimizing variance
    for i in range(9999):
        print(N)

        # generate a new iteration of Z_alpha
        _, I, _, = stochastic_SIR(S0, I0, beta, gamma, T)
        Z_alpha_i = int(I[-1] == 0) - alpha * (int(I[1] == I0 - 1) - E_Y)

        # update the mean and the variance
        mean_Z_alpha = N / (N + 1) * mean_Z_alpha + 1 / (N + 1) * Z_alpha_i
        var_Z_alpha = (N - 1) / N * var_Z_alpha + 1 / (N + 1) * (Z_alpha_i - mean_Z_alpha) ** 2

        # update iteration count
        N += 1
   
    # Compute confidence interval 1% significance level
    upper_99 = mean_Z_alpha + c_99 * np.sqrt(var_Z_alpha / N)
    lower_99 = mean_Z_alpha - c_99 * np.sqrt(var_Z_alpha / N)
    conf_int_99 = (lower_99, upper_99)
    rel_err = c_99 * np.sqrt(var_Z_alpha / N) / lower_99

    return mean_Z_alpha, var_Z_alpha, conf_int_99, rel_err


##########  Stochastic SIR-d ##########
def stochastic_SIR_d(S0, I0, beta, gamma, m, v, T):
    '''
    Simulates the SIR model with demographic effect stochastically

    Inputs:
    - S0, I0: intial number of suceptible and infected individual (int)
    - beta, gamma: rate of infection and recovery (float)
    - m, v: rate of birth/death and pathogen induced rate of death
    - T: time horizon of the simulation (float)

    Outputs: 
    - susceptible evolution
    - infected evolution
    - recovered evolution
    - time of events
    '''
    s = S0
    susceptible = [s]
    i = I0
    infected = [i]
    r = 0
    recovered = [r]
    t = 0
    time = [t]

    while t < T:

        # generate possible occurence time
        # we use a generating function based on the scipy stats one
        # that yields the appropriate rate when some of the sub pop. are 0.
        a1 = exp_rvs_gen(m * (s + i + r))
        a2 = exp_rvs_gen(m * s)
        a3 = exp_rvs_gen(beta * i * s)
        a4 = exp_rvs_gen((m + v) * i)
        a5 = exp_rvs_gen(gamma * i)
        a6 = exp_rvs_gen(m * r)
        event_time_list = [a1, a2, a3, a4, a5, a6]

        # pick minimum out of the lot
        # a1 is a birth in the whole pop.
        if a1 == min(event_time_list):
            t += a1
            time.append(t)
            s += 1
            susceptible.append(s)
            i = i
            infected.append(i)
            r = r
            recovered.append(r)
        
        # a2 is a death within the susceptible
        elif a2 == min(event_time_list):
            t += a2
            time.append(t)
            s -= 1
            susceptible.append(s)
            i = i
            infected.append(i)
            r = r
            recovered.append(r)

        # a3 is an infection
        elif a3 == min(event_time_list):
            t += a3
            time.append(t)
            s -= 1
            susceptible.append(s)
            i += 1
            infected.append(i)
            r = r
            recovered.append(r)

        # a4 is death within the infected
        elif a4 == min(event_time_list):
            t += a4
            time.append(t)
            susceptible.append(s)
            i -= 1
            infected.append(i)
            r = r
            recovered.append(r)

        # a5 is a recovery
        elif a5 == min(event_time_list):
            t += a5
            time.append(t)
            s = s
            susceptible.append(s)
            i -= 1
            infected.append(i)
            r += 1
            recovered.append(r)

        # a6 is a death within the recovered
        elif a6 == min(event_time_list):
            t += a6
            time.append(t)
            s = s
            susceptible.append(s)
            i = i
            infected.append(i)
            r -= 1
            recovered.append(r)

    # remove the last instance from chain if it occurs after T
    if time[-1] > T:
        time[-1] = T
        susceptible[-1] = susceptible[-2]
        infected[-1] = infected[-2]
        recovered[-1] = recovered[-2]
  
    return susceptible, infected, recovered, time


def SIR_d_mean_field(S0, I0, beta, gamma, T, m, v, num_step):
    '''
    Simulates the SIR model with demographic effect deterministically

    Inputs:
    - S0, I0: intial number of suceptible and infected individual (int)
    - beta, gamma: rate of infection and recovery (float)
    - m, v: rate of birth/death and pathogen induced rate of death
    - T: time horizon of the simulation (float)
    - num_step: number of step with which we should simulate the process
    with.

    Outputs: 
    - susceptible evolution
    - infected evolution
    - recovered evolution
    - time of events
    '''
    dt = T / num_step
    susceptible = [S0]
    infected = [I0]
    recovered = [0]
    time = np.linspace(0, T, num_step)

    # Using euler's method, simulate process
    for t in range(num_step - 1):
        s = susceptible[t]
        i = infected[t]
        r = recovered[t]

        susceptible.append(s + (m*(s+i+r) - m*s - beta*s*i)*dt)
        infected.append(i + (beta*s*i - (m+v)*i - gamma*i)*dt)
        recovered.append(r + (gamma*i - m*r)*dt)
    
    return susceptible, infected, recovered, time


def RMSE_SIR_d_mean_field(S0, I0, beta, gamma, m, v, T, ref_step):
    '''
    Compute the RMSE of the SIR-d mean field with [10e1, 10e5] step and a simulation 
    with ref_step step.

    Input:
    - S0, I0: initial state of the population (int)
    - beta, gamma: rate of infection and recovery (float)
    - m, v: rate of birth/death and pathogen induced rate of death
    - T: time horizon of the simulation (float)
    - ref_step: number of step in the reference simulation (int)

    Output:
    - graph of the RMSE for the infected and susceptible
    - RMSE susceptible (list)
    - RMSE infected (list)
    '''
    step_list = [10, 100, 1000, 5000, 10000, 25000, 50000, 75000, 100000]
    RMSE_S = []
    RMSE_I = []
    S_ref, I_ref, _, time_ref = SIR_d_mean_field(S0, I0, beta, gamma, T, m, v, ref_step)

    for t in range(len(step_list)):
        S_1, I_1, _, time_1 = SIR_d_mean_field(S0, I0, beta, gamma, T, m, v, step_list[t])

        # using interpolation, evaluates lower step size data at smaller step count.
        interpolator_S = interp1d(time_ref, S_ref, kind='linear', fill_value="extrapolate")
        interpolator_I = interp1d(time_ref, I_ref, kind='linear', fill_value="extrapolate")
        interp_S = interpolator_S(time_1)
        interp_I = interpolator_I(time_1)

        # evaluate the RMSE of the larger step size with interpolated data
        RMSE_S.append(RMSE(S_1, interp_S))
        RMSE_I.append(RMSE(I_1, interp_I))

    # plot result.
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # plot RMSE of infected values
    axs[0].plot(step_list, RMSE_I)
    axs[0].set_title('RMSE infected')
    axs[0].set_xlabel("Number of Step")
    axs[0].set_ylabel("RMSE")

    # plot RMSE of susceptible values
    axs[1].plot(step_list, RMSE_S)
    axs[1].set_title('RMSE susceptible')
    axs[1].set_xlabel("Number of Step")
    axs[1].set_ylabel("RMSE")

    plt.tight_layout()
    plt.savefig('RMSE_mean_field_SIR_d.png')
    plt.show()
    return RMSE_S, RMSE_I

##########  Extinction Stochastic SIR-d  ##########
def prob_extinct_SIR_d(S0, I0, beta, gamma, m, v, T, rel_err, N):
    '''
    Return the probability (using Monte Carlo) of the disease going extinct
    in the stochastic SIR with demographic effect, using N iteration

    Inputs:
    - S0, I0: initial state of the population (int)
    - beta, gamma, m and v are the infection, recovery, birth/death and pathogen
    induced death rates.
    - T is the amount of time for which we run the process for.
    - tol_rel_err denote the the tolereance for the relative error, i.e., our
    estimate will have a relative error smaller than this inputs.

    Outputs:
    - The Monte Carlo estimation of the probability of the disease to go
    extinct.
    '''
    is_extinct = np.zeros(N)
    c_99 = abs(st.norm.ppf(0.995))

    # pilot run the stochastic SIR alg. 1000 times.
    for i in range(N):
        _, I, _, _ = stochastic_SIR_d(S0, I0, beta, gamma, m, v, T)

        # add 1 to the sample the disease went extinct within time T
        # add 0 otherwise
        is_extinct[i] = (int(I[-1] == 0))

    # Compute the mean and variance of the 1000 iteratrion
    mean_CMC = np.mean(is_extinct)
    var_sample = np.var(is_extinct, ddof=1)

    # compute confidence interval value for relative error
    # condition
    half_int = c_99 * np.sqrt(var_sample / N)

    # iterate the population model until the desired
    # relative error is met.
    while half_int / mean_CMC > rel_err:

        # generate one more simulation
        Z = int(stochastic_SIR_d(S0, I0, beta, gamma, m, v, T)[1][-1] == 0)

        # update mean and variance accordingly
        mean_CMC = N / (N + 1) * mean_CMC + 1 / (N + 1) * Z
        var_sample = (N - 1) / N * var_sample + 1 / (N + 1) * (Z - mean_CMC) ** 2

        # update iteration count accordingly
        N += 1

        # update confidence interval
        lower_99 = mean_CMC - c_99 * np.sqrt(var_sample / N)
        half_int = c_99 * np.sqrt(var_sample / N)

    # Compute 1% significance confidence interval
    upper_99 = mean_CMC + c_99 * np.sqrt(var_sample / N)
    lower_99 = mean_CMC - c_99 * np.sqrt(var_sample / N)
    conf_int_99 = (lower_99, upper_99)

    return mean_CMC, var_sample, N, conf_int_99


##########  Alternative Implementation Not Used in Project  ##########
def sto_SIR_dt(S0, I0, R0, gamma, beta, T, num_step):
    '''
    Simulates SIR model stochastically using predfiend time intervals.

    Inputs:
    -S0, I0: intial number of suceptible and infected individual (int)
    - beta, gamma: rate of infection and recovery (float)
    - T: time horizon of the simulation (float)

    Outputs:
    - susceptible evolution
    - infected evolution
    - time of events
    '''
    dt = T / num_step
    t = 0
    s = S0
    i = I0
    r = R0
    susceptible = [s]
    infected = [i]
    recovered = [r]
    time = [t]
    for j in range(num_step):
        U = st.uniform.rvs()
        prob_inf = beta * i * s * dt
        prob_rec = gamma * i * dt
        prob_ev = prob_inf + prob_rec

        # infection occurs
        if U <= prob_inf:
            s -= 1
            i += 1
            t += dt
            susceptible.append(s)
            infected.append(i)
            time.append(t)

        # recovery occurs.
        elif U > prob_inf and U <= prob_ev:
            i -= 1
            r += 1
            t += dt
            susceptible.append(s)
            infected.append(i)
            time.append(t)
      
        # nothing happens.
        elif U >= prob_ev:
            t += dt
            susceptible.append(s)
            infected.append(i)
            time.append(t)
    return susceptible, infected, recovered, time


###  code used to produce document content.  ###
if __name__ == '__main__':

    ###  2.1  ###
    # simulate stochastic SIR 3 times and save simulation for doc.
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for i in range(3):
        # plot result.
        S, I, T = stochastic_SIR(99, 1, 0.02, 0.4, 10)
        axs[i].plot(T, S, label='Susceptible')
        axs[i].plot(T, I, label='Infected')
        axs[i].set_title(f'Simulation {1 + i}')
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Pop.")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig('stochastic_SIR.png')
    plt.show()

    ###  2.2  ###
    # plot RMSE of SIR mean field approximation and
    # recover and report RMSE for each number of step considered.
    S, I = RMSE_SIR_mean_field(SIR_mean_field, 99, 1, 0.02, 0.4, 10, 1000000)
    print(f'the RMSE for susceptible is {S}')
    print(f'the RMSE for infected is {I}')

    # simulate and plot process
    S, I, T = SIR_mean_field(99, 1, 0.02, 0.4, 10, 5000)
    plt.plot(T, S, label='Susceptible')
    plt.plot(T, I, label='Infected')
    plt.xlabel('Time')
    plt.ylabel('Pop.')
    plt.savefig('SIR_mean_field.png')
    plt.legend()
    plt.show()


    ###  2.3  ###

    # simulate process thrice with appropiate number of step
    ffig, axs = plt.subplots(1, 3, figsize=(10, 4))
    num_step = [100, 500, 1000]
    for i, j in enumerate(num_step):
        # plot result.
        S, I, T = diff_approxim(99, 1, 0.02, 0.4, 10, j)
        axs[i].plot(T, S, label='Susceptible')
        axs[i].plot(T, I, label='Infected')
        axs[i].set_title(f'Simulation with {j} Step')
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Pop.")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig('diffusion_approximation.png')
    plt.show()

    ###  3.2  ###
    # estimate probability of extinction
    # T = 1
    T = [1, 2, 10]
    for i in T:
        prob, var, N, conf_int = prob_extinct_SIR(99, 1, 0.02, 0.4, 1, 0.05)
        print(f'for T = {i} Prob. of extinction is {prob}')
        print(f'for T = {i} variance of sample is {var}')
        print(f'for T = {i} number of iteration is {N}')
        print(f'for T = {i} 99% confidence interval is {conf_int}')


    ### 3.3 ###
    # estimate the probability of extinction for S0 = 99, I0 = 1
    T = [1, 2, 10]
    for i in T:
        prob, var, N, conf_int = prob_extinct_control_err(99, 1, 0.02, 0.4, 1, 0.05, 1000)
        print(f'for T = {i} Prob. of extinction is {prob}')
        print(f'for T = {i} variance of sample is {var}')
        print(f'for T = {i} number of iteration is {N}')
        print(f'for T = {i} 99% confidence interval is {conf_int}')

    ###  3.4  ###
    # estimate the probability of extinction for S0 = 95, I0 = 5
    # running algorithm for N = 10000 iteration.
    prob, var, conf_int, rel_err = prob_extinct_control_weak(95, 5, 0.02, 0.4, 2)
    print(f'for T = 2 Prob. of extinction is {prob}')
    print(f'for T = 2 variance of sample is {var}')
    print(f'for T = 2 99% confidence interval is {conf_int}')
    print(f'for T = 2 relative error is {rel_err}')


    ###  5.1  ###
    # plot SIR-d
    dem_rate = [(10e-4, 10e-3), (10e-4, 10e-2), (10e-3, 10e-3), (10e-3, 10e-2)]
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, (m, v) in enumerate(dem_rate):

        # indexing subplots
        row = i // 2
        col = i % 2

        # generating and plotting
        s, i, r, t, = stochastic_SIR_d(99, 1, 0.02, 0.4, 10, m, v)
        axs[row, col].plot(t, s, label='Susceptible')
        axs[row, col].plot(t, i, label='Infected')
        axs[row, col].plot(t, r, label='Recovered')
        axs[row, col].set_title(f'm = {m}, v = {v}')
        axs[row, col].set_xlabel("Time")
        axs[row, col].set_ylabel("Pop.")
        axs[row, col].legend()
    plt.tight_layout()
    plt.savefig('stochastic_SIR_d.png')
    plt.show()


    ###  5.2  ###
    # compute RMSE of mean field approximation
    dem_rate = [(10e-4, 10e-3), (10e-4, 10e-2), (10e-3, 10e-3), (10e-3, 10e-2)]
    for i, (m, v) in enumerate(dem_rate):
        RMSE_S, RMSE_I = RMSE_SIR_d_mean_field(99, 1, 0.02, 0.4, m, v, 10, 1000000)
        print(f'for (m, v) = ({m}, {v}) RMSE susceptible = {RMSE_S}')
        print(f'for (m, v) = ({m}, {v}) RMSE Infected = {RMSE_I}')

    # plot mean field approximation
    dem_rate = [(10e-4, 10e-3), (10e-4, 10e-2), (10e-3, 10e-3), (10e-3, 10e-2)]
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, (m, v) in enumerate(dem_rate):

        # indexing subplots
        row = i // 2
        col = i % 2

        # generating and plotting
        s, i, r, t, = SIR_d_mean_field(99, 1, 0.02, 0.4, 10, m, v, 1000)
        axs[row, col].plot(t, s, label='Susceptible')
        axs[row, col].plot(t, i, label='Infected')
        axs[row, col].plot(t, r, label='Recovered')
        axs[row, col].set_title(f'm = {m}, v = {v}')
        axs[row, col].set_xlabel("Time")
        axs[row, col].set_ylabel("Pop.")
        axs[row, col].legend()
    plt.tight_layout()
    plt.savefig('mean_field_SIR_d.png')
    plt.show()git 

    
    ###  5.3  ###
    # Estimate probability of extinction
    T = [1, 2, 10]
    beta = [0.00414201, 0.00430605, 0.0045111, 0.0061515]
    for i in beta:
        for j in T:
            prob, var, N, conf_int = prob_extinct_SIR_d(95, 5, i, 0.4, 10e-4, 10e-2, j, 0.05, 10000)
            print(f'for beta = {i}, T = {j} Prob. of extinction is {prob}')
            print(f'for beta = {i}, T = {j} variance of sample is {var}')
            print(f'for beta = {i}, T = {j} number of iteration is {N}')
            print(f'for beta = {i}, T = {j} 99% confidence interval is {conf_int}')



