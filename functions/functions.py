###################################################################################
## Copyright (c) 2022 Simon Muntwiler
##
## See attached LICENSE file
###################################################################################

import math
import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

###################################################################################
## generate system matrices of cooling system
###################################################################################
def generate_cooling_system():
    nx = 4
    nu = 4
    ny = 2
    dt = 0.1

    # continuous time system dynamics
    thermal_coupling = 1
    Ac_local = 0.005 * np.eye(nx)
    Ac_coupling = 0.001 * np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
    Ac = Ac_local + thermal_coupling * Ac_coupling
    Bc = np.eye(nx)

    # measure temperature at subsystem 1
    Cc = np.array([[1/3,1/3,1/3,0],[1/3,0,1/3,1/3]])

    # discrete time system
    Ad = np.eye(4) + dt * Ac
    Bd = dt * Bc
    Cd = Cc

    # noise and disturbance parameters
    disturbance_var = 0.01
    noise_var = 0.1
    disturbance_cov = disturbance_var * np.eye(nx)
    noise_cov = noise_var * np.eye(ny)
    
    # tuning cost regularizer
    gamma = round(disturbance_var / noise_var,2)

    # threshold constraint
    temp_threshold = 103
    h_x = temp_threshold * np.ones((nx,1))
    H_x = np.eye(nx)
    
    # initial sate distribution
    x0_mean = 100*np.ones((nx,))
    P0 = np.eye(nx)
    h_x0 = temp_threshold * np.ones((nx,1))
    H_x0 = np.eye(nx)

    # bounded disturbances:
    bound = np.sqrt(disturbance_var)
    h_w = bound * np.ones((2*nx,1))
    H_w = np.array([[-1,0,0,0],[1,0,0,0],[0,-1,0,0],[0,1,0,0],[0,0,-1,0],
               [0,0,1,0],[0,0,0,-1],[0,0,0,1]])
    
    # MHE constraint
    h_x_MHE = Ad@h_x + bound * np.ones((nx,1))
    
    # safe input
    safety_factor = 1
    u_s = safety_factor * (h_x - Ad@(h_x_MHE) - bound * np.ones((nx,1)))[0,0] / dt
    u_s = math.floor(u_s)

    # parameter constraint set
    thermal_coupling_min = 0.1
    thermal_coupling_max = 50

    # generate system dict
    system = {}
    system['Ad'] = Ad
    system['Bd'] = Bd
    system['Cd'] = Cd
    system['Ac_local'] = Ac_local
    system['Ac_coupling'] = Ac_coupling
    system['nx'] = nx
    system['nu'] = nu
    system['ny'] = ny
    system['H_x'] = H_x
    system['h_x'] = h_x
    system['x0_mean'] = x0_mean
    system['P0'] = P0
    system['H_x0'] = H_x0
    system['h_x0'] = h_x0
    system['h_x_MHE'] = h_x_MHE
    system['u_s'] = u_s
    system['safety_factor'] = safety_factor
    system['H_w'] = H_w
    system['h_w'] = h_w
    system['disturbance_cov'] = disturbance_cov
    system['noise_cov'] = noise_cov
    system['gamma'] = gamma
    system['dt'] = dt
    system['temp_threshold'] = temp_threshold
    system['thermal_coupling'] = thermal_coupling
    system['thermal_coupling_min'] = thermal_coupling_min
    system['thermal_coupling_max'] = thermal_coupling_max
    
    return system

###################################################################################
# Generate MHE CvxpyLayers problem for given linear system model
###################################################################################
def generate_mhe_estimator(system,T):

    # extract system parameters
    Ad = system['Ad']
    Bd = system['Bd']
    Cd = system['Cd']
    nx = system['nx']
    nu = system['nu']
    ny = system['ny']
    H_x = system['H_x']
    h_x_MHE = system['h_x_MHE']
    H_w = system['H_w']
    h_w = system['h_w']
    disturbance_cov = system['disturbance_cov']
    noise_cov = system['noise_cov']

    # fixed parameters
    y_param = cp.Parameter((ny, T+1))
    u_param = cp.Parameter((nu,T))
    Bd_mhe = Bd
    Cd_mhe = Cd
    Q_inv = np.linalg.inv(disturbance_cov)
    R_inv = np.linalg.inv(noise_cov)

    # parameters to update
    Ad_mhe = cp.Parameter((nx,nx))
    
    # prior weighting
    x_prior = cp.Parameter(nx)
    prior_delta = cp.Variable(nx)
    P_prior_inv_sqrt = cp.Parameter((nx,nx), PSD=True)

    # optimization variables
    x_opt = [cp.Variable(nx) for _ in range(T+1)]
    v_opt = cp.Variable((ny, T+1))
    w_opt = cp.Variable((nx, T))

    # objective and constraint definition
    objective = 0
    constraints = []

    for k in range(T):
        objective += cp.quad_form(v_opt[:,k], R_inv) + cp.quad_form(w_opt[:,k], Q_inv)
        constraints.append(cp.reshape(x_opt[k+1],(nx,1)) == Ad_mhe@cp.reshape(x_opt[k],(nx,1)) + Bd_mhe@cp.reshape(u_param[:,k],(nu,1)) + cp.reshape(w_opt[:,k],(nx,1)))
        constraints.append(H_x @ cp.reshape(x_opt[k],(nx,1)) <= h_x_MHE)
        constraints.append(cp.reshape(y_param[:,k],(ny,1)) == Cd_mhe@cp.reshape(x_opt[k],(nx,1)) + cp.reshape(v_opt[:,k],(ny,1)))
        constraints.append(H_w @ cp.reshape(w_opt[:,k],(nx,1)) <= h_w)
    objective += cp.quad_form(v_opt[:,T], R_inv)
    objective += cp.sum_squares(P_prior_inv_sqrt@prior_delta)
    constraints.append(prior_delta == x_opt[0] - x_prior)
    constraints.append(cp.reshape(y_param[:,T],(ny,1)) == Cd_mhe@cp.reshape(x_opt[T],(nx,1)) + cp.reshape(v_opt[:,T],(ny,1)))
    constraints.append(H_x @ cp.reshape(x_opt[T],(nx,1)) <= h_x_MHE)

    # define MHE problem
    mhe = cp.Problem(cp.Minimize(objective), constraints)
    assert mhe.is_dpp()

    # define MHE problem as CvxpyLayer
    mhe_layer = CvxpyLayer(mhe, parameters=[Ad_mhe, u_param, y_param, P_prior_inv_sqrt, x_prior], variables=[x_opt[T], v_opt, w_opt])
    
    return mhe_layer

###################################################################################
# Simulate linear dynamics with temperature threshold
###################################################################################
def dynamics(system, x, u):
    Ad = system['Ad']
    Bd = system['Bd']
    nx = system['nx']
    H_w = system['H_w']
    h_w = system['h_w']
    disturbance_cov = system['disturbance_cov']
    temp_threshold = system['temp_threshold']
    u_s = system['u_s']

    for i in range(nx):
        if x[i,0] > temp_threshold:
            u[i,0] = u_s
    
    mu = np.zeros((nx,))
    while True:
        w = np.random.multivariate_normal(mu, disturbance_cov).reshape(nx,1)
        if (H_w @ w <= h_w).all():
            break

    return Ad@x + Bd@u + w , u

###################################################################################
# Generate noisy measurements
###################################################################################
def measurement(system, x):
    ny = system['ny']
    mu = np.zeros((ny,))
    v = np.random.multivariate_normal(mu, system['noise_cov']).reshape(ny,1)
    return system['Cd']@x + v

###################################################################################
# Simulate system for nsim steps with given initial position and input
###################################################################################
def simulate(system, x0, inputs, nsim):
    nu = system['nu']
    
    states = []
    measurements = []
    
    for i in range(nsim):
        states.append(x0)
        measurements.append(measurement(system, x0))
        x0, u = dynamics(system,x0,inputs[:,i].reshape((nu,1)))
        inputs[:,i] = u.reshape(nu)
    
    return states, measurements, inputs

###################################################################################
# Generate random sinusoidal inputs
###################################################################################
def generate_inputs(nsim, nu):
    inputs = np.zeros((nu,nsim))
    for i in range(nu):
        amplitude = np.random.uniform(0,1)
        frequency = np.random.uniform(0, 1 / (2 * math.pi))
        shift = np.random.uniform(-math.pi, math.pi)
        inputs[i,:] = amplitude * np.sin(frequency * np.arange(0, nsim) + shift) - amplitude
    return inputs

###################################################################################
# Obtain ground truth loss between true states and estimates
###################################################################################
def gt_loss(gt_states, estimates):
    loss = 0
    if gt_states.shape[1] != estimates.shape[1]:
        raise ValueError("Number of ground truth states and estimates different")
    for k in range(gt_states.shape[1]):
        difference = gt_states[:,k] - estimates[:,k]
        loss += np.linalg.norm(difference, ord=2)**2
    return loss / gt_states.shape[1]

###################################################################################
# Generate sequence of MHE estimates and loss
###################################################################################
def generate_estimates_mhe(nsim, system, mhe_layer, inputs, measurements, Ad_mhe_tch, T):
    
    from pytorch_sqrtm.sqrtm import sqrtm

    Bd = system['Bd']
    Cd = system['Cd']
    nx = system['nx']
    nu = system['nu']
    ny = system['ny']
    disturbance_cov = system['disturbance_cov']
    noise_cov = system['noise_cov']
    gamma = system['gamma']
    x0_mean = system['x0_mean']
    P0 = system['P0']
    
    x_prior = torch.tensor(x0_mean)
    P_prior = torch.tensor(P0)
    P_prior_inv = torch.inverse(P_prior)
    P_prior_inv_sqrt = sqrtm(P_prior_inv)
    
    estimates = [x_prior]
    outputs = []
    
    Q = torch.tensor(disturbance_cov, requires_grad=False)
    R = torch.tensor(noise_cov, requires_grad=False)
    
    Cd_tch = torch.tensor(Cd, requires_grad=False).double()
    Bd_tch = torch.tensor(Bd, requires_grad=False)
    
    loss_mhe = torch.tensor(0, dtype=torch.float64)

    
    for i in range(2,inputs.shape[1] + 1):
        if i <= T:
            y_horizon_T = np.concatenate(measurements[0:i], axis=1)
            y_horizon_T = torch.tensor(y_horizon_T, dtype=torch.float64)
            u_horizon_T = torch.tensor(inputs[:,0:i-1], dtype=torch.float64)
            results = mhe_layer[i-2](Ad_mhe_tch, u_horizon_T, y_horizon_T, P_prior_inv_sqrt, x_prior)
        elif i > T:
            P_prior =  Q + Ad_mhe_tch @ P_prior @ Ad_mhe_tch.transpose(0,1) \
                - Ad_mhe_tch @ P_prior @ Cd_tch.transpose(0,1) @ torch.inverse(R + Cd_tch @ P_prior @ Cd_tch.transpose(0,1)) \
                @ Cd_tch @ P_prior @ Ad_mhe_tch.transpose(0,1)
            P_prior = P_prior.transpose(0,1)/2 + P_prior/2
            P_prior_inv = torch.inverse(P_prior)
            P_prior_inv_sqrt = sqrtm(P_prior_inv)
            y_horizon_T = np.concatenate(measurements[i-T-1:i], axis=1)
            y_horizon_T = torch.tensor(y_horizon_T, dtype=torch.float64)
            u_horizon_T = torch.tensor(inputs[:,i-T-1:i-1], dtype=torch.float64)
            results = mhe_layer[-1](Ad_mhe_tch, u_horizon_T, y_horizon_T, P_prior_inv_sqrt, estimates[i-T-1])
        estimates.append(results[0].reshape(nx,))
        outputs.append(measurements[i-1] - results[1].data[:,-1].numpy().reshape((ny,1)))
        
        x_est_step = results[0].reshape(nx,1).double()
        #w_last = results[2][:,-1]
        w_last = estimates[-1] - Ad_mhe_tch @ estimates[-2] - Bd_tch @ inputs[:,i-2].reshape((nu,1))

        difference = y_horizon_T[:,-1].reshape(ny,1).double() - torch.tensor(Cd).double()@x_est_step
        loss_mhe += torch.norm(difference)**2 + gamma * torch.norm(w_last)**2 
            
    return estimates, outputs, loss_mhe / nsim

###################################################################################
# Generate sequence of Kalman filter estimates and loss
###################################################################################
def generate_estimates_kalman_pytorch(nsim, system, inputs, measurements, Ad_kf_tch):
    Bd = system['Bd']
    Cd = system['Cd']
    nx = system['nx']
    nu = system['nu']
    disturbance_cov = system['disturbance_cov']
    noise_cov = system['noise_cov']
    gamma = system['gamma']
    x0_mean = system['x0_mean']
    P0 = system['P0']
    
    # define torch tensors
    Cd_tch = torch.tensor(Cd, requires_grad=False).double()
    Bd_tch = torch.tensor(Bd, requires_grad=False)
    u_tch = torch.tensor(np.array(inputs), requires_grad=False)
    y_tch = torch.tensor(np.array(measurements), requires_grad=False)
    
    # initial variance and state
    P_init = torch.tensor(P0, requires_grad=False).double()
    x_init = torch.tensor(100*np.ones((nx,1)))
    
    estimates = [x_init]
    P = P_init
    
    # initialize loss
    loss = torch.tensor(0, dtype=torch.float64)
    difference = y_tch[0] - Cd_tch @ estimates[-1]
    loss = torch.norm(difference)**2
    
    for i in range(len(measurements)):
        # predict
        x_hat = Ad_kf_tch @ estimates[-1].double() + Bd_tch @ inputs[:,i].reshape((nu,1))
        P_hat = Ad_kf_tch @ P @ Ad_kf_tch.transpose(0,1) + torch.tensor(disturbance_cov).double()

        # update
        y_tilde = y_tch[i] - Cd_tch @ x_hat
        dummy = torch.inverse(Cd_tch.double() @ P_hat.double() @ Cd_tch.transpose(0,1).double() + torch.tensor(noise_cov).double())
        K_gain = P_hat @ Cd_tch.transpose(0,1).double() @ dummy
        estimates.append(x_hat + K_gain @ y_tilde.double())
        P = (torch.eye(nx).double() - K_gain @ Cd_tch.double()) @ P_hat
        
        # loss
        w_last = estimates[-1] - Ad_kf_tch @ estimates[-2].double() - Bd_tch @ inputs[:,i].reshape((nu,1))
        difference = y_tch[i] - Cd_tch @ estimates[-1]
        loss += torch.norm(difference)**2 + gamma * torch.norm(w_last)**2 
        
    return estimates, loss/nsim

###################################################################################
# Initialize estimators
###################################################################################
def initialize_estimators(system, T, thermal_coupling_0, Ac_local_tch, Ac_coupling_tch):
    nx = system['nx']
    dt = system['dt']

    ## Kalman filter
    tc_kf_tch = torch.tensor(thermal_coupling_0).requires_grad_(True)
    Ad_kf_tch = torch.eye(nx) + dt * (Ac_local_tch + tc_kf_tch * Ac_coupling_tch)
    Ad_kf_tch.double()

    # MHE
    tc_mhe_tch = torch.tensor(thermal_coupling_0).requires_grad_(True)
    Ad_mhe_tch = torch.eye(nx) + dt * (Ac_local_tch + tc_mhe_tch * Ac_coupling_tch)
    Ad_mhe_tch.double()

    mhe_layer = []
    for i in range(1,T+1):
        mhe_layer.append(generate_mhe_estimator(system, i))
    
    return mhe_layer, tc_kf_tch, Ad_kf_tch, tc_mhe_tch, Ad_mhe_tch

###################################################################################
# Learn KF and MHE
###################################################################################
def learn_kf_mhe(system, inputs):
    # extract system data
    Ad = system['Ad']
    Bd = system['Bd']
    Cd = system['Cd']
    nx = system['nx']
    nu = system['nu']
    ny = system['ny']
    H_x = system['H_x']
    h_x = system['h_x']
    H_w = system['H_w']
    h_w = system['h_w']
    disturbance_cov = system['disturbance_cov']
    noise_cov = system['noise_cov']
    gamma = system['gamma']
    dt = system['dt']
    x0_mean = system['x0_mean']
    P0 = system['P0']

    # extract other input data
    epochs = inputs['epochs']
    n_sim_epoch = inputs['n_sim_epoch']
    T = inputs['T']
    thermal_coupling_0 = inputs['thermal_coupling_0']
    Ac_local_tch = inputs['Ac_local_tch']
    Ac_coupling_tch = inputs['Ac_coupling_tch']
    val_inputs = inputs['val_inputs']
    val_measurements = inputs['val_measurements']
    val_states = inputs['val_states']
    nsim = inputs['nsim']

    # initialize estimators
    mhe_layer, tc_kf_tch, Ad_kf_tch, tc_mhe_tch, Ad_mhe_tch = initialize_estimators(system, T, thermal_coupling_0, Ac_local_tch, Ac_coupling_tch)
    
    # initialization
    val_losses_mhe = []
    val_losses_kf = []
    losses_mhe = []
    losses_kf = []
    tc_values_mhe = [tc_mhe_tch.item()]
    tc_values_kf = [tc_kf_tch.item()]

    # initial validation loss
    val_states_mat = np.concatenate(val_states).reshape(-1,nx).transpose()
    with torch.no_grad():
        # initial validation loss kf
        estimates_kf, _ = generate_estimates_kalman_pytorch(nsim, system, val_inputs, val_measurements, Ad_kf_tch)
        x_est_kf_val_init = torch.cat(estimates_kf[1:], 0).numpy().reshape(-1,nx).transpose()
        val_losses_kf.append(gt_loss(val_states_mat,x_est_kf_val_init))
    
        # initial validation loss mhe
        estimates_mhe, _ , _ = generate_estimates_mhe(nsim, system, mhe_layer, val_inputs, val_measurements, Ad_mhe_tch, T)
        x_est_mhe_val_init = torch.cat(estimates_mhe, 0).numpy().reshape(-1,nx).transpose()
        val_losses_mhe.append(gt_loss(val_states_mat,x_est_mhe_val_init))
    
    alpha_0_kf = inputs['alpha_0_kf']
    alpha_0_mhe = inputs['alpha_0_mhe']
    
    # optimizer and learning rate definition
    lr_kf = lambda epoch: alpha_0_kf / (epoch + 1)
    lr_mhe = lambda epoch: alpha_0_mhe / (epoch + 1)
    opt_kf = torch.optim.SGD([tc_kf_tch], lr=lr_kf(0))
    opt_mhe = torch.optim.SGD([tc_mhe_tch], lr=lr_mhe(0))
    lr_scheduler_kf = torch.optim.lr_scheduler.LambdaLR(opt_kf, lr_kf)
    lr_scheduler_mhe = torch.optim.lr_scheduler.LambdaLR(opt_mhe, lr_mhe)

    for epoch in range(epochs):
        print('Epoch: ', epoch)

        opt_kf.zero_grad()
        opt_mhe.zero_grad()

        epoch_loss_kf = torch.tensor(0, dtype=torch.float64)
        epoch_loss_mhe = torch.tensor(0, dtype=torch.float64)
    
        for sim in range(n_sim_epoch):
            while True:
                x0_epoch = np.random.multivariate_normal(mean=system['x0_mean'],
                                                   cov=system['P0']).reshape(nx,1)
                if (system['H_x0'] @ x0_epoch <= system['h_x0']).all():
                    break
            epoch_inputs = generate_inputs(nsim,nu)
            epoch_states, epoch_measurements, epoch_inputs = simulate(system, x0_epoch,epoch_inputs,nsim)
    
            ## estimate using kf
            _, sim_loss_kf = generate_estimates_kalman_pytorch(nsim, system, epoch_inputs, epoch_measurements, Ad_kf_tch)
            epoch_loss_kf += sim_loss_kf
    
            ## estimate using mhe
            _, _, sim_loss_mhe = generate_estimates_mhe(nsim, system, mhe_layer, epoch_inputs, epoch_measurements, Ad_mhe_tch, T)
            epoch_loss_mhe += sim_loss_mhe
        
        ## learn parameter in kf
        epoch_loss_kf = epoch_loss_kf / n_sim_epoch
        epoch_loss_kf.backward(retain_graph=True)
        losses_kf.append(epoch_loss_kf.item())
        opt_kf.step()
        
    
        with torch.no_grad():
            lr_scheduler_kf.step()
            tc_kf_tch.data = tc_kf_tch.clamp(system['thermal_coupling_min'], system['thermal_coupling_max'])
            tc_values_kf.append(tc_kf_tch.item())
    
            # update Ad_kf_tch
            Ad_kf_tch.data = torch.eye(nx) + dt * (Ac_local_tch + tc_kf_tch * Ac_coupling_tch)
    
        ## learn parameter in MHE
        epoch_loss_mhe = epoch_loss_mhe / n_sim_epoch
        epoch_loss_mhe.backward(retain_graph=True)
        losses_mhe.append(epoch_loss_mhe.item())
        opt_mhe.step()

        with torch.no_grad():
            lr_scheduler_mhe.step()
            tc_mhe_tch.data = tc_mhe_tch.clamp(system['thermal_coupling_min'], system['thermal_coupling_max'])
            tc_values_mhe.append(tc_mhe_tch.item())
    
            # update Ad_mhe_tch
            Ad_mhe_tch.data = torch.eye(nx) + dt * (Ac_local_tch + tc_mhe_tch * Ac_coupling_tch)
    
        with torch.no_grad():
            # validation loss KF
            estimates_kf, _ = generate_estimates_kalman_pytorch(nsim, system, val_inputs, val_measurements, Ad_kf_tch)
            x_est_kf = torch.cat(estimates_kf[1:], 0).numpy().reshape(-1,nx).transpose()
            val_losses_kf.append(gt_loss(val_states_mat,x_est_kf))
    
            # validation loss mhe
            estimates_mhe, _ , _ = generate_estimates_mhe(nsim, system, mhe_layer, val_inputs, val_measurements, Ad_mhe_tch, T)
            x_est_mhe = torch.cat(estimates_mhe, 0).numpy().reshape(-1,nx).transpose()
            val_losses_mhe.append(gt_loss(val_states_mat,x_est_mhe))
        
    # final validation
    x_est_kf_val_final = x_est_kf
    x_est_mhe_val_final = x_est_mhe

    outputs = {}
    outputs['x_est_kf_val_init'] = x_est_kf_val_init
    outputs['x_est_mhe_val_init'] = x_est_mhe_val_init
    outputs['x_est_kf_val_final'] = x_est_kf_val_final
    outputs['x_est_mhe_val_final'] = x_est_mhe_val_final
    outputs['tc_values_kf'] = tc_values_kf
    outputs['tc_values_mhe'] = tc_values_mhe
    outputs['losses_kf'] = losses_kf
    outputs['losses_mhe'] = losses_mhe
    outputs['val_losses_kf'] = val_losses_kf
    outputs['val_losses_mhe'] = val_losses_mhe
    return outputs
