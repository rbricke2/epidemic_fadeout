import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

# this function sets parameters for graphs
def set_rcParams():
    font_size    = 10
    widths       = 1
    line_width   = widths + (widths * 0.5)
    font_leg     = font_manager.FontProperties(family="Arial", style='normal', size=font_size)
    rcParameters = {'font.size'                    : font_size,    # font size
                    'font.family'                  : "sans-serif", # specify the font
                    'font.sans-serif'              : "Arial",
                    'axes.linewidth'               : widths,       # set width of axes
                    'xtick.major.width'            : widths,       # set tick widths
                    'xtick.minor.width'            : widths,
                    'ytick.major.width'            : widths,
                    'ytick.minor.width'            : widths,
                    'grid.color'                   : 'lightgray',  # set grid color
                    'grid.linewidth'               : widths,       # set grid width
                    'grid.linestyle'               : '--',         # dashed grid
                    'lines.linewidth'              : line_width,   # line width
                    'patch.linewidth'              : widths,       # legend line width
                    'axes.axisbelow'               : True,         # plot grid behind data
                    'axes.titlesize'               : font_size,    # font size of plot title
                    'figure.constrained_layout.use': True}
    plt.rcParams.update(rcParameters)

    return font_leg

####################################################################
# simulate the original unscaled SIRS model using the Euler method

def F_org(t, b, A, p, w):
    return 1 + A * np.cos(2 * np.pi * (t + p) / w)

def update_flows_org(t, S, I, N, b, d, g, A, p, w):

    force = F_org(t, b, A, p, w) # seasonal forcing
    
    # update flows from one population to another
    flow_to_S = d * (N - S - I)        # waning immunity
    flow_to_I = b * S * I * force / N  # infection
    flow_to_R = g * I                  # recovery from infection

    return flow_to_S, flow_to_I, flow_to_R

def sim_SIRS_org(sim_len, dt, initials, N,
                 b, d, g, A = 0, p = 0, w = 1):
    # initialize the number of iterations
    n_iterations = int(sim_len / dt) + 1

    # initialize population sizes
    S = initials[0]; I = initials[1]

    # initalize the time
    t = 0.0

    # initialize flows between compartments
    flow_to_S, flow_to_I, flow_to_R = update_flows_forced(t, S, I, N, b, d, g, A, p, w)

    # initialize lists which record the population at each time step dt
    t_list = [t]; S_list = [S]; I_list = [I]

    for i in range(1, n_iterations):
        # update time
        t = i * dt

        # update the compartments
        S = S + ( flow_to_S - flow_to_I ) * dt
        I = I + ( flow_to_I - flow_to_R ) * dt

        # update the flows
        flow_to_S, flow_to_I, flow_to_R = update_flows_forced(t, S, I, N, b, d, g, A, p, w)

        # append data corresponding to time i*dt to lists
        t_list.append(t)
        S_list.append(S)
        I_list.append(I)
 
    return t_list, x_list, y_list
    
####################################################################
# simulate the scaled SIRS model using the Euler method,
# that is x = S / N, y = I / N, z = R / N, tau = delta * t, \tilde{y} = y / delta

def F_scaled(t, b, d, A, p, w):
    return 1 + A * np.cos(2 * np.pi * (t / d + p) / w)

def update_flows_forced(t, x, y, b, d, g, A, p, w):

    force = F_scaled(t, b, d, A, p, w) # seasonal forcing
    
    # update flows from one population to another
    flow_to_x = 1 - x - (d * y)    # waning immunity
    flow_to_y = b * x * y * force  # infection
    flow_to_z = g * y              # recovery from infection

    return flow_to_x, flow_to_y, flow_to_z

def sim_SIRS_scaled(sim_len, dt, initials,
                    b, d, g, A = 0, p = 0, w = 1):
    # initialize the number of iterations
    n_iterations = int(sim_len / dt) + 1

    # initialize population sizes
    x = initials[0]; y = initials[1]

    # initalize the time
    t = 0.0

    # initialize flows between compartments
    flow_to_x, flow_to_y, flow_to_z = update_flows_forced(t, x, y, b, d, g, A, p, w)

    # initialize lists which record the population at each time step dt
    t_list = [t]; x_list = [x]; y_list = [y]

    for i in range(1, n_iterations):
        # update time
        t = i * dt

        # update the compartments
        x = x + ( flow_to_x - flow_to_y ) * dt
        y = y + ( flow_to_y - flow_to_z ) * dt / d

        # update the flows
        flow_to_x, flow_to_y, flow_to_z = update_flows_forced(t, x, y, b, d, g, A, p, w)

        # append data corresponding to time i*dt to lists
        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
 
    return t_list, x_list, y_list

####################################################################
# compute endemic equilibrium of autonomous system

def get_EE_org(b, d, g):
    xEE = g / b
    yEE = d * (1 - g / b) / (d + g)

    return xEE, yEE
    
def get_EE_scaled(b, d, g):
    xEE = g / b
    yEE = (1 - g / b) / (d + g)

    return xEE, yEE
    