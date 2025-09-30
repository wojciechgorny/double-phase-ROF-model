# Code for automated computations for different parameters within weight 3


import time
from pickle import TUPLE3
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io, util, color
from numba import njit
from scipy.ndimage import convolve
from google.colab import files
import io as python_io
from numpy import linalg as LA


# Definition of test functions with parameters

def zero_function(grid_size):
    auxiliary = np.zeros((1, grid_size))
    output = auxiliary.reshape(1, grid_size)
    return output

def linear_function(grid_size):
    auxiliary = np.linspace(0, 1, grid_size)
    output = auxiliary.reshape(1, grid_size)
    return output

def characteristic_function_interval(grid_size):
    half_grid_size = round(grid_size/2)
    auxiliary1 = np.zeros((1, half_grid_size))
    auxiliary2 = np.ones((1, half_grid_size))
    auxiliary = np.hstack((auxiliary1, auxiliary2))
    output = auxiliary.reshape(1, grid_size)
    return output

def step_function(grid_size):
    half_grid_size = round(grid_size/2)
    auxiliary1 = np.ones((1, half_grid_size))
    auxiliary1 *= 0.2
    auxiliary2 = np.ones((1, half_grid_size))
    auxiliary2 *= 0.8
    auxiliary = np.hstack((auxiliary1, auxiliary2))
    output = auxiliary.reshape(1, grid_size)
    return output

def two_jumps(grid_size):
    onefourth_grid_size = round(grid_size/4)
    auxiliary1 = np.ones((1, onefourth_grid_size))
    auxiliary1 *= 0
    auxiliary2 = np.ones((1, 2*onefourth_grid_size))
    auxiliary2 *= 1
    auxiliary3 = np.ones((1, onefourth_grid_size))
    auxiliary3 *= 0
    auxiliary = np.hstack((auxiliary1, auxiliary2, auxiliary3))
    output = auxiliary.reshape(1, grid_size)
    return output


def saw(grid_size):
    onefourth_grid_size = round(grid_size/4)
    auxiliary1 = np.linspace(0, 1/4, onefourth_grid_size)
    auxiliary2 = np.linspace(0, 1/2, onefourth_grid_size)
    auxiliary3 = np.linspace(0, 3/4, onefourth_grid_size)
    auxiliary4 = np.linspace(0, 1, onefourth_grid_size)
    auxiliary = np.hstack((auxiliary1, auxiliary2, auxiliary3,auxiliary4))
    output = auxiliary.reshape(1, grid_size)
    return output




# Definition of different weights with parameters

def weight1(argument, a_const = 1000, b_const = 10000):
    first_cutoff = a_const/(2*b_const)
    #second_cutoff = a_const/(b_const)
    output = np.maximum(0, a_const - b_const * np.maximum(argument, first_cutoff))
    return output

def weight2(argument, a_const = 1000, b_const = 10000):
    #cutoff = a_const/(b_const)
    output = np.maximum(0, a_const - b_const * argument)
    return output

def weight3(argument, height = 1000, radius = 0.03):
    output = height*(argument < radius)
    return output


weight_labels = {
    weight1: "1",
    weight2: "2",
    weight3: "3"}

test_function_labels = {
    saw: "saw",
    two_jumps: "jump",
    linear_function: "lin"}


# Set the current parameters

current_lambda = 0.24
current_noise = 0.01
grid_size = 1000
error_parameter = 1e-4

# Choose current test functions and weights

current_weight = weight3
current_test_function = saw

# Number of iterations for averaging

number_for_averaging = 1

# Spacing of parameters and number of tested values

number_for_comparison = 41
number_for_height = 25
number_for_radius = 0.05



# Two options below


# Option 1: Upload an image (in grayscale)

#uploaded = files.upload()
#filename = list(uploaded.keys())[0]
#image = io.imread(filename, as_gray=True)



# Option 2: Load a predefined image

image = current_test_function(grid_size)
filename = test_function_labels[current_test_function]

# Redefine image as float

image = img_as_float(image)





# Definitions of auxiliary functions

def compute_gradient_magnitude(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return np.sqrt(gx**2 + gy**2)

@njit
def gradient(u):
    grad_u = np.zeros(u.shape + (2,))
    grad_u[:-1,:,0] = u[1:,:] - u[:-1,:]
    grad_u[:,:-1,1] = u[:,1:] - u[:,:-1]
    return grad_u

@njit
def divergence(p):
    div = np.zeros(p.shape[:2])
    div[:-1,:] += p[:-1,:,0]
    div[1:,:]  -= p[:-1,:,0]
    div[:,:-1] += p[:,:-1,1]
    div[:,1:]  -= p[:,:-1,1]
    return div

@njit
def norm2(p):
    return np.sqrt(np.sum(p**2, axis=-1) + 1e-12)

@njit
def norm1(p):
    return np.sum(np.sqrt(p**2), axis=-1) + 1e-12


# Average on ball kernel for mollification

def ball_kernel(radius):
    L = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(float)
    kernel /= np.sum(kernel)
    return kernel

def mollify_function(func, radius=2):
    kernel = ball_kernel(radius)
    padded_func = np.pad(func, pad_width=radius, mode='reflect')
    mollified = convolve(padded_func, kernel, mode='reflect')
    return mollified[radius:-radius, radius:-radius]


# Custom Resolvent for modified ROF

def custom_resolvent(p_tilde, a, sigma):
    norm_p = norm2(p_tilde)
    p = np.zeros_like(p_tilde)

    mask_zero = (a == 0)
    mask_small = (a > 0) & (norm_p <= 1)
    mask_large = (a > 0) & (norm_p > 1)

    p[mask_zero] = p_tilde[mask_zero] / np.maximum(1, norm_p[mask_zero])[..., None]
    p[mask_small] = p_tilde[mask_small]
    s = norm_p[mask_large]
    a_vals = a[mask_large]
    factor = (a_vals * s + sigma) / (a_vals * s + sigma * s)
    p[mask_large] = (factor[..., None]) * p_tilde[mask_large]

    return p






# Original ROF
@njit
def chambolle_pock(image, tau, lam=0.2, max_iter=10000, tol=1.0e-8):

    # Initialise the variables

    m, n = image.shape                      # Take the dimensions of the image
    p = np.zeros((m, n, 2))                 # The vector field
    g = np.zeros((m, n, 2))                 # The gradient
    x = image                               # Input image
    x_bar = x.copy()                        # Again the input image, because Chambolle-Pock tracks two copies of it

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2 * tau * L**2)

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * div_p + (tau / lam) * image  )/(1 + (tau / lam))

        # Update x_bar

        x_bar = (2*x - x_prev)

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            break

    return x, i+1


# Original ROF, accelerated
@njit
def chambolle_pock_accelerated(image, tau, lam=0.2, max_iter=10000, tol=1.0e-8):

    # Initialise the variables

    m, n = image.shape                      # Take the dimensions of the image
    p = np.zeros((m, n, 2))                 # The vector field
    g = np.zeros((m, n, 2))                 # The gradient
    x = image                               # Input image
    x_bar = x.copy()                        # Again the input image, because Chambolle-Pock tracks two copies of it

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2 * tau * L**2)
    theta = 1

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * div_p + (tau / lam) * image  )/(1 + (tau / lam))

        # Update theta, tau, sigma and x_bar

        theta = 1/(np.sqrt(1 + 2*tau/lam))
        tau *= theta
        sigma /= theta
        
        x_bar = (1+theta)*x - theta*x_prev

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            break

    return x, i+1


# Modified ROF
def chambolle_pock_modified(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-8):

    # Initialise the variables

    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2*tau * L**2)

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = p + sigma * g
        p = custom_resolvent(p_new, a_weight, sigma)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))

        # Update x_bar

        x_bar = 2 * x - x_prev

        # Check convergence

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x, i+1




# Modified ROF, accelerated
def chambolle_pock_modified_accelerated(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-8):

    # Initialise the variables

    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2*tau * L**2)
    theta = 1

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = p + sigma * g
        p = custom_resolvent(p_new, a_weight, sigma)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))

        # Update theta, tau, sigma and x_bar

        theta = 1/(np.sqrt(1 + 2*tau/lam))
        tau *= theta
        sigma /= theta
        
        x_bar = (1+theta)*x - theta*x_prev

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x, i+1



# Huber ROF
def chambolle_pock_huber(image, alpha, tau, lam=0.2, max_iter=10000, tol=1e-8):

    # Initialise the variables

    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2*tau * L**2)

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = (1 + sigma * alpha)*np.maximum(1.0,(1/(1 + sigma * alpha))*norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))

        # Update x_bar

        x_bar = 2 * x - x_prev

        # Check convergence

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x, i+1




# Huber ROF, accelerated
def chambolle_pock_huber_accelerated(image, alpha, tau, lam=0.2, max_iter=10000, tol=1e-8):

    # Initialise the variables

    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2*tau * L**2)
    theta = 1

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = (1 + sigma * alpha)*np.maximum(1.0,(1/(1 + sigma * alpha))*norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))

        # Update theta, tau, sigma and x_bar

        theta = 1/(np.sqrt(1 + 2*tau/lam))
        tau *= theta
        sigma /= theta
        
        x_bar = (1+theta)*x - theta*x_prev

        # Check convergence

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x, i+1






# Loop for automating the computations



print(f"Automated Chambolle-Pock: comparing weights, {number_for_averaging} iterations")
print(f"Value of lambda: {current_lambda}")
print(f"Noise level: {current_noise}")
print(f"Error parameter: {error_parameter}")
print(f"Grid size: {grid_size}")
print()



# Initialising the variables for averaging

technical_sum_ROF_acc = 0
l2distance_sum_ROF_acc = 0
l2distance_noisy_sum_ROF_acc = 0
iterations_sum_ROF_acc = 0
technical_sum_mROF = np.zeros(number_for_comparison)
l2distance_sum_mROF = np.zeros(number_for_comparison)
l2distance_noisy_sum_mROF = np.zeros(number_for_comparison)
iterations_sum_mROF = np.zeros(number_for_comparison)

for i in range(1, number_for_comparison):
   technical_sum_mROF[i] = 0
   l2distance_sum_mROF[i] = 0
   l2distance_noisy_sum_mROF[i] = 0
   iterations_sum_mROF[i] = 0


# Total variation of the initial datum for control

grad = compute_gradient_magnitude(image)
grad_norm_image = norm1(grad)


print(f"TV for initial datum: {grad_norm_image}")
print()


# For control: print the L2 norm of the image


#l2norm = norm2(norm2(image))
#print(f"L2 norm of the image: {l2norm}")

l2norm_image = LA.norm(image)
print(f"L2 norm of the image: {l2norm_image}")



# Start the loop for averaging



for i in range(number_for_averaging):

    # Step 0: Create a new picture by adding Gaussian noise
    noisy = util.random_noise(image, mode='gaussian', var=current_noise)




    # Step 1: Run accelerated classical ROF

    denoised_v1_acc, iter_v1_acc = chambolle_pock_accelerated(noisy, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v1_acc.copy()
    difference_2d_noisy = denoised_v1_acc.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_ROF_acc += grad_norm
    l2distance_sum_ROF_acc += l2distance
    l2distance_noisy_sum_ROF_acc += l2distance_noisy
    iterations_sum_ROF_acc += iter_v1_acc


    for i in range(1, number_for_comparison):

        # Step i+1: Run the accelerated double-phase adaptive ROF

        # First generate the weight from the accelerated classical ROF

        mollified_v1_acc = mollify_function(denoised_v1_acc, radius=2)
        grad_v1 = gradient(mollified_v1_acc)
        grad_norm = norm2(grad_v1)

        # Two options for the weight depending whether we test for the height or the radius

        # If we test for heights

        a_weight = current_weight(argument=grad_norm, height = number_for_height * i, radius = number_for_radius)

        # If we test for radii

        #a_weight = current_weight(argument=grad_norm, height = number_for_height, radius = number_for_radius * i)

        # From now on the same


        # Then apply the algorithm

        denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

        # Compute the differences to the original and noisy images

        difference_2d = denoised_v2_acc.copy()
        difference_2d_noisy = denoised_v2_acc.copy()
        difference_2d -= image
        difference_2d_noisy -= noisy

        # Compute the gradient of the difference to the original image

        grad = compute_gradient_magnitude(difference_2d)
        grad_norm = norm1(grad)

        # Compute the L2 norm of the difference to the original image

        l2distance = norm2(norm2(difference_2d))

        # Compute the L2 norm of the difference to the noisy image

        l2distance_noisy = norm2(norm2(difference_2d_noisy))

        # Add to the average: the gradient, the L2 distances, the number of iterations

        technical_sum_mROF[i] += grad_norm
        l2distance_sum_mROF[i] += l2distance
        l2distance_noisy_sum_mROF[i] += l2distance_noisy
        iterations_sum_mROF[i] += iter_v2_acc


# Take the averages



average_ROF_acc = technical_sum_ROF_acc / (number_for_averaging * grad_norm_image)
average_l2distance_ROF_acc = l2distance_sum_ROF_acc / (number_for_averaging * l2norm_image)
average_l2distance_noisy_ROF_acc = l2distance_noisy_sum_ROF_acc / (number_for_averaging * l2norm_image)
average_iterations_ROF_acc = round(iterations_sum_ROF_acc / number_for_averaging)

average_mROF = np.zeros(number_for_comparison)
average_l2distance_mROF = np.zeros(number_for_comparison)
average_l2distance_noisy_mROF = np.zeros(number_for_comparison)
average_iterations_mROF = np.zeros(number_for_comparison)

average_mROF = technical_sum_mROF / (number_for_averaging * grad_norm_image)
average_l2distance_mROF = l2distance_sum_mROF / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF = l2distance_noisy_sum_mROF / (number_for_averaging * l2norm_image)

for i in range(1, number_for_comparison):
    average_iterations_mROF[i] = round(iterations_sum_mROF[i] / number_for_averaging)



# Print the averages: first classical ROF, then modified ROF for different weights

print()

print(f"Classical ROF (acc) - average TV error: {average_ROF_acc}")

print(f"Classical ROF (acc) - average l2 distance (image): {average_l2distance_ROF_acc}")

print(f"Classical ROF (acc) - average l2 distance (noisy): {average_l2distance_noisy_ROF_acc}")

print(f"Classical ROF (acc) - number of iterations {average_iterations_ROF_acc}")

for i in range(1, number_for_comparison):

    print()

    # Two options for the results depending whether we test for the height or the radius

    # If we test for heights

    print(f"Modified ROF, h = {number_for_height * i}, r = {number_for_radius} (acc) - average TV error: {average_mROF[i]}")

    print(f"Modified ROF, h = {number_for_height * i}, r = {number_for_radius} (acc) - average l2 distance (image): {average_l2distance_mROF[i]}")

    print(f"Modified ROF, h = {number_for_height * i}, r = {number_for_radius} (acc) - average l2 distance (noisy): {average_l2distance_noisy_mROF[i]}")

    print(f"Modified ROF, h = {number_for_height * i}, r = {number_for_radius} (acc) - number of iterations: {average_iterations_mROF[i]}")

    # If we test for radii

    #print(f"Modified ROF, h = {number_for_height}, r = {number_for_radius * i} (acc) - average TV error: {average_mROF[i]}")

    #print(f"Modified ROF, h = {number_for_height}, r = {number_for_radius * i} (acc) - average l2 distance (image): {average_l2distance_mROF[i]}")

    #print(f"Modified ROF, h = {number_for_height}, r = {number_for_radius * i} (acc) - average l2 distance (noisy): {average_l2distance_noisy_mROF[i]}")

    #print(f"Modified ROF, h = {number_for_height}, r = {number_for_radius * i} (acc) - number of iterations: {average_iterations_mROF[i]}")

    # From now on the same



weight_labels = {
    weight1: "1",
    weight2: "2",
    weight3: "3"}

test_function_labels = {
    saw: "saw",
    two_jumps: "jump",
    linear_function: "lin"
    }



average_l2distance_ROF_acc_plot = np.zeros(number_for_comparison)

for i in range(1, number_for_comparison):
    average_l2distance_ROF_acc_plot[i] = average_l2distance_ROF_acc



average_ROF_acc_plot = np.zeros(number_for_comparison)

for i in range(1, number_for_comparison):
    average_ROF_acc_plot[i] = average_ROF_acc

#plt.plot(average_ROF_acc_plot)
#plt.plot(average_mROF)
#plt.show()



# Two options for the plot depending whether we test for the height or the radius

# If we test for heights

weight_values = [number_for_height * i for i in range(1, number_for_comparison)]
plt.plot(weight_values, average_ROF_acc_plot[1:], label="ROF")
plt.plot(weight_values, average_mROF[1:], label=f"dpROF, r = {number_for_radius}")
plt.xlabel("Values of h")

# If we test for radii

#weight_values = [number_for_radius * i for i in range(1, number_for_comparison)]
#plt.plot(weight_values, average_ROF_acc_plot[1:], label="ROF")
#plt.plot(weight_values, average_mROF[1:], label=f"dpROF, h = {number_for_height}")
#plt.xlabel("Values of r")

# From now on the same

plt.ylabel("Average TV Error")
plt.title(f"Weight {weight_labels[current_weight]}, σ = {current_noise}, f = {filename}")
plt.grid(True)
plt.legend()

plt.xlim(left=0)
plt.ylim(bottom=0,top=1.15)

plt.show()



plt.plot(weight_values, average_l2distance_ROF_acc_plot[1:], label="ROF")
plt.plot(weight_values, average_l2distance_mROF[1:], label=f"dpROF, r = {number_for_radius}")
plt.xlabel("Values of h")

plt.ylabel("Average L2 Error")
plt.title(f"Weight {weight_labels[current_weight]}, σ = {current_noise}, f = {filename}")
plt.grid(True)
plt.legend()

plt.xlim(left=0)
plt.ylim(bottom=0,top=0.1)

plt.show()