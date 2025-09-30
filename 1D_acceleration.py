# Code for automated computations comparing non-accelerated and accelerated algorithms

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

def weight1(argument, a_const = 60, b_const = 900):
    first_cutoff = a_const/(2*b_const)
    #second_cutoff = a_const/(b_const)
    output = np.maximum(0, a_const - b_const * np.maximum(argument, first_cutoff))
    return output

def weight2(argument, a_const = 50, b_const = 1000):
    #cutoff = a_const/(b_const)
    output = np.maximum(0, a_const - b_const * argument)
    return output

def weight3(argument, a_const = 1000, b_const = 0.1):
    output = a_const*(argument < b_const)
    return output





# Set the current parameters

current_lambda = 0.24
current_noise = 0.01
grid_size = 1000
error_parameter = 1e-6


# Choose current test functions and weights

current_weight = weight1
current_test_function = saw

# Number of iterations for averaging

number_for_averaging = 100



# Two options below


# Option 1: Upload an image (in grayscale)

#uploaded = files.upload()
#filename = list(uploaded.keys())[0]
#image = io.imread(filename, as_gray=True)



# Option 2: Load a predefined image

image = current_test_function(grid_size)

# Redefine image as float

image = img_as_float(image)





# Optional: plot the shape of the weight

#x = np.linspace(0, 0.2, 1000)
#y = current_weight(x)

#print(f"Current weight")

#plt.plot(x, y)
#plt.show()

#print()




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


print(f"Automated Chambolle-Pock: verification of acceleration, {number_for_averaging} iterations")
print(f"Value of lambda: {current_lambda}")
print(f"Noise level: {current_noise}")
print(f"Error parameter: {error_parameter}")
print(f"Grid size: {grid_size}")
print()



# Initialising the variables for averaging


technical_sum_ROF = 0
technical_sum_mROF1 = 0
technical_sum_mROF2 = 0
technical_sum_mROF3 = 0
technical_sum_hROF = 0
technical_sum_ROF_acc = 0
technical_sum_mROF1_acc = 0
technical_sum_mROF2_acc = 0
technical_sum_mROF3_acc = 0
technical_sum_hROF_acc = 0

l2distance_sum_ROF = 0
l2distance_sum_mROF1 = 0
l2distance_sum_mROF2 = 0
l2distance_sum_mROF3 = 0
l2distance_sum_hROF = 0
l2distance_sum_ROF_acc = 0
l2distance_sum_mROF1_acc = 0
l2distance_sum_mROF2_acc = 0
l2distance_sum_mROF3_acc = 0
l2distance_sum_hROF_acc = 0

l2distance_noisy_sum_ROF = 0
l2distance_noisy_sum_mROF1 = 0
l2distance_noisy_sum_mROF2 = 0
l2distance_noisy_sum_mROF3 = 0
l2distance_noisy_sum_hROF = 0
l2distance_noisy_sum_ROF_acc = 0
l2distance_noisy_sum_mROF1_acc = 0
l2distance_noisy_sum_mROF2_acc = 0
l2distance_noisy_sum_mROF3_acc = 0
l2distance_noisy_sum_hROF_acc = 0

l2distance_ROF_sum_ROF = 0
l2distance_ROF_sum_mROF1 = 0
l2distance_ROF_sum_mROF2 = 0
l2distance_ROF_sum_mROF3 = 0
l2distance_ROF_sum_hROF = 0
l2distance_ROF_acc_sum_ROF_acc = 0
l2distance_ROF_acc_sum_mROF1_acc = 0
l2distance_ROF_acc_sum_mROF2_acc = 0
l2distance_ROF_acc_sum_mROF3_acc = 0
l2distance_ROF_acc_sum_hROF_acc = 0

iterations_sum_ROF = 0
iterations_sum_mROF1 = 0
iterations_sum_mROF2 = 0
iterations_sum_mROF3 = 0
iterations_sum_hROF = 0
iterations_sum_ROF_acc = 0
iterations_sum_mROF1_acc = 0
iterations_sum_mROF2_acc = 0
iterations_sum_mROF3_acc = 0
iterations_sum_hROF_acc = 0

l2distance_acceleration_sum_ROF = 0
l2distance_acceleration_sum_mROF1 = 0
l2distance_acceleration_sum_mROF2 = 0
l2distance_acceleration_sum_mROF3 = 0
l2distance_acceleration_sum_hROF = 0

technical_sum_acceleration_ROF = 0
technical_sum_acceleration_mROF1 = 0
technical_sum_acceleration_mROF2 = 0
technical_sum_acceleration_mROF3 = 0
technical_sum_acceleration_hROF = 0


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




    # Step 1: Run the classical ROF

    denoised_v1, iter_v1 = chambolle_pock(noisy, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v1.copy()
    difference_2d_noisy = denoised_v1.copy()
    difference_2d_ROF = denoised_v1.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_ROF += grad_norm
    l2distance_sum_ROF += l2distance
    l2distance_noisy_sum_ROF += l2distance_noisy
    l2distance_ROF_sum_ROF += l2distance_ROF
    iterations_sum_ROF += iter_v1




    # Step 2: Run accelerated classical ROF

    denoised_v1_acc, iter_v1_acc = chambolle_pock_accelerated(noisy, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v1_acc.copy()
    difference_2d_noisy = denoised_v1_acc.copy()
    difference_2d_ROF = denoised_v1_acc.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1_acc

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_ROF_acc += grad_norm
    l2distance_sum_ROF_acc += l2distance
    l2distance_noisy_sum_ROF_acc += l2distance_noisy
    l2distance_ROF_acc_sum_ROF_acc += l2distance_ROF
    iterations_sum_ROF_acc += iter_v1_acc




    # Conclusion of Step 2: Comparing the accelerated and non-accelerated classical ROF

    difference_acc = denoised_v1_acc.copy()
    difference_acc -= denoised_v1

    l2distance_acc = norm2(norm2(difference_acc))
    l2distance_acceleration_sum_ROF += l2distance_acc

    grad_acc = compute_gradient_magnitude(difference_acc)
    grad_norm_acc = norm1(grad_acc)

    technical_sum_acceleration_ROF += grad_norm_acc


    # Step 3: Run the double-phase adaptive ROF, first weight

    # First generate the weight from the classical ROF

    mollified_v1 = mollify_function(denoised_v1, radius=2)
    grad_v1 = gradient(mollified_v1)
    grad_norm = norm2(grad_v1)

    current_weight = weight1
    a_weight = current_weight(argument=grad_norm)

    # Then apply the algorithm

    denoised_v2, iter_v2 = chambolle_pock_modified(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v2.copy()
    difference_2d_noisy = denoised_v2.copy()
    difference_2d_ROF = denoised_v2.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_mROF1 += grad_norm
    l2distance_sum_mROF1 += l2distance
    l2distance_noisy_sum_mROF1 += l2distance_noisy
    l2distance_ROF_sum_mROF1 += l2distance_ROF
    iterations_sum_mROF1 += iter_v2



    # Step 4: Run the accelerated double-phase adaptive ROF, first weight

    # First generate the weight from the accelerated classical ROF

    mollified_v1_acc = mollify_function(denoised_v1_acc, radius=2)
    grad_v1 = gradient(mollified_v1_acc)
    grad_norm = norm2(grad_v1)

    current_weight = weight1
    a_weight = current_weight(argument=grad_norm)

    # Then apply the algorithm

    denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v2_acc.copy()
    difference_2d_noisy = denoised_v2_acc.copy()
    difference_2d_ROF = denoised_v2_acc.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1_acc

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_mROF1_acc += grad_norm
    l2distance_sum_mROF1_acc += l2distance
    l2distance_noisy_sum_mROF1_acc += l2distance_noisy
    l2distance_ROF_acc_sum_mROF1_acc += l2distance_ROF
    iterations_sum_mROF1_acc += iter_v2_acc



    # Conclusion of Step 4: Comparing the accelerated and non-accelerated double-phase adaptive ROF, first weight

    difference_acc = denoised_v2_acc.copy()
    difference_acc -= denoised_v2

    l2distance_acc = norm2(norm2(difference_acc))
    l2distance_acceleration_sum_mROF1 += l2distance_acc

    grad_acc = compute_gradient_magnitude(difference_acc)
    grad_norm_acc = norm1(grad_acc)

    technical_sum_acceleration_mROF1 += grad_norm_acc



    # Step 5: Run the double-phase adaptive ROF, second weight

    # First generate the weight from the classical ROF

    mollified_v1 = mollify_function(denoised_v1, radius=2)
    grad_v1 = gradient(mollified_v1)
    grad_norm = norm2(grad_v1)

    current_weight = weight2
    a_weight = current_weight(argument=grad_norm)

    # Then apply the algorithm

    denoised_v2, iter_v2 = chambolle_pock_modified(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v2.copy()
    difference_2d_noisy = denoised_v2.copy()
    difference_2d_ROF = denoised_v2.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_mROF2 += grad_norm
    l2distance_sum_mROF2 += l2distance
    l2distance_noisy_sum_mROF2 += l2distance_noisy
    l2distance_ROF_sum_mROF2 += l2distance_ROF
    iterations_sum_mROF2 += iter_v2



    # Step 6: Run the accelerated double-phase adaptive ROF, second weight

    # First generate the weight from the accelerated classical ROF

    mollified_v1_acc = mollify_function(denoised_v1_acc, radius=2)
    grad_v1 = gradient(mollified_v1_acc)
    grad_norm = norm2(grad_v1)

    current_weight = weight2
    a_weight = current_weight(argument=grad_norm)

    # Then apply the algorithm

    denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v2_acc.copy()
    difference_2d_noisy = denoised_v2_acc.copy()
    difference_2d_ROF = denoised_v2_acc.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1_acc

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_mROF2_acc += grad_norm
    l2distance_sum_mROF2_acc += l2distance
    l2distance_noisy_sum_mROF2_acc += l2distance_noisy
    l2distance_ROF_acc_sum_mROF2_acc += l2distance_ROF
    iterations_sum_mROF2_acc += iter_v2_acc



    # Conclusion of Step 6: Comparing the accelerated and non-accelerated double-phase adaptive ROF, second weight

    difference_acc = denoised_v2_acc.copy()
    difference_acc -= denoised_v2

    l2distance_acc = norm2(norm2(difference_acc))
    l2distance_acceleration_sum_mROF2 += l2distance_acc

    grad_acc = compute_gradient_magnitude(difference_acc)
    grad_norm_acc = norm1(grad_acc)

    technical_sum_acceleration_mROF2 += grad_norm_acc



    # Step 7: Run the double-phase adaptive ROF, third weight

    # First generate the weight from the classical ROF

    mollified_v1 = mollify_function(denoised_v1, radius=2)
    grad_v1 = gradient(mollified_v1)
    grad_norm = norm2(grad_v1)

    current_weight = weight3
    a_weight = current_weight(argument=grad_norm)

    # Then apply the algorithm

    denoised_v2, iter_v2 = chambolle_pock_modified(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v2.copy()
    difference_2d_noisy = denoised_v2.copy()
    difference_2d_ROF = denoised_v2.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_mROF3 += grad_norm
    l2distance_sum_mROF3 += l2distance
    l2distance_noisy_sum_mROF3 += l2distance_noisy
    l2distance_ROF_sum_mROF3 += l2distance_ROF
    iterations_sum_mROF3 += iter_v2



    # Step 8: Run the accelerated double-phase adaptive ROF, third weight

    # First generate the weight from the accelerated classical ROF

    mollified_v1_acc = mollify_function(denoised_v1_acc, radius=2)
    grad_v1 = gradient(mollified_v1_acc)
    grad_norm = norm2(grad_v1)

    current_weight = weight3
    a_weight = current_weight(argument=grad_norm)

    # Then apply the algorithm

    denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v2_acc.copy()
    difference_2d_noisy = denoised_v2_acc.copy()
    difference_2d_ROF = denoised_v2_acc.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1_acc

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_mROF3_acc += grad_norm
    l2distance_sum_mROF3_acc += l2distance
    l2distance_noisy_sum_mROF3_acc += l2distance_noisy
    l2distance_ROF_acc_sum_mROF3_acc += l2distance_ROF
    iterations_sum_mROF3_acc += iter_v2_acc



    # Conclusion of Step 8: Comparing the accelerated and non-accelerated double-phase adaptive ROF, third weight

    difference_acc = denoised_v2_acc.copy()
    difference_acc -= denoised_v2

    l2distance_acc = norm2(norm2(difference_acc))
    l2distance_acceleration_sum_mROF3 += l2distance_acc

    grad_acc = compute_gradient_magnitude(difference_acc)
    grad_norm_acc = norm1(grad_acc)

    technical_sum_acceleration_mROF3 += grad_norm_acc



    # Step 9: Run the Huber ROF

    denoised_v3, iter_v3 = chambolle_pock_huber(noisy, alpha = 0.001, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v3.copy()
    difference_2d_noisy = denoised_v3.copy()
    difference_2d_ROF = denoised_v3.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_hROF += grad_norm
    l2distance_sum_hROF += l2distance
    l2distance_noisy_sum_hROF += l2distance_noisy
    l2distance_ROF_sum_hROF += l2distance_ROF
    iterations_sum_hROF += iter_v3




    # Step 10: Run the accelerated Huber ROF

    denoised_v3_acc, iter_v3_acc = chambolle_pock_huber_accelerated(noisy, alpha = 0.001, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

    # Compute the differences to the original and noisy images

    difference_2d = denoised_v3_acc.copy()
    difference_2d_noisy = denoised_v3_acc.copy()
    difference_2d_ROF = denoised_v3_acc.copy()
    difference_2d -= image
    difference_2d_noisy -= noisy
    difference_2d_ROF -= denoised_v1_acc

    # Compute the gradient of the difference to the original image

    grad = compute_gradient_magnitude(difference_2d)
    grad_norm = norm1(grad)

    # Compute the L2 norm of the difference to the original image

    l2distance = norm2(norm2(difference_2d))

    # Compute the L2 norm of the difference to the noisy image

    l2distance_noisy = norm2(norm2(difference_2d_noisy))

    # Compute the L2 norm of the difference to the classical ROF minimiser

    l2distance_ROF = norm2(norm2(difference_2d_ROF))

    # Add to the average: the gradient, the L2 distances, the number of iterations

    technical_sum_hROF_acc += grad_norm
    l2distance_sum_hROF_acc += l2distance
    l2distance_noisy_sum_hROF_acc += l2distance_noisy
    l2distance_ROF_acc_sum_hROF_acc += l2distance_ROF
    iterations_sum_hROF_acc += iter_v3_acc




    # Conclusion of Step 10: Comparing the accelerated and non-accelerated Huber ROF

    difference_acc = denoised_v3_acc.copy()
    difference_acc -= denoised_v3

    l2distance_acc = norm2(norm2(difference_acc))
    l2distance_acceleration_sum_hROF += l2distance_acc

    grad_acc = compute_gradient_magnitude(difference_acc)
    grad_norm_acc = norm1(grad_acc)

    technical_sum_acceleration_hROF += grad_norm_acc



# Take the averages



average_ROF = technical_sum_ROF / (number_for_averaging * grad_norm_image)
average_mROF1 = technical_sum_mROF1 / (number_for_averaging * grad_norm_image)
average_mROF2 = technical_sum_mROF2 / (number_for_averaging * grad_norm_image)
average_mROF3 = technical_sum_mROF3 / (number_for_averaging * grad_norm_image)
average_hROF = technical_sum_hROF / (number_for_averaging * grad_norm_image)
average_ROF_acc = technical_sum_ROF_acc / (number_for_averaging * grad_norm_image)
average_mROF1_acc = technical_sum_mROF1_acc / (number_for_averaging * grad_norm_image)
average_mROF2_acc = technical_sum_mROF2_acc / (number_for_averaging * grad_norm_image)
average_mROF3_acc = technical_sum_mROF3_acc / (number_for_averaging * grad_norm_image)
average_hROF_acc = technical_sum_hROF_acc / (number_for_averaging * grad_norm_image)

average_l2distance_ROF = l2distance_sum_ROF / (number_for_averaging * l2norm_image)
average_l2distance_mROF1 = l2distance_sum_mROF1 / (number_for_averaging * l2norm_image)
average_l2distance_mROF2 = l2distance_sum_mROF2 / (number_for_averaging * l2norm_image)
average_l2distance_mROF3 = l2distance_sum_mROF3 / (number_for_averaging * l2norm_image)
average_l2distance_hROF = l2distance_sum_hROF / (number_for_averaging * l2norm_image)
average_l2distance_ROF_acc = l2distance_sum_ROF_acc / (number_for_averaging * l2norm_image)
average_l2distance_mROF1_acc = l2distance_sum_mROF1_acc / (number_for_averaging * l2norm_image)
average_l2distance_mROF2_acc = l2distance_sum_mROF2_acc / (number_for_averaging * l2norm_image)
average_l2distance_mROF3_acc = l2distance_sum_mROF3_acc / (number_for_averaging * l2norm_image)
average_l2distance_hROF_acc = l2distance_sum_hROF_acc / (number_for_averaging * l2norm_image)

average_l2distance_noisy_ROF = l2distance_noisy_sum_ROF / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF1 = l2distance_noisy_sum_mROF1 / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF2 = l2distance_noisy_sum_mROF2 / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF3 = l2distance_noisy_sum_mROF3 / (number_for_averaging * l2norm_image)
average_l2distance_noisy_hROF = l2distance_noisy_sum_hROF / (number_for_averaging * l2norm_image)
average_l2distance_noisy_ROF_acc = l2distance_noisy_sum_ROF_acc / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF1_acc = l2distance_noisy_sum_mROF1_acc / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF2_acc = l2distance_noisy_sum_mROF2_acc / (number_for_averaging * l2norm_image)
average_l2distance_noisy_mROF3_acc = l2distance_noisy_sum_mROF3_acc / (number_for_averaging * l2norm_image)
average_l2distance_noisy_hROF_acc = l2distance_noisy_sum_hROF_acc / (number_for_averaging * l2norm_image)

average_l2distance_ROF_ROF = l2distance_ROF_sum_ROF / (number_for_averaging * l2norm_image)
average_l2distance_ROF_mROF1 = l2distance_ROF_sum_mROF1 / (number_for_averaging * l2norm_image)
average_l2distance_ROF_mROF2 = l2distance_ROF_sum_mROF2 / (number_for_averaging * l2norm_image)
average_l2distance_ROF_mROF3 = l2distance_ROF_sum_mROF3 / (number_for_averaging * l2norm_image)
average_l2distance_ROF_hROF = l2distance_ROF_sum_hROF / (number_for_averaging * l2norm_image)
average_l2distance_ROF_acc_ROF_acc = l2distance_ROF_acc_sum_ROF_acc / (number_for_averaging * l2norm_image)
average_l2distance_ROF_acc_mROF1_acc = l2distance_ROF_acc_sum_mROF1_acc / (number_for_averaging * l2norm_image)
average_l2distance_ROF_acc_mROF2_acc = l2distance_ROF_acc_sum_mROF2_acc / (number_for_averaging * l2norm_image)
average_l2distance_ROF_acc_mROF3_acc = l2distance_ROF_acc_sum_mROF3_acc / (number_for_averaging * l2norm_image)
average_l2distance_ROF_acc_hROF_acc = l2distance_ROF_acc_sum_hROF_acc / (number_for_averaging * l2norm_image)

average_iterations_ROF = round(iterations_sum_ROF / number_for_averaging)
average_iterations_mROF1 = round(iterations_sum_mROF1 / number_for_averaging)
average_iterations_mROF2 = round(iterations_sum_mROF2 / number_for_averaging)
average_iterations_mROF3 = round(iterations_sum_mROF3 / number_for_averaging)
average_iterations_hROF = round(iterations_sum_hROF / number_for_averaging)
average_iterations_ROF_acc = round(iterations_sum_ROF_acc / number_for_averaging)
average_iterations_mROF1_acc = round(iterations_sum_mROF1_acc / number_for_averaging)
average_iterations_mROF2_acc = round(iterations_sum_mROF2_acc / number_for_averaging)
average_iterations_mROF3_acc = round(iterations_sum_mROF3_acc / number_for_averaging)
average_iterations_hROF_acc = round(iterations_sum_hROF_acc / number_for_averaging)

average_l2distance_acceleration_ROF = l2distance_acceleration_sum_ROF / (number_for_averaging * l2norm_image)
average_l2distance_acceleration_mROF1 = l2distance_acceleration_sum_mROF1 / (number_for_averaging * l2norm_image)
average_l2distance_acceleration_mROF2 = l2distance_acceleration_sum_mROF2 / (number_for_averaging * l2norm_image)
average_l2distance_acceleration_mROF3 = l2distance_acceleration_sum_mROF3 / (number_for_averaging * l2norm_image)
average_l2distance_acceleration_hROF = l2distance_acceleration_sum_hROF / (number_for_averaging * l2norm_image)

average_acceleration_ROF = technical_sum_acceleration_ROF / (number_for_averaging * grad_norm_image)
average_acceleration_mROF1 = technical_sum_acceleration_mROF1 / (number_for_averaging * grad_norm_image)
average_acceleration_mROF2 = technical_sum_acceleration_mROF2 / (number_for_averaging * grad_norm_image)
average_acceleration_mROF3 = technical_sum_acceleration_mROF3 / (number_for_averaging * grad_norm_image)
average_acceleration_hROF = technical_sum_acceleration_hROF / (number_for_averaging * grad_norm_image)


# Print the averages: first classical ROF, then modified ROF, then Huber ROF

print()

print(f"Classical ROF - average TV error: {average_ROF}")

print(f"Classical ROF - average l2 distance (image): {average_l2distance_ROF}")

print(f"Classical ROF - average l2 distance (noisy): {average_l2distance_noisy_ROF}")

print(f"Classical ROF - average l2 distance (ROF): {average_l2distance_ROF_ROF}")

print(f"Classical ROF - number of iterations: {average_iterations_ROF}")

print()

print(f"Classical ROF (acc) - average TV error: {average_ROF_acc}")

print(f"Classical ROF (acc) - average l2 distance (image): {average_l2distance_ROF_acc}")

print(f"Classical ROF (acc) - average l2 distance (noisy): {average_l2distance_noisy_ROF_acc}")

print(f"Classical ROF (acc) - average l2 distance (ROF): {average_l2distance_ROF_acc_ROF_acc}")

print(f"Classical ROF (acc) - number of iterations: {average_iterations_ROF_acc}")

print()

print(f"Classical ROF - TV distance between non-acc and acc: {average_acceleration_ROF}")

print(f"Classical ROF - L2 distance between non-acc and acc: {average_l2distance_acceleration_ROF}")

print()

print(f"Modified ROF, weight 1 - average TV error: {average_mROF1}")

print(f"Modified ROF, weight 1 - average l2 distance (image): {average_l2distance_mROF1}")

print(f"Modified ROF, weight 1 - average l2 distance (noisy): {average_l2distance_noisy_mROF1}")

print(f"Modified ROF, weight 1 - average l2 distance (ROF): {average_l2distance_ROF_mROF1}")

print(f"Modified ROF, weight 1 - number of iterations: {average_iterations_mROF1}")

print()

print(f"Modified ROF, weight 1 (acc) - average TV error: {average_mROF1_acc}")

print(f"Modified ROF, weight 1 (acc) - average l2 distance (image): {average_l2distance_mROF1_acc}")

print(f"Modified ROF, weight 1 (acc) - average l2 distance (noisy): {average_l2distance_noisy_mROF1_acc}")

print(f"Modified ROF, weight 1 (acc) - average l2 distance (ROF): {average_l2distance_ROF_acc_mROF1_acc}")

print(f"Modified ROF, weight 1 (acc) - number of iterations: {average_iterations_mROF1_acc}")

print()

print(f"Modified ROF, weight 1 - TV distance between non-acc and acc: {average_acceleration_mROF1}")

print(f"Modified ROF, weight 1 - L2 distance between non-acc and acc: {average_l2distance_acceleration_mROF1}")

print()

print(f"Modified ROF, weight 2 - average TV error: {average_mROF2}")

print(f"Modified ROF, weight 2 - average l2 distance (image): {average_l2distance_mROF2}")

print(f"Modified ROF, weight 2 - average l2 distance (noisy): {average_l2distance_noisy_mROF2}")

print(f"Modified ROF, weight 2 - average l2 distance (ROF): {average_l2distance_ROF_mROF2}")

print(f"Modified ROF, weight 2 - number of iterations: {average_iterations_mROF2}")

print()

print(f"Modified ROF, weight 2 (acc) - average TV error: {average_mROF2_acc}")

print(f"Modified ROF, weight 2 (acc) - average l2 distance (image): {average_l2distance_mROF2_acc}")

print(f"Modified ROF, weight 2 (acc) - average l2 distance (noisy): {average_l2distance_noisy_mROF2_acc}")

print(f"Modified ROF, weight 2 (acc) - average l2 distance (ROF): {average_l2distance_ROF_acc_mROF2_acc}")

print(f"Modified ROF, weight 2 (acc) - number of iterations: {average_iterations_mROF2_acc}")

print()

print(f"Modified ROF, weight 2 - TV distance between non-acc and acc: {average_acceleration_mROF2}")

print(f"Modified ROF, weight 2 - L2 distance between non-acc and acc: {average_l2distance_acceleration_mROF2}")

print()

print(f"Modified ROF, weight 3 - average TV error: {average_mROF3}")

print(f"Modified ROF, weight 3 - average l2 distance (image): {average_l2distance_mROF3}")

print(f"Modified ROF, weight 3 - average l2 distance (noisy): {average_l2distance_noisy_mROF3}")

print(f"Modified ROF, weight 3 - average l2 distance (ROF): {average_l2distance_ROF_mROF3}")

print(f"Modified ROF, weight 3 - number of iterations: {average_iterations_mROF3}")

print()

print(f"Modified ROF, weight 3 (acc) - average TV error: {average_mROF3_acc}")

print(f"Modified ROF, weight 3 (acc) - average l2 distance (image): {average_l2distance_mROF3_acc}")

print(f"Modified ROF, weight 3 (acc) - average l2 distance (noisy): {average_l2distance_noisy_mROF3_acc}")

print(f"Modified ROF, weight 3 (acc) - average l2 distance (ROF): {average_l2distance_ROF_acc_mROF3_acc}")

print(f"Modified ROF, weight 3 (acc) - number of iterations: {average_iterations_mROF3_acc}")

print()

print(f"Modified ROF, weight 3 - TV distance between non-acc and acc: {average_acceleration_mROF3}")

print(f"Modified ROF, weight 3 - L2 distance between non-acc and acc: {average_l2distance_acceleration_mROF3}")

print()

print(f"Huber ROF - average TV error: {average_hROF}")

print(f"Huber ROF - average l2 distance (image): {average_l2distance_hROF}")

print(f"Huber ROF - average l2 distance (noisy): {average_l2distance_noisy_hROF}")

print(f"Huber ROF - average l2 distance (ROF): {average_l2distance_ROF_hROF}")

print(f"Huber ROF - number of iterations: {average_iterations_hROF}")

print()

print(f"Huber ROF (acc) - average TV error: {average_hROF_acc}")

print(f"Huber ROF (acc) - average l2 distance (image): {average_l2distance_hROF_acc}")

print(f"Huber ROF (acc) - average l2 distance (noisy): {average_l2distance_noisy_hROF_acc}")

print(f"Huber ROF (acc) - average l2 distance (ROF): {average_l2distance_ROF_acc_hROF_acc}")

print(f"Huber ROF (acc) - number of iterations: {average_iterations_hROF_acc}")

print()

print(f"Huber ROF - TV distance between non-acc and acc: {average_acceleration_hROF}")

print(f"Huber ROF - L2 distance between non-acc and acc: {average_l2distance_acceleration_hROF}")