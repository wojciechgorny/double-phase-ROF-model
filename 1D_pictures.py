# Code for generating 1D pictures comparing the results of the adaptive double-phase ROF to other models

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



# Definition of several test functions with parameters

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

def barcode(grid_size):
    onehundredth_grid_size = round(grid_size/100)
    auxiliary1 = np.ones((1, 2*onehundredth_grid_size))
    auxiliary1 *= 1
    auxiliary2 = np.ones((1, 7*onehundredth_grid_size))
    auxiliary2 *= 0
    auxiliary3 = np.ones((1, 4*onehundredth_grid_size))
    auxiliary3 *= 1
    auxiliary4 = np.ones((1, 5*onehundredth_grid_size))
    auxiliary4 *= 0
    auxiliary5 = np.ones((1, 6*onehundredth_grid_size))
    auxiliary5 *= 1
    auxiliary6 = np.ones((1, 2*onehundredth_grid_size))
    auxiliary6 *= 0
    auxiliary7 = np.ones((1, 14*onehundredth_grid_size))
    auxiliary7 *= 1
    auxiliary8 = np.ones((1, 5*onehundredth_grid_size))
    auxiliary8 *= 0
    auxiliary9 = np.ones((1, 10*onehundredth_grid_size))
    auxiliary9 *= 1
    auxiliary10 = np.ones((1, 7*onehundredth_grid_size))
    auxiliary10 *= 0
    auxiliary11 = np.ones((1, 10*onehundredth_grid_size))
    auxiliary11 *= 1
    auxiliary12 = np.ones((1, 8*onehundredth_grid_size))
    auxiliary12 *= 0
    auxiliary13 = np.ones((1, 3*onehundredth_grid_size))
    auxiliary13 *= 1
    auxiliary14 = np.ones((1, 9*onehundredth_grid_size))
    auxiliary14 *= 0
    auxiliary15 = np.ones((1, 8*onehundredth_grid_size))
    auxiliary15 *= 1
    auxiliary = np.hstack((auxiliary1, auxiliary2, auxiliary3,auxiliary4,auxiliary5,auxiliary6,auxiliary7,auxiliary8,auxiliary9,auxiliary10,auxiliary11,auxiliary12,auxiliary13,auxiliary14,auxiliary15))
    output = auxiliary.reshape(1, grid_size)
    return output


# Definition of different weights with parameters

def weight1(argument, a_const = 40, b_const = 800):
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
    barcode: "barcode",
    linear_function: "lin"}


# Set the current parameters

current_lambda = 0.24
current_noise = 0.01
grid_size = 1000
error_parameter = 1e-6

# Choose current test functions and weights

current_weight = weight1
current_test_function = saw

# Choice of parameters within the weight (1 or 2)

number_for_scaling = 40
number_for_ratio = 30



# Two options below


# Option 1: Upload an image (in grayscale)

uploaded = files.upload()
filename = list(uploaded.keys())[0]
image = io.imread(filename, as_gray=True)



# Option 2: Load a predefined image

#image = current_test_function(grid_size)
#filename = test_function_labels[current_test_function]

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






# Total variation of the initial datum for control

grad = compute_gradient_magnitude(image)
grad_norm = norm1(grad)


print(f"TV for initial datum: {grad_norm}")
print()


# For control: print the L2 norm of the image

l2norm_image = LA.norm(image)
print(f"L2 norm of the image: {l2norm_image}")

# Different equivalent way of computing L2 norm

#l2norm = norm2(norm2(image))
#print(f"L2 norm of the image: {l2norm}")



# Initialise: create a noisy image

# Add Gaussian noise
noisy = util.random_noise(image, mode='gaussian', var=current_noise)

# Convert the original and noisy images to 1D

image_1d = image.flatten()
noisy_1d = noisy.flatten()

# Define two figures with the original and noisy images

figure1 = image_1d
figure5 = noisy_1d

print()




# Step 1: Apply the original ROF (accelerated)

start = time.time()
denoised_v1_acc, iter_v1_acc = chambolle_pock_accelerated(noisy, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)
t1 = time.time() - start


# Show the result of original ROF (accelerated) and compute the difference to the original image

denoised_v1_acc_1d = denoised_v1_acc.flatten()
figure2 = denoised_v1_acc_1d.copy()

image_1d = image.flatten()

difference_2d = denoised_v1_acc - image

grad = compute_gradient_magnitude(difference_2d)
grad_norm = norm1(grad)

print(f"Classical ROF, accelerated - difference to image: {grad_norm}")



# Step 2: Apply the Huber ROF (accelerated)

start = time.time()
denoised_v3_acc, iter_v3_acc = chambolle_pock_huber_accelerated(noisy, alpha = 0.002, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)
t2 = time.time() - start

denoised_v3_acc_1d = denoised_v3_acc.flatten()
figure3 = denoised_v3_acc_1d.copy()

difference_1d = denoised_v3_acc_1d
difference_1d -= image_1d

difference_2d = denoised_v3_acc
difference_2d -= image

grad = compute_gradient_magnitude(difference_2d)
grad_norm = norm1(grad)

print(f"Huber ROF (acc), alpha = 0.002: {grad_norm}")



# Step 3: Apply Huber-ROF (accelerated) for a larger alpha

start = time.time()
denoised_v3_acc, iter_v3_acc = chambolle_pock_huber_accelerated(noisy, alpha = 0.01, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)
t2 = time.time() - start

denoised_v3_acc_1d = denoised_v3_acc.flatten()
figure4 = denoised_v3_acc_1d.copy()

difference_1d = denoised_v3_acc_1d
difference_1d -= image_1d

difference_2d = denoised_v3_acc
difference_2d -= image

grad = compute_gradient_magnitude(difference_2d)
grad_norm = norm1(grad)

print(f"Huber ROF (acc), alpha = 0.1: {grad_norm}")




# Step 4: Apply the modified ROF

# Compute the weight from the result of the original ROF

defined_radius = 2
mollified_rof = mollify_function(denoised_v1_acc, radius=defined_radius)
grad_rof = gradient(mollified_rof)
grad_norm_rof = norm2(grad_rof)

a_const = number_for_scaling
b_const = number_for_scaling * number_for_ratio
a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm_rof, 1/(2*number_for_ratio)))



# Optional: Show the weight for control

#a_weight_1d = a_weight.flatten()
#plt.plot(a_weight_1d)
#plt.show()



# Now we apply the modified ROF (accelerated)

start = time.time()
denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)
t2 = time.time() - start

denoised_v2_acc_1d = denoised_v2_acc.flatten()
figure6 = denoised_v2_acc_1d.copy()

# Compute and show the gradient of the distance

difference_2d = denoised_v2_acc
difference_2d -= image

grad = compute_gradient_magnitude(difference_2d)
grad_norm = norm1(grad)

print(f"Modified ROF (acc), radius {defined_radius}, weights: a = {a_const}, b = {b_const} - result: {grad_norm}")



# Step 5: Apply the modified ROF for a larger weight

# Compute the weight from the result of the original ROF

a_const = 2 * number_for_scaling
b_const = 2 * number_for_scaling * number_for_ratio
a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm_rof, 1/(2*number_for_ratio)))


# Optional: show the weight for control

#a_weight_1d = a_weight.flatten()
#plt.plot(a_weight_1d)
#plt.show()


# Now apply the modified ROF (accelerated)

start = time.time()
denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)
t2 = time.time() - start

denoised_v2_acc_1d = denoised_v2_acc.flatten()
figure7 = denoised_v2_acc_1d.copy()

difference_1d = denoised_v2_acc_1d
difference_1d -= image_1d

difference_2d = denoised_v2_acc
difference_2d -= image

grad = compute_gradient_magnitude(difference_2d)
grad_norm = norm1(grad)

print(f"Modified ROF (acc), radius {defined_radius}, weights: a = {a_const}, b = {b_const} - result: {grad_norm}")



# Step 4: Apply the modified ROF for a larger weight

# Compute the weight from the result of the original ROF

a_const = 3 * number_for_scaling
b_const = 3 * number_for_scaling * number_for_ratio
a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm_rof, 1/(2*number_for_ratio)))


# Optional: Show the weight for control

#a_weight_1d = a_weight.flatten()
#plt.plot(a_weight_1d)
#plt.show()


# Now apply the modified ROF (accelerated)

start = time.time()
denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)
t2 = time.time() - start

denoised_v2_acc_1d = denoised_v2_acc.flatten()
figure8 = denoised_v2_acc_1d.copy()

difference_1d = denoised_v2_acc_1d
difference_1d -= image_1d

difference_2d = denoised_v2_acc
difference_2d -= image

grad = compute_gradient_magnitude(difference_2d)
grad_norm = norm1(grad)

print(f"Modified ROF (acc), radius {defined_radius}, weights: a = {a_const}, b = {b_const} - result: {grad_norm}")





# Show all images
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

plt.ylim(bottom=0,top=1)
axes[0].plot(figure1)
axes[0].set_title('Original image')
axes[1].plot(figure2)
axes[1].set_title('Classical ROF')
axes[2].plot(figure3)
axes[2].set_title('Huber ROF, α = 0.002')
axes[3].plot(figure4)
axes[3].set_title('Huber ROF, α = 0.01')


plt.show()


fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

plt.ylim(bottom=0,top=1)
axes[0].plot(figure5)
axes[0].set_title('Noisy image')
axes[1].plot(figure6)
axes[1].set_title(f'a = {number_for_scaling}, b = {number_for_scaling * number_for_ratio}')
axes[2].plot(figure7)
axes[2].set_title(f'a = {2 * number_for_scaling}, b = {2 * number_for_scaling * number_for_ratio}')
axes[3].plot(figure8)
axes[3].set_title(f'a = {3 * number_for_scaling}, b = {3 * number_for_scaling * number_for_ratio}')

plt.show()