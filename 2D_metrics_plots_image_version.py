import time
from pickle import TUPLE3
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io, util, color
!pip install --upgrade numba
from numba import njit
from scipy.ndimage import convolve
from google.colab import files
import io as python_io
from numpy import linalg as LA

# Code for producing plots of the metrics: TV error, SSIM and PSNR for the classical, adaptive double-phase and Huber ROF models


import time
from io import BytesIO
import os
from pickle import TUPLE3
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io, util, color
from numba import njit
from scipy.ndimage import convolve
from google.colab import files
import io as python_io
from numpy import linalg as LA
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# Upload image
uploaded = files.upload()
filename = list(uploaded.keys())[0]
image = io.imread(python_io.BytesIO(uploaded[filename]))

# Handle different image formats
if image.ndim == 2:
    pass  # Already grayscale
elif image.ndim == 3:
    if image.shape[2] == 4:
        image = color.rgb2gray(color.rgba2rgb(image))  # Convert RGBA -> RGB -> grayscale
    elif image.shape[2] == 3:
        image = color.rgb2gray(image)  # Convert RGB -> grayscale
    else:
        raise ValueError(f"Unsupported channel format: image.shape = {image.shape}")
else:
    raise ValueError(f"Unsupported image format: image.ndim = {image.ndim}")

image = img_as_float(image)


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

def square_zero(grid_size):
    output = np.zeros((grid_size,grid_size))
    return output

def square(grid_size):
    half_grid_size = round(grid_size/2)
    output = np.zeros((grid_size,grid_size))
    for i in range(0,half_grid_size):
      for j in range(0,half_grid_size):
        output[i,j] = 1
    return output

def square_linear(grid_size):
    output = np.zeros((grid_size,grid_size))
    for i in range(0,grid_size):
      for j in range(0,grid_size):
        output[i,j] = (i + j)/(2*grid_size)
    return output




# Definition of different weights with parameters

def weight_two_cutoffs(argument, a_const = 60, b_const = 1200):
    first_cutoff = a_const/(2*b_const)
    #second_cutoff = a_const/(b_const)
    output = np.maximum(0, a_const - b_const * np.maximum(argument, first_cutoff))
    return output

def weight_one_cutoff(argument, a_const = 50, b_const = 1000):
    #cutoff = a_const/(b_const)
    output = np.maximum(0, a_const - b_const * argument)
    return output

def weight_characteristic_function(argument, a_const = 40, b_const = 0.07):
    output = a_const*(argument < b_const)
    return output


# Set the current parameters

current_lambda = 0.16
current_noise = 0.04
error_parameter = 1e-4
current_weight = weight_two_cutoffs
number_for_averaging = 1
number_for_comparison = 22
number_for_scaling = 0.02
huber_alpha = 0.01


# Optional: use one of the defined test images

#grid_size = 10
#image = square(grid_size)


# Show what function is chosen

#x = np.linspace(0, 0.2, 1000)
#y = current_weight(x)

#print(f"Current weight")

#plt.plot(x, y)
#plt.show()

#print()


# Helper functions: Gradient, Divergence, Norm
# Compute Gradient Magnitude
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
def custom_resolvent(p_tilde, a, sigma, lam=current_lambda):
    norm_p = norm2(p_tilde)
    p = np.zeros_like(p_tilde)

    mask_zero = (a == 0)
    mask_small = (a > 0) & (norm_p <= lam)
    mask_large = (a > 0) & (norm_p > lam)

    p[mask_zero] = p_tilde[mask_zero] / np.maximum(1, norm_p[mask_zero]/lam)[..., None]
    p[mask_small] = p_tilde[mask_small]
    s = norm_p[mask_large]
    a_vals = a[mask_large]

    factor = (sigma/s + a_vals) / (sigma/lam + a_vals)


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
            #print (f"Original ROF: {i+1} iterations")
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
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None]/lam)
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()

        x = ( x + tau * (div_p + image) )/(1 + tau)

        # Update theta, tau, sigma and x_bar


        theta = 1/(np.sqrt(1 + tau/2))
        tau *= theta
        sigma /= theta
        x_bar = (1+theta)*x - theta*x_prev

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            #print(f"Original ROF (acc): {i+1} iterations")
            break

    return x, i+1


# Modified ROF(adaptive double-phase)
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
            #print(f"Modified ROF: {i+1} iterations")
            break

    return x, i+1




# Modified ROF (adaptive double-phase), accelerated
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
        p = custom_resolvent(p_new, a_weight, sigma, lam)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()

        x = ( x + tau * (div_p + image) )/(1 + tau)

        # Update theta, tau, sigma and x_bar


        theta = 1/(np.sqrt(1+tau/2))
        tau *= theta
        sigma /= theta
        x_bar = (1+theta)*x - theta*x_prev

        if np.linalg.norm(x - x_prev) < tol:
            #print(f"Modified ROF (acc): {i+1} iterations")
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
            #print(f"Huber ROF: {i+1} iterations")
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
            #print(f"Huber ROF (acc): {i+1} iterations")
            break

    return x, i+1






# Total variation of the initial datum for control

grad = compute_gradient_magnitude(image)
grad_norm = norm1(norm1(grad))


print(f"TV for initial datum: {grad_norm}")
print()

TV_original = grad_norm


# Loop for automating the computations

print(f"Automated Chambolle-Pock, {number_for_averaging} iterations")
print(f"Value of lambda: {current_lambda}")
print(f"Noise level: {current_noise}")
print(f"Error parameter: {error_parameter}")
#print(f"Grid size: {grid_size}")   # Optional for the defined test functions
print()


# Initialising the variables for averaging


technical_sum_ROF_acc = np.zeros(number_for_comparison)
l2distance_sum_ROF_acc = np.zeros(number_for_comparison)
l2distance_noisy_sum_ROF_acc = np.zeros(number_for_comparison)
iterations_sum_ROF_acc = np.zeros(number_for_comparison)

technical_sum_mROF = np.zeros(number_for_comparison)
l2distance_sum_mROF = np.zeros(number_for_comparison)
l2distance_noisy_sum_mROF = np.zeros(number_for_comparison)
iterations_sum_mROF = np.zeros(number_for_comparison)

technical_sum_hROF_acc = np.zeros(number_for_comparison)
l2distance_sum_hROF_acc = np.zeros(number_for_comparison)
l2distance_noisy_sum_hROF_acc = np.zeros(number_for_comparison)
iterations_sum_hROF_acc = np.zeros(number_for_comparison)


ssim_sum_ROF_acc = np.zeros(number_for_comparison)
psnr_sum_ROF_acc = np.zeros(number_for_comparison)

ssim_sum_mROF = np.zeros(number_for_comparison)
psnr_sum_mROF = np.zeros(number_for_comparison)

ssim_sum_hROF_acc = np.zeros(number_for_comparison)
psnr_sum_hROF_acc = np.zeros(number_for_comparison)



for i in range(0, number_for_comparison):
   technical_sum_mROF[i] = 0
   l2distance_sum_mROF[i] = 0
   l2distance_noisy_sum_mROF[i] = 0
   iterations_sum_mROF[i] = 0


# For control: print the L2 norm of the image


l2norm = norm2(norm2(image))
print(f"L2 norm of the image (v1): {l2norm}")

l2norm = LA.norm(image)
print(f"L2 norm of the image (v2): {l2norm}")



# Start the loop for averaging

for i in range(1, number_for_comparison):

    current_lambda = number_for_scaling*i

    for j in range(number_for_averaging):

        # Step 0: Create a new image by adding Gaussian noise
        noisy = util.random_noise(image, mode='gaussian', var=current_noise)




        # Step 2i-1: Run accelerated classical ROF

        denoised_v1_acc, iter_v1_acc = chambolle_pock_accelerated(noisy, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

        # Compute the differences to the original and noisy images

        difference_2d = denoised_v1_acc.copy()
        difference_2d_noisy = denoised_v1_acc.copy()
        difference_2d -= image
        difference_2d_noisy -= noisy

        # Compute the gradient of the difference to the original image

        grad = compute_gradient_magnitude(difference_2d)
        grad_norm = norm1(norm1(grad))

        #grad_norm = norm1(norm1(grad)) / TV_original # Optional: compute normalized TV error

        # Compute the L2 norm of the difference to the original image

        l2distance = norm2(norm2(difference_2d))

        # Compute the L2 norm of the difference to the noisy image

        l2distance_noisy = norm2(norm2(difference_2d_noisy))

        # Compute the SSIM & PSNR metrics for Classical ROF

        ssim_val = ssim(image, denoised_v1_acc, data_range=denoised_v1_acc.max() - denoised_v1_acc.min())
        psnr_val = psnr(image, denoised_v1_acc, data_range=denoised_v1_acc.max() - denoised_v1_acc.min())

        ssim_sum_ROF_acc[i] += ssim_val
        psnr_sum_ROF_acc[i] += psnr_val


        # Add to the average: the gradient, the L2 distances, the number of iterations

        technical_sum_ROF_acc[i] += grad_norm
        l2distance_sum_ROF_acc[i] += l2distance
        l2distance_noisy_sum_ROF_acc[i] += l2distance_noisy
        iterations_sum_ROF_acc[i] += iter_v1_acc




        # Step 2i: Run the accelerated adaptive double-phase ROF

        # First generate the weight from the classical ROF

        mollified_v1_acc = mollify_function(denoised_v1_acc, radius=2)
        grad_v1 = gradient(mollified_v1_acc)
        grad_norm = norm2(grad_v1)

        a_weight = current_weight(argument=grad_norm)


        # Optional: construct the weight from the classical ROF (without mollification)
        #grad_v1 = gradient(denoised_v1v_acc)
        #grad_norm = norm2(grad_v1)
        #a_weight = current_weight(argument=grad_norm)

        # Optional: construct the weight from noisy image
        #mollified_v1_acc = mollify_function(noisy, radius=1)
        #grad_v1 = gradient(mollified_v1_acc)
        #grad_norm = norm2(grad_v1)
        #a_weight = current_weight(argument=grad_norm)

        # Then apply the algorithm

        denoised_v2_acc, iter_v2_acc = chambolle_pock_modified_accelerated(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

        # Compute the differences to the original and noisy images

        difference_2d = denoised_v2_acc.copy()
        difference_2d_noisy = denoised_v2_acc.copy()
        difference_2d -= image
        difference_2d_noisy -= noisy

        # Compute the gradient of the difference to the original image

        grad = compute_gradient_magnitude(difference_2d)
        grad_norm = norm1(norm1(grad))

        #grad_norm = norm1(norm1(grad)) / TV_original # Optional: compute normalized TV error

        # Compute the L2 norm of the difference to the original image

        l2distance = norm2(norm2(difference_2d))

        # Compute the L2 norm of the difference to the noisy image

        l2distance_noisy = norm2(norm2(difference_2d_noisy))

        # Compute the SSIM & PSNR metrics for modified ROF

        ssim_val_mROF = ssim(image, denoised_v2_acc, data_range=denoised_v2_acc.max() - denoised_v2_acc.min())
        psnr_val_mROF = psnr(image, denoised_v2_acc, data_range=denoised_v2_acc.max() - denoised_v2_acc.min())

        ssim_sum_mROF[i] += ssim_val_mROF
        psnr_sum_mROF[i] += psnr_val_mROF

        # Add to the average: the gradient, the L2 distances, the number of iterations

        technical_sum_mROF[i] += grad_norm
        l2distance_sum_mROF[i] += l2distance
        l2distance_noisy_sum_mROF[i] += l2distance_noisy
        iterations_sum_mROF[i] += iter_v2_acc


        # Step 2i+1: Run the accelerated Huber ROF
        denoised_v3_acc, iter_v3_acc = chambolle_pock_huber_accelerated(
        noisy, alpha=huber_alpha, tau=0.25, lam=current_lambda, max_iter=50000, tol=error_parameter)

        # Compute differences
        difference_2d = denoised_v3_acc.copy()
        difference_2d_noisy = denoised_v3_acc.copy()
        difference_2d -= image
        difference_2d_noisy -= noisy

        # Compute gradient of the difference to original image
        grad = compute_gradient_magnitude(difference_2d)
        grad_norm = norm1(norm1(grad))


        #grad_norm = norm1(norm1(grad)) / TV_original # Optional: compute normalized TV error

        # Compute L2 distances
        l2distance = norm2(norm2(difference_2d))
        l2distance_noisy = norm2(norm2(difference_2d_noisy))

        # Compute SSIM & PSNR
        ssim_val_hROF = ssim(image, denoised_v3_acc, data_range=denoised_v3_acc.max() - denoised_v3_acc.min())
        psnr_val_hROF = psnr(image, denoised_v3_acc, data_range=denoised_v3_acc.max() - denoised_v3_acc.min())

        # Accumulate
        technical_sum_hROF_acc[i] += grad_norm
        l2distance_sum_hROF_acc[i] += l2distance
        l2distance_noisy_sum_hROF_acc[i] += l2distance_noisy
        iterations_sum_hROF_acc[i] += iter_v3_acc

        ssim_sum_hROF_acc[i] += ssim_val_hROF
        psnr_sum_hROF_acc[i] += psnr_val_hROF




# Take the averages


average_ROF_acc = technical_sum_ROF_acc / number_for_averaging
average_l2distance_ROF_acc = l2distance_sum_ROF_acc / number_for_averaging
average_l2distance_noisy_ROF_acc = l2distance_noisy_sum_ROF_acc / number_for_averaging
average_iterations_ROF_acc = np.zeros(number_for_comparison)

for i in range(0, number_for_comparison):
    average_iterations_ROF_acc[i] = round(iterations_sum_ROF_acc[i] / number_for_averaging)



average_mROF = np.zeros(number_for_comparison)
average_l2distance_mROF = np.zeros(number_for_comparison)
average_l2distance_noisy_mROF = np.zeros(number_for_comparison)
average_iterations_mROF = np.zeros(number_for_comparison)

average_mROF = technical_sum_mROF / number_for_averaging
average_l2distance_mROF = l2distance_sum_mROF / number_for_averaging
average_l2distance_noisy_mROF = l2distance_noisy_sum_mROF / number_for_averaging

for i in range(0, number_for_comparison):
    average_iterations_mROF[i] = round(iterations_sum_mROF[i] / number_for_averaging)

    average_ssim_ROF_acc = ssim_sum_ROF_acc / number_for_averaging
    average_psnr_ROF_acc = psnr_sum_ROF_acc / number_for_averaging

    average_ssim_mROF = ssim_sum_mROF / number_for_averaging
    average_psnr_mROF = psnr_sum_mROF / number_for_averaging

average_hROF_acc = technical_sum_hROF_acc / number_for_averaging
average_l2distance_hROF_acc = l2distance_sum_hROF_acc / number_for_averaging
average_l2distance_noisy_hROF_acc = l2distance_noisy_sum_hROF_acc / number_for_averaging

average_iterations_hROF_acc = np.zeros(number_for_comparison)

for i in range(0, number_for_comparison):
    average_iterations_hROF_acc[i] = round(iterations_sum_hROF_acc[i] / number_for_averaging)

average_ssim_hROF_acc = ssim_sum_hROF_acc / number_for_averaging
average_psnr_hROF_acc = psnr_sum_hROF_acc / number_for_averaging


# Print the averages: first classical ROF, then modified ROF for different weights


for i in range(0, number_for_comparison):

    print()

    print(f"Classical ROF, lambda {number_for_scaling*i} (acc) - average TV error: {average_ROF_acc[i]}")

    print(f"Classical ROF, lambda {number_for_scaling*i} (acc) - average l2 distance (image): {average_l2distance_ROF_acc[i]}")

    print(f"Classical ROF, lambda {number_for_scaling*i} (acc) - average l2 distance (noisy): {average_l2distance_noisy_ROF_acc[i]}")

    print(f"Classical ROF, lambda {number_for_scaling*i} (acc) - number of iterations {average_iterations_ROF_acc[i]}")

    print(f"Classical ROF, lambda {number_for_scaling*i} (acc) - average SSIM: {average_ssim_ROF_acc[i]}")

    print(f"Classical ROF, lambda {number_for_scaling*i} (acc) - average PSNR: {average_psnr_ROF_acc[i]}")

    print()

    print(f"Modified ROF, lambda {number_for_scaling*i} (acc) - average TV error: {average_mROF[i]}")

    print(f"Modified ROF, lambda {number_for_scaling*i} (acc) - average l2 distance (image): {average_l2distance_mROF[i]}")

    print(f"Modified ROF, lambda {number_for_scaling*i} (acc) - average l2 distance (noisy): {average_l2distance_noisy_mROF[i]}")

    print(f"Modified ROF, lambda {number_for_scaling*i} (acc) - number of iterations: {average_iterations_mROF[i]}")

    print(f"Modified ROF, lambda {number_for_scaling*i} (acc) - average SSIM: {average_ssim_mROF[i]}")

    print(f"Modified ROF, lambda {number_for_scaling*i} (acc) - average PSNR: {average_psnr_mROF[i]}")

    print()

    print(f"Huber ROF, lambda {number_for_scaling*i} (acc) - average TV error: {average_hROF_acc[i]}")

    print(f"Huber ROF, lambda {number_for_scaling*i} (acc) - average l2 distance (image): {average_l2distance_hROF_acc[i]}")

    print(f"Huber ROF, lambda {number_for_scaling*i} (acc) - average l2 distance (noisy): {average_l2distance_noisy_hROF_acc[i]}")

    print(f"Huber ROF, lambda {number_for_scaling*i} (acc) - number of iterations: {average_iterations_hROF_acc[i]}")

    print(f"Huber ROF, lambda {number_for_scaling*i} (acc) - average SSIM: {average_ssim_hROF_acc[i]}")

    print(f"Huber ROF, lambda {number_for_scaling*i} (acc) - average PSNR: {average_psnr_hROF_acc[i]}")


# Plots of the metrics vs fixed L2 distance from the noisy image

# Set the name of the image

image_name = os.path.splitext(filename)[0] # Automated label of the plot according to the image

# Label for current weight function

weight_labels = {
    weight_two_cutoffs: "1",
    weight_one_cutoff: "2",
    weight_characteristic_function: "3"
}


# Set average L2 distance from the noisy image for the x-axis

x_axis_dist_ROF = average_l2distance_noisy_ROF_acc[1:]
x_axis_dist_mROF = average_l2distance_noisy_mROF[1:]
x_axis_dist_hROF = average_l2distance_noisy_hROF_acc[1:]


# Plot TV-error vs L2 distance from noisy image

plt.plot(x_axis_dist_ROF, average_ROF_acc[1:], label="ROF")
plt.plot(x_axis_dist_mROF, average_mROF[1:], label="dpROF")
plt.plot(x_axis_dist_hROF, average_hROF_acc[1:], label="Huber ROF")


plt.xlabel(r"$\mathrm{dist}_{L^2}(\text{noisy}, \text{denoised})$")
plt.ylabel(r"TV(original, denoised)")
plt.title(
    f""
    f"Weight {weight_labels[current_weight]}, "
    f"σ = {current_noise}, Image: girlface_512"
)
plt.grid(True)
plt.legend()
plt.xlim(left=0)
plt.ylim(bottom=0)

# Compute min TV error and corresponding L2 distances
min_tv_ROF = min(average_ROF_acc[1:])
idx_min_ROF = np.argmin(average_ROF_acc[1:])
x_min_ROF = x_axis_dist_ROF[idx_min_ROF]

min_tv_mROF = min(average_mROF[1:])
idx_min_mROF = np.argmin(average_mROF[1:])
x_min_mROF = x_axis_dist_mROF[idx_min_mROF]

min_tv_hROF = min(average_hROF_acc[1:])
idx_min_hROF = np.argmin(average_hROF_acc[1:])
x_min_hROF = x_axis_dist_hROF[idx_min_hROF]

# Prepare the summary box text for the plot of TV error
summary_text_tv = (
    f"Min TV Error:\n"
    f"ROF: ({number_for_scaling * (idx_min_ROF + 1):.2f}, {x_min_ROF:.2f}, {min_tv_ROF:.3f})\n"
    f"dpROF: ({number_for_scaling * (idx_min_mROF + 1):.2f}, {x_min_mROF:.2f}, {min_tv_mROF:.3f})\n"
    f"Huber ROF: ({number_for_scaling * (idx_min_hROF + 1):.2f}, {x_min_hROF:.2f}, {min_tv_hROF:.3f})"
)

# Add summary box to the plot of TV error
plt.gca().text(
    0.03, 0.02, summary_text_tv,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='left',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray")
)
plt.tight_layout()
plt.show()



# Plot SSIM metric vs L2 distance from noisy image


plt.plot(x_axis_dist_ROF, average_ssim_ROF_acc[1:], label="ROF")
plt.plot(x_axis_dist_mROF, average_ssim_mROF[1:], label="dpROF")
plt.plot(x_axis_dist_hROF, average_ssim_hROF_acc[1:], label="Huber ROF")

plt.xlabel(r"$\mathrm{dist}_{L^2}(\text{noisy}, \text{denoised})$")
plt.ylabel(r"SSIM(original, denoised)")
plt.title(
    f"Weight {weight_labels[current_weight]}, "
    f"σ = {current_noise}, Image: girlface_512"
)
plt.grid(True)
plt.xlim(left=0)
plt.ylim(0, 1)
plt.legend()

# Compute max SSIM values and corresponding L2 distances
max_ssim_ROF = max(average_ssim_ROF_acc[1:])
idx_max_ROF = np.argmax(average_ssim_ROF_acc[1:])
x_max_ROF = x_axis_dist_ROF[idx_max_ROF]

max_ssim_mROF = max(average_ssim_mROF[1:])
idx_max_mROF = np.argmax(average_ssim_mROF[1:])
x_max_mROF = x_axis_dist_mROF[idx_max_mROF]

max_ssim_hROF = max(average_ssim_hROF_acc[1:])
idx_max_hROF = np.argmax(average_ssim_hROF_acc[1:])
x_max_hROF = x_axis_dist_hROF[idx_max_hROF]

# Prepare the summary box text for the plot of the SSIM metric
summary_text = (
    f"Max SSIM:\n"
    f"ROF: ({number_for_scaling * (idx_max_ROF + 1):.2f}, {x_max_ROF:.2f}, {max_ssim_ROF:.3f})\n"
    f"dpROF: ({number_for_scaling * (idx_max_mROF + 1):.2f}, {x_max_mROF:.2f}, {max_ssim_mROF:.3f})\n"
    f"Huber ROF: ({number_for_scaling * (idx_max_hROF + 1):.2f}, {x_max_hROF:.2f}, {max_ssim_hROF:.3f})"
)

# Add the summary text box to the plot of the SSIM metric
plt.gca().text(
    0.98, 0.02, summary_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray")
)

plt.tight_layout()
plt.show()


# Plot PSNR metric vs L2 distance from noisy image

plt.plot(x_axis_dist_ROF, average_psnr_ROF_acc[1:], label="ROF")
plt.plot(x_axis_dist_mROF, average_psnr_mROF[1:], label="dpROF")
plt.plot(x_axis_dist_hROF, average_psnr_hROF_acc[1:], label="Huber ROF")


plt.xlabel(r"$\mathrm{dist}_{L^2}(\text{noisy}, \text{denoised})$")
plt.ylabel("PSNR(original, denoised)")
plt.title(
    f""
    f"Weight {weight_labels[current_weight]}, "
    f"σ = {current_noise}, Image: girlface_512"
)
plt.grid(True)
plt.legend()
plt.xlim(left=0)
plt.ylim(bottom=0, top=50.0)

# Compute max PSNR and corresponding L2 distances
max_psnr_ROF = max(average_psnr_ROF_acc[1:])
idx_max_psnr_ROF = np.argmax(average_psnr_ROF_acc[1:])
x_max_psnr_ROF = x_axis_dist_ROF[idx_max_psnr_ROF]

max_psnr_mROF = max(average_psnr_mROF[1:])
idx_max_psnr_mROF = np.argmax(average_psnr_mROF[1:])
x_max_psnr_mROF = x_axis_dist_mROF[idx_max_psnr_mROF]

max_psnr_hROF = max(average_psnr_hROF_acc[1:])
idx_max_psnr_hROF = np.argmax(average_psnr_hROF_acc[1:])
x_max_psnr_hROF = x_axis_dist_hROF[idx_max_psnr_hROF]

# Prepare the summary box text for the PSNR metric
summary_text_psnr = (
    f"Max PSNR:\n"
    f"ROF: ({number_for_scaling * (idx_max_psnr_ROF + 1):.2f}, {x_max_psnr_ROF:.2f}, {max_psnr_ROF:.2f})\n"
    f"dpROF: ({number_for_scaling * (idx_max_psnr_mROF + 1):.2f}, {x_max_psnr_mROF:.2f}, {max_psnr_mROF:.2f})\n"
    f"Huber ROF: ({number_for_scaling * (idx_max_psnr_hROF + 1):.2f}, {x_max_psnr_hROF:.2f}, {max_psnr_hROF:.2f})"
)

# Add summary box to the plot of the PSNR metric
plt.gca().text(
    0.98, 0.02, summary_text_psnr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray")
)

plt.tight_layout()
plt.show()