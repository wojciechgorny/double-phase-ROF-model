import time
from pickle import TUPLE3
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io, util, color
from numba import njit
from scipy.ndimage import convolve
from google.colab import files
import io as python_io

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


# Original Chambolle-Pock, acclerated v1
@njit
def chambolle_pock(image, tau, lam=0.2, max_iter=10000, tol=1.0e-5):
    # Initialise the variables

    m, n = image.shape                      # Take the dimensions of the image
    p = np.zeros((m, n, 2))                # The vector field
    g = np.zeros((m, n, 2))                # The gradient
    x = image                               # Input image
    x_bar = x.copy()                        # Again the input image, because Chambolle-Pock tracks two copies of it

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma_cp * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * div_p + (tau / lam) * image  )/(1 + (tau / lam))

        # Update theta, tau, sigma and x_bar
        theta = 1 / np.sqrt(1 + tau /(2*lam))
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            print ( f" Converged in { i + 1 } iterations. - Chambolle-Pock " )
            break

    return x


#Original Chambolle-Pock, accelerated v2
@njit
def chambolle_pock_v(image, tau, lam=0.2, max_iter=10000, tol=1.0e-5):
    # Initialise the variables

    m, n = image.shape                      # Take the dimensions of the image
    p = np.zeros((m, n, 2))                # The vector field
    g = np.zeros((m, n, 2))                # The gradient
    x = image                               # Input image
    x_bar = x.copy()                        # Again the input image, because Chambolle-Pock tracks two copies of it

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma_cp * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None]/lam)
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * (div_p + image) )/(1 + tau)

        # Update theta, tau, sigma and x_bar
        theta = 1 / np.sqrt(1 + tau/2)
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            print ( f" Converged in { i + 1 } iterations. - Chambolle-Pock_v " )
            break

    return x

# Average on ball kernel for mollification
def ball_kernel(radius):
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

# Custom Resolvent
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

# Modified Chambolle-Pock, accelerated (adaptive double-phase) v1
def chambolle_pock_modified(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-5):
    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()


    # Set the values of auxiliary constants
    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1


    for i in range(max_iter):

        # Compute the gradient g and update the vector field p
        g = gradient(x_bar)
        p_new = p + sigma_cp * g
        p = custom_resolvent(p_new, a_weight, sigma_cp)


        # Compute divergence and update x
        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))


        # Update theta, tau, sigma and x_bar

        theta = 1 / np.sqrt(1 + tau / (2*lam))
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            print(f"Converged in {i+1} iterations. - modified Chambolle-Pock")
            break

    return x

# Custom Resolvent
def custom_resolvent_v(p_tilde, a, sigma, lam):
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
    p[mask_large] = p_tilde[mask_large] * (factor[..., None])

    return p

# Modified Chambolle-Pock, accelerated (adaptive double-phase) v2
def chambolle_pock_modified_v(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-5):
    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()


    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1

    for i in range(max_iter):



        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = p + sigma_cp * g
        p = custom_resolvent_v(p_new, a_weight, sigma_cp, lam)


        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * (div_p + image) )/(1 + tau)

        # Update theta, tau, sigma and x_bar

        theta = 1 / np.sqrt(1 + tau/2)
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            print(f"Converged in {i+1} iterations. - modified Chambolle-Pock_v")
            break

    return x



# Add Gaussian noise
noisy = util.random_noise(image, mode='gaussian', var=0.01)


# Generate plots
lam = 0.24 # chosen λ

# Apply Chambolle-Pock algorithm for classical ROF model
start = time.time()
denoised_classical = chambolle_pock(noisy, tau=0.25, lam=lam, max_iter=20000)
t1 = time.time() - start

grad_class = compute_gradient_magnitude(denoised_classical)
classical_energy = np.sum(grad_class)
classical_distance = np.linalg.norm(denoised_classical - noisy)

# Mollified gradient of minimizer of classical ROF
mollified = mollify_function(denoised_classical, radius=2)
grad_mollified = gradient(mollified)
grad_norm = norm2(grad_mollified)


a_const = 150
b_const = 1500
a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

# Apply modified Chambolle-Pock algorithm for adaptive double-phase ROF model

start = time.time()
u_mod = chambolle_pock_modified(noisy, a_weight, tau=0.25, lam=lam, max_iter=20000)
t2 = time.time() - start

mod_distance = np.linalg.norm(u_mod - noisy)


# Evaluation of performance
print("\n--- Performance Summary ---")

print(f"Classical ROF runtime: {t1:.2f} s")

print(f"Double-phase ROF runtime: {t2:.2f} s")



# Plot 1: Denoised image with adaptive double-phase ROF model
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(u_mod, cmap='gray')
ax.set_title(f'\nλ={lam:.3f}\nD={mod_distance:.3f}')
ax.axis('off')
plt.tight_layout()
plt.show()

# Plot 2: Denoised image with classical ROF model
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(denoised_classical, cmap='gray')
ax.set_title(f'\nλ={lam:.3f}\nD={classical_distance:.3f}')
ax.axis('off')
plt.tight_layout()
plt.show()

# Plot 3: Gradient of minimizer of classical ROF model
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(grad_class, cmap='magma')
ax.set_title(f'TV={classical_energy:.2f}')
ax.axis('off')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('|∇u| classical')
plt.tight_layout()
plt.show()

# Plot 4: Mollified gradient of minimizer of classical ROF model
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(grad_norm, cmap='magma')
ax.set_title(fr'$|\nabla \tilde{{u}}_{{ROF}}|_{{max}} = {np.max(grad_norm):.2f}$')
ax.axis('off')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Plot 5: The weight
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(a_weight, cmap='viridis')
ax.set_title(f'weight: min={np.min(a_weight):.2f}, max={np.max(a_weight):.2f}')
ax.axis('off')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

#Plot 6: Dependence of weight function from magnitude of mollified gradient
fig, ax = plt.subplots(figsize=(4, 3))
gnorm = grad_norm.ravel()
wvals = a_weight.ravel()
ax.plot(gnorm, wvals, '.', markersize=1, alpha=0.3)
ax.set_xlabel(fr'$|\nabla \tilde{{u}}_{{ROF}}|$')
ax.set_ylabel(fr'W($|\nabla \tilde{{u}}_{{ROF}}|$)')
plt.tight_layout()
plt.show()