import time
from pickle import TUPLE3
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io, util, color
from numba import njit
from scipy.ndimage import convolve
from google.colab import files
import io as python_io
import matplotlib.patches as patches

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
            print ( f" Converged in { i + 1 } iterations. - classical ROF model (acc) " )
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
            print ( f" Converged in { i + 1 } iterations. - classical ROF model (acc) " )
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
            print(f"Converged in {i+1} iterations. - adaptive double-phase ROF model (acc)")
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
            print(f"Converged in {i+1} iterations. - adaptive double-phase ROF model (acc)")
            break

    return x



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
            print(f"Huber ROF (acc): {i+1} iterations")
            break

    return x

# Add Gaussian noise
noisy = util.random_noise(image, mode='gaussian', var=0.01)

# Step 1: Apply Chambolle-Pock algorithm for classical ROF model

start = time.time()
denoised_v1v = chambolle_pock_v(noisy, tau=0.25, lam=0.20, max_iter=20000)
t2 = time.time() - start

# Step 2: Compute weight from gradient norm
#mollified_v1 = mollify_function(denoised_v1v, radius=1)
#mollified_v1 = mollify_function(noisy, radius=1)
#grad_v1 = gradient(mollified_v1)
grad_v1 = gradient(denoised_v1v)
grad_norm = norm2(grad_v1)

# Use weight 1

a_const = 200.0
b_const = 1000.0
a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

# Optional: Use weight 3

#a_const = 30.0
#b_const = 0.09
#a_weight = a_const*(grad_norm < b_const)


# Step 3: Apply modified Chambolle-Pock algorithm for adaptive double-phase ROF model

start = time.time()
denoised_v2v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=0.08, max_iter=20000)
t4 = time.time() - start

# Step 4: Apply Chambolle-Pock algorithm for Huber ROF model
start = time.time()
denoised_v3v = chambolle_pock_huber_accelerated(noisy, alpha=0.01, tau=0.25, lam=0.22, max_iter=20000)
t5 = time.time() - start


# Plots of denoised versions
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
titles = ['Original', 'Noisy (Gaussian)', 'classical ROF', 'double-phase ROF', 'Huber ROF']
images = [image, noisy, denoised_v1v, denoised_v2v, denoised_v3v]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Evaluation of performance
print("\n--- Performance Summary ---")

print(f"Classical ROF runtime: {t2:.2f} s")

print(f"Double-phase ROF runtime: {t4:.2f} s")

print(f"Huber ROF runtime: {t5:.2f} s")

import matplotlib.patches as patches


fig, axes = plt.subplots(1, 5, figsize=(18, 5))
titles = ['Original', 'Noisy (Gaussian)', 'classical ROF', 'double-phase ROF', 'Huber ROF']
images = [image, noisy, denoised_v1v, denoised_v2v, denoised_v3v]

# Zoom coordinates (adjust for the image content)
r0, r1 = 120, 470  # rows
c0, c1 = 120, 300  # columns



for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()


# Zoomed comparison

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, img, title in zip(axes, images, titles):
    zoom_region = img[r0:r1, c0:c1]
    ax.imshow(zoom_region, cmap='gray')
    ax.set_title(f"Zoom - {title}")
    ax.axis('off')

plt.tight_layout()
plt.show()



# Denoising with increasing parameter λ - comparison of adaptive double-phase ROF vs classical ROF

mod_images = [image, noisy]
mod_blurred_gradients = [None, None]
mod_weights = [None, None]
mod_distances = [0.0, 0.0]

classical_images = [image, noisy]
classical_gradients = [None, None]
classical_energies = [None, None]
classical_distances = [0.0, 0.0]


w_values = [0.02 + 0.02 * x for x in range(1, 6)]

for w in w_values:
    print(f"λ = {w:.3f}")

    # Compute mollified image from original Chambolle-Pock for adaptive weight
    denoised_classical = chambolle_pock_v(noisy, tau=0.25, lam=w, max_iter=20000)
    mollified = mollify_function(denoised_classical, radius=2)
    grad_mollified = gradient(mollified)
    grad_norm = norm2(grad_mollified)

    # Use weight 1

    a_const = 90.0
    b_const = 1500.0
    a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))


    # Optional: Use weight 3
    #a_const = 30.0
    #b_const = 0.09
    #a_weight = a_const*(grad_norm < b_const)

    # Apply modified Chambolle-Pock algorithm for adaptive double-phase ROF model
    u_mod = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=w, max_iter=20000)
    mod_images.append(u_mod)
    mod_blurred_gradients.append(norm2(gradient(mollified)))
    mod_weights.append(a_weight)
    mod_distances.append(np.linalg.norm(u_mod - noisy))

    # Apply Chambolle-Pock algorithm for the classical ROF model
    classical_images.append(denoised_classical)
    grad_class = compute_gradient_magnitude(denoised_classical)
    classical_gradients.append(grad_class)
    classical_energies.append(np.sum(grad_class))
    classical_distances.append(np.linalg.norm(denoised_classical - noisy))

fig, axes = plt.subplots(5, 7, figsize=(36, 26))

w_values = [0.02 + 0.02 * x for x in range(1, 6)]

# Row 1: Denoised images with adaptive double-phase ROF model
for i in range(7):
    axes[0, i].imshow(mod_images[i], cmap='gray')
    if i > 1:
        w = w_values[i - 2]
        d = mod_distances[i]
        axes[0, i].set_title(f'λ={w:.2f}\nD={d:.3f}')
    else:
        axes[0, i].set_title('Original' if i == 0 else 'Noisy')
    axes[0, i].axis('off')

# Row 2: Denoised images with classical ROF model
for i in range(7):
    axes[1, i].imshow(classical_images[i], cmap='gray')
    if i > 1:
        w = w_values[i - 2]
        d = classical_distances[i]
        axes[1, i].set_title(f'λ={w:.2f}\nD={d:.3f}')
    else:
        axes[1, i].set_title('Original' if i == 0 else 'Noisy')
    axes[1, i].axis('off')

# Row 3: Gradient magnitude of the minimizer of classical ROF model
im5 = None
for i in range(7):
    if classical_gradients[i] is not None:
        g = classical_gradients[i]
        im = axes[2, i].imshow(g, cmap='magma')
        if im5 is None:
            im5 = im
        tv = classical_energies[i]
        axes[2, i].set_title(f'TV={tv:.1f}')
    axes[2, i].axis('off')

# Row 4: Mollified gradient of the minimizer of classical ROF model
im2 = None
for i in range(7):
    if mod_blurred_gradients[i] is not None:
        g = mod_blurred_gradients[i]
        im = axes[3, i].imshow(g, cmap='magma')
        if im2 is None:
            im2 = im
        axes[3, i].set_title(f'|∇u|_max={np.max(g):.2f}')
    axes[3, i].axis('off')

# Row 5: The weight
im3 = None
for i in range(7):
    if mod_weights[i] is not None:
        wmap = mod_weights[i]
        im = axes[4, i].imshow(wmap, cmap='viridis')
        if im3 is None:
            im3 = im
        axes[4, i].set_title(f'a: min={np.min(wmap):.2f}, max={np.max(wmap):.2f}')
    axes[4, i].axis('off')

# Colorbars with precise alignment
plt.subplots_adjust(right=0.9)  # Leave room for colorbars

# Helper: get position of a row (use first column as reference)
def get_row_bbox(row_idx):
    pos = axes[row_idx][0].get_position()
    return pos.y0, pos.y1

cb_left = 0.91
cb_width = 0.015

# Row 5 (gradient of minimizer of classical ROF)
y0, y1 = get_row_bbox(2)
cax5 = fig.add_axes([cb_left, y0, cb_width, y1 - y0])
fig.colorbar(im5, cax=cax5, label='|∇u| classical')

# Row 2 (mollified gradient of minimizer of classical ROF)
y0, y1 = get_row_bbox(3)
cax2 = fig.add_axes([cb_left, y0, cb_width, y1 - y0])
fig.colorbar(im2, cax=cax2, label='|∇u| (mollified)')

# Row 3 (the weight)
y0, y1 = get_row_bbox(4)
cax3 = fig.add_axes([cb_left, y0, cb_width, y1 - y0])
fig.colorbar(im3, cax=cax3, label='a(x) weight')

plt.show()