import cv2
import numpy as np
import os
import time
import datetime
import itertools
import traceback

import torch
import lpips

from scipy.ndimage import convolve
from pathlib import Path
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# Helper Functions

def gradient_vectorial(u):

    grad_u = np.zeros(u.shape + (2,), dtype=np.float32)

    grad_u[:-1, :, :, 0] = u[1:, :, :] - u[:-1, :, :]
    grad_u[:, :-1, :, 1] = u[:, 1:, :] - u[:, :-1, :]

    return grad_u


def divergence_vectorial(p):

    div = np.zeros(p.shape[:3], dtype=np.float32)

    div[:-1, :, :] += p[:-1, :, :, 0]
    div[1:, :, :] -= p[:-1, :, :, 0]

    div[:, :-1, :] += p[:, :-1, :, 1]
    div[:, 1:, :] -= p[:, :-1, :, 1]

    return div


def frobenius_norm(p):

    return np.sqrt(np.sum(p**2, axis=(2, 3)) + 1e-12)


def chambolle_pock_vectorial_rof(image, tau, lam=0.2, max_iter=10000, tol=1e-5):

    m, n, channels = image.shape

    p = np.zeros((m, n, channels, 2), dtype=np.float32)

    x = image.copy()
    x_bar = x.copy()

    L = np.sqrt(8.0)
    sigma_cp = 1.0 / (tau * L * L)

    for i in range(max_iter):

        g = gradient_vectorial(x_bar)

        p_new = p + sigma_cp * g

        norm_p = frobenius_norm(p_new)

        p = p_new / np.maximum(1.0, norm_p / lam)[..., None, None]

        div_p = divergence_vectorial(p)

        x_prev = x.copy()

        x = (x + tau * (div_p + image)) / (1.0 + tau)

        theta = 1.0 / np.sqrt(1.0 + tau / 2.0)

        tau *= theta
        sigma_cp /= theta

        x_bar = x + theta * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x


def ball_kernel(radius):

    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    kernel /= np.sum(kernel)

    return kernel


def mollify_rgb(image, radius=2):

    kernel = ball_kernel(radius)
    mollified = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[2]):

        padded = np.pad(image[:, :, channel], pad_width=radius, mode="reflect")
        mollified_channel = convolve(padded, kernel, mode="reflect")
        mollified[:, :, channel] = mollified_channel[radius:-radius, radius:-radius]

    return mollified


def custom_resolvent_vectorial(p_tilde, a, sigma, lam):

    norm_p = frobenius_norm(p_tilde)

    p = np.zeros_like(p_tilde)

    mask_zero = a == 0
    mask_small = (a > 0) & (norm_p <= lam)
    mask_large = (a > 0) & (norm_p > lam)

    p[mask_zero] = p_tilde[mask_zero] / np.maximum(1.0, norm_p[mask_zero] / lam)[..., None, None]
    p[mask_small] = p_tilde[mask_small]

    s = norm_p[mask_large]
    a_values = a[mask_large]

    factor = (sigma / s + a_values) / (sigma / lam + a_values)

    p[mask_large] = p_tilde[mask_large] * factor[..., None, None]

    return p


def chambolle_pock_vectorial_dprof(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-5):

    m, n, channels = image.shape

    p = np.zeros((m, n, channels, 2), dtype=np.float32)

    x = image.copy()
    x_bar = x.copy()

    L = np.sqrt(8.0)
    sigma_cp = 1.0 / (tau * L * L)

    for i in range(max_iter):

        g = gradient_vectorial(x_bar)

        p_new = p + sigma_cp * g

        p = custom_resolvent_vectorial(p_new, a_weight, sigma_cp, lam)

        div_p = divergence_vectorial(p)

        x_prev = x.copy()

        x = (x + tau * (div_p + image)) / (1.0 + tau)

        theta = 1.0 / np.sqrt(1.0 + tau / 2.0)

        tau *= theta
        sigma_cp /= theta

        x_bar = x + theta * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x


# Utility Functions

def convert_image_to_rgb_float(image):

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def conversion_for_lpips(image):

    image_lpips = np.transpose(image, (2, 0, 1))
    image_lpips = torch.from_numpy(image_lpips.copy()).float()
    image_lpips = image_lpips.unsqueeze(0)
    image_lpips = image_lpips * 2.0 - 1.0
    image_lpips = image_lpips.to(device)

    return image_lpips




##############################################################################
############################ USER CONFIGURATION ##############################
##############################################################################

# Dataset folder containing the clean images.

folder_path = Path("/home/matsoukas/Desktop/experiments/BSDS500/35010.jpg")

# Base output folder and unique experiment name.

base_output_directory = Path(
    r"C:/Users/matsoukas/Desktop/dp/vectorial_dpROF_optimization"
)

base_output_directory = Path(r"RGB_dpROF_optimization")
experiment_name = "RGB_dpROF_variance_0.01_BSDS500_test"



# Gaussian noise variance and random seed.

given_variance = 0.01
noise_sigma = np.sqrt(given_variance)
random_seed = 0


# Full vectorial ROF parameter.
#
# Only one vectorial ROF is solved per image.
# The same ROF result is:
# 1. evaluated as the classical vectorial ROF output, and
# 2. used to construct the adaptive dpROF weight.

lambda_ROF = 0.7 * noise_sigma



# Tested vectorial dpROF parameter grid.

set_of_lambdas = [0.4 * noise_sigma, 0.5 * noise_sigma, 0.6 * noise_sigma, 0.7 * noise_sigma, 0.8 * noise_sigma]
set_of_a = [30.0, 40.0, 50.0, 60.0]
set_of_ba = [25.0, 50.0, 75.0, 100.0]


# Chambolle-Pock settings.

given_precision_ROF = 1e-4
given_precision_dpROF = 1e-4

tau_initial = 0.25
max_iter_ROF = 20000
max_iter_dpROF = 20000
mollifier_radius = 2


# Checkpoint settings.
#
# resume_preprocessing = True:
# Load existing per-image vectorial ROF preprocessing checkpoints and compute
# only the missing images.
#
# resume_grid = True:
# Load the latest vectorial dpROF grid checkpoint and continue automatically.
#
# For a completely new experiment, use a new experiment_name and set both
# values to False.

resume_preprocessing = True
resume_grid = False

# Save the grid checkpoint every saved_per_set completed dpROF iterations.

saved_per_set = 10


##############################################################################
######################## Derived paths and parameters ########################
##############################################################################

product_set = list(
    itertools.product(
        set_of_lambdas,
        set_of_a,
        set_of_ba
    )
)

product_set_print = [
    (
        round(float(current_lambda), 6),
        float(a_const),
        float(ba)
    )
    for current_lambda, a_const, ba in product_set
]

set_of_lambdas_print = [
    round(float(current_lambda), 6)
    for current_lambda in set_of_lambdas
]

output_directory = base_output_directory / experiment_name

preprocessing_checkpoint_directory = (
    output_directory / "preprocessing_checkpoints"
)

grid_checkpoint_directory = (
    output_directory / "grid_checkpoints"
)

preprocessing_progress_log_file = (
    output_directory / "vectorial_dpROF_preprocessing_progress.log"
)

grid_progress_log_file = (
    output_directory / "vectorial_dpROF_grid_progress.log"
)

log_file = (
    output_directory / "vectorial_dpROF_optimisation.log"
)

extended_log_file = (
    output_directory / "vectorial_dpROF_optimisation_extendedlog.log"
)

output_directory.mkdir(parents=True, exist_ok=True)

print(f"Noise standard deviation: {noise_sigma}")
print(f"Noise variance: {given_variance}")
print(f"lambda_ROF: {lambda_ROF}")
print(f"Considered dpROF lambdas: {set_of_lambdas_print}")
print(f"Considered a: {set_of_a}")
print(f"Considered b/a: {set_of_ba}")


##############################################################################
######################## Preprocessing checkpoint helpers ####################
##############################################################################

def safe_image_checkpoint_name(image_index, image_name):

    stem = Path(image_name).stem

    return f"{image_index:05d}_{stem}.npz"


def preprocessing_checkpoint_path(image_index, image_name):

    return (
        preprocessing_checkpoint_directory
        / safe_image_checkpoint_name(image_index, image_name)
    )


def save_preprocessing_checkpoint(image_index, image_name, prepared_image, rof_time):

    preprocessing_checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = preprocessing_checkpoint_path(
        image_index,
        image_name
    )

    temporary_path = checkpoint_path.with_suffix(".tmp.npz")

    np.savez_compressed(
        temporary_path,
        image_name=np.array(image_name),
        clean=prepared_image["clean"],
        noisy=prepared_image["noisy"],
        rof=prepared_image["rof"],
        grad_frobenius=prepared_image["grad_frobenius"],
        rof_time=np.float64(rof_time)
    )

    os.replace(temporary_path, checkpoint_path)

    return checkpoint_path


def load_preprocessing_checkpoint(image_index, image_name):

    checkpoint_path = preprocessing_checkpoint_path(
        image_index,
        image_name
    )

    if not checkpoint_path.is_file():
        return None

    checkpoint = np.load(checkpoint_path)

    stored_image_name = str(checkpoint["image_name"])

    if stored_image_name != image_name:
        raise ValueError(
            f"Preprocessing checkpoint mismatch: expected {image_name}, "
            f"found {stored_image_name}."
        )

    return {
        "name": image_name,
        "clean": checkpoint["clean"].astype(np.float32),
        "noisy": checkpoint["noisy"].astype(np.float32),
        "rof": checkpoint["rof"].astype(np.float32),
        "grad_frobenius": checkpoint["grad_frobenius"].astype(np.float32),
        "rof_time": float(checkpoint["rof_time"]),
        "checkpoint_path": checkpoint_path
    }


def write_preprocessing_progress(completed_images, total_images, image_name, message):

    with open(
        preprocessing_progress_log_file,
        "a",
        encoding="utf-8"
    ) as f:

        timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        f.write(f"TIME: [{timestamp}]\n")
        f.write(f"{message}\n")
        f.write(
            f"Preprocessing status: "
            f"{completed_images}/{total_images}\n"
        )
        f.write(f"Current image: {image_name}\n")
        f.write("======================================\n\n")


##############################################################################
############################ Grid checkpoint helpers #########################
##############################################################################

def latest_grid_checkpoint_path():

    return grid_checkpoint_directory / "latest_checkpoint.npz"


def save_grid_checkpoint(completed_iterations, lpips_sum_ROF_image, lpips_sum_dpROF_image, ssim_sum_ROF, ssim_sum_dpROF, psnr_sum_ROF, psnr_sum_dpROF, time_sum_dpROF, total_time_ROF, noise_technical_sum):

    grid_checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = latest_grid_checkpoint_path()
    temporary_path = checkpoint_path.with_suffix(".tmp.npz")

    np.savez_compressed(
        temporary_path,
        completed_iterations=np.int64(completed_iterations),
        lpips_sum_ROF_image=lpips_sum_ROF_image,
        lpips_sum_dpROF_image=lpips_sum_dpROF_image,
        ssim_sum_ROF=ssim_sum_ROF,
        ssim_sum_dpROF=ssim_sum_dpROF,
        psnr_sum_ROF=psnr_sum_ROF,
        psnr_sum_dpROF=psnr_sum_dpROF,
        time_sum_dpROF=time_sum_dpROF,
        total_time_ROF=np.float64(total_time_ROF),
        noise_technical_sum=np.float64(noise_technical_sum)
    )

    os.replace(temporary_path, checkpoint_path)

    return checkpoint_path


def load_grid_checkpoint():

    if not resume_grid:
        return None

    checkpoint_path = latest_grid_checkpoint_path()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Grid checkpoint not found: {checkpoint_path}\n"
            f"Set resume_grid = False for a new grid run."
        )

    checkpoint = np.load(checkpoint_path)

    return {
        "completed_iterations": int(
            checkpoint["completed_iterations"]
        ),
        "lpips_sum_ROF_image": checkpoint[
            "lpips_sum_ROF_image"
        ],
        "lpips_sum_dpROF_image": checkpoint[
            "lpips_sum_dpROF_image"
        ],
        "ssim_sum_ROF": checkpoint["ssim_sum_ROF"],
        "ssim_sum_dpROF": checkpoint["ssim_sum_dpROF"],
        "psnr_sum_ROF": checkpoint["psnr_sum_ROF"],
        "psnr_sum_dpROF": checkpoint["psnr_sum_dpROF"],
        "time_sum_dpROF": checkpoint["time_sum_dpROF"],
        "total_time_ROF": float(checkpoint["total_time_ROF"]),
        "noise_technical_sum": float(checkpoint["noise_technical_sum"]),
        "checkpoint_path": checkpoint_path
    }


def write_grid_progress(completed_iterations, total_iterations, current_lambda, a_const, ba, image_name, message):

    with open(
        grid_progress_log_file,
        "a",
        encoding="utf-8"
    ) as f:

        timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        f.write(f"TIME: [{timestamp}]\n")
        f.write(f"{message}\n")
        f.write(
            f"Current iteration status: "
            f"{completed_iterations}/{total_iterations}\n"
        )
        f.write(f"saved_per_set: {saved_per_set}\n")
        f.write(
            f"Current parameters: "
            f"lambda={float(current_lambda)}, "
            f"a={float(a_const)}, "
            f"b/a={float(ba)}\n"
        )
        f.write(f"Current image: {image_name}\n")
        f.write("======================================\n\n")


##############################################################################
######################## Dataset file list only ##############################
##############################################################################

if not folder_path.is_dir():
    raise FileNotFoundError(f"Folder not found: {folder_path}")

all_files = sorted(os.listdir(folder_path))

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".ppm",
    ".pgm"
}

image_files = [
    file_name
    for file_name in all_files
    if os.path.splitext(file_name)[1].lower() in image_extensions
]

number_of_images = len(image_files)

print(f"Found {number_of_images} image files in the folder.")
print("Image files:", image_files)

if number_of_images == 0:
    raise ValueError("No image files were found.")

print("Images will be loaded one at a time during the optimization.")


##############################################################################
######################## LPIPS initialisation ################################
##############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net="alex").to(device)
loss_fn.eval()

print(f"LPIPS import successful. Device: {device}")


##############################################################################
############################ Metric arrays ###################################
##############################################################################

metric_shape = (
    len(set_of_lambdas),
    len(set_of_a),
    len(set_of_ba)
)

lpips_technical_sum_ROF_image = np.zeros(
    metric_shape,
    dtype=np.float64
)

lpips_technical_sum_dpROF_image = np.zeros(
    metric_shape,
    dtype=np.float64
)

ssim_sum_ROF = np.zeros(
    metric_shape,
    dtype=np.float64
)

ssim_sum_dpROF = np.zeros(
    metric_shape,
    dtype=np.float64
)

psnr_sum_ROF = np.zeros(
    metric_shape,
    dtype=np.float64
)

psnr_sum_dpROF = np.zeros(
    metric_shape,
    dtype=np.float64
)

time_sum_dpROF = np.zeros(
    metric_shape,
    dtype=np.float64
)

total_time_ROF = 0.0
noise_technical_sum = 0.0


##############################################################################
######################## Load grid checkpoint if requested ###################
##############################################################################

grid_checkpoint_state = load_grid_checkpoint()

if grid_checkpoint_state is not None:

    iterations_number = grid_checkpoint_state[
        "completed_iterations"
    ]

    lpips_technical_sum_ROF_image = (
        grid_checkpoint_state[
            "lpips_sum_ROF_image"
        ]
    )

    lpips_technical_sum_dpROF_image = (
        grid_checkpoint_state[
            "lpips_sum_dpROF_image"
        ]
    )

    ssim_sum_ROF = grid_checkpoint_state[
        "ssim_sum_ROF"
    ]

    ssim_sum_dpROF = grid_checkpoint_state[
        "ssim_sum_dpROF"
    ]

    psnr_sum_ROF = grid_checkpoint_state[
        "psnr_sum_ROF"
    ]

    psnr_sum_dpROF = grid_checkpoint_state[
        "psnr_sum_dpROF"
    ]

    time_sum_dpROF = grid_checkpoint_state[
        "time_sum_dpROF"
    ]

    total_time_ROF = grid_checkpoint_state[
        "total_time_ROF"
    ]

    noise_technical_sum = grid_checkpoint_state[
        "noise_technical_sum"
    ]

    print(
        f"Loaded grid checkpoint: "
        f"{grid_checkpoint_state['checkpoint_path']}"
    )

    print(
        f"Resuming vectorial dpROF grid "
        f"from iteration: {iterations_number}"
    )

else:

    iterations_number = 0

    save_grid_checkpoint(
        0,
        lpips_technical_sum_ROF_image,
        lpips_technical_sum_dpROF_image,
        ssim_sum_ROF,
        ssim_sum_dpROF,
        psnr_sum_ROF,
        psnr_sum_dpROF,
        time_sum_dpROF,
        total_time_ROF,
        noise_technical_sum
    )


##############################################################################
################ Explicit image-first streaming optimization ##################
##############################################################################

parameters_per_image = len(product_set)
total_iterations = number_of_images * parameters_per_image

if iterations_number > total_iterations:

    raise ValueError(
        f"Grid checkpoint iteration "
        f"{iterations_number} exceeds total "
        f"iterations {total_iterations}."
    )

start_image_index = (
    iterations_number // parameters_per_image
    if iterations_number < total_iterations
    else number_of_images
)

start_parameter_index = (
    iterations_number % parameters_per_image
    if iterations_number < total_iterations
    else 0
)

print(f"Total grid iterations: {total_iterations}")
print(
    f"Starting vectorial dpROF grid "
    f"from iteration: {iterations_number}"
)
print(
    f"Each image has {parameters_per_image} "
    f"parameter combinations."
)


##############################################################################
######## One image -> preprocessing -> full grid -> next image ###############
##############################################################################

rng = np.random.default_rng(random_seed)

# Advance the random generator for all fully completed images so that the
# fixed noisy realization of the resume image remains exactly reproducible.

for skipped_image_index in range(start_image_index):

    skipped_path = folder_path / image_files[skipped_image_index]

    skipped_image = io.imread(skipped_path)
    skipped_image = convert_image_to_rgb_float(skipped_image)

    rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=skipped_image.shape
    )

    del skipped_image


last_lambda = None
last_a = None
last_ba = None
last_image_name = "None"

prepared = None

try:

    for image_index in range(start_image_index, number_of_images):

        image_name = image_files[image_index]
        full_path = folder_path / image_name

        clean_image = io.imread(full_path)
        clean_image = convert_image_to_rgb_float(clean_image)

        gaussian_noise = rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=clean_image.shape
        ).astype(np.float32)

        loaded_preprocessing = None

        if resume_preprocessing:

            loaded_preprocessing = load_preprocessing_checkpoint(
                image_index,
                image_name
            )

        if loaded_preprocessing is not None:

            prepared = {
                "name": loaded_preprocessing["name"],
                "clean": loaded_preprocessing["clean"],
                "noisy": loaded_preprocessing["noisy"],
                "rof": loaded_preprocessing["rof"],
                "grad_frobenius": loaded_preprocessing[
                    "grad_frobenius"
                ],
                "rof_time": loaded_preprocessing["rof_time"]
            }

            print(
                f"Loaded vectorial ROF preprocessing checkpoint: "
                f"{image_index + 1}/{number_of_images} - "
                f"{image_name}"
            )

        else:

            noisy_image = np.clip(
                clean_image + gaussian_noise,
                0.0,
                1.0
            ).astype(np.float32)

            print(
                f"Running full vectorial RGB ROF: "
                f"{image_index + 1}/{number_of_images} - "
                f"{image_name}"
            )

            rof_start = time.perf_counter()

            denoised_rof = chambolle_pock_vectorial_rof(
                noisy_image,
                tau=tau_initial,
                lam=lambda_ROF,
                max_iter=max_iter_ROF,
                tol=given_precision_ROF
            )

            rof_time = time.perf_counter() - rof_start

            denoised_rof = np.clip(
                denoised_rof,
                0.0,
                1.0
            ).astype(np.float32)

            # Only one full vectorial ROF is solved. The same result is used
            # for the ROF metrics and for constructing the adaptive weight.

            mollified_rof = mollify_rgb(
                denoised_rof,
                radius=mollifier_radius
            )

            grad_rof = gradient_vectorial(mollified_rof)

            grad_frobenius = frobenius_norm(
                grad_rof
            ).astype(np.float32)

            prepared = {
                "name": image_name,
                "clean": clean_image,
                "noisy": noisy_image,
                "rof": denoised_rof,
                "grad_frobenius": grad_frobenius,
                "rof_time": rof_time
            }

            if resume_preprocessing:

                preprocessing_path = save_preprocessing_checkpoint(
                    image_index,
                    image_name,
                    prepared,
                    rof_time
                )

                write_preprocessing_progress(
                    image_index + 1,
                    number_of_images,
                    image_name,
                    message=(
                        "Vectorial ROF preprocessing checkpoint saved: "
                        f"{preprocessing_path.name}"
                    )
                )

                print(
                    f"Saved vectorial ROF preprocessing checkpoint: "
                    f"{image_index + 1}/{number_of_images} - "
                    f"{image_name}"
                )

        parameter_start = (
            start_parameter_index
            if image_index == start_image_index
            else 0
        )

        # Image-level quantities are added only when the grid for this image
        # starts from its first parameter combination.

        if parameter_start == 0:

            total_time_ROF += prepared["rof_time"]

            noise_technical_sum += float(
                np.mean(
                    (
                        prepared["clean"]
                        - prepared["noisy"]
                    )**2
                )
            ) / number_of_images

        print(
            f"Running complete vectorial dpROF grid for image "
            f"{image_index + 1}/{number_of_images} - {image_name} "
            f"from parameter {parameter_start + 1}/"
            f"{parameters_per_image}"
        )

        for parameter_index in range(
            parameter_start,
            parameters_per_image
        ):

            current_lambda, a_const, ba = product_set[parameter_index]

            last_lambda = current_lambda
            last_a = a_const
            last_ba = ba
            last_image_name = image_name

            clean_image = prepared["clean"]
            noisy_image = prepared["noisy"]
            denoised_rof = prepared["rof"]
            grad_frobenius = prepared["grad_frobenius"]

            b_const = a_const * ba

            a_weight = np.maximum(
                0.0,
                a_const
                - b_const
                * np.maximum(
                    grad_frobenius,
                    a_const / (2.0 * b_const)
                )
            ).astype(np.float32)

            dprof_start = time.perf_counter()

            denoised_dprof = chambolle_pock_vectorial_dprof(
                noisy_image,
                a_weight,
                tau=tau_initial,
                lam=current_lambda,
                max_iter=max_iter_dpROF,
                tol=given_precision_dpROF
            )

            time_dprof = time.perf_counter() - dprof_start

            denoised_dprof = np.clip(
                denoised_dprof,
                0.0,
                1.0
            ).astype(np.float32)

            lambdas_index = set_of_lambdas.index(current_lambda)
            a_index = set_of_a.index(a_const)
            b_index = set_of_ba.index(ba)

            clean_lpips = conversion_for_lpips(clean_image)
            rof_lpips = conversion_for_lpips(denoised_rof)
            dprof_lpips = conversion_for_lpips(denoised_dprof)

            with torch.no_grad():

                distance = loss_fn(
                    clean_lpips,
                    rof_lpips
                )

            lpips_technical_sum_ROF_image[
                lambdas_index,
                a_index,
                b_index
            ] += distance.item() / number_of_images

            with torch.no_grad():

                distance = loss_fn(
                    clean_lpips,
                    dprof_lpips
                )

            lpips_technical_sum_dpROF_image[
                lambdas_index,
                a_index,
                b_index
            ] += distance.item() / number_of_images

            ssim_value = ssim(
                clean_image,
                denoised_rof,
                data_range=1.0,
                channel_axis=-1
            )

            psnr_value = psnr(
                clean_image,
                denoised_rof,
                data_range=1.0
            )

            ssim_sum_ROF[
                lambdas_index,
                a_index,
                b_index
            ] += float(ssim_value) / number_of_images

            psnr_sum_ROF[
                lambdas_index,
                a_index,
                b_index
            ] += float(psnr_value) / number_of_images

            ssim_value = ssim(
                clean_image,
                denoised_dprof,
                data_range=1.0,
                channel_axis=-1
            )

            psnr_value = psnr(
                clean_image,
                denoised_dprof,
                data_range=1.0
            )

            ssim_sum_dpROF[
                lambdas_index,
                a_index,
                b_index
            ] += float(ssim_value) / number_of_images

            psnr_sum_dpROF[
                lambdas_index,
                a_index,
                b_index
            ] += float(psnr_value) / number_of_images

            time_sum_dpROF[
                lambdas_index,
                a_index,
                b_index
            ] += time_dprof / number_of_images

            iterations_number = (
                image_index * parameters_per_image
                + parameter_index
                + 1
            )

            if (
                iterations_number % saved_per_set == 0
                or iterations_number == total_iterations
            ):

                checkpoint_path = save_grid_checkpoint(
                    iterations_number,
                    lpips_technical_sum_ROF_image,
                    lpips_technical_sum_dpROF_image,
                    ssim_sum_ROF,
                    ssim_sum_dpROF,
                    psnr_sum_ROF,
                    psnr_sum_dpROF,
                    time_sum_dpROF,
                    total_time_ROF,
                    noise_technical_sum
                )

                write_grid_progress(
                    iterations_number,
                    total_iterations,
                    current_lambda,
                    a_const,
                    ba,
                    image_name,
                    message=(
                        f"Grid checkpoint saved: "
                        f"{checkpoint_path.name}"
                    )
                )

                print(
                    f"Grid checkpoint saved for image "
                    f"{image_index + 1}/{number_of_images}, "
                    f"parameter {parameter_index + 1}/"
                    f"{parameters_per_image} "
                    f"({iterations_number}/{total_iterations} "
                    f"total iterations)."
                )

        print(
            f"Completed complete vectorial dpROF grid for image "
            f"{image_index + 1}/{number_of_images} - {image_name}"
        )

        del prepared
        prepared = None

        start_parameter_index = 0

except KeyboardInterrupt:

    write_grid_progress(
        iterations_number,
        total_iterations,
        last_lambda,
        last_a,
        last_ba,
        last_image_name,
        message=(
            "Execution interrupted by user. "
            "Resume from the latest grid checkpoint."
        )
    )

    print()
    print(
        f"Execution interrupted at vectorial dpROF iteration "
        f"{iterations_number}/{total_iterations}."
    )
    print(
        f"See grid progress in: "
        f"{grid_progress_log_file}"
    )

    raise

except Exception:

    write_grid_progress(
        iterations_number,
        total_iterations,
        last_lambda,
        last_a,
        last_ba,
        last_image_name,
        message=(
            "Execution stopped because of an exception. "
            "Resume from the latest grid checkpoint."
        )
    )

    with open(
        grid_progress_log_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(traceback.format_exc())
        f.write(
            "\n======================================\n\n"
        )

    print()
    print(
        f"Execution stopped at vectorial dpROF iteration "
        f"{iterations_number}/{total_iterations}."
    )
    print(
        f"See grid progress and traceback in: "
        f"{grid_progress_log_file}"
    )

    raise

finally:

    if prepared is not None:
        del prepared


##############################################################################
############################ Final Outputs ###################################
##############################################################################

average_time_ROF = (
    total_time_ROF
    / number_of_images
)

average_total_time_dpROF = float(
    np.mean(time_sum_dpROF)
)

print()
print(product_set_print)
print()
print(
    f"Average noise variance: "
    f"{noise_technical_sum:.8f}"
)
print()
print(
    f"Average LPIPS, ROF-image:\n"
    f"{np.round(lpips_technical_sum_ROF_image, 6)}"
)
print()
print(
    f"Average LPIPS, dpROF-image:\n"
    f"{np.round(lpips_technical_sum_dpROF_image, 6)}"
)
print()
print(
    f"Average SSIM, ROF:\n"
    f"{np.round(ssim_sum_ROF, 6)}"
)
print()
print(
    f"Average SSIM, dpROF:\n"
    f"{np.round(ssim_sum_dpROF, 6)}"
)
print()
print(
    f"Average PSNR, ROF:\n"
    f"{np.round(psnr_sum_ROF, 4)}"
)
print()
print(
    f"Average PSNR, dpROF:\n"
    f"{np.round(psnr_sum_dpROF, 4)}"
)
print()
print(
    f"Average total vectorial ROF time "
    f"per image: {average_time_ROF:.6f}s"
)
print(
    f"Average total vectorial dpROF time "
    f"per image: {average_total_time_dpROF:.6f}s"
)
print()


def best_for_lpips_dpROF(params):

    current_lambda, a_const, ba = params

    lambdas_index = set_of_lambdas.index(
        current_lambda
    )
    a_index = set_of_a.index(
        a_const
    )
    b_index = set_of_ba.index(
        ba
    )

    return lpips_technical_sum_dpROF_image[
        lambdas_index,
        a_index,
        b_index
    ]


def best_for_ssim_dpROF(params):

    current_lambda, a_const, ba = params

    lambdas_index = set_of_lambdas.index(
        current_lambda
    )
    a_index = set_of_a.index(
        a_const
    )
    b_index = set_of_ba.index(
        ba
    )

    return ssim_sum_dpROF[
        lambdas_index,
        a_index,
        b_index
    ]


def best_for_psnr_dpROF(params):

    current_lambda, a_const, ba = params

    lambdas_index = set_of_lambdas.index(
        current_lambda
    )
    a_index = set_of_a.index(
        a_const
    )
    b_index = set_of_ba.index(
        ba
    )

    return psnr_sum_dpROF[
        lambdas_index,
        a_index,
        b_index
    ]


best_lpips = min(
    product_set,
    key=best_for_lpips_dpROF
)

best_ssim = max(
    product_set,
    key=best_for_ssim_dpROF
)

best_psnr = max(
    product_set,
    key=best_for_psnr_dpROF
)

best_lpips_value = best_for_lpips_dpROF(
    best_lpips
)

best_ssim_value = best_for_ssim_dpROF(
    best_ssim
)

best_psnr_value = best_for_psnr_dpROF(
    best_psnr
)

best_lpips_print = (
    round(float(best_lpips[0]), 6),
    float(best_lpips[1]),
    float(best_lpips[2])
)

best_ssim_print = (
    round(float(best_ssim[0]), 6),
    float(best_ssim[1]),
    float(best_ssim[2])
)

best_psnr_print = (
    round(float(best_psnr[0]), 6),
    float(best_psnr[1]),
    float(best_psnr[2])
)

print(
    f"Optimal parameters using LPIPS: "
    f"{best_lpips_print}"
)

print(
    f"Optimal value of LPIPS: "
    f"{best_lpips_value:.6f}"
)

print()

print(
    f"Optimal parameters using SSIM: "
    f"{best_ssim_print}"
)

print(
    f"Optimal value of SSIM: "
    f"{best_ssim_value:.6f}"
)

print()

print(
    f"Optimal parameters using PSNR: "
    f"{best_psnr_print}"
)

print(
    f"Optimal value of PSNR: "
    f"{best_psnr_value:.4f}"
)

print()


##############################################################################
######################## Add outputs to log file #############################
##############################################################################

with open(
    log_file,
    "a",
    encoding="utf-8"
) as f:

    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    f.write(f"TIME: [{timestamp}]\n\n")

    f.write(
        f"Current iteration status: "
        f"{iterations_number}/"
        f"{total_iterations}\n"
    )

    f.write(
        f"saved_per_set: "
        f"{saved_per_set}\n\n"
    )

    f.write(
        f"Prescribed noise standard deviation: "
        f"{float(noise_sigma)}\n"
    )

    f.write(
        f"Prescribed noise variance: "
        f"{float(given_variance)}\n"
    )

    f.write(
        f"Estimated average noise variance: "
        f"{noise_technical_sum:.8f}\n\n"
    )

    f.write(
        f"Full vectorial ROF lambda: "
        f"{float(lambda_ROF)}\n"
    )

    f.write(
        f"Considered dpROF lambdas: "
        f"{set_of_lambdas_print}\n"
    )

    f.write(
        f"Considered a: "
        f"{set_of_a}\n"
    )

    f.write(
        f"Considered b/a: "
        f"{set_of_ba}\n\n"
    )

    f.write(
        f"Average total vectorial ROF time "
        f"per image: "
        f"{average_time_ROF:.6f}s\n"
    )

    f.write(
        f"Average total vectorial dpROF time "
        f"per image: "
        f"{average_total_time_dpROF:.6f}s\n\n"
    )

    f.write(
        f"Optimal parameters using LPIPS: "
        f"{best_lpips_print}\n"
    )

    f.write(
        f"Optimal value of LPIPS: "
        f"{best_lpips_value:.6f}\n\n"
    )

    f.write(
        f"Optimal parameters using SSIM: "
        f"{best_ssim_print}\n"
    )

    f.write(
        f"Optimal value of SSIM: "
        f"{best_ssim_value:.6f}\n\n"
    )

    f.write(
        f"Optimal parameters using PSNR: "
        f"{best_psnr_print}\n"
    )

    f.write(
        f"Optimal value of PSNR: "
        f"{best_psnr_value:.4f}\n\n"
    )

    f.write(
        "======================================\n\n"
    )

print(
    f"Successfully added to "
    f"{log_file}"
)


##############################################################################
################ Add extended outputs to supplemental log ####################
##############################################################################

with open(
    extended_log_file,
    "a",
    encoding="utf-8"
) as f:

    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    f.write(f"TIME: [{timestamp}]\n\n")

    f.write(
        f"Current iteration status: "
        f"{iterations_number}/"
        f"{total_iterations}\n"
    )

    f.write(
        f"saved_per_set: "
        f"{saved_per_set}\n\n"
    )

    f.write(
        f"Parameter product set: "
        f"{product_set_print}\n\n"
    )

    f.write(
        f"Average noise variance: "
        f"{noise_technical_sum:.8f}\n\n"
    )

    f.write(
        f"Average LPIPS, ROF-image: "
        f"{lpips_technical_sum_ROF_image}\n\n"
    )

    f.write(
        f"Average LPIPS, dpROF-image: "
        f"{lpips_technical_sum_dpROF_image}\n\n"
    )

    f.write(
        f"Average SSIM, ROF: "
        f"{ssim_sum_ROF}\n\n"
    )

    f.write(
        f"Average SSIM, dpROF: "
        f"{ssim_sum_dpROF}\n\n"
    )

    f.write(
        f"Average PSNR, ROF: "
        f"{psnr_sum_ROF}\n\n"
    )

    f.write(
        f"Average PSNR, dpROF: "
        f"{psnr_sum_dpROF}\n\n"
    )

    f.write(
        f"Average total vectorial ROF time "
        f"per image: "
        f"{average_time_ROF:.6f}s\n"
    )

    f.write(
        f"Average total vectorial dpROF time "
        f"per image: "
        f"{average_total_time_dpROF:.6f}s\n\n"
    )

    f.write(
        f"Prescribed noise standard deviation: "
        f"{float(noise_sigma)}\n"
    )

    f.write(
        f"Prescribed noise variance: "
        f"{float(given_variance)}\n"
    )

    f.write(
        f"Estimated average noise variance: "
        f"{noise_technical_sum:.8f}\n\n"
    )

    f.write(
        f"Full vectorial ROF lambda: "
        f"{float(lambda_ROF)}\n"
    )

    f.write(
        f"Considered dpROF lambdas: "
        f"{set_of_lambdas_print}\n"
    )

    f.write(
        f"Considered a: "
        f"{set_of_a}\n"
    )

    f.write(
        f"Considered b/a: "
        f"{set_of_ba}\n\n"
    )

    f.write(
        f"Optimal parameters using LPIPS: "
        f"{best_lpips_print}\n"
    )

    f.write(
        f"Optimal value of LPIPS: "
        f"{best_lpips_value:.6f}\n\n"
    )

    f.write(
        f"Optimal parameters using SSIM: "
        f"{best_ssim_print}\n"
    )

    f.write(
        f"Optimal value of SSIM: "
        f"{best_ssim_value:.6f}\n\n"
    )

    f.write(
        f"Optimal parameters using PSNR: "
        f"{best_psnr_print}\n"
    )

    f.write(
        f"Optimal value of PSNR: "
        f"{best_psnr_value:.4f}\n\n"
    )

    f.write(
        "======================================\n\n"
    )

print(
    f"Successfully added to "
    f"{extended_log_file}"
)

print("Done.")
