# Adaptive double-phase ROF model

We present a new image denoising algorithm called the adaptive double-phase ROF model. We describe the results of the numerical experiments and present the source files for these experiments.

# List of files

1D_supplement.pdf: The file describing briefly the model and presenting the results of numerical experiments in one dimension, comparing several image denoising methods, i.e., classical ROF, Huber-ROF and adaptive double-phase ROF.

2D_supplement_part1.pdf: The file presenting the results of numerical experiments in two dimensions (part 1).

2D_supplement_part2.pdf: The file presenting the results of numerical experiments in two dimensions (part 2).

1D_acceleration.py: The code comparing the results of accelerated and non-accelerated Chambolle-Pock algorithm for the tested models (in one dimension).

1D_l2_distance_plots.py: The code for presenting the comparison between the tested models as a function of the L2 distance from the noisy datum (in one dimension).

1D_lambdas.py: The code for presenting the comparison between the tested models as a function of the parameter lambda (in one dimension).

1D_noise.py: The code for presenting the comparison between the tested models for different noise levels (in one dimension).

1D_pictures.py: The code for presenting visually the results of the tested models in one dimension.

1D_radii.py: The code for presenting the comparison between the tested models for different mollification radii for the weight (in one dimension).

1D_tolerance.py: The code for presenting the comparison between the tested models for different stop parameters (in one dimension).

1D_weights3.py: The code for presenting the comparison between the tested models for different weights (in one dimension, Weight 3).

1D_weights12.py: The code for presenting the comparison between the tested models for different weights (in one dimension, Weights 1 and 2).

2D_visual_results.py: The code for presenting visually the results of the tested models in two dimensions and construction of the weight in the adaptive double-phase ROF model (with increasing parameter lambda).

2D_weight_plots.py: The code for presenting visually the results of the tested models in two dimensions and construction of the weight in the adaptive double-phase ROF model (with single parameter lambda).

2D_metrics_image_version.py: The code for comparing the values of SSIM and PSNR in two dimensions for the tested models.

cut1.jpg, cut2.png, cut3.jpg: one-dimensional images used in the supplements.

LICENSE.txt: The license file (Apache License 2.0 for the codes; CC BY 4.0 for the description of numerical experiments).
