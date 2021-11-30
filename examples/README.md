
##af_accel_gan_example
###Filename
af_accel_gan_example.py
###Capabilities
* Load models
* Generate data
* Visualize generated data
### How to use
#### Simple use
enter "python af_accel_gan_example.py" or "./af_accel_gan_example"
#### Not simple use
Call the function or script as before with the following additional arguments:

* model_path: path to desired model
  * model path should lead to model folder; i.e. ./conditional_af_accel_generator_v3/
* input_paths: paths to noise and label data respectively separated by space
  * paths must lead to .npy or .npz files as loading function uses numpy.load()
* --output_path: Optional path for program to output predictions to

Example:

python af_accel_gan_example.py "my_model/" "foo_noise.npy" "bar_labels.npy" --output "output.npy"
#### Extra info
Input data:

Latent dimension composed of array of shape (any positive size x 128)

Class labels composed of array of shape (same positive size x 4)
* Class label only need if conditional generator, values for class labels can be 0 or 1
<!--* latent dimension: ($, &)
  * $ = any positive integer
  * & = a 128 point vector of decimals of value [0, 1]
* class labels: ($, [_, _, _, _])
  * $ = 0 or 1

** Class labels only needed for conditional generator->

