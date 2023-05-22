# SolarGAN
 Generating synthetic urban solar irradiance time-series via Deep Generative Networks

Data can be shared upon request.

(20.05.2023) 4 notebooks with some necessary utility functions uploaded:
fisheye_image_processing: Transform fisheye renderings into grayscale cubemap images as input for the image encorder.
att_processing: load the image encoder to compress cubemap images into a vector (dim = 32), draw weather statistics & sensor-point location information as another vector (Table 2), and concat. both as input attribute vector.
gen_time_series: load the time series generator and draw time series samples according to the input attribute vector.
gen_parametric: generate images when traversing latent dim. associated to WWR (might need to change necessary path varaibles)  
