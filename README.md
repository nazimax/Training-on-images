



requirement basic knowledge in Data.Analysis "KMeans Algorithm" we will need from sklearn library :
			* "cluster" because KMeans is clustering algorithm.
			* " shuffl" to get a random order of pixels in array
			* "metrics " to choose one metric also w need numpay and image finally the time library to calculate the time execution of each way which renders a specific figure


Goal : *** Get a non exponential algorithm to calculate number of distinct color of an image
		** Preserve a good quality of  4K image with a very low number of colors (less than hundred instead of millions)
		* Image Compression

note : the current algorithm (commit "cf6a3ac79b0a6649b74b52a8c98e778232510c09") which calculate number of colors gives the result of "doe.jpg" in 35 min
adnd it was about 30835