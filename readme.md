<h2>Georgia Tech : CS-7641 - Machine Learning : Spring 2020 </h1>
<h2>Assignment 3 - Clustering and Dimensionality Reduction</h3>

<h3>Running the code</h3>
<p>
Running main.py will run the analysis that forms the bulk of this assignment. Output consists of plots (pasted into the report) and run stats.


Everything is kicked off from main.py. Please change the function main() at the bottom of the file. You can choose to run any of the functions from main(). All function calls have commented out other than the call to the function that runs the neural network on a PCA reduced dataset that includes cluster labels.

</p>
<h3>Code Environment</h3>
<p>
This code was developed on a mac, using VSCode, and python 3.7.6. Where file paths are involved, you might need to tweak things just a bit based on how you run it, and your file system. All of that is possible within main.py

</p>
<h3>Dependencies</h3>
<ul>
<li>numpy
<li>matplotlib
<li>sklearn(0.22)
<li>PyTorch (1.3.1)
<li>mlrose_hiive
</ul>

<h3>Code Organization</h3>
<p>
The various Solver and Algorithm classes implement all the code required by this assignment. Code for Assignment 1, 2 has been left in.

The files to review are:
The main files to review are main.py, BaseLearner.py. KMeansClusterer.py and GMMClusterer.py.

<ul>
	<li>main.py
	<li>BaseLearner.py
	<li>KMeansClusterer.py
	<li>GMMClusterer.py
</ul>


Apart from these pieces, there are a number of utility files for ancillary operations like plotting, timing, etc which have little bearing on the main purpose of this project, but help me profile and debug. This project builds on code that I developed during CS-7642 during  Fall '19.


</p>


