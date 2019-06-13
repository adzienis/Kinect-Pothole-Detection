# Kinect Pothole Detection

## Arthur Dzieniszewski & William Mitchell

Final project created for CMU's spring 15-418,
Parallel Computer Architecture & Programming course.

Primary goal of the project was to investigate the usage of MPI,
specifically OpenMPI, in the context of a real world setting. Our real
world setting is composed by trying to identify potholes given the
depth image from a Kinect V1 (Xbox 360), and then clustering points
corresponding to particular potholes. Originally,
the goal of the project was to utilize Raspberry Pi's in a scatter-gather
pattern to speed up our implementation of k-means using Lloyd's Method.
Logistically, we found this difficult to implement since we had to face issues
outside of the project scope (power, wire connections, ethernet controllers).
Instead, we focused on the analysis of an MPI implementation on one of our laptops
utilizing an Intel Core i5-4200M, with 2 physical cores, each hyperthreaded. We ran
tests using our identical implementation against other k-means implementations, with
results included in our report.

Our implementation of k-means is written in C, with a wrapper in Python for us to communicate
with the workers.

Our findings and a more in-depth explanation are found in our poster file:

[Poster](https://drive.google.com/file/d/1OULftGTT2nOWm3ECtMFWzxOkLVPzsTF-/view?usp=sharing)

Some detections:
![](https://i.imgur.com/XG3y9Hz.png)

![](https://i.imgur.com/s1JYtvq.png)

In addition, a video highlighting some of our real life evaluations is provided:

[Video](https://drive.google.com/open?id=1k-rRtn9M_zvdkkj7QfxSN9mWZmkKB762)

Built off work from an MPI implementation of kmeans.c designed for another purpose.
We added the capability to handle work not divisible by processors, communication
with a master Python process, improved scatter and gather uses, and some OpenMP pragmas such
as explicit vectorization (still done regardless with -O3) [link][https://github.com/rexdwyer/MPI-K-means-clustering]
