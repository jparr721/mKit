# mKit Library v0.1.0
The MKit library is a series of python classes dedicated to machine learning algorithms and their "from scratch" implementation. This should not be used as a formal replacement for the already extremely well done libraries of scikit-learn, tensorflow, etc. These are simply my explorations into the how the underlying math behind this stuff works and should serve only as a reference.

## Features
Most things are implemented in python but if time allows I will do some paralell stuff in C++/Cuda to optimize performance of some of the algorithms. Ideally I would get some stuff performing comprable to services like pytorch etc.

### Compiling
For the C++ section compilation should be pretty straight forward. If you download this repo and want to run the tests, do the following:

`$ cd cc/`

`$ mkdir build/`

`$ cd build`

`$ cmake ..`

Then you simply run the executable. If something breaks feel free to post an issue. Most things here will be pretty much only run on Arch Linux so I cannot guarantee backwards compatibility.

Happy hacking!
