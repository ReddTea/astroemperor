# FAQ

## Why Emperor?


## What is APT?


## How many steps?


## How many walkers?

- At least double the dimensions.
- I recommend a multiple of the threads you are using to minimise idle-core time.
- I like to scale them as an exponential of ndim.
- You can never go wrong with more
- More walkers means more concurrent memory usage.


## How many temperatures?

- Really problem dependant.
- At least 5, more than 20 seems overkill.
