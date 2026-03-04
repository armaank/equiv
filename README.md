# constraints

Experiments w/ equivariance constraints/soft inductive biases and ml algorithms. Also an exercise in learning Jax

Based on talk from Dr. Andrew Gordon Wilson from June 2025 at LoG NYC meetup.


## Main ideas
- Embrace an expansive hypothesis space w/ softer inductive biases instead of hard restrictions
- Compressability of solution space goes hand-in-hand with model size (larger models can have more 'efficient' solution spaces than smaller models). This is what makes neural nets unique
-  'Occam's Razor' Bias
- Good q: should we use a different model if we have less/more data? (ans: no, b/c do we think that the data generation process is dependent on the number of samples? obv not)
- For example: in CNNs, instead of explicit parameter sharing that enforces transliation invariances, use a softer restriction that encourages
translation parameter sharing but not a hard restriction


### Papers & References

https://arxiv.org/pdf/2503.02113

https://arxiv.org/pdf/2304.05366 

Very interesting blog post on an exchange between Randford Neal, Andrew Gelman and David MacKay on this topic: https://statmodeling.stat.columbia.edu/2011/12/04/david-mackay-and-occams-razor/

