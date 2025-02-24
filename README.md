# wPRF-Collision-Attack

This repository contains Python code designed to simulate and test a key-recovery attack on the standard One-to-One wPRF introduced by Alamati et al. The attack operates by iteratively refining a guessed key K through collision finding and exhaustive search. It also incorporates theoretical models to determine the optimal transition point between these phases, aiming to minimize overall complexity.

The program requires Python 3 and the numpy library.

To run the program, execute the script in your terminal or preferred Python environment. You will be prompted to input a security parameter L, which determines the size of the problem space.  One experiment will have complexity on the order of log_2(L)*2^{L/2}.  The program then conducts 100 independent experiments, measuring the complexities of the collision finding and exhaustive search phases, as well as the overall efficiency of the attack.

For each experiment, the program reports:
- The number of collisions required to recover the key.
- The complexities of both the collision finding and exhaustive search steps.
- Whether the theoretical maximum Hamming distance bound was sufficient to recover the key.
- At the end of the execution, average results across all experiments are displayed, providing insights into the overall performance and success of the attack.
  
The code is written to be adaptable for various experimental setups. You can modify the value of L and the number of experiments by altering the num_experiments variable in the main() function. Intermediate results and detailed outputs are printed during the execution to allow you to monitor the progress of the attack.

## Licence
The Collision Attack is licensed under the MIT License.

* MIT license ([LICENSE](../LICENSE) or http://opensource.org/licenses/MIT)
