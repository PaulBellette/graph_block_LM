# Adam-metric curvature correction

It is impossible to practically get Levenberg-Marquardt to work on a decent sized neural network due to the size of the Jacobian. This repo is trying to get something LM-like work on a neural net (based on the ideas of Gower-Richtárik sketch-and-project).

Instead:

* use Adam for the base step
* add a small curvature correction
* get the correction from a sketch
* keep the sketch coherent
* reject bad corrections with a fresh batch

## Motivation

Levenberg-Marquardt is attractive, it can make large moves based on the curvature (if it trusts it).

For a least-squares problem it solves

$$
(J^\top J + \lambda I)\delta = -J^\top r.
$$

For neural nets this is not practical, the parameter space is way too large. Also the batch curvature is noisy and the solve is expensive.

The stochastic setting also changes the problem, a minibatch Newton step is not an unbiased Newton step.

$$
\mathbb{E}[(H_B+\lambda I)^{-1}g_B] \neq (H+\lambda I)^{-1}g.
$$

This changed the goal to something more modest. This work landed on using a small projected problem and then  subsequently use it as a correction to Adam.

## This idea

Adam already gives a good scaled direction.

Let

$$
D = \sqrt{\hat v} + \epsilon.
$$

The Adam-like base step is

$$
\delta_A = -\eta \frac{\hat m}{D}.
$$

In the experiments here,

$$
\eta \approx \frac{1}{\lambda}.
$$

Then build a small correction space. Use a rank (k) Krylov-like basis, the first direction is the Adam direction amd the next directions come from local curvature.

In whitened coordinates:

$$
K = D^{-1/2} J^\top J D^{-1/2}.
$$

A rank-2 version is roughly

$$
U = [u_0, Ku_0].
$$

Then solve a tiny damped problem in that subspace.

$$
(U^\top H U + \lambda_c I)z = -U^\top(g + H\delta_A)
$$

The final candidate step is

$$
\delta = \delta_A + Uz.
$$

The correction is clipped and gated.

If the candidate step improves a fresh training batch, accept it otherwise use the Adam step.

## Insights

The curvature object must be coherent, this mattered more than expected.

Bad version:

* build the basis from batch A
* solve coefficients on batch B

This was much worsem, it made the correction off-policy. The directions came from one stochastic objective and the coefficients came from another.

Good version:

* build the basis from the same accumulated curvature used for the solve
* include the Adam batch
* add one extra curvature batch

This worked better.

The useful setting so far is:

```bash
--mode adam_curv
--correction_rank 2
--correction_accum_batches 2
--correction_include_base_batch 1
--acceptance_mode fresh_train
```

Interestingly one extra batch helped and too many extra batches seemed to not help. I do niot quite understand this yet. I think it is something like;

The correction is not a full population Newton step, it is norm bounded correction to an Adam trajectory and too much averaging can wash out that coupling.

## What worked

On the squared-logit Fashion-MNIST:

* Adam is good, as expected.
* The curvature correction is competitive.
* On seed 0, long runs found a better validation envelope than Adam.
* Across a small seed sweep, the effect was small.
* The method was not a blowout.
* It was also probably not a fluke.

This mothed is scalable to neural network sized problems but it is still a little slow. But it ias still cool that a coherent low-rank curvature correction can get similar performance as Adam.

## Cross entropy check

The cross-entropy version also seems to workish. For CE the curvature is not the squared residual curvature, it uses the softmax CE generalized Gauss-Newton / Fisher form:

$$
J^\top(\mathrm{diag}(p)-pp^\top)J.
$$

This is all new and still to be investigated so take any preliminary results with a grain of salt. It is competitive with Adam on best validation CE loss, but both methods overfit more than the squared loss.

The CE version probably needs ordinary regularisation etc.

Likely moves:

* weight decay
* label smoothing
* lower learning rate
* smaller correction cap
* correction decay or stop
* data augmentation

## Dead ends

### Plain block LM

The first version tried tiled LM updates instead of sketching.

It converged but it mostly became a slow block-coordinate optimiser. The block visitation pattern seemed to matter too much.

It did not get close to Adam.

### Diagonal damping without the right metric

Adding Adam’s second moment only as a damping term was not enough. If the metric is Adam-like, the solver must also use that metric otherwise one CG step starts in the wrong geometry.

The useful fix was preconditioned CG.

With one PCG step, the method recovers the Adam-like direction.

### More CG

Running more CG iterations was not better. I guess because the batch curvature was too local.

Interestingly the correction moved away from the Adam direction.

This was an important negative result.

### Random parameter sketches

Blind random low-rank sketches were weak, they captured too little of the useful direction.

Unbiasing fixed scale, but it did not fix alignment.

Adding the Adam direction to the sketch helped.

Random correction directions helped only a little.

### Krylov-only correction

Krylov directions were more meaningful than random directions, but using them as the whole step was too aggressive.

This led to the final form

* Adam is the base step
* curvature provides a bounded correction

### Clever row batches

I tried the following idea "If we sketch parameters, why not sketch data rows too?"

This involved residual and leverage-based row selection. But alas, it didn't help.

The likely reason:

* row selection made the batch more informative
* but not more useful for generalisation
* and sometimes broke basis/solve coherence

## Current script

The attached script keeps only the final useful path.

Modes:

```bash
--mode adam
--mode adam_curv
```

Objectives:

```bash
--objective mse
--objective ce
```

The main experimental script is:

```bash
uv run graph_block_lm_experiment.py \
  --device cuda \
  --problem fashion_mnist \
  --objective mse \
  --mode adam_curv \
  --steps 6000 \
  --batch_size 64 \
  --conv_channels 16 \
  --damping 300 \
  --correction_rank 2 \
  --correction_damping_mult 10 \
  --correction_max_norm_frac 0.25 \
  --correction_accum_batches 2 \
  --correction_accum_batch_size 64 \
  --correction_include_base_batch 1 \
  --acceptance_mode fresh_train \
  --acceptance_batch_size 256
```

For CE:

```bash
uv run graph_block_lm_experiment.py \
  --device cuda \
  --problem fashion_mnist \
  --objective ce \
  --mode adam_curv \
  --steps 2000 \
  --batch_size 64 \
  --conv_channels 16 \
  --damping 300 \
  --correction_rank 2 \
  --correction_damping_mult 10 \
  --correction_max_norm_frac 0.25 \
  --correction_accum_batches 2 \
  --correction_accum_batch_size 64 \
  --correction_include_base_batch 1 \
  --acceptance_mode fresh_train \
  --acceptance_batch_size 256
```

## Current belief

So nothing amazing yet, but it does appear like it is plausible to smuggle in LM steps on neural Network sized problems, however the following tricks were necessary:

* Adam-metric
* low rank
* coherent across basis and solve
* bounded
* gated on fresh data
