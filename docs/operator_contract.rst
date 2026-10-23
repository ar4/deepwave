Forward operators and their derivatives
=======================================

This page derives the shared composition of a Deepwave calculation. It separates
physical source quantities, discrete tensor coordinates, time resampling and
differentiation. The family-specific equations are on the propagator pages;
:doc:`units` defines source and receiver quantities, and :doc:`discretisation`
gives the interior stencils. These are complementary specifications.

Symbols, units and scope
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 48 34

   * - Symbol
     - Meaning
     - Dimensions
   * - :math:`\Delta t`
     - Public source/receiver sample interval, ``dt``
     - Time
   * - :math:`R`, :math:`h=\Delta t/R`
     - CFL ``step_ratio`` and actual internal time step
     - Integer count; time
   * - :math:`m`
     - ``model_gradient_sampling_interval``
     - Integer count of public samples
   * - :math:`\Delta x_i`
     - Grid spacing along spatial axis :math:`i`
     - Length
   * - :math:`q`, :math:`\theta`
     - User material parameters and the prepared material coefficients
     - Parameter dependent: speed, density, stiffness, etc.
   * - :math:`x_n`
     - Complete propagation state before internal step :math:`n`, including
       auxiliary memory
     - Each component has its own field units
   * - :math:`s_n`, :math:`y_n`
     - Internal-rate source coordinates and receiver samples
     - Source density or rate; measured field (see :doc:`units`)
   * - :math:`f`, :math:`\omega=2\pi f`
     - Cyclic and angular temporal frequency
     - Cycles per time; radians per time

Floating source/receiver locations supplied to propagators are in **cell index
coordinates**, including the documented component offsets; they are not
physical lengths. Convert physical coordinates to cell indices using each
axis's grid spacing and the model origin. Integer locations are discrete
choices, not differentiable real
variables. Position derivatives through Hicks weights are piecewise derivatives;
stencil changes and reflecting-face restrictions matter.

The derivation below holds with the numerical configuration fixed: extraction
region, grid, time step, PML profiles, boundary choice and source/receiver
support. An inferred maximum speed or source-spectrum peak is a setup decision,
not a differentiable physical parameter. :ref:`gradient-guarantees` describes
those qualifications. Arbitrary callback edits are not part of the unmodified
operator considered here.

One step and its transpose
--------------------------

At fixed material and configuration, all retained propagators are linear in
sources and initial states. Include both scalar time levels, or the staggered
velocity/stress levels, and all PML/attenuation memory in :math:`x_n`. One
internal step and its recording can then be written

.. math::

   x_{n+1} &= A(\theta)x_n+B(\theta)s_n, \\
   y_n &= C(\theta)x_n+D(\theta)s_n.

The matrices here denote operators; the implementation need not construct them.
The :math:`D` term allows a recording made after a source-dependent sub-update.
It must not be discarded merely because another channel records before that
update. Spatial and temporal staggering are part of these operators, not extra
shifts to be guessed from a receiver name.

Let :math:`J` be a scalar loss, :math:`r_n=\partial J/\partial y_n`, and
:math:`\lambda_N=\partial J/\partial x_N`. Differentiating the two equations,
substituting into :math:`dJ`, and collecting the coefficients of :math:`dx_n`
and :math:`ds_n` gives the reverse recurrence

.. math::

   \lambda_n &= A^T\lambda_{n+1}+C^T r_n, \\
   \frac{\partial J}{\partial s_n} &= B^T\lambda_{n+1}+D^T r_n.

A loss directly observing an intermediate state adds its derivative to
:math:`\lambda_n`. For a prepared material parameter :math:`\theta_k`, the
corresponding contribution is

.. math::

   \frac{\partial J}{\partial\theta_k}
   =\sum_n\left[
      \lambda_{n+1}^{T}(A_{,k}x_n+B_{,k}s_n)
      +r_n^{T}(C_{,k}x_n+D_{,k}s_n)\right],

plus explicit loss or initial-state dependence when present. Subscript
:math:`{,k}` means differentiation with respect to :math:`\theta_k`.
This derivation is the discrete adjoint-state construction; see
`Plessix (2006), A review of the adjoint-state method
<https://doi.org/10.1111/j.1365-246X.2006.02978.x>`_ for the general method.
It does not assume that a separately discretised continuous adjoint will match
this recurrence.

These transposes use the **ordinary tensor dot product**. If :math:`J` has units
:math:`[J]`, a returned derivative with respect to an entry of units :math:`[z]`
has units :math:`[J]/[z]`. A continuum space-time inner product instead contains
quadrature weights such as :math:`h\prod_i\Delta x_i`. Those weights belong in
the discretised objective or in an explicit conversion of its gradient; they
are not silently inserted by autograd. In particular, components of a state
vector with different physical units do not define a physical energy merely by
summing their squares.

The complete composition
------------------------

Write the receiver map schematically as

.. math::

   d = H_r\,D_t\,F_{\theta(q)}\,U_t\,H_s\,M_s\,P_t\,a.

Here :math:`a` is the public source array, :math:`P_t` explicitly pads or truncates
it to the requested length, :math:`M_s` disables masked sources,
:math:`H_s` spreads off-grid sources, :math:`U_t` upsamples in time,
:math:`F` includes propagation, physical source factors and on-grid recording,
:math:`D_t` downsamples, and :math:`H_r` gathers off-grid receivers. Source-free
initial-state contributions are an additional linear term. Multiple channels
are represented by block operators.

The source derivative is the transpose of this entire composition in reverse
order. It is not obtained simply by exchanging physical source and receiver
names. For example, scalar injection contains :math:`-v^2h^2`, acoustic volume
rate injection contains :math:`hK`, and force-density injection contains
:math:`h/\rho`. Each has the units needed to produce its updated field.
Geometrical scatter and gather on the same grid are transposes, but these
physical factors must also be differentiated or transposed.

Hicks source spreading and receiver gathering use the same spatial weights.
Masking is a diagonal zero/one map; its transpose is itself. Length padding and
cropping are transposes of one another. Model extraction is a restriction map;
its transpose scatters into the full model. Replicated model padding has a
transpose that **sums** padded-cell derivatives onto their originating edge
cell. Parameter conversions and source scaling contribute through the ordinary
chain rule. Consequently a native model-gradient accumulator, even at the last
backward callback, need not yet be the complete gradient of the public model.
See :doc:`callback_names` before modifying gradients.

Finite-record Fourier resampling
--------------------------------

Consider a coarse record of :math:`N` samples and a fine record of :math:`RN`
samples. Let :math:`Q_N` be the unitary full discrete Fourier transform, and
:math:`E` embed the retained signed frequency bins into the longer spectrum.
For real signals conjugate bins are embedded together. The even-length coarse
Nyquist bin is projected out; an odd-length record has no such bin. With no
extra taper, padding or shift,

.. math::

   U_t=\sqrt R\,Q_{RN}^{*}E Q_N,\qquad
   D_t=\frac{1}{\sqrt R}\,Q_N^{*}E^{*}Q_{RN}.

The factors follow by applying each transform to a constant signal: changing
length changes the unitary DC coefficient by the square root of the length
ratio. Therefore amplitudes are preserved, and on real coordinates

.. math::

   U_t^T=R D_t,\qquad D_t^T=U_t/R.

Upsampling and downsampling are thus **not each other's unscaled Euclidean
adjoints**. They are adjoints under the time-weighted inner products because
:math:`\Delta t=Rh`. PyTorch differentiates the actual FFT operations and their
scaling. Its `real FFT documentation
<https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_ describes
Hermitian storage and the ``ortho`` normalisation used here.

The record represents a periodic Fourier series. Fourier interpolation does not
know an unrecorded causal past or future. Zero padding changes the assumed
periodic extension; time windows change the signal. To transpose a resampler
with these options, reverse the order of every padding, cropping, window and
spectral operation, taking each transpose. Do not assume an inverse resampler
with arbitrary options is its adjoint. An explicit frequency-domain shift is
transposed by conjugating its phase factor.

A fixed integer downsampling rate requires a nonempty fine length divisible by
that rate. Otherwise spectral resizing to :math:`\lfloor N/R\rfloor` samples
has a different time interval and the fixed :math:`1/\sqrt R` scaling does not
even preserve a constant. Deepwave rejects that case. A positive frequency-taper
fraction that rounds to zero bins still obeys the same Nyquist projection.

Why sampled model gradients are retained
----------------------------------------

The exact material derivative above sums correlations over internal steps.
The compiled model-gradient path samples selected history-dependent
correlations every :math:`Rm` steps and weights them by that stride. Source and
initial-state adjoint propagation still visits every internal step. This is a
numerical approximation to the correlation integral with a sound bandlimited
sampling rationale, not an assertion of exact differentiation of every finite
record. Eager autograd differentiates every internal step.

An exact finite periodic identity shows the relevant error. Set
:math:`N=RM`, :math:`T=Nh`, and let :math:`c_j=a_j\overline{b_j}` be one
correlation sequence. For its normalised DFT

.. math::

   C_k=\frac1N\sum_{j=0}^{N-1}c_j e^{-2\pi i jk/N},

Fourier orthogonality gives

.. math::

   h\sum_j c_j &= T C_0, \\
   Rh\sum_{j=0}^{M-1}c_{jR} &= T\sum_{\ell=0}^{R-1}C_{\ell M}.

Their difference is exactly the sum of the nonzero-frequency terms on the
right. If both factors have signed DFT support strictly below the coarse
Nyquist, their product has no support at those aliased bins, and the sums agree.
The full-time sinc inner-product identity in :doc:`what_deepwave_calculates`
is the corresponding continuous statement. Finite causal records, arbitrary
initial states and terminal objectives need not satisfy this bandlimit.
Increasing a stride does not imply monotonic error, since aliased terms have
signs and phases. The same identity applies with stride :math:`Rm` when
additional gradient subsampling is selected.

For duration :math:`T`, every-internal-step storage has about :math:`T/h`
history entries; the default user-step storage has about :math:`T/\Delta t`.
Changing the default would therefore multiply history capacity and traffic by
approximately :math:`R`, while fixed model and live-state memory stay fixed.
This accuracy/memory contract is intentional. Effective stride one and lossless
history are required for the native exact finite-discrete model-derivative
contract and supported compiled second derivatives.

Continuation, checkpointing and evidence
----------------------------------------

On the internal grid, complete-state propagation composes across a split:
:math:`F_{0:N}=F_{K:N}\circ F_{0:K}` with the same coefficients and state
convention. Omitting PML or constitutive memory changes that state and therefore
the problem. Changing the grid, internal time step or inferred coefficients can
also change the operator. A global Fourier interpolator generally does not
commute with splitting a record. Accordingly, ``run_segments`` resamples the
whole source record once and the concatenated receiver record once, and aligns
model-gradient sampling phases at segment boundaries. Checkpointing recomputes
these same segment maps; it does not supply a missing transpose or repair an
inconsistent split.

The independent checks needed for these claims are distinct. Analytic physical
responses validate equations and source meaning. Operator dot products validate
transpose pairs. Directional finite differences validate parameter derivatives
with setup held fixed. Single Fourier modes validate resampling amplitude,
phase and bin selection independently of a second implementation. Split-run
checks validate complete-state composition. Representative native/eager
comparisons then check implementations of those established contracts.
None of these alone establishes every other property, and a successful source
check does not qualify an installed artifact or another platform.
