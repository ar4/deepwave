The scalar wave equation implemented in Deepwave is

.. math::
    :label: wave_equation

    \nabla u - \frac{1}{v^2}\ddot{u} = -f,

where :math:`u` is a scalar field that obeys the wave equation, such as pressure, :math:`v` is the wave speed, and :math:`f` is a source term. 

Deepwave uses central finite differences in time (with second-order accuracy) and space (with a user-specified order of accuracy) to approximately solve this equation on a given time and space domain.

To prevent unwanted reflections from the spatial boundaries of this domain, Deepwave uses a `Perfectly Matched Layer (PML) <https://en.wikipedia.org/wiki/Perfectly_matched_layer>`_. For the scalar wave equation, Deepwave uses the method of `Pasalic and McGarry <https://doi.org/10.1190/1.3513453>`_. In this approach, spatial derivatives are replaced by

.. math::
    :label: pml_deriv

    \frac{\partial}{\partial \tilde{x}} = \frac{\partial}{\partial x} + \psi,

where

.. math::
    :label: psi_update

    \psi_t = a\psi_{t-1} + b\left(\frac{\partial}{\partial x}\right)_t.

therefore
\begin{align}
c^2\frac{\partial^2 u_t}{\partial \tilde{x}^2} - \frac{\partial^2 u_t}{\partial t^2} &= c^2 f_t \\
c^2\frac{\partial}{\partial \tilde{x}}\left(\frac{\partial u_t}{\partial x} + p_t\right) - \frac{\partial^2 u_t}{\partial t^2} &= c^2 f_t \\
c^2\left(\frac{\partial^2 u_t}{\partial x^2} + \frac{\partial p_t}{\partial x} + z_t\right) - \frac{\partial^2 u_t}{\partial t^2} &= c^2 f_t
\end{align}
where
\begin{equation}
p_t = \psi u_t
\end{equation}
and
\begin{equation}
z_t = \psi \left(\frac{\partial u_t}{\partial x} + p_t\right)
\end{equation}

Using \ref{psi_update}, we can calculate $p$ and $z$ using
\begin{equation}
p_t = ap_{t-1} + b\frac{\partial u_t}{\partial_x}
\end{equation}
and
\begin{equation}
z_t = az_{t-1} + b\left(\frac{\partial^2 u_t}{\partial_x^2} + \frac{\partial p_t}{\partial x}\right)
\end{equation}

The update equation for $u$ is thus
\begin{equation}
u_{t+1} = c^2dt^2\left(\frac{\partial^2 u_t}{\partial x^2} + \frac{\partial p_t}{\partial x} + z_t\right) + 2u_t - u_{t-1} - c^2dt^2 f_t
\end{equation}
