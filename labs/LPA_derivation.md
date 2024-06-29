## The idea
Approximate the signal locally with a polynomial.
Given a noisy signal $s\in\mathbb{R}^l$, we can write it as:
$$s = y+\eta$$
where $y\in\mathbb R^l$ is the true signal and $\eta\sim\mathcal{N}(0,\sigma^2 \mathbb I_l)$ is the noise.
We want to estimate $y$ from $s$.

To estimate the point $y_c\in\mathbb R$ for a given $c\in\{1,\ldots,l\}$, we can use a polynomial of degree $d$, fitted to the $m$ points ($m$ odd) $s_{c-(m-1)/2}, s_{c-(m-1)/2+1}$, $\ldots$, $s_{c+(m-1)/2}$.

So we can extract the window of the signal around $s_c$ and call it $s[c]\in\mathbb R^m$. We can then fit a polynomial of degree $d$ to $s[c]$ and call it $p[c]$.

The problem reduces to the minimization of the following function:

$$\hat x[c]=\argmin_x{\sum_{k=1}^m}(s[c]_k-p_x(k))$$
where $p_x(k)$ is the polynomial of degree $d$ evaluated at $k$:
$$p_x(k)=\sum_{i=0}^d x_i k^i$$

Now we can translate the problem in a matrix form.
Let:
$$p_x:=\begin{bmatrix}p_x(1) & p_x(2) & p_x(3) & \cdots & p_x(m)\end{bmatrix}^T$$

$$x:=\begin{bmatrix}x_0 & x_1 & x_2 & \cdots & x_d\end{bmatrix}^T$$

$$T:=\begin{bmatrix}1 & 1 & 1 & \cdots & 1\\
1 & 2 & 4 & \cdots & 2^d\\
1 & 3 & 9 & \cdots & 3^d\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & m & m^2 & \cdots & m^d\end{bmatrix}$$

Note that $T$ is a Vandermonde matrix.

Then we can write the problem as:
$$\hat x[c]=\argmin_x{\frac 1 2 \|s[c]-T x\|_2^2}$$
In general, it may be useful to add weights to the error, so we can write the problem as:
$$\hat x[c]=\argmin_x{\frac 1 2 \|W(s[c]-T x)\|_2^2}$$
where $W$ is a diagonal matrix of weights:
$$W=\begin{bmatrix}w_1 & 0 & 0 & \cdots & 0\\
0 & w_2 & 0 & \cdots & 0\\
0 & 0 & w_3 & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
0 & 0 & 0 & \cdots & w_m\end{bmatrix}$$
The problem is convex, therefore it admits a unique solution, which can be evaluated in closed form.

$$\begin{align*}&\frac{\partial}{\partial x}\frac 1 2 \|W(s[c]-T x)\|_2^2\\
&=-T^T W^T W s[c] + T^T W^T W T x=0\\
&\iff \hat x[c] = (T^T W^T W T)^{-1} T^T W^T W s[c]\end{align*}$$

Now reconstruct the signal:
$$
\begin{align*}
\hat s[c] &= T \hat x[c]\\
&= T (T^T W^T W T)^{-1} T^T W^T W s[c]
\end{align*}
$$

Multiply both sides by $W$:
$$
\begin{align*}
W\hat s[c] &= W T (T^T W^T W T)^{-1} T^T W^T W s[c]
\end{align*}
$$

Now, perform the QR decomposition of $WT$:
$$WT=QR$$
where $Q$ is an orthogonal matrix and $R$ is an upper triangular matrix.

Then we can write:
$$
\begin{align*}
W\hat s[c] &= W T (T^T W^T W T)^{-1} T^T W^T W s[c]\\
&=QR(R^T \cancel{Q^T Q} R)^{-1} R^T Q^T W s[c]\\
&=Q\cancel{RR^{-1}}\cancel{(R^T)^{-1} R^T}Q^T W s[c]\\
&=QQ^T W s[c]
\end{align*}
$$

We are interested in the estimation of the middle point $y_c$ of the signal, which is equal to $\hat s[c]_{\frac{m-1} 2}$, Therefore we take the middle row of each side of the equation:

$$
\begin{align*}
w_{\frac{m-1} 2}\hat s[c]_{\frac{m-1} 2} &= q_{\frac{m-1} 2}^TQ^T W s[c] \\
w_{\frac{m-1} 2}\hat y_c &= (Q q_{\frac{m-1} 2})^TW s[c] \\
\hat y_c &= s[c]^T\underbrace{\frac{1}{w_{\frac{m-1} 2}}WQq_{\frac{m-1} 2}}_{\text{filter}}
\end{align*}
$$

The transformation may be implemented as a convolution with a filter $k$ of the form:
$$k = \text{flip}\left(\frac{1}{w_{\frac{m-1}{2}}}WQq_{\frac{m-1}{2}}\right)$$
$$\hat y = s * k$$
where $L$ is the window length (i.e., $m$).


