# DNA — Mathematical Foundations
**Dynamic Nonlinear Adaptive Time Series Model**

> Full derivations for every formula in the `universitybox` package.
> Rendered best on GitHub. All equations are numbered in the right column.

---

## 1. Problem Statement

Given a univariate series $y = (y_1, \ldots, y_n)^\top \in \mathbb{R}^n$, produce a point forecast $\hat{y}_{n+h}$ and predictive distribution $p(y_{n+h} \mid y_{1:n})$ for any horizon $h \geq 1$.

---

## 2. The DNA Equation

DNA proposes a hierarchical additive model:

| Equation | # |
|----------|---|
| $y_t = \mu_t + s_t + f(\Phi(\mathbf{x}_t)) + \ell_t + \eta_t$ | (DNA) |

where $\mu_t$ is the trend, $s_t$ the seasonal component, $f(\Phi(\cdot))$ a nonlinear correction, $\ell_t$ an adaptive correction, and $\eta_t \sim \mathcal{N}(0, \sigma^2_\eta)$ irreducible noise. Each stage is fit on the residual of the stage before it.

---

## 3. Stage D — Decomposition

### 3.1 Additive decomposition

| Equation | # |
|----------|---|
| $y_t = \mu_t + s_t + \varepsilon_t$ | (1) |

### 3.2 Henderson filter (trend)

The trend is estimated by a symmetric moving average: $\hat{\mu}_t = \sum_{j=-m}^{m} w_j \, y_{t+j}$

Weights $\{w_j\}$ minimise the third-difference roughness of the trend subject to reproducing polynomials up to degree 2. The optimisation problem is:

| Equation | # |
|----------|---|
| $\min_{\{w_j\}} \sum_t (\nabla^3 \hat{\mu}_t)^2 \quad \text{s.t.} \quad \sum_j w_j = 1,\ \sum_j j\,w_j = 0,\ \sum_j j^2 w_j = 0$ | (2) |

Closed-form weights (Doherty 2001), with $h = m+1$, $h_1 = m+2$, $h_2 = m+3$:

| Equation | # |
|----------|---|
| $w_j \;\propto\; (h^2 - j^2)(h_1^2 - j^2)(h_2^2 - j^2)(3h^2 - 11j^2 - 16)$ | (3) |

Normalised so $\sum_j w_j = 1$. Endpoints use symmetric reflect-padding. Default half-length: $m = \max(3,\, \lfloor P/2 \rfloor \cdot 2 + 1)$.

### 3.3 Fourier seasonal model

Detrended series: $d_t = y_t - \hat{\mu}_t$. Seasonal model of order $K$:

| Equation | # |
|----------|---|
| $s_t = \sum_{k=1}^{K} \left[ a_k \cos\!\left(\tfrac{2\pi k t}{P}\right) + b_k \sin\!\left(\tfrac{2\pi k t}{P}\right) \right]$ | (4) |

Design matrix $\mathbf{F} \in \mathbb{R}^{n \times 2K}$ with $F_{t,2k-1} = \cos(2\pi k t / P)$, $F_{t,2k} = \sin(2\pi k t / P)$. OLS solution:

| Equation | # |
|----------|---|
| $[\mathbf{a}, \mathbf{b}]^* = (\mathbf{F}^\top \mathbf{F})^{-1} \mathbf{F}^\top \mathbf{d}, \qquad \hat{s}_t = \mathbf{F}[\mathbf{a},\mathbf{b}]^*$ | (5) |

Seasonal component is normalised so $\sum_{p=0}^{P-1} s_{t+p} = 0$ over each full period.

### 3.4 Period estimation (periodogram)

When $P$ is unknown, estimate from the sample periodogram:

| Equation | # |
|----------|---|
| $I(\omega) = \tfrac{1}{n}\left\|\sum_{t=1}^n y_t e^{-i\omega t}\right\|^2, \qquad \hat{P} = \operatorname*{argmax}_{P \in \{2,\ldots,\lfloor n/2\rfloor\}} I\!\left(\tfrac{2\pi}{P}\right)$ | (6) |

A linear trend is removed from $y$ before computing the periodogram to suppress DC contamination.

---

## 4. Stage N — Nonlinear Basis Expansion

### 4.1 Feature map

Three feature dictionaries are concatenated into $\Phi(t) \in \mathbb{R}^D$, $D = (p+1) + L + J$:

**Polynomial** (degree $p$, time normalised to $[0,1]$):

| Equation | # |
|----------|---|
| $\phi^\text{poly}(t) = \bigl[1,\ t/n,\ (t/n)^2,\ \ldots,\ (t/n)^p\bigr]^\top \in \mathbb{R}^{p+1}$ | (7) |

**Autoregressive lags** ($L$ lags of the D-stage residual, standardised by $\hat{\sigma} = \text{std}(\varepsilon)$):

| Equation | # |
|----------|---|
| $\phi^\text{lag}(t) = \bigl[\varepsilon_{t-1}/\hat{\sigma},\ \ldots,\ \varepsilon_{t-L}/\hat{\sigma}\bigr]^\top \in \mathbb{R}^L$ | (8) |

**Radial Basis Functions** ($J$ centres $c_1,\ldots,c_J$, bandwidth $\gamma$):

| Equation | # |
|----------|---|
| $\phi^\text{rbf}_j(t) = \exp\!\bigl(-\gamma(\varepsilon_t - c_j)^2\bigr), \quad j=1,\ldots,J$ | (9) |

Full feature vector: $\Phi(t) = [\phi^\text{poly}(t) \mid \phi^\text{lag}(t) \mid \phi^\text{rbf}(t)]^\top$

### 4.2 k-means++ seed selection

RBF centres are chosen by k-means++ (Arthur & Vassilvitskii, 2007):

| Equation | # |
|----------|---|
| $c_1 \sim \text{Uniform}(\{\varepsilon_t\})$ | (10) |
| $c_{j+1} \sim \text{Categorical}\!\left(p_t \propto \min_{i \leq j}\|\varepsilon_t - c_i\|^2\right)$ | (11) |

This gives an $O(\log J)$ approximation ratio over random initialisation (Arthur & Vassilvitskii 2007, Theorem 3.1).

### 4.3 Bandwidth — median heuristic

| Equation | # |
|----------|---|
| $\gamma = \dfrac{1}{2\cdot\text{median}^2\!\bigl(\{\|\varepsilon_i - \varepsilon_j\|\}_{i \neq j}\bigr)}$ | (12) |

### 4.4 Ridge regression

Design matrix $\boldsymbol{\Phi} \in \mathbb{R}^{n \times D}$. Ridge estimator (Tikhonov 1963):

| Equation | # |
|----------|---|
| $\boldsymbol{\theta}^* = \operatorname*{argmin}_\theta \|\hat{\boldsymbol{\varepsilon}} - \boldsymbol{\Phi}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|^2$ | (13) |
| $\boldsymbol{\theta}^* = (\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \lambda\mathbf{I}_D)^{-1}\boldsymbol{\Phi}^\top\hat{\boldsymbol{\varepsilon}}$ | (14) |

Solved via Cholesky decomposition: $O(D^3 + nD^2)$ time. Dual form when $D > n$:

| Equation | # |
|----------|---|
| $\boldsymbol{\theta}^* = \boldsymbol{\Phi}^\top(\boldsymbol{\Phi}\boldsymbol{\Phi}^\top + \lambda\mathbf{I}_n)^{-1}\hat{\boldsymbol{\varepsilon}}$ | (15) |

### 4.5 RKHS interpretation

Ridge regression with $\Phi$ is equivalent to minimum-norm interpolation in the RKHS $\mathcal{H}_K$ of the composite kernel $K = K_\text{poly} + K_\text{RBF}$. The regulariser $\lambda$ controls the RKHS norm: $\|f\|_{\mathcal{H}_K} \leq \|\boldsymbol{\theta}^*\| / \sqrt{\lambda}$.

---

## 5. Stage A — Adaptive Kalman Filter

### 5.1 Local Linear Trend state-space model

State $\mathbf{x}_t = [\ell_t,\ b_t]^\top$ (level, slope). System matrices:

$\mathbf{F} = \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix}1 & 0\end{bmatrix}, \quad \mathbf{Q} = \text{diag}(q_\ell, q_b), \quad R = \sigma^2_v$

| Equation | # |
|----------|---|
| $\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$ | (16) |
| $r_t = \mathbf{H}\mathbf{x}_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$ | (17) |

### 5.2 Kalman filter recursion

Initialisation: $\hat{\mathbf{x}}_{1|0} = (r_1, 0)^\top$, $\mathbf{P}_{1|0} = 10^6 \mathbf{I}$ (diffuse prior).

For $t = 1, \ldots, n$:

| Step | Equation | # |
|------|----------|---|
| Predict state | $\hat{\mathbf{x}}_{t\|t-1} = \mathbf{F}\hat{\mathbf{x}}_{t-1\|t-1}$ | (P1) |
| Predict covariance | $\mathbf{P}_{t\|t-1} = \mathbf{F}\mathbf{P}_{t-1\|t-1}\mathbf{F}^\top + \mathbf{Q}$ | (P2) |
| Innovation | $v_t = r_t - \mathbf{H}\hat{\mathbf{x}}_{t\|t-1}$ | (I1) |
| Innovation variance | $S_t = \mathbf{H}\mathbf{P}_{t\|t-1}\mathbf{H}^\top + R$ | (I2) |
| Kalman gain | $\mathbf{K}_t = \mathbf{P}_{t\|t-1}\mathbf{H}^\top S_t^{-1}$ | (G) |
| Update state | $\hat{\mathbf{x}}_{t\|t} = \hat{\mathbf{x}}_{t\|t-1} + \mathbf{K}_t v_t$ | (U1) |
| Update covariance (Joseph form) | $\mathbf{P}_{t\|t} = (\mathbf{I} - \mathbf{K}_t\mathbf{H})\mathbf{P}_{t\|t-1}(\mathbf{I}-\mathbf{K}_t\mathbf{H})^\top + \mathbf{K}_t R\mathbf{K}_t^\top$ | (U2) |

### 5.3 h-step forecast

| Equation | # |
|----------|---|
| $\hat{\mathbf{x}}_{n+h\|n} = \mathbf{F}^h\hat{\mathbf{x}}_{n\|n}$ | (18) |
| $\hat{\ell}_{n+h} = \mathbf{H}\hat{\mathbf{x}}_{n+h\|n} = \ell_n + h\cdot b_n$ | (19) |

### 5.4 MLE noise estimation

Log-likelihood via prediction-error decomposition:

| Equation | # |
|----------|---|
| $\log\mathcal{L}(q_\ell, q_b, R) = -\tfrac{n}{2}\log(2\pi) - \tfrac{1}{2}\sum_{t=1}^n\bigl[\log S_t + v_t^2/S_t\bigr]$ | (20) |

Optimised over $(\log q_\ell, \log q_b, \log R)$ via Nelder-Mead. Log-parameterisation enforces positivity.

---

## 6. Ensemble Combination

Final h-step forecast:

| Equation | # |
|----------|---|
| $\hat{y}_{n+h} = \alpha\,\hat{\mu}_{n+h} + \beta\,\hat{s}_{n+h} + \gamma\,\hat{f}_{n+h} + \delta\,\hat{\ell}_{n+h}$ | (21) |

**Inverse-variance weights** (Bates & Granger 1969), where $\sigma^2_i = \text{Var}(y_t - \hat{c}^{(i)}_t)$:

| Equation | # |
|----------|---|
| $w_i = \dfrac{1/\sigma^2_i}{\sum_j 1/\sigma^2_j}$ | (22) |

**OLS stacking** (alternative): $\mathbf{w}^* = \operatorname*{argmin}_{w \geq 0} \|y - \mathbf{C}w\|^2$, where $\mathbf{C} \in \mathbb{R}^{n \times 4}$ stacks in-sample component fits.

---

## 7. Prediction Intervals

**Analytical** (random-walk propagation of in-sample RMSE $\hat{\sigma}_\varepsilon$):

| Equation | # |
|----------|---|
| $\hat{y}_{n+h} \pm z_{\alpha/2}\,\hat{\sigma}_\varepsilon\,\sqrt{h}$ | (23) |

**Bootstrap** (B replications, empirical quantiles):

| Equation | # |
|----------|---|
| $\hat{y}^{*(b)}_{n+h} = \hat{y}_{n+h} + \sum_{k=1}^h \varepsilon^{*(b)}_{n+k}, \quad \varepsilon^{*(b)} \sim \text{Empirical}(\{\hat{\varepsilon}_t\})$ | (24) |

---

## 8. Evaluation Metrics

| Metric | Formula |
|--------|---------|
| MAE | $\frac{1}{n}\sum\|y_t - \hat{y}_t\|$ |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_t-\hat{y}_t)^2}$ |
| MAPE | $\frac{100}{n}\sum\frac{\|y_t-\hat{y}_t\|}{\|y_t\|}$ |
| sMAPE | $\frac{200}{n}\sum\frac{\|y_t-\hat{y}_t\|}{\|y_t\|+\|\hat{y}_t\|}$ |
| MASE | $\text{MAE} \;/\; \frac{1}{n-P}\sum_{t>P}\|y_t - y_{t-P}\|$ |
| CRPS (Gaussian) | $\sigma\bigl[z(2\Phi(z)-1) + 2\phi(z) - 1/\sqrt{\pi}\bigr],\quad z=(y-\mu)/\sigma$ |

---

## 9. Consistency

- **Ridge (Stage N):** Under bounded feature map and $\lambda_n \to \infty$, $\lambda_n = o(n)$: $\|\hat{f} - f^*\|_{L_2} \xrightarrow{p} 0$ as $n \to \infty$ (Steinwart & Christmann 2008, Theorem 9.1).
- **Kalman MLE (Stage A):** Under identifiability and observability of $(\mathbf{F}, \mathbf{H})$, MLE of $(q_\ell, q_b, R)$ is consistent and asymptotically normal (Shumway & Stoffer 2011, Theorem 6.1).

---

## 10. Computational Complexity

| Operation | Complexity |
|-----------|-----------|
| Henderson filter | $O(n \cdot m)$ |
| Fourier OLS | $O(nK^2 + K^3)$ |
| k-means++ seeding | $O(nJ)$ |
| Feature matrix | $O(nD)$,  $D = p+1+L+J$ |
| Ridge (primal) | $O(nD^2 + D^3)$ |
| Kalman filter | $O(n)$ (2-D state) |
| Kalman MLE | $O(n \cdot \text{iter})$, typically < 2000 iterations |
| Bootstrap CI | $O(Bh)$ |

Default parameters ($p=2, L=4, J=10$): $D=17$. Ridge step < 0.1 ms for $n=10^4$.

---

## 11. References

- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA 2007*.
- Bates, J.M. & Granger, C.W.J. (1969). The combination of forecasts. *Operational Research Quarterly*.
- Doherty, M. (2001). The surrogate Henderson filters in X-11. *Australian & New Zealand J. Statistics*.
- Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules. *JASA*.
- Henderson, R. (1916). Note on graduation by adjusted average. *Trans. Actuarial Soc. America*.
- Lawson, C.L. & Hanson, R.J. (1974). *Solving Least Squares Problems*. SIAM.
- Schölkopf, B. & Smola, A.J. (2002). *Learning with Kernels*. MIT Press.
- Shumway, R.H. & Stoffer, D.S. (2011). *Time Series Analysis and Its Applications*. Springer.
- Steinwart, I. & Christmann, A. (2008). *Support Vector Machines*. Springer.
- Tikhonov, A.N. (1963). On the solution of ill-posed problems. *Soviet Mathematics Doklady*.
