# DNA Forecaster — Mathematical Foundations
## Dynamic Nonlinear Adaptive Time Series Model
**UniversityBox Research | April 2026**

---

## 1. Notation and Problem Statement

Let $y = (y_1, y_2, \ldots, y_n)^\top \in \mathbb{R}^n$ be a univariate, regularly-spaced time series observed at integer times $t = 1, \ldots, n$.

**Goal:** Construct a point forecast $\hat{y}_{t+h|t}$ and a predictive distribution $p(y_{t+h} \mid y_{1:t})$ for any horizon $h \geq 1$.

**Assumptions:**
1. $y_t$ admits an additive decomposition into components of different frequency.
2. The irregular residual after decomposition is in the RKHS of a Matérn-type kernel (covered by the RBF basis).
3. The second-order residual (after N-stage) follows a locally linear Gaussian process.

---

## 2. The DNA Decomposition

DNA proposes a three-stage hierarchical decomposition:

$$
y_t = \underbrace{\mu_t + s_t}_{\text{Stage D}} + \underbrace{f(\Phi(\mathbf{x}_t))}_{\text{Stage N}} + \underbrace{\ell_t}_{\text{Stage A}} + \eta_t
\tag{DNA}
$$

where $\eta_t \sim \mathcal{N}(0, \sigma^2_\eta)$ is irreducible noise.

Each stage is fitted sequentially on the residual of the previous stage, ensuring orthogonality of corrections.

---

## 3. Stage D — Decomposition

### 3.1 Additive Model

$$
y_t = \mu_t + s_t + \varepsilon_t \tag{1}
$$

**Trend** $\mu_t$: slowly-varying, estimated by a Henderson moving average.

**Seasonal** $s_t$: periodic with known or estimated period $P$, estimated via Fourier OLS.

**Residual** $\varepsilon_t$: passed to Stage N.

### 3.2 Henderson Moving Average (Trend)

The Henderson filter (Henderson, 1916) minimises the roughness of the third difference of the trend while reproducing polynomials of degree up to 2:

$$
\hat{\mu}_t = \sum_{j=-m}^{m} w_j \, y_{t+j} \tag{2}
$$

The weights $\{w_j\}$ are the unique solution to:

$$
\min_{\{w_j\}} \sum_{t} (\nabla^3 \hat{\mu}_t)^2 \quad \text{subject to} \quad
\sum_j w_j = 1, \quad \sum_j j \, w_j = 0, \quad \sum_j j^2 w_j = 0
$$

**Closed-form weights** (Doherty, 2001): let $h = m+1$, $h_1 = m+2$, $h_2 = m+3$:

$$
w_j \propto (h^2 - j^2)(h_1^2 - j^2)(h_2^2 - j^2)\bigl[3h^2 - 11j^2 - 16\bigr] \tag{3}
$$

Normalised so $\sum_j w_j = 1$. At series endpoints, a symmetric pad (reflect mode) is applied.

**Default half-length:** $m = \max\!\left(3,\; \left\lfloor P/2 \right\rfloor \cdot 2 + 1\right)$.

### 3.3 Fourier Seasonal Model

Conditional on $\hat{\mu}_t$, fit a Fourier regression on the detrended series $d_t = y_t - \hat{\mu}_t$:

$$
s_t = \sum_{k=1}^{K} \left[ a_k \cos\!\left(\frac{2\pi k t}{P}\right) + b_k \sin\!\left(\frac{2\pi k t}{P}\right) \right] \tag{4}
$$

Design matrix $\mathbf{F} \in \mathbb{R}^{n \times 2K}$:

$$
F_{t,2k-1} = \cos\!\left(\frac{2\pi k t}{P}\right), \qquad F_{t,2k} = \sin\!\left(\frac{2\pi k t}{P}\right)
$$

OLS estimator:

$$
[\mathbf{a}, \mathbf{b}]^* = (\mathbf{F}^\top \mathbf{F})^{-1} \mathbf{F}^\top \mathbf{d}, \qquad \hat{s}_t = \mathbf{F}[\mathbf{a},\mathbf{b}]^* \tag{5}
$$

**Normalisation:** subtract per-phase mean to enforce $\sum_{p=0}^{P-1} s_{t+p} = 0$.

### 3.4 Period Estimation

When $P$ is unknown, estimate from the sample periodogram:

$$
I(\omega) = \frac{1}{n} \left| \sum_{t=1}^n y_t e^{-i\omega t} \right|^2 \tag{6}
$$

$$
\hat{P} = \operatorname*{argmax}_{P \in \{2, \ldots, \lfloor n/2 \rfloor\}} I\!\left(\frac{2\pi}{P}\right)
$$

After removing a linear trend from $y$ to avoid DC contamination of the periodogram.

---

## 4. Stage N — Nonlinear Basis Expansion

### 4.1 Feature Map

Define a composite feature vector $\Phi(\mathbf{x}_t) \in \mathbb{R}^D$ from three dictionaries:

**1. Polynomial basis** (degree $p$):

$$
\phi^{\text{poly}}(t) = \left[1,\; \frac{t}{n},\; \left(\frac{t}{n}\right)^2,\; \ldots,\; \left(\frac{t}{n}\right)^p \right]^\top \in \mathbb{R}^{p+1} \tag{7}
$$

Time is normalised to $[0,1]$ for numerical stability.

**2. Autoregressive lags** ($L$ lags):

$$
\phi^{\text{lag}}(t) = \left[\frac{\varepsilon_{t-1}}{\hat{\sigma}},\; \frac{\varepsilon_{t-2}}{\hat{\sigma}},\; \ldots,\; \frac{\varepsilon_{t-L}}{\hat{\sigma}} \right]^\top \in \mathbb{R}^L \tag{8}
$$

where $\hat{\sigma} = \operatorname{std}(\varepsilon)$. Zero-padded before $t = 1$.

**3. Radial Basis Functions** ($J$ centres):

$$
\phi^{\text{rbf}}_j(t) = \exp\!\left(-\gamma \left(\varepsilon_t - c_j\right)^2\right), \quad j = 1, \ldots, J \tag{9}
$$

Concatenated feature vector (dimension $D = p + 1 + L + J$):

$$
\Phi(t) = \left[\phi^{\text{poly}}(t) \;\Big|\; \phi^{\text{lag}}(t) \;\Big|\; \phi^{\text{rbf}}(t)\right]^\top \tag{10}
$$

### 4.2 k-means++ Centre Selection (Seed Protocol)

RBF centres $\{c_j\}_{j=1}^J$ are selected via k-means++ (Arthur & Vassilvitskii, 2007):

$$
c_1 \sim \text{Uniform}(\{\varepsilon_t\}) \tag{11}
$$

$$
c_{j+1} \sim \text{Categorical}\!\left(p_t \propto \min_{i \leq j} \|\varepsilon_t - c_i\|^2\right) \tag{12}
$$

This **seeding protocol** provides an $\mathcal{O}(\log J)$ approximation ratio over random initialisation and is the k-means++ guarantee (Theorem 3.1 of Arthur & Vassilvitskii, 2007).

### 4.3 Bandwidth Estimation — Median Heuristic

The RBF bandwidth $\gamma$ is set by the median heuristic (Schölkopf & Smola, 2002):

$$
\gamma = \frac{1}{2 \cdot \text{median}^2\!\left(\left\{\|\varepsilon_i - \varepsilon_j\|\right\}_{i \neq j}\right)} \tag{13}
$$

This ensures the feature map has a length scale matched to the data distribution, avoiding the over-smoothing / under-smoothing extremes.

### 4.4 Ridge Regression

Let $\boldsymbol{\Phi} \in \mathbb{R}^{n \times D}$ be the design matrix. The Stage N estimator solves:

$$
\boldsymbol{\theta}^* = \operatorname*{argmin}_{\boldsymbol{\theta} \in \mathbb{R}^D} \left\| \hat{\boldsymbol{\varepsilon}} - \boldsymbol{\Phi}\boldsymbol{\theta} \right\|^2 + \lambda \|\boldsymbol{\theta}\|^2 \tag{14}
$$

**Closed-form solution** (Tikhonov, 1963):

$$
\boldsymbol{\theta}^* = \left(\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I}_D\right)^{-1} \boldsymbol{\Phi}^\top \hat{\boldsymbol{\varepsilon}} \tag{15}
$$

Solved via Cholesky decomposition: $\mathcal{O}(D^3 + nD^2)$ time, $\mathcal{O}(D^2)$ space.

**Primal vs dual form:** When $D > n$, use the kernel trick (dual form):

$$
\boldsymbol{\theta}^* = \boldsymbol{\Phi}^\top \left(\boldsymbol{\Phi}\boldsymbol{\Phi}^\top + \lambda \mathbf{I}_n\right)^{-1} \hat{\boldsymbol{\varepsilon}} \tag{16}
$$

In-sample fitted values: $\hat{f}(\Phi(t)) = \boldsymbol{\Phi}\boldsymbol{\theta}^*$.

### 4.5 RKHS Interpretation

Ridge regression with the composite feature map $\Phi$ is equivalent to minimum-norm interpolation in the RKHS $\mathcal{H}_K$ induced by the composite kernel:

$$
K(x, x') = K_{\text{poly}}(x,x') + K_{\text{RBF}}(x, x') \tag{17}
$$

where $K_{\text{poly}}(x,x') = \langle \phi^{\text{poly}}(x), \phi^{\text{poly}}(x') \rangle$ and $K_{\text{RBF}}$ is the standard squared-exponential kernel. The regulariser $\lambda$ controls the RKHS norm bound $\|f\|_{\mathcal{H}_K} \leq \|\boldsymbol{\theta}^*\| / \sqrt{\lambda}$.

---

## 5. Stage A — Adaptive Kalman Filter

### 5.1 Local Linear Trend Model

The second-order residual $r_t = \hat{\varepsilon}_t - \hat{f}(\Phi(t))$ is modelled by the Local Linear Trend (LLT) state-space model:

$$
\mathbf{x}_t = \begin{pmatrix} \ell_t \\ b_t \end{pmatrix}, \qquad
\mathbf{F} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \qquad
\mathbf{H} = \begin{pmatrix} 1 & 0 \end{pmatrix} \tag{18}
$$

**Transition equation:**
$$
\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{G} \mathbf{w}_t, \qquad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}) \tag{19}
$$

**Observation equation:**
$$
r_t = \mathbf{H} \mathbf{x}_t + v_t, \qquad v_t \sim \mathcal{N}(0, R) \tag{20}
$$

with $\mathbf{Q} = \operatorname{diag}(q_\ell, q_b)$, $\mathbf{G} = \mathbf{I}_2$.

### 5.2 Kalman Filter Recursion

**Diffuse initialisation:**
$$
\hat{\mathbf{x}}_{1|0} = (r_1, 0)^\top, \qquad \mathbf{P}_{1|0} = \text{diag}(10^6, 10^6)
$$

**Prediction step** ($t = 1, \ldots, n$):
$$
\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\,\hat{\mathbf{x}}_{t-1|t-1} \tag{P1}
$$
$$
\mathbf{P}_{t|t-1} = \mathbf{F}\,\mathbf{P}_{t-1|t-1}\,\mathbf{F}^\top + \mathbf{Q} \tag{P2}
$$

**Innovation:**
$$
v_t = r_t - \mathbf{H}\,\hat{\mathbf{x}}_{t|t-1} \tag{I1}
$$
$$
S_t = \mathbf{H}\,\mathbf{P}_{t|t-1}\,\mathbf{H}^\top + R \tag{I2}
$$

**Kalman gain:**
$$
\mathbf{K}_t = \mathbf{P}_{t|t-1}\,\mathbf{H}^\top S_t^{-1} \tag{G}
$$

**Update step (Joseph form for numerical stability):**
$$
\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t v_t \tag{U1}
$$
$$
\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H})\,\mathbf{P}_{t|t-1}\,(\mathbf{I} - \mathbf{K}_t \mathbf{H})^\top + \mathbf{K}_t R \mathbf{K}_t^\top \tag{U2}
$$

### 5.3 h-Step Forecast

$$
\hat{\mathbf{x}}_{n+h|n} = \mathbf{F}^h \,\hat{\mathbf{x}}_{n|n} \tag{21}
$$

$$
\hat{\ell}_{n+h} = \mathbf{H}\,\hat{\mathbf{x}}_{n+h|n} = \ell_n + h \cdot b_n \tag{22}
$$

**Forecast covariance:**
$$
\mathbf{P}_{n+h|n} = \mathbf{F}^h\,\mathbf{P}_{n|n}\,(\mathbf{F}^\top)^h + \sum_{j=0}^{h-1} \mathbf{F}^j \mathbf{Q} (\mathbf{F}^\top)^j \tag{23}
$$

Predictive variance: $\sigma^2_{n+h} = \mathbf{H}\,\mathbf{P}_{n+h|n}\,\mathbf{H}^\top + R$.

### 5.4 MLE Parameter Estimation

The marginal log-likelihood (prediction-error decomposition) is:

$$
\log \mathcal{L}(q_\ell, q_b, R) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^n \left[\log S_t + \frac{v_t^2}{S_t}\right] \tag{24}
$$

Maximised over $\theta = (\log q_\ell, \log q_b, \log R) \in \mathbb{R}^3$ via Nelder-Mead (gradient-free, robust for small $n$). Log-parameterisation enforces positivity.

---

## 6. Ensemble Combination

### 6.1 Component Forecasts

Denote the four h-step component forecasts:

| Component | Symbol | Source |
|-----------|--------|--------|
| Trend | $\hat{\mu}_{n+h}$ | D-stage linear extrapolation |
| Seasonal | $\hat{s}_{n+h}$ | Fourier evaluation at future time |
| Nonlinear | $\hat{f}_{n+h}$ | Ridge regression at future features |
| Adaptive | $\hat{\ell}_{n+h}$ | Kalman h-step forecast |

### 6.2 Final Forecast

$$
\hat{y}_{n+h} = \alpha\,\hat{\mu}_{n+h} + \beta\,\hat{s}_{n+h} + \gamma\,\hat{f}_{n+h} + \delta\,\hat{\ell}_{n+h} \tag{25}
$$

with weights $(\alpha, \beta, \gamma, \delta)$ summing to 1.

### 6.3 Inverse-Variance Weighting (default)

Let $\sigma^2_i = \operatorname{Var}(y_t - \hat{c}^{(i)}_t)$ be the in-sample variance of the residual when only component $i$ is used:

$$
w_i = \frac{1/\sigma^2_i}{\sum_j 1/\sigma^2_j} \tag{26}
$$

This is the minimum-variance combination of unbiased estimators (Bates & Granger, 1969).

### 6.4 OLS Stacking (alternative)

$$
[\alpha, \beta, \gamma, \delta]^* = \operatorname*{argmin}_{w \geq 0} \left\| y - \mathbf{C} w \right\|^2 \tag{27}
$$

where $\mathbf{C} \in \mathbb{R}^{n \times 4}$ stacks component in-sample fits column-wise. Solved by non-negative least squares (Lawson & Hanson, 1974).

---

## 7. Prediction Intervals

### 7.1 Analytical Gaussian Intervals

Under the LLT model, the $h$-step predictive distribution is approximately Gaussian. We estimate the marginal forecast variance by propagating the in-sample RMSE:

$$
\sigma_h = \hat{\sigma}_\varepsilon \cdot \sqrt{h} \tag{28}
$$

where $\hat{\sigma}_\varepsilon = \sqrt{n^{-1}\sum_t (y_t - \hat{y}_t)^2}$ is the in-sample RMSE.

**Coverage-$\alpha$ interval:**
$$
\hat{y}_{n+h} \pm z_{\alpha/2}\,\sigma_h, \qquad z_{\alpha/2} = \Phi^{-1}\!\left(\frac{1+\alpha}{2}\right) \tag{29}
$$

### 7.2 Bootstrap Prediction Intervals

Let $\hat{\varepsilon}_1, \ldots, \hat{\varepsilon}_n$ be the in-sample residuals. For $b = 1, \ldots, B$:

$$
\varepsilon^{*(b)}_{n+1}, \ldots, \varepsilon^{*(b)}_{n+h} \;\overset{\text{iid}}{\sim}\; \text{Empirical}\!\left(\{\hat{\varepsilon}_t\}\right)
$$

$$
\hat{y}^{*(b)}_{n+h} = \hat{y}_{n+h} + \sum_{k=1}^h \varepsilon^{*(b)}_{n+k} \tag{30}
$$

**Empirical quantile interval:**
$$
\left[\hat{Q}_{\alpha/2}\!\left(\hat{y}^{*(1)}, \ldots, \hat{y}^{*(B)}\right),\; \hat{Q}_{1-\alpha/2}\!\left(\hat{y}^{*(1)}, \ldots, \hat{y}^{*(B)}\right)\right]
$$

Bootstrap intervals are distribution-free and capture skewness in the residual distribution.

---

## 8. Evaluation Metrics

| Metric | Formula |
|--------|---------|
| MAE | $n^{-1}\sum\|y_t - \hat{y}_t\|$ |
| RMSE | $\sqrt{n^{-1}\sum(y_t - \hat{y}_t)^2}$ |
| MAPE | $100 \cdot n^{-1}\sum\|y_t - \hat{y}_t\|/\|y_t\|$ |
| sMAPE | $200 \cdot n^{-1}\sum\frac{\|y_t-\hat{y}_t\|}{\|y_t\|+\|\hat{y}_t\|}$ |
| MASE | $\text{MAE} / \bar{d}$,   $\bar{d} = (n-P)^{-1}\sum_{t>P}\|y_t - y_{t-P}\|$ |
| CRPS | $\mathbb{E}[\|F - \mathbf{1}(Y \leq y)\|^2]$ (Gneiting & Raftery, 2007) |

**CRPS closed form for Gaussian predictive $\mathcal{N}(\mu, \sigma^2)$:**

$$
\text{CRPS}\bigl(\mathcal{N}(\mu,\sigma^2),\, y\bigr) = \sigma\left\{z\left[2\Phi(z) - 1\right] + 2\phi(z) - \frac{1}{\sqrt{\pi}}\right\}, \quad z = \frac{y-\mu}{\sigma}
$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF.

---

## 9. Identifiability and Consistency

**Identifiability of the additive decomposition** requires that trend, seasonal, and residual components lie in orthogonal function spaces. This is ensured by:
- Trend $\mu_t$ is smooth (bounded third differences).
- Seasonal $s_t$ has zero mean over each complete period.
- Residual $\varepsilon_t$ is weakly stationary with zero mean.

**Consistency of Ridge regression (Stage N):**
Under regularity conditions (bounded feature map, $\lambda_n = o(n)$, $\lambda_n \to \infty$), the Ridge estimator is consistent in $L_2$:

$$
\|\hat{f} - f^*\|_{L_2} \xrightarrow{p} 0 \quad \text{as } n \to \infty \tag{31}
$$

(Steinwart & Christmann, 2008, Theorem 9.1).

**Consistency of Kalman filter:**
Under identifiability of $(Q, R)$ and observability of $(\mathbf{F}, \mathbf{H})$ (trivially satisfied for LLT), the MLE of $(q_\ell, q_b, R)$ is consistent and asymptotically normal (Shumway & Stoffer, 2011, Theorem 6.1).

---

## 10. Computational Complexity

| Operation | Complexity |
|-----------|-----------|
| Henderson filter | $\mathcal{O}(n \cdot m)$ |
| Fourier OLS | $\mathcal{O}(n K^2 + K^3)$ |
| k-means++ seeding | $\mathcal{O}(nJ)$ per iteration |
| Feature map construction | $\mathcal{O}(nD)$, $D = p+1+L+J$ |
| Ridge regression (primal) | $\mathcal{O}(nD^2 + D^3)$ |
| Kalman filter | $\mathcal{O}(n)$ (2-dimensional state) |
| Kalman MLE (Nelder-Mead) | $\mathcal{O}(n \cdot \text{iter})$, typically $< 2000$ iter |
| Bootstrap intervals | $\mathcal{O}(Bh)$ |

**Total fit:** $\mathcal{O}(nD^2 + D^3 + n)$ — dominated by Ridge regression when $D \gg 1$.

For default parameters ($p=2, L=4, J=10$): $D = 17$, Ridge step is $< 0.1$ms for $n = 10^4$.

---

## 11. References

- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA 2007*.
- Bates, J.M. & Granger, C.W.J. (1969). The combination of forecasts. *Operational Research Quarterly*.
- Doherty, M. (2001). The surrogate Henderson filters in X-11. *Australian & New Zealand J. Statistics*.
- Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
- Henderson, R. (1916). Note on graduation by adjusted average. *Trans. Actuarial Soc. America*.
- Lawson, C.L. & Hanson, R.J. (1974). *Solving Least Squares Problems*. SIAM.
- Schölkopf, B. & Smola, A.J. (2002). *Learning with Kernels*. MIT Press.
- Shumway, R.H. & Stoffer, D.S. (2011). *Time Series Analysis and Its Applications*. Springer.
- Steinwart, I. & Christmann, A. (2008). *Support Vector Machines*. Springer.
- Tikhonov, A.N. (1963). On the solution of ill-posed problems. *Soviet Mathematics Doklady*.

---

*UniversityBox Research — April 2026*
