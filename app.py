import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.stats import multivariate_normal

####################################################

st.sidebar.title("Table of contents")
pages = ["A bit of theory...", "Univariate case", "Bivariate case"]
page = st.sidebar.radio("Go to page...", pages)

st.sidebar.markdown("---")  
st.sidebar.write("Author: Hugo Rameil")

#################### PAGE 0 #####################

if page == pages[0]:
    st.title("Univariate Gaussian Distribution")

    st.write("""
    The **Gaussian distribution**, also known as the Normal distribution, is a continuous probability distribution characterized by its symmetric bell-shaped curve. 
    It is defined by two parameters:
    - **Mean (μ)**: The center or the location of the distribution.
    - **Standard deviation (σ)**: The spread or scale of the distribution.
    """)

    st.write("### Probability Density Function (PDF):")
    st.latex(r"""
    f(x | \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
    """)

    st.write("""
    Where
    - **x** is the random variable.
    - **μ** is the mean.
    - **σ** is the standard deviation.
    """)

    st.write("### Key Properties of Gaussian Distribution:")
    st.write("""
    1. **Symmetry**: The distribution is symmetric around the mean **μ**.
    2. **Mean = Median = Mode**: In a Gaussian distribution, the mean, median, and mode are all the same.
    3. **68-95-99.7 Rule**: 
    """)
    
    st.latex(r"""
    \begin{align*}
    &68\% \quad : \quad \mu - \sigma \leq X \leq \mu + \sigma \\
    &95\% \quad : \quad \mu - 2\sigma \leq X \leq \mu + 2\sigma \\
    &99.7\% \quad : \quad \mu - 3\sigma \leq X \leq \mu + 3\sigma
    \end{align*}
    """)

    st.write("### Standard Normal Distribution:")
    st.write("""
    A special case of the normal distribution is the **Standard Normal Distribution**, where the mean **μ** = 0 and the standard deviation **σ** = 1.
    The PDF of a standard normal distribution is:
    """)
    st.latex(r"""
    f(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)
    """)
    
    st.title("Multivariate Gaussian Distribution")
    st.write("""
    The **Multivariate Gaussian Distribution** is an extension of the univariate Gaussian distribution to multiple dimensions. 
    In this case, the distribution is defined by a **mean vector** and a **covariance matrix**.
    """)

    st.write("### Mean Vector **μ**:")
    st.write("""
    The mean vector **μ** is a **k**-dimensional vector that represents the expected values of each random variable. The mean vector is given by:
    """)
    st.latex(r"""
    \mu = \begin{bmatrix} 
    \mu_1 \\ 
    \mu_2 \\ 
    \vdots \\ 
    \mu_k 
    \end{bmatrix}
    """)
    
    st.write("### Covariance Matrix ( \( \Sigma \) ):")
    st.write("""
    The covariance matrix \( \Sigma \) represents the variances of each variable along the diagonal and the covariances between pairs of variables off the diagonal.
    For a \(k\)-dimensional Gaussian vector, the covariance matrix is given by:
    """)
    st.latex(r"""
    \Sigma = \begin{bmatrix} 
    \sigma_{11} & \sigma_{12} & \dots  & \sigma_{1k} \\ 
    \sigma_{21} & \sigma_{22} & \dots  & \sigma_{2k} \\ 
    \vdots     & \vdots     & \ddots & \vdots \\ 
    \sigma_{k1} & \sigma_{k2} & \dots  & \sigma_{kk}
    \end{bmatrix}
    """)

    st.write("### Probability Density Function (PDF):")
    st.write("""
    The PDF of the multivariate Gaussian distribution is given by:
    """)
    st.latex(r"""
    f(\mathbf{x} | \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)
    """)

    st.write("""
    In this formula:
    - \( \mathbf{x} \) is a \( k \)-dimensional vector representing the random variables.
    - \( \mu \) is the mean vector, a \( k \times 1 \) column vector representing the means of the variables.
    - \( \Sigma \) is the covariance matrix, a \( k \times k \) matrix describing variances and covariances.
    - \( |\Sigma| \) is the determinant of the covariance matrix.
    - \( \Sigma^{-1} \) is the inverse of the covariance matrix.
    """)

    st.write("### Properties of the Multivariate Gaussian Distribution:")
    st.write("""
    1. **Mean vector**: The mean vector \( \mu \) represents the central point of the distribution.
    2. **Covariance matrix**: The covariance matrix \( \Sigma \) describes the shape of the distribution, its spread, and the correlations between variables.
    3. **Independence**: If the off-diagonal entries of the covariance matrix are zero, the variables are uncorrelated (and independent in the case of Gaussian distributions).
    """)

    st.write("### Standard Multivariate Normal Distribution:")
    st.write("""
    A special case of the multivariate normal distribution is when the mean vector \( \mu \) is a zero vector, and the covariance matrix \( \Sigma \) is the identity matrix.
    In this case, the PDF simplifies to:
    """)
    st.latex(r"""
    f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k}} \exp\left(-\frac{1}{2} \mathbf{x}^T \mathbf{x}\right)
    """)
    st.write("""
    This corresponds to the case where all variables are independent standard normal variables.
    """)

#################### PAGE 1 #####################

if page == pages[1]:
    
    st.title('Univariate Gaussian Distribution')
    st.header('Theoretical vs Empirical Density')
    
    st.latex(r'''
    f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)
    ''')
    
    n_samples = st.number_input('Number of samples (n)', value=100)

    mean = st.number_input('Mean (μ)', value=0.0)
    variance = st.number_input('Variance (σ²)', value=1.0)
    std_dev = np.sqrt(variance) 

    if st.button("Show Plots"):

        samples = np.random.normal(loc=mean, scale=std_dev, size=n_samples)

        quantiles = [0.05, 0.5, 0.95]  # Percentiles: 5%, 50%, 95%
        theoretical_quantiles = norm.ppf(quantiles, loc=mean, scale=std_dev)  # Theoretical quantiles
        empirical_quantiles = np.percentile(samples, [5, 50, 95])  # Empirical quantiles
        
        st.write(f"Generated {n_samples} sample from a Univariate Gaussian distribution with mean {mean} and variance {variance}.")

        # bins for histogram and calculate the theoretical density curve
        x_vals = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 1000)
        theoretical_density = norm.pdf(x_vals, mean, std_dev)
        theoretical_cdf = norm.cdf(x_vals, loc=mean, scale=std_dev)  
        
        # ecdf
        sorted_samples = np.sort(samples)  # sort the samples for CDF
        empirical_cdf = np.arange(1, n_samples + 1) / n_samples

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=samples, nbinsx=50, histnorm='probability density', 
                                name='Empirical Density', marker_color='blue', opacity=0.6))

        fig.add_trace(go.Scatter(x=x_vals, y=theoretical_density, mode='lines', 
                                name='Theoretical Density', line=dict(color='red', width=2)))

        for i, quantile in enumerate(quantiles):
            fig.add_trace(go.Scatter(x=[theoretical_quantiles[i], theoretical_quantiles[i]], 
                                    y=[0, norm.pdf(theoretical_quantiles[i], mean, std_dev)],
                                    mode='lines', line=dict(color='red', dash='dash'), 
                                    name=f'Theoretical Quantile {quantile}'))

            fig.add_trace(go.Scatter(x=[empirical_quantiles[i], empirical_quantiles[i]], 
                                    y=[0, norm.pdf(empirical_quantiles[i], mean, std_dev)],
                                    mode='lines', line=dict(color='purple', dash='dot'), 
                                    name=f'Empirical Quantile {quantile}'))

        fig.update_layout(
            title="Theoretical vs Empirical Density",
            xaxis_title="X",
            yaxis_title="Density",
            showlegend=True,
            autosize=False,
            width=800, height=500
        )

        st.plotly_chart(fig)
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x_vals, y=theoretical_cdf, mode='lines', 
                                name='Theoretical CDF', line=dict(color='red', width=2)))

        fig.add_trace(go.Scatter(x=sorted_samples, y=empirical_cdf, mode='lines', 
                                name='Empirical CDF', line=dict(color='blue', width=2)))

        fig.update_layout(
            title="Theoretical vs Empirical Cumulative Distribution Function",
            xaxis_title="X",
            yaxis_title="CDF",
            showlegend=True,
            autosize=False,
            width=800, height=500
        )

        st.plotly_chart(fig)

    st.write(f"Generated {n_samples} samples from a Univariate Gaussian distribution with mean {mean} and variance {variance}.")

#################### PAGE 2 #####################

if page == pages[2]:

    st.title("Bivariate Gaussian Distribution")
    st.latex(r"f(\mathbf{X}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)")


    mean_1 = st.number_input('Mean of X1 (μ₁)', value=0.0)
    mean_2 = st.number_input('Mean of X2 (μ₂)', value=0.0)
    sigma2_1 = st.number_input('Variance of X1 (σ₁²)', value=1.0)
    sigma2_2 = st.number_input('Variance of X2 (σ₂²)', value=1.0)
    cov = st.number_input('Covariance (cov)', value=0.0)
    n_samples = st.number_input('Number of samples (n)', value=100)

    cov_matrix = [[sigma2_1, cov], [cov, sigma2_2]]

    mean_vector = [mean_1, mean_2]
    
    if st.button("Show Plots"):
        
        st.write(f"Generated {n_samples} samples from a Bivariate Gaussian distribution with:")
        st.latex(r'''
        \mu = \begin{pmatrix}
        ''' + str(mean_1) + r''' \\
        ''' + str(mean_2) + r'''
        \end{pmatrix}
        ''')
        st.latex(r'''
            \Sigma = \begin{bmatrix}
            \sigma_1^2 & \text{cov} \\
            \text{cov} & \sigma_2^2
            \end{bmatrix}
            =
            \begin{bmatrix}
            ''' + str(sigma2_1) + r''' & ''' + str(cov) + r''' \\
            ''' + str(cov) + r''' & ''' + str(sigma2_2) + r'''
            \end{bmatrix}
        ''')
        samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_samples)

        x_samples = samples[:, 0]
        y_samples = samples[:, 1]

        # Step 3: Theoretical Density (PDF) on a grid
        x_vals = np.linspace(np.min(x_samples), np.max(x_samples), 100)
        y_vals = np.linspace(np.min(y_samples), np.max(y_samples), 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        pos = np.dstack((X, Y))

        # theoretical density
        theoretical_pdf = multivariate_normal(mean=mean_vector, cov=cov_matrix).pdf(pos)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_samples, 
            y=y_samples, 
            mode='markers', 
            marker=dict(size=5, opacity=1), 
            name='Samples'
        ))

        fig.add_trace(go.Contour(
            x=x_vals, 
            y=y_vals, 
            z=theoretical_pdf, 
            colorscale='Reds', 
            contours=dict(showlabels=True), 
            name='Theoretical Density Contour'
        ))

        fig.update_layout(
            title="Bivariate Gaussian: Empirical and Theoretical Density with Contour Plots",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True,
            autosize=False,
            width=800, height=600
        )

        st.plotly_chart(fig)
