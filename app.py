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
    st.title("Gaussian Vectors")
    st.latex(r"f(\mathbf{X}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)")

#################### PAGE 1 #####################

if page == pages[1]:
    
    st.title('Univariate Gaussian Distribution')
    st.header('Theoretical vs Empirical Density')
    
    st.latex(r'''
    f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)
    ''')
    
    # Input number of samples (n)
    n_samples = st.number_input('Number of samples (n)', value=100)

    # Input mean (mu) and variance (sigma^2)
    mean = st.number_input('Mean (μ)', value=0.0)
    variance = st.number_input('Variance (σ²)', value=1.0)
    std_dev = np.sqrt(variance)  # Standard deviation

    if st.button("Show Plots"):

        # Step 2: Generate the Univariate Gaussian samples
        samples = np.random.normal(loc=mean, scale=std_dev, size=n_samples)

        # Calculate the Theoretical and Empirical Quantiles
        quantiles = [0.05, 0.5, 0.95]  # Percentiles: 5%, 50%, 95%
        theoretical_quantiles = norm.ppf(quantiles, loc=mean, scale=std_dev)  # Theoretical quantiles
        empirical_quantiles = np.percentile(samples, [5, 50, 95])  # Empirical quantiles
        
        st.write(f"Generated {n_samples} sample from a Univariate Gaussian distribution with mean {mean} and variance {variance}.")

        # Step 3: Create bins for histogram and calculate the theoretical density curve
        x_vals = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 1000)
        theoretical_density = norm.pdf(x_vals, mean, std_dev)
        theoretical_cdf = norm.cdf(x_vals, loc=mean, scale=std_dev)  
        
        # ecdf
        sorted_samples = np.sort(samples)  # Sort the samples for CDF
        empirical_cdf = np.arange(1, n_samples + 1) / n_samples

        # Step 4: Plot the histogram (empirical density) and overlay the theoretical density curve
        fig = go.Figure()

        # Histogram of the samples (empirical density)
        fig.add_trace(go.Histogram(x=samples, nbinsx=50, histnorm='probability density', 
                                name='Empirical Density', marker_color='blue', opacity=0.6))

        # Theoretical density curve
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

        # Update layout for better visibility
        fig.update_layout(
            title="Theoretical vs Empirical Density",
            xaxis_title="X",
            yaxis_title="Density",
            showlegend=True,
            autosize=False,
            width=800, height=500
        )

        # Step 5: Display the plot in Streamlit
        st.plotly_chart(fig)
        
        fig = go.Figure()

        # Plot Theoretical CDF
        fig.add_trace(go.Scatter(x=x_vals, y=theoretical_cdf, mode='lines', 
                                name='Theoretical CDF', line=dict(color='red', width=2)))

        # Plot Empirical CDF
        fig.add_trace(go.Scatter(x=sorted_samples, y=empirical_cdf, mode='lines', 
                                name='Empirical CDF', line=dict(color='blue', width=2)))

        # Update layout for better visibility
        fig.update_layout(
            title="Theoretical vs Empirical Cumulative Distribution Function",
            xaxis_title="X",
            yaxis_title="CDF",
            showlegend=True,
            autosize=False,
            width=800, height=500
        )

        # Step 6: Display the plot in Streamlit
        st.plotly_chart(fig)

    # Display additional information
    st.write(f"Generated {n_samples} samples from a Univariate Gaussian distribution with mean {mean} and variance {variance}.")

#################### PAGE 2 #####################

if page == pages[2]:

    # Streamlit Inputs
    st.title("Bivariate Gaussian Distribution")
    st.latex(r"f(\mathbf{X}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)")


    # Input parameters for bivariate Gaussian distribution
    mean_1 = st.number_input('Mean of X1 (μ₁)', value=0.0)
    mean_2 = st.number_input('Mean of X2 (μ₂)', value=0.0)
    sigma2_1 = st.number_input('Variance of X1 (σ₁²)', value=1.0)
    sigma2_2 = st.number_input('Variance of X2 (σ₂²)', value=1.0)
    cov = st.number_input('Covariance (cov)', value=0.0)
    n_samples = st.number_input('Number of samples (n)', value=100)

    # Step 1: Create the covariance matrix
    cov_matrix = [[sigma2_1, cov], [cov, sigma2_2]]

    # Step 2: Generate the Bivariate Gaussian samples
    mean_vector = [mean_1, mean_2]
    
    if st.button("Show Plots"):
        
        # Display additional information
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

        # Extract X and Y components from the samples
        x_samples = samples[:, 0]
        y_samples = samples[:, 1]

        # Step 3: Calculate the Theoretical Density (PDF) on a grid
        x_vals = np.linspace(np.min(x_samples), np.max(x_samples), 100)
        y_vals = np.linspace(np.min(y_samples), np.max(y_samples), 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        pos = np.dstack((X, Y))

        # Multivariate normal for theoretical density
        theoretical_pdf = multivariate_normal(mean=mean_vector, cov=cov_matrix).pdf(pos)

        # Step 4: Plot Empirical Density (using a 2D histogram) and Theoretical Density
        fig = go.Figure()

        # Plot scatter plot for sample points
        fig.add_trace(go.Scatter(
            x=x_samples, 
            y=y_samples, 
            mode='markers', 
            marker=dict(size=5, opacity=1), 
            name='Samples'
        ))

        # Theoretical contour plot
        fig.add_trace(go.Contour(
            x=x_vals, 
            y=y_vals, 
            z=theoretical_pdf, 
            colorscale='Reds', 
            contours=dict(showlabels=True), 
            name='Theoretical Density Contour'
        ))

        # Update layout
        fig.update_layout(
            title="Bivariate Gaussian: Empirical and Theoretical Density with Contour Plots",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True,
            autosize=False,
            width=800, height=600
        )

        # Step 5: Display the plot in Streamlit
        st.plotly_chart(fig)
