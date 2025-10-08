import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import minimize
import plotly.graph_objects as graph_obj


st.set_page_config(page_title="Maximum Likelihood Estimation (MLE)", layout="centered")

st.title('MLE Visualization')

def load_data(file_name="./weatherHistory.csv"):
    df = pd.read_csv('weatherHistory.csv')[:100]
    return df


intermediate_results = []
def neg_log_likelihood(params):
    intermediate_results.append(params.copy())
    a, b, log_sigma = params
    sigma = np.exp(log_sigma)
    resid = y - (a + b*x)
    return 0.5 * n * np.log(2*np.pi) + n * log_sigma + 0.5 * np.sum((resid**2) / (sigma**2))


data = load_data()


data = data[["Temperature (C)", "Humidity"]]

x = data["Temperature (C)"].values
y = data["Humidity"].values
n = len(y)

init = np.array([2, 2, np.log(2)])
res = minimize(neg_log_likelihood, init, method="L-BFGS-B")
mle_a, mle_b, mle_log_sigma = res.x
mle_sigma = np.exp(mle_log_sigma)

print("\n--- Numerical MLE estimates ---")
print(f"Intercept (a): {mle_a:.4f}")
print(f"Slope     (b): {mle_b:.4f}")
print(f"Sigma        : {mle_sigma:.4f}")
print("Optimization success:", res.success)

# -------------------------------
# 4. Plots
# -------------------------------

print("Number of iterations:", len(intermediate_results))
intermediate_results = intermediate_results[-50:]

xx = np.linspace(x.min(), x.max(), 200)
yy = mle_a + mle_b * xx

# Initialize session state to keep track of current line index
if 'line_index' not in st.session_state:
    st.session_state.line_index = 0

# Button to add next line
if st.button("Add next probable line"):
    if st.session_state.line_index < len(intermediate_results):
        st.session_state.line_index += 1


fig = graph_obj.Figure()
# Define fixed axis limits (field of view)
x_min, x_max = x.min() - 1, x.max() + 1
y_min, y_max = 0, 1

# Add scatter plot (points)
fig.add_trace(graph_obj.Scatter(
    x=x,
    y=y,
    mode='markers',  # markers only
    name='Scatter'
))


# Lines added according to current line_index
colors = px.colors.sequential.Viridis  # color palette
xx = np.linspace(x.min(), x.max(), 200)




for i in range(st.session_state.line_index):
    # fig_placeholder.plotly_chart(fig, use_container_width=True)
    a, b, _ = intermediate_results[i]
    yy = a + b * xx
    fig.add_trace(graph_obj.Scatter(
        x=xx,
        y=yy,
        mode='lines',
        name=f'Iteration {i+1}',
        line=dict(color=colors[i % len(colors)], width=2)
    ))

# Layout
fig.update_layout(
    title="Scatter plot showing multiple iterations fit on MLE Equation using scipy.optimize.minimize",
    xaxis=dict(title="Temperature  (C)", range=[x_min, x_max]),
    yaxis=dict(title="Humidity", range=[y_min, y_max])
)



st.plotly_chart(fig, use_container_width=True)



# Final Optimal Line

fig2 = graph_obj.Figure()

# Add scatter plot (points)
fig2.add_trace(graph_obj.Scatter(
    x=x,
    y=y,
    mode='markers',  # markers only
    name='Scatter'
))

# Add line plot
fig2.add_trace(graph_obj.Scatter(
    x=xx,
    y=yy,
    mode='lines',  # line only
    name='Line',
))

# ±σ shaded region (data fluctuation area)
fig2.add_trace(graph_obj.Scatter(
    x=np.concatenate([xx, xx[::-1]]),
    y=np.concatenate([yy + mle_sigma, (yy - mle_sigma)[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',  # light red transparent area
    line=dict(color='rgba(255,255,255,0)'),
    name='±1σ range'
))

fig2.update_layout(
    title="Best Fit Line with region showing for ±1σ ",
    xaxis=dict(title="Temperature  (C)", range=[x_min, x_max]),
    yaxis=dict(title="Humidity", range=[y_min, y_max])
)


# Use session state to track button click
if "show_plot" not in st.session_state:
    st.session_state.show_plot = False

if st.button("Show Figure"):
    st.session_state.show_plot = True

# Only display if button pressed
if st.session_state.show_plot:
    st.plotly_chart(fig2)



# slides


# --- Initialize session state for navigation ---
if "slide" not in st.session_state:
    st.session_state.slide = 1

# --- Navigation buttons ---
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("◀ Previous", disabled=st.session_state.slide == 1):
        st.session_state.slide -= 1
with col3:
    if st.button("Next ▶", disabled=st.session_state.slide == 6):
        st.session_state.slide += 1

st.markdown("---")

# --- SLIDE 1: TITLE ---
if st.session_state.slide == 1:
    st.title("📊 Maximum Likelihood Estimation (MLE)")
    st.write("Choose parameters that maximizes the probability of observing the data.")

# --- SLIDE 2: INTUITION ---
elif st.session_state.slide == 2:
    st.header("🎯 The Intuition")
    st.markdown("""
    Suppose we have data points \( x1, x2, ....., xn \)  
    and we believe they come from a distribution with parameters $\\theta$.

    **Goal:** Find the parameter $\\theta$ that makes the observed data *most likely*.
    """)
    st.latex(r"L(\theta) = P(x_1, x_2, \dots, x_n \mid \theta)")
    st.info("In simple terms: Which parameters make our data most probable?")

# --- SLIDE 3: MATH FORMULATION ---
elif st.session_state.slide == 3:
    st.header("📐 Mathematical Formulation")
    st.markdown("If data points are independent:")
    st.latex(r"L(\theta) = \prod_{i=1}^n P(x_i \mid \theta)")
    st.markdown("MLE estimate:")
    st.latex(r"\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta)")

# --- SLIDE 4: NORMAL DISTRIBUTION EXAMPLE ---
elif st.session_state.slide == 4:
    st.header("📊 Example — Normal Distribution")
    st.latex(r"x_i \sim \mathcal{N}(\mu, \sigma^2)")
    st.latex(r"P(x_i \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i - \mu)^2}{2\sigma^2}}")
    st.latex(r"L(x_n \mid \mu, \sigma) = P(x_1 \mid \mu, \sigma) * P(x_2 \mid \mu, \sigma) * \cdots * P(x_n \mid \mu, \sigma)")

    st.latex(r"""
        \text{Maximize: } \ \log(L(\theta)) 
        \;\;\Longrightarrow\;\;
        \text{Minimize: } -\log(L(\theta))
        """)

    st.latex(r"""
        \text{Gradient Descent (partial derivatives)} 
        \;\;\Longrightarrow\;\;
        \text{Optimal Parameters } \theta
        \;\;\Longrightarrow\;\;
        (\mu, \sigma)
        """)

# --- SLIDE 5: OPTIMIZATION DEMO ---
elif st.session_state.slide == 5:
    st.header("⚙️ MLE via Optimization (Numerical Example)")

    np.random.seed(42)
    true_mu, true_sigma = 5, 2
    data = np.random.normal(true_mu, true_sigma, size=100)

    def neg_log_likelihood(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        n = len(data)
        return 0.5 * n * np.log(2 * np.pi) + n * log_sigma + np.sum((data - mu) ** 2) / (2 * sigma ** 2)

    init = np.array([0.0, np.log(1.0)])
    res = minimize(neg_log_likelihood, init, method="L-BFGS-B")
    mle_mu, mle_log_sigma = res.x
    mle_sigma = np.exp(mle_log_sigma)

    st.markdown(f"**True μ:** {true_mu:.2f} | **MLE μ:** {mle_mu:.2f}")
    st.markdown(f"**True σ:** {true_sigma:.2f} | **MLE σ:** {mle_sigma:.2f}")

    # Plot histogram + fitted curve
    fig, ax = plt.subplots()
    x_vals = np.linspace(min(data), max(data), 100)
    y_true = (1/(true_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-true_mu)/true_sigma)**2)
    y_est = (1/(mle_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-mle_mu)/mle_sigma)**2)
    ax.hist(data, bins=20, density=True, alpha=0.3, label="Data")
    ax.plot(x_vals, y_true, 'g--', label="True PDF")
    ax.plot(x_vals, y_est, 'r-', label="MLE PDF")
    ax.legend()
    st.pyplot(fig)

# --- SLIDE 6: ROLE IN APPLICATIONS ---
elif st.session_state.slide == 6:
    st.header("💡 Roles")
    st.markdown("""
     - Understanding Distribution
     - Predict Probabilities, Testing Hypothesis
    - Finding "Best Fit" Parameters to the Model
    """)

elif st.session_state.slide == 7:
    st.title("🙏 Thank You!")
    st.write("")
    st.subheader("It was great sharing my work with you.")
    st.write("---")
    st.write("**Questions or feedback are welcome.**")
    st.write("Bishal Gautam")

st.markdown("---")
st.write(f"Slide {st.session_state.slide} / 6")






