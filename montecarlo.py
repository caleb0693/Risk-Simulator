
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from scipy.stats import qmc

st.set_page_config(page_title="Quantitative Risk Simulator", layout="wide")
st.title("Quantitative Risk Simulator")

st.markdown("Created by [Caleb Ginorio González](https://www.linkedin.com/in/caleb-ginorio-gonzález-58b243b4/)")

            

tab1, tab2 = st.tabs(["📊 Simulation", "📘 Instructions"])

with tab1:


    st.sidebar.header("Simulation Setup")
    n_trials = st.sidebar.number_input("Number of Trials", 1000, 100000, 10000, step=1000)
    sampling_method = st.sidebar.radio("Sampling Method", ["Monte Carlo", "Latin Hypercube"])
    n_vars = st.sidebar.number_input("Number of Variables", 1, 20, 3, step=1)

    st.sidebar.markdown("---")
    st.sidebar.header("Define Variables")

    user_inputs = {}
    var_names = []
    var_units = {}
    preview_plots = {}

    for i in range(n_vars):
        name = st.sidebar.text_input(f"Variable {i+1} Name", key=f"name_{i}").strip()
        if not name:
            continue

        unit = st.sidebar.text_input(f"Units for {name}", key=f"unit_{i}")
        input_method = st.sidebar.selectbox(f"Input Method for {name}", ["Fixed", "Distribution", "Upload"], key=f"input_{i}")

        var_names.append(name)
        var_units[name] = unit

        if input_method == "Fixed":
            val = st.sidebar.number_input(f"Fixed Value for {name}", key=f"fixed_{i}")
            user_inputs[name] = val

        elif input_method == "Distribution":
            dist_type = st.sidebar.selectbox(f"Distribution for {name}", ["Normal", "Triangular", "Uniform", "Lognormal"], key=f"dist_{i}")

            if dist_type == "Normal":
                mean = st.sidebar.number_input(f"Mean ({name})", key=f"mean_{i}")
                std = st.sidebar.number_input(f"Std Dev ({name})", key=f"std_{i}")
                min_val = st.sidebar.number_input(f"Min Clip ({name})", value=0.0, key=f"minclip_{i}")
                max_val = st.sidebar.number_input(f"Max Clip ({name})", value=mean + 4 * std, key=f"maxclip_{i}")
                user_inputs[name] = ("Normal", (mean, std, min_val, max_val))
                dist_data = np.clip(np.random.normal(mean, std, n_trials), min_val, max_val)

            elif dist_type == "Triangular":
                left = st.sidebar.number_input(f"Min ({name})", key=f"min_{i}")
                mode = st.sidebar.number_input(f"Most Likely ({name})", key=f"mode_{i}")
                right = st.sidebar.number_input(f"Max ({name})", key=f"max_{i}")
                user_inputs[name] = ("Triangular", (left, mode, right))
                dist_data = np.random.triangular(left, mode, right, n_trials) if left < mode < right else np.zeros(n_trials)

            elif dist_type == "Uniform":
                low = st.sidebar.number_input(f"Min ({name})", key=f"low_{i}")
                high = st.sidebar.number_input(f"Max ({name})", key=f"high_{i}")
                user_inputs[name] = ("Uniform", (low, high))
                dist_data = np.random.uniform(low, high, n_trials) if low < high else np.zeros(n_trials)

            elif dist_type == "Lognormal":
                median = st.sidebar.number_input(f"Median ({name})", key=f"median_{i}")
                gsd = st.sidebar.number_input(f"GSD ({name})", min_value=0, key=f"gsd_{i}")

                if median > 0 and gsd >= 1:
                    mu = np.log(median)
                    sigma = np.log(gsd)
                    user_inputs[name] = ("Lognormal", (median, gsd))
                    dist_data = np.random.lognormal(mu, sigma, n_trials)
                    preview_plots[name] = px.histogram(dist_data, nbins=30, title=f"{name} Distribution Preview")
                else:
                    st.sidebar.warning(f"Enter valid Median and GSD values for {name}")
                    user_inputs[name] = ("Lognormal", (None, None))
                                    
            preview_plots[name] = px.histogram(dist_data, nbins=30, title=f"{name} Distribution Preview")
            user_inputs[name] = (dist_type, user_inputs[name][1])  # override to keep consistent

        elif input_method == "Upload":
            uploaded_file = st.sidebar.file_uploader(f"Upload data for {name}", type=["csv", "xlsx", "xls"], key=f"upload_{i}")
            if uploaded_file:
                if uploaded_file.name.endswith("csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
                    sheet_names = list(all_sheets.keys())
                    selected_sheet = st.sidebar.selectbox(f"Select sheet for {name}", sheet_names, key=f"sheet_{i}")
                    df = all_sheets[selected_sheet]
                selected_column = st.sidebar.selectbox(f"Select column for {name}", df.columns, key=f"col_{i}")
                custom_data = df[selected_column].dropna().values
                user_inputs[name] = custom_data
                preview_plots[name] = px.histogram(custom_data, nbins=30, title=f"{name} Uploaded Distribution Preview")

    st.subheader("Define the Forecast Target")
    forecast_var = st.text_input("Name of the variable you're forecasting (e.g., Dose, Concentration)", value="Output")

    st.subheader("Enter Equation")
    equation = st.text_area("Enter your equation using variable names (e.g., (C * IR * EF * ED) / (BW * AT))")
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", equation)
    undefined_vars = [tok for tok in tokens if tok not in var_names and tok not in ["abs", "min", "max", "np", "log", "exp", "ln"]]
    if undefined_vars:
        st.error(f"❌ Equation references undefined variable(s): {undefined_vars}")
        st.stop()

    st.subheader("Equation Preview")
    unit_expr = equation
    for var in var_names:
        unit_expr = unit_expr.replace(var, f"{var} [{var_units.get(var, '')}]")

    try:
        from sympy import symbols, latex
        from sympy.parsing.sympy_parser import parse_expr
        import sympy
        expr = parse_expr(equation, evaluate=False)
        var_syms = symbols(var_names)
        latex_expr = latex(expr)
        st.latex(f"{forecast_var} = " + latex_expr)
    except Exception as e:
        st.markdown(f"**Equation Preview Error:** {e}")


    st.divider()
    if preview_plots:
        st.subheader("Distribution Previews")
        for var, fig in preview_plots.items():
            st.plotly_chart(fig, use_container_width=True)

    if st.button("Run Simulation"):
        samples = {}
        for var in var_names:
            params = user_inputs[var]
            try:
                if isinstance(params, tuple):
                    dist_type, args = params
                    if dist_type == "Normal":
                        mean, std, min_val, max_val = args
                        base = np.random.normal(mean, std, n_trials)
                        samples[var] = np.clip(base, min_val, max_val)
                    elif dist_type == "Triangular":
                        left, mode, right = args
                        if left < mode < right:
                            samples[var] = np.random.triangular(left, mode, right, n_trials)
                        else:
                            samples[var] = np.full(n_trials, np.nan)
                    elif dist_type == "Uniform":
                        low, high = args
                        samples[var] = np.random.uniform(low, high, n_trials) if low < high else np.full(n_trials, np.nan)
                    elif dist_type == "Lognormal":
                        median, gsd = args
                        mu = np.log(median)
                        sigma = np.log(gsd)
                        samples[var] = np.random.lognormal(mu, sigma, n_trials)
                elif isinstance(params, np.ndarray):
                    samples[var] = np.random.choice(params, size=n_trials, replace=True)
                else:
                    samples[var] = np.full(n_trials, params)
            except Exception as e:
                st.error(f"Error in {var}: {e}")
                samples[var] = np.full(n_trials, np.nan)

        df = pd.DataFrame(samples)

        if sampling_method == "Latin Hypercube":
            sampler = qmc.LatinHypercube(d=len(df.columns))
            lhs_sample = sampler.random(n=n_trials)
            for j, var in enumerate(df.columns):
                col = df[var]
                if np.issubdtype(col.dtype, np.number):
                    sorted_vals = np.sort(col)
                    idxs = np.floor(lhs_sample[:, j] * len(sorted_vals)).astype(int)
                    idxs = np.clip(idxs, 0, len(sorted_vals) - 1)
                    df[var] = sorted_vals[idxs]

        try:
            safe_dict = {"__builtins__": None, "np": np, "log": np.log10, "exp": np.exp, "ln": np.log}
            parsed_equation = equation.replace("ln(", "np.log(")
            df[forecast_var] = eval(parsed_equation, safe_dict, df.to_dict(orient="series"))
        except Exception as e:
            st.error(f"Equation error: {e}")
            st.stop()

        st.subheader(f"📊 Forecast: {forecast_var}")
        st.plotly_chart(px.histogram(df, x=forecast_var, nbins=50, title=f"{forecast_var} Distribution"), use_container_width=True)

        st.divider()

        st.markdown("### Percentiles")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_vals = np.percentile(df[forecast_var].dropna(), percentiles)
        pct_df = pd.DataFrame({f"{forecast_var}": pct_vals}, index=[f"{p}%" for p in percentiles])
        st.table(pct_df.style.format(precision=4).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'left')]
        }]))

        st.divider()

        st.markdown("### Tornado Chart")
        corrs = df.corr(numeric_only=True)[forecast_var].drop(forecast_var)
        tornado = corrs.abs().sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(x=tornado.values, y=tornado.index, orientation='h', marker_color='darkorange'))
        fig2.update_layout(title="Tornado Chart", xaxis_title="|Correlation|", yaxis_title="Variable")
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button("Download CSV", df.to_csv(index=False), f"{forecast_var}_results.csv", "text/csv")

with tab2:
    st.markdown("""
    ## How to Use the Monte Carlo / Latin Hypercube Simulator

    1. **Set Simulation Parameters**  
       Use the sidebar to define the number of trials and whether to use Monte Carlo or Latin Hypercube sampling.

    2. **Define Variables**  
       - Name each variable and specify its unit.
       - Choose between:
         - Fixed Value
         - Probability Distribution (Normal, Uniform, Triangular, Lognormal)
         - Upload from file (CSV, XLSX, or XLS)
       - For distributions, you'll see a histogram preview.

    3. **Enter Equation**  
       - Write your custom equation using the variable names defined above.
       - Use standard Python math syntax (`*`, `/`, `**` for exponentiation).
       - Example: `(C * IR * EF * ED) / (BW * AT)`

    4. **Run Simulation**  
       Click “Run Simulation” to generate forecast output. You'll see:
       - Distribution chart of the forecast
       - Summary percentiles
       - Tornado chart showing variable influence

    > **Tip:** Make sure variable names in the equation exactly match those defined in the sidebar.
    """)
