# Link to access the pricer : pricer-6pwzvjpyb4zxg5wxvga5ej.streamlit.app

# Black-Scholes Option Strategy Dashboard

An interactive Streamlit dashboard for pricing and analyzing option strategies using the Black-Scholes model.

## Features

-   **Pricing & Greeks**: Computes theoretical prices and Greeks (Delta, Gamma, Theta, Vega, Rho) for individual legs and combined strategies.
-   **Predefined Strategies**: Quickly load common strategies like Straddles, Iron Condors, Butterflies, and Spreads.
-   **Custom Leg Builder**: Add any number of custom call/put legs with specific strikes and positions (Long/Short).
-   **Interactive Plotting**:
    -   Visualize Payoff, Time Value, Premium, and Greeks vs. Underlying Price.
    -   Overlay multiple metrics on a single graph.
    -   Generate separate comparative graphs for each metric.
-   **Break-even Analysis**: Automatically identifies and marks break-even points on the payoff diagram.
-   **Dark Theme**: Sleek, modern interface with a premium dark theme.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/option-greeks-simulator.git
    cd option-greeks-simulator
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Accès à l'application en ligne

[6pwzvjpyb4zxg5wxvga5ej.streamlit.app](https://6pwzvjpyb4zxg5wxvga5ej.streamlit.app)

## Technologies Used

-   **Python**: Core logic and financial calculations.
-   **Streamlit**: Web interface and interactivity.
-   **NumPy & SciPy**: Vectorized Black-Scholes calculations.
-   **Matplotlib**: Custom styled financial plotting.
-   **Pandas**: Data handling.

## License

This project is open-source and available under the [MIT License](LICENSE).
