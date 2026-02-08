# Kiiking Simulation Model

This folder contains the Python scripts to simulate the motion of a Kiiking swing (variable-length pendulum).

## 📂 File Structure

* **`run_experiment.py`**: **Run this file.** It sets the parameters and launches the simulation.
* **`model.py`**: Contains the physics equations (equations of motion and energy).
* **`controls.py`**: Defines the "Squat-Stand" strategy ($r$ vs. $\theta, \dot{\theta}$).
* **`plotting.py`**: Helper functions for graphs and animation.

## 🚀 How to Run

1.  **Install dependencies** (if you haven't already):
    ```bash
    pip install numpy scipy matplotlib
    ```

2.  **Run the simulation**:
    Open your terminal or command prompt in this folder and type:
    ```bash
    python main.py
    ```
    * *Note:* The animation window might pop up behind your editor.

## ⚙️ Key Parameters to Change (in `main.py`)

If you want to experiment, edit these variables inside the `main()` function in `main.py`:

* **`r_arm`**: Length of the swing arms (e.g., `7.0` meters).
* **`A`**: Squat amplitude (e.g., `0.145` = 29cm range of motion).
* **`k`**: Sharpness of the squat (Higher = robotic/fast, Lower = smooth/human).
* **`theta0`**: Initial angle in radians (e.g., `0.785` is approx 45 degrees).
* **`t_span`**: How long the simulation runs (e.g., `(0.0, 200.0)` seconds).

## 🐞 Common Issues

* **"Overflow Warning":** If you see a warning about `cosh` or `overflow`, ignore it. It just means the transition between squat/stand is very sharp. The code handles it safely.
