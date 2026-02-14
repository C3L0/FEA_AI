# FEA-GNN: Structural Analysis Acceleration using Graph Attention Networks

This project implements a Deep Learning surrogate model designed to predict the deformation of mechanical structures (plates with holes) under tensile stress. By replacing traditional Finite Element Analysis (FEA) solvers like FEniCSx with a Graph Attention Network (GAT), this tool achieves acceleration factors between 125x and 350x while maintaining a relative error rate below 10%.

## Model Architecture

The architecture is designed to capture both local stress concentrations and global equilibrium:

Graph Attention Network (GATv2): Uses dynamic attention mechanisms to weight the importance of neighboring nodes. This allows the model to focus on high-stress gradients around geometric singularities (holes) and handle non-structured meshes effectively.

Global Context Layer: A global pooling and broadcast module that facilitates instantaneous information flow between boundary conditions (applied force and fixed supports) across the entire mesh, overcoming the slow diffusion limits of standard GCNs.

Physics-Informed Loss (PINN): Training is guided by both ground-truth data and physical residuals. The loss function includes components to minimize the violation of linear elasticity governing equations.

## Installation

### Prerequisites

Python 3.12 or higher

NVIDIA GPU with CUDA drivers (Highly recommended for training)

GMSH system library

uv (Fast Python package manager)

#### Option 1: Local Installation

The project uses uv to manage Deep Learning dependencies while relying on system-level libraries for FEniCSx data generation.
```
# Clone the repository
git clone https://github.com/C3L0/FEA_AI.git
cd FEA_AI

# Install dependencies
uv sync
```

#### Option 2: Docker Installation

Docker is the preferred method for deployment to avoid manual installation of FEniCSx and GMSH.

```
# Build the image
docker build -t fea-gnn .

# Run the inference interface (accessible on  0.0.0.0:8501)
docker run --gpus all -p 8501:8501 fea-gnn
```

## Complete Workflow

Follow these steps in order to replicate the full project pipeline.

#### Step 1: Data Generation (FEniCS)

This step uses the FEM solver to create the ground-truth dataset. It requires the system Python environment where dolfinx is installed.
(The data generation following this way have only worked on linux/Macos)

```
# Generates db.csv and connectivity.csv in the raw/ folder
export DOLFINX_ALLOW_USER_SITE_IMPORTS=1
mpirun -n 4 python3 -m data.data_generator_2d
```

```
# Generates db.csv and connectivity.csv in the generalization/ folder
export DOLFINX_ALLOW_USER_SITE_IMPORTS=1
mpirun -n 4 python3 -m data.generalized_shape

```
#### Step 2: Data Pipeline (Graph Construction)

Converts raw CSV simulation results into normalized PyTorch Geometric graph objects.

```
# Clean previous processed data if necessary
rm -rf data/processed/

# Run the conversion pipeline
uv run python -m data.data_pipeline
```

#### Step 3: Training

Starts the training process using the GAT architecture and the hybrid loss (Data MSE + Physics Residuals). Configuration parameters are defined in config.yaml.

```
uv run python -m src.fea_gnn.trainer
```

The trained weights are saved in the directory specified by the configuration file (typically models/gnn_hybrid_scaled.pth).

#### Step 4: Metrics and Evaluation

Launch the interactive dashboard to analyze model performance, view error heatmaps, and inspect individual predictions.

```
# Observe training metrics & results on a sample of inferences
uv run streamlit run src/fea_gnn/metric_dashboard.py

# Observe evaluation on one by one sample on regular and extrem stress conditions
uv run streamlit run src/fea_gnn/evaluation_dashboard.py

# To study the benefits of the model in time consumation
uv run python -m src.fea_gnn.benchmark --mode gnn
uv run python -m src.fea_gnn.benchmark --mode fenics
```

```
# Oberve generalization shape 
uv run streamlit run src/fea_gnn/test_generalization.py
```

#### Step 5: Live Inference (User Application)

Test the model on new, custom geometries generated on the fly using the integrated GMSH logic.

```
uv run streamlit run src/fea_gnn/app.py
```


## Project Structure

FEA_AI/ <br>
├── config.yaml                         # Hyperparameters (Architecture, Physics, Normalization) <br>
├── .github/workflows/python-test.yml   # GitAction config <br>
├── data/ <br>
│   ├── connectivity.csv                # (wrongly placed)Dataset : relation between nodes <br>
│   ├── db.csv                          # (wrongly placed)Dataset : nodes parameters <br>
│   ├── data_generator_2d.py            # FEniCSx script for FEM generation <br>
│   ├── generalized_shape.py            # FEniCSx script for FEM generation on untrained shape <br>
│   ├── data_pipeline.py                # CSV to PyG Graph converter <br>
│   ├── raw/                            # Raw simulation data (CSV format)(not used yet due to error in path) <br>
│   ├── generalization/                 # Raw simulation data (CSV format) for untrained shape <br>
│   └── processed/                      # Processed binary dataset (.pt) <br>
├── models/                             # Trained model weights (.pth) <br>
├── raw/                                # Raw simulation data (CSV format)(don't use/ can be remove) <br>
├── src/fea_gnn/ <br>
│   ├── model.py                        # GATv2 and Global Context architecture <br>
│   ├── trainer.py                      # Training loop and PINN loss logic <br>
│   ├── benchmark.py                    # Script to mesure the time of a FEM simulation <br>
│   ├── data_loader.py                  # PyTorch Dataset classes <br>
│   ├── eval_dashboard.py               # Metric calculation scripts <br>
│   ├── metric_dashboard.py             # Streamlit Analytics Dashboard <br>
│   ├── evaluator.py                    # (not usefull anymore)  <br>
│   ├── visualizer.py                   # (not usefull anymore) <br>
│   ├── utils.py                        # Utils methods <br>
│   ├── test_generalization.py          # Streamlit Analytics Generalized Shape <br>
│   └── app.py                          # Streamlit Live Simulation App <br>
├── tests/ <br>
│   ├── test_data.py                    # (not up-to-date) Test Data <br>
│   └── test_model.py                   # (not up-to-date) Test Model <br>
├── visualization/                      # (save img result) <br>
├── Dockerfile                          # Container configuration <br>
└── pyproject.toml                      # Project dependencies <br>


## Performance Results

Benchmark results on a test set of plates with varying dimensions, hole radii, and materials (Young's Modulus between 10 and 210 GPa):

Mean Absolute Error (MAE): ~0.03 mm

Median Relative Error: ~9.5%

Inference Speed: ~28ms (GAT) vs ~1500ms (FEniCSx)

## Authors

Antoine LOPEZ @C3L0
Paul ROUXEL @PaulRouxel
Victor JOUET @VictorJouet
Marwan Bemmousat @marouz94
Alexandra Cocuron @alexandracocuron
Mohamed Boukaoui @mohamedboukaoui

Final Year Projct (PFE) development.
