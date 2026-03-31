import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import math

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 1. EMA Helper
# ==========================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow, self.backup = {}, {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ==========================================
# 2. Data Generator
# ==========================================

def generate_concentric_hyperspheres(num_samples: int, intrinsic_dim: int, ambient_dim: int, seed=999) -> torch.Tensor:
    """
    Generates an inner sphere and an outer sphere.
    Creates a severe 'topological hole' (a radial vacuum) that traps standard SGMs.
    """
    n_half = num_samples // 2
    
    # Inner Sphere (Radius 1)
    x_inner = torch.randn(n_half, intrinsic_dim + 1)
    x_inner /= torch.norm(x_inner, dim=1, keepdim=True)
    
    # Outer Sphere (Radius 4)
    x_outer = torch.randn(num_samples - n_half, intrinsic_dim + 1)
    x_outer /= torch.norm(x_outer, dim=1, keepdim=True)
    x_outer *= 4.0 
    
    x_combined = torch.cat([x_inner, x_outer], dim=0)
    
    # Embed into ambient space
    padding = torch.zeros(num_samples, ambient_dim - (intrinsic_dim + 1))
    x_embedded = torch.cat([x_combined, padding], dim=1)
    
    # Apply fixed random rotation
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim))
    torch.set_rng_state(rng_state)
    
    return x_embedded @ q

def sliced_wasserstein_distance(x: torch.Tensor, y: torch.Tensor, num_projections: int = 1000):
    dim = x.shape[1]
    projections = torch.randn(dim, num_projections, device=x.device)
    projections /= torch.norm(projections, dim=0, keepdim=True)
    x_proj, _ = torch.sort(x @ projections, dim=0)
    y_proj, _ = torch.sort(y @ projections, dim=0)
    return torch.mean(torch.abs(x_proj - y_proj)).item()


# ==========================================
# 3. SDE Schedule
# ==========================================
class AdaptiveDriftSchedule:
    def __init__(self, intrinsic_dim: int, beta_smoothness: float = 2, t_0: float = 1e-6):
        self.d = intrinsic_dim
        self.beta = beta_smoothness
        self.t_0 = t_0
        self.gamma = (self.beta * max(self.d - 2, 0)) / (self.d * (self.beta + 1))

    def get_drift_coef(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t, min=self.t_0) ** (-self.gamma)

    def get_exact_marginals(self, t: torch.Tensor):
        t_safe = torch.clamp(t, min=self.t_0)
        integrated_alpha = (
            t_safe if self.gamma == 0.0
            else (t_safe ** (1 - self.gamma)) / (1 - self.gamma)
        )
        m_t = torch.exp(-integrated_alpha)
        sigma_t_sq = 1.0 - torch.exp(-2.0 * integrated_alpha)
        return m_t, sigma_t_sq

    def integrated_alpha_interval(self, t_low: torch.Tensor, t_high: torch.Tensor) -> torch.Tensor:
        t_lo = torch.clamp(t_low,  min=self.t_0)
        t_hi = torch.clamp(t_high, min=self.t_0)
        if self.gamma == 0.0:
            return t_hi - t_lo
        else:
            return (t_hi ** (1 - self.gamma) - t_lo ** (1 - self.gamma)) / (1 - self.gamma)

    def get_time_grid(self, steps: int, device=None) -> torch.Tensor:
        if self.gamma == 0.0:
            grid = torch.linspace(1.0, self.t_0, steps + 1)
        else:
            phi_T  = 1.0 ** (1 - self.gamma) / (1 - self.gamma)     
            phi_t0 = self.t_0 ** (1 - self.gamma) / (1 - self.gamma) 
            phi_grid = torch.linspace(phi_T, phi_t0, steps + 1)
            grid = ((1 - self.gamma) * phi_grid) ** (1 / (1 - self.gamma))

        if device is not None:
            grid = grid.to(device)
        return grid


# ==========================================
# 4. Score Network
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        time = time.squeeze(-1)                 
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]   
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ScoreMLP(nn.Module):
    def __init__(self, ambient_dim: int, hidden_dim: int = 1024, time_emb_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2), nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(ambient_dim + time_emb_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ambient_dim)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_mlp(t)], dim=1))


# ==========================================
# 5. Sampler
# ==========================================
def euler_maruyama_sampler(model, schedule: AdaptiveDriftSchedule, shape, steps=1500):
    model.eval()
    x = torch.randn(shape, device=DEVICE)
    time_steps = torch.linspace(1.0, schedule.t_0, steps, device=DEVICE)
    dt = (1.0 - schedule.t_0) / steps

    with torch.no_grad():
        for t_val in time_steps:
            t = torch.full((shape[0], 1), t_val.item(), device=DEVICE)
            alpha_t = schedule.get_drift_coef(t)
            score   = model(x, t)
            drift   = alpha_t * x + 2.0 * alpha_t * score
            z       = torch.randn_like(x) if t_val.item() > schedule.t_0 else torch.zeros_like(x)
            x       = x + drift * dt + torch.sqrt(2 * alpha_t) * z * math.sqrt(dt)
    return x.cpu()


# ==========================================
# 6. Training Loop
# ==========================================
def train_model_robust(data, schedule, ambient_dim, iterations=5000, desc="Training"):
    model     = ScoreMLP(ambient_dim=ambient_dim).to(DEVICE)
    ema       = EMA(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-6)
    data      = data.to(DEVICE)
    batch_size = min(512, data.shape[0])

    model.train()
    for _ in range(iterations):
        idx  = torch.randint(0, data.shape[0], (batch_size,), device=DEVICE)
        x_0  = data[idx]
        t    = torch.rand(batch_size, 1, device=DEVICE) * (1.0 - schedule.t_0) + schedule.t_0
        m_t, sigma_t_sq = schedule.get_exact_marginals(t)
        sigma_t = torch.sqrt(sigma_t_sq)
        z    = torch.randn_like(x_0)
        x_t  = m_t * x_0 + sigma_t * z
        loss = torch.mean(sigma_t_sq * (model(x_t, t) - (-z / sigma_t)) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()

    return model, ema


# ==========================================
# 7. Execution & Statistical Plotting
# ==========================================
if __name__ == "__main__":
    n_values = [100, 500, 1000, 5000, 10000]
    num_runs = 10  # Added 10 runs

    # Initialize matrices to store results: [runs, n_values]
    raw_results = {
        "Standard OU": np.zeros((num_runs, len(n_values))),
        "Adaptive Drift (Ours)": np.zeros((num_runs, len(n_values))),
    }

    intrinsic_dim = 4
    ambient_dim = 40
    iterations = 5000

    print(f"\nStarting {num_runs} separate runs for statistical validation...")
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        for i, n in enumerate(tqdm(n_values, desc=f"Evaluating Sizes (Run {run+1})")):
            # Generate fresh data per run to capture data sampling variance
            data = generate_concentric_hyperspheres(n, intrinsic_dim=intrinsic_dim, ambient_dim=ambient_dim, seed=run)
            true_data = generate_concentric_hyperspheres(2000, intrinsic_dim=intrinsic_dim, ambient_dim=ambient_dim, seed=run)

            ou_sched   = AdaptiveDriftSchedule(intrinsic_dim=2)   
            adp_sched  = AdaptiveDriftSchedule(intrinsic_dim=intrinsic_dim)   

            _, ema_ou  = train_model_robust(data, ou_sched,  ambient_dim=ambient_dim, iterations=iterations, desc="Training OU")
            _, ema_adp = train_model_robust(data, adp_sched, ambient_dim=ambient_dim, iterations=iterations, desc="Training Adp")

            ema_ou.apply_shadow()
            ema_adp.apply_shadow()

            def swd(samples):
                return sliced_wasserstein_distance(samples, true_data)

            # Store the SWD metric in our numpy arrays
            raw_results["Standard OU"][run, i] = swd(euler_maruyama_sampler(ema_ou.model, ou_sched, (2000, ambient_dim), steps=1000))
            raw_results["Adaptive Drift (Ours)"][run, i] = swd(euler_maruyama_sampler(ema_adp.model, adp_sched, (2000, ambient_dim), steps=1000))

    # ---- Plot ---------------------------------------------------------------
    styles = {
        "Standard OU": dict(color="red", marker="o", linestyle="-", linewidth=2.5),
        "Adaptive Drift (Ours)": dict(color="royalblue", marker="^", linestyle="-", linewidth=2.5),
    }

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    print("\n--- Final Statistical Results (10 Runs) ---")
    
    for label, data_matrix in raw_results.items():
        # Calculate statistics
        means = np.mean(data_matrix, axis=0)
        variances = np.var(data_matrix, axis=0)
        stds = np.std(data_matrix, axis=0)
        
        # Print variances for the paper/tables
        print(f"{label} Mean: {np.round(means, 5)}")
        print(f"{label} Variance: {np.round(variances, 7)}")
        
        # Plot mean line
        line, = ax.plot(n_values, means, label=label, **styles[label])
        
        # Add shaded region for standard deviation (confidence interval)
        ax.fill_between(n_values, means - stds, means + stds, color=line.get_color(), alpha=0.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (n)", fontsize=16)
    ax.set_ylabel("Sliced Wasserstein Distance (Mean ± Std)", fontsize=16)
    ax.set_title(f"SWD Scaling OU vs Adaptive (d={intrinsic_dim}, D={ambient_dim})\nAveraged over 10 runs", fontsize=16)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig("exp1_sphere.pdf", dpi=300)
    print("\nSaved as 'exp1_sphere.pdf'")
    plt.show()


