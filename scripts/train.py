"""
train.py
--------
Main training loop for Backprop-NEAT on classification datasets.

Algorithm per generation:
    1. For each genome in population:
        a. train_genome() builds JAX forward fn, runs B Adam steps,
           writes weights back into genome, returns (w, loss, fwd, conn_keys)
        b. Fitness = -cross_entropy - lambda * sqrt(K+)
    2. neat-python handles speciation, selection, crossover, mutation
    3. After every generation: print stats, append to history CSV

Usage:
    python train.py --dataset xor --generations 50 --b_steps 300 --lam 0.01
    python train.py --dataset spiral --generations 100 --b_steps 600 --lam 0.005
"""

import sys, os, argparse, time, json, pickle, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neat
import jax.numpy as jnp

from datasets.datasets import get_dataset
from neat_core.genome_to_jax import extract_weights
from training.backprop import train_genome, clear_cache
from training.fitness import compute_fitness, compute_accuracy


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Backprop-NEAT trainer")
    p.add_argument("--dataset",     type=str,   default="xor",
                   choices=["xor", "circle", "spiral", "gaussian"])
    p.add_argument("--generations", type=int,   default=50)
    p.add_argument("--b_steps",     type=int,   default=300,
                   help="Adam steps per genome per generation (B)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--lam",         type=float, default=0.01,
                   help="Complexity penalty coefficient lambda")
    p.add_argument("--penalty",     type=str,   default="sqrt",
                   choices=["sqrt", "linear"])
    p.add_argument("--n_data",      type=int,   default=200)
    p.add_argument("--noise",       type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--config",      type=str,   default="config/xor.cfg")
    p.add_argument("--out_dir",     type=str,   default="results",
                   help="Directory to save winner, history, logs")
    return p.parse_args()


# ------------------------------------------------------------------
# Evaluation function (neat-python calls this each generation)
# ------------------------------------------------------------------

def make_eval_fn(data, config, b_steps, lr, lam, penalty):
    """
    Closure over dataset and hyperparameters.
    neat-python calls eval_genomes(genomes, config) each generation.

    Uses new train_genome signature:
        w, loss, forward_fn, conn_keys = train_genome(genome, config, X, y, ...)
    """
    X = data["X_train"]
    y = data["y_train"]

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            w_opt, loss, fwd, conn_keys = train_genome(
                genome, config, X, y,
                b_steps=b_steps, lr=lr
            )
            genome.fitness = compute_fitness(
                fwd, w_opt, genome, X, y,
                lam=lam, penalty=penalty
            )

    return eval_genomes


# ------------------------------------------------------------------
# Per-generation stats (called manually in the run loop)
# ------------------------------------------------------------------

def generation_stats(gen, population, species_set, config,
                     data, lam, penalty, elapsed):
    """
    Compute and return stats dict for one generation.
    Uses weights already written into each genome by train_genome —
    no second forward pass needed.
    """
    fitnesses = [g.fitness for g in population.values()
                 if g.fitness is not None]
    best_genome  = max(
        (g for g in population.values() if g.fitness is not None),
        key=lambda g: g.fitness
    )

    # Re-run forward fn on best genome using its current (backpropped) weights.
    # b_steps=0 skips the training loop entirely — just builds fwd and extracts w.
    _, _, fwd, conn_keys = train_genome(
        best_genome, config,
        data["X_train"], data["y_train"],
        b_steps=0
    )
    w = extract_weights(best_genome, conn_keys)

    train_acc = compute_accuracy(fwd, w, data["X_train"], data["y_train"])
    test_acc  = compute_accuracy(fwd, w, data["X_test"],  data["y_test"])
    n_nodes   = len(best_genome.nodes)
    n_conns   = sum(1 for c in best_genome.connections.values() if c.enabled)
    n_species = len(species_set.species)

    stats = {
        "generation":    gen,
        "best_fitness":  max(fitnesses),
        "mean_fitness":  sum(fitnesses) / len(fitnesses),
        "best_train_acc": train_acc,
        "best_test_acc":  test_acc,
        "best_nodes":    n_nodes,
        "best_conns":    n_conns,
        "n_species":     n_species,
        "elapsed_s":     round(elapsed, 1),
    }

    print(
        f"gen={gen:3d} | "
        f"fit={stats['best_fitness']:7.4f} (mean={stats['mean_fitness']:7.4f}) | "
        f"train={train_acc:.3f}  test={test_acc:.3f} | "
        f"nodes={n_nodes}  conns={n_conns} | "
        f"species={n_species} | "
        f"t={elapsed:.1f}s"
    )

    return stats, best_genome


# ------------------------------------------------------------------
# Save utilities
# ------------------------------------------------------------------

def save_winner(winner, out_dir, dataset_name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"winner_{dataset_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(winner, f)
    print(f"  Winner saved -> {path}")


def save_history(history, out_dir, dataset_name):
    os.makedirs(out_dir, exist_ok=True)

    # JSON (full)
    json_path = os.path.join(out_dir, f"history_{dataset_name}.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # CSV (for quick plotting / spreadsheet)
    csv_path = os.path.join(out_dir, f"history_{dataset_name}.csv")
    if history:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print(f"  History saved -> {json_path}  {csv_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Backprop-NEAT  |  dataset={args.dataset.upper()}")
    print(f"  generations={args.generations}  B={args.b_steps}  "
          f"lr={args.lr}  lambda={args.lam}  penalty={args.penalty}")
    print(f"{'='*60}\n")

    # --- Dataset ---
    data = get_dataset(args.dataset, n=args.n_data,
                       noise=args.noise, seed=args.seed)
    print(f"Dataset: {data['name']}  "
          f"train={data['X_train'].shape[0]}  "
          f"test={data['X_test'].shape[0]}\n")

    # --- Config ---
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        args.config
    )

    # --- Population ---
    clear_cache()   # fresh JIT cache for this run
    pop = neat.Population(config)

    # --- Eval function ---
    eval_fn = make_eval_fn(
        data, config, args.b_steps, args.lr, args.lam, args.penalty
    )

    # --- Generation loop ---
    # --- Generation loop ---
    history = []
    winner  = None
    best_ever_test_acc = 0.0
    best_ever = None
    t_run   = time.time()

    print("gen   | best_fit  (mean_fit) | train  test  | nodes conns | species | time")
    print("-" * 75)

    for gen in range(1, args.generations + 1):
        t0 = time.time()
        pop.run(eval_fn, 1)
        elapsed = time.time() - t0

        stats, best = generation_stats(
            gen, pop.population, pop.species, config,
            data, args.lam, args.penalty, elapsed
        )
        history.append(stats)
        winner = best

        if stats["best_test_acc"] >= best_ever_test_acc:
            best_ever_test_acc = stats["best_test_acc"]
            best_ever = best
            save_winner(best_ever, args.out_dir, f"{args.dataset}_best_test")

        # Save checkpoint every 10 generations
        if gen % 10 == 0:
            save_winner(winner, args.out_dir, args.dataset)
            save_history(history, args.out_dir, args.dataset)

    # --- Final save ---
    save_winner(winner, args.out_dir, args.dataset)
    save_history(history, args.out_dir, args.dataset)

    total = time.time() - t_run
    print(f"\n{'='*60}")
    print(f"  Done. Total time: {total/60:.1f} min")
    print(f"  Winner: nodes={len(winner.nodes)}  "
          f"conns={sum(1 for c in winner.connections.values() if c.enabled)}")
    print(f"  Results in: {args.out_dir}/")
    print(f"{'='*60}\n")

    return winner, history, config


if __name__ == "__main__":
    main()
    print(f"  Winner: nodes={len(winner.nodes)}  "
          f"conns={sum(1 for c in winner.connections.values() if c.enabled)}")
    print(f"  Best test acc: {best_ever_test_acc:.3f}  "
          f"(saved as winner_{args.dataset}_best_test.pkl)")
