import argparse


def main():
    ap = argparse.ArgumentParser(description="Estimate training wall-time and cost from a measured throughput.")
    ap.add_argument("--samples-per-second", type=float, default=None,
                    help="measured train_samples_per_second (from HF Trainer output)")
    ap.add_argument("--steps", type=int, default=None, help="alternative: steps run in a short measured run")
    ap.add_argument("--seconds", type=float, default=None, help="alternative: seconds that run took")
    ap.add_argument("--batch", type=int, default=8, help="per-step batch size (for the --steps/--seconds route)")
    ap.add_argument("--num-examples", type=int, required=True, help="training-set size")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--gpu-rate", type=float, default=1.80, help="$ per GPU-hour (cloud on-demand)")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--overhead", type=float, default=1.3, help="multiplier for download / checkpoint / restarts")
    args = ap.parse_args()

    if args.samples_per_second:
        sps = args.samples_per_second
    elif args.steps and args.seconds:
        sps = args.steps * args.batch / args.seconds
    else:
        ap.error("give --samples-per-second, or both --steps and --seconds")

    total_samples = args.num_examples * args.epochs
    compute_hours = total_samples / (sps * 3600)
    wall_hours = compute_hours * args.overhead
    gpu_hours = wall_hours * args.num_gpus
    cost = gpu_hours * args.gpu_rate

    print(f"measured throughput = {sps:.3f} samples/s")
    print(f"work                = {int(total_samples):,} samples ({args.num_examples:,} x {args.epochs} epochs)")
    print(f"compute time        = {compute_hours:.2f} h")
    print(f"wall time (x{args.overhead} ovh)  = {wall_hours:.2f} h")
    print(f"gpu-hours           = {gpu_hours:.2f}  ({args.num_gpus} gpu)")
    print(f"estimated cost      = ${cost:,.2f}  (@ ${args.gpu_rate}/gpu-h)")


if __name__ == "__main__":
    main()
