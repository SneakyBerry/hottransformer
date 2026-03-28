"""
Large model training with curriculum: simple → medium → hard.

python -m proof_gate_poc.train_large
"""
from .train import train, TrainConfig


def main():
    config = TrainConfig(
        # Model
        d_model=512,
        n_heads=8,
        n_encoder_layers=8,
        n_decoder_layers=8,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=128,
        max_proof_len=64,

        # Phase 1: Supervised — curriculum from simple to complex
        supervised_epochs=60,
        supervised_lr=3e-4,
        supervised_batch_size=256,

        # Phase 2: REINFORCE on full difficulty
        rl_epochs=30,
        rl_lr=5e-5,
        rl_batch_size=128,
        temperature=0.7,
        baseline_decay=0.99,
        length_penalty_scale=0.02,

        # Start with simple data — Phase 1 uses this
        n_train=20000,
        n_val=5000,

        # Simple: mostly basic connectives
        supervised_weights={
            "simple": 0.35, "medium": 0.25, "hard": 0.10,
            "trap": 0.05, "conjunction": 0.08, "disjunction": 0.08,
            "long_chain": 0.03, "nested_arrow": 0.03, "lambda_intro": 0.03,
            "mixed_connectives": 0.0, "deep_elimination": 0.0,
            "nested_case": 0.0, "complex_trap": 0.0,
        },

        # RL: full difficulty
        rl_weights={
            "simple": 0.08, "medium": 0.10, "hard": 0.10,
            "trap": 0.08, "conjunction": 0.06, "disjunction": 0.06,
            "long_chain": 0.10, "nested_arrow": 0.08, "lambda_intro": 0.08,
            "mixed_connectives": 0.08, "deep_elimination": 0.06,
            "nested_case": 0.05, "complex_trap": 0.07,
        },
    )

    print(f"Config: d_model={config.d_model}, layers={config.n_encoder_layers}+{config.n_decoder_layers}, "
          f"d_ff={config.d_ff}, heads={config.n_heads}")
    print(f"Data: {config.n_train} train, {config.n_val} val")
    print(f"Training: {config.supervised_epochs} supervised + {config.rl_epochs} RL epochs")
    print(f"Curriculum: simple→complex, then RL on full difficulty")
    print()

    model, metrics = train(config)

    # Now run online self-improvement on the trained model
    print("\n" + "=" * 60)
    print("Starting online self-improvement...")
    print("=" * 60)

    from .online import online_train, OnlineConfig
    online_config = OnlineConfig(
        batch_size=64,
        n_rounds=500,
        log_every=10,
        save_every=100,
        lr=1e-5,
        supervised_lr=5e-5,
        temperature=0.7,
        max_proof_len=64,
        update_mode="both",
    )
    online_train(model, online_config, config.get_device(), config)


if __name__ == "__main__":
    main()
