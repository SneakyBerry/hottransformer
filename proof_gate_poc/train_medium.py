"""
5M model — full pipeline with all tools.

python -m proof_gate_poc.train_medium
"""
from .train import train, TrainConfig
from .online import online_train, OnlineConfig


def main():
    config = TrainConfig(
        # Model
        d_model=256,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        dropout=0.1,
        max_seq_len=64,
        max_proof_len=32,

        # Phase 1: Supervised — learn the language of proofs
        supervised_epochs=30,
        supervised_lr=3e-4,
        supervised_batch_size=128,

        # Phase 2: RL — learn to search (iterative, with backtracking)
        rl_epochs=30,
        rl_lr=5e-5,
        rl_batch_size=32,
        temperature=0.8,

        # Data
        n_train=20000,
        n_val=2000,

        supervised_weights={
            "simple": 0.25, "medium": 0.20, "hard": 0.10,
            "trap": 0.05, "conjunction": 0.06, "disjunction": 0.06,
            "long_chain": 0.06, "nested_arrow": 0.06, "lambda_intro": 0.06,
            "mixed_connectives": 0.04, "deep_elimination": 0.03,
            "nested_case": 0.02, "complex_trap": 0.01,
        },
        rl_weights={
            "simple": 0.10, "medium": 0.15, "hard": 0.12,
            "trap": 0.08, "conjunction": 0.06, "disjunction": 0.06,
            "long_chain": 0.08, "nested_arrow": 0.08, "lambda_intro": 0.08,
            "mixed_connectives": 0.07, "deep_elimination": 0.05,
            "nested_case": 0.04, "complex_trap": 0.03,
        },
    )

    model, metrics = train(config)

    # Phase 3: Online self-improvement with all tools
    print("\n" + "=" * 60)
    print("PHASE 3: Online self-improvement")
    print("Lemma memory + adaptive curriculum + infinite data")
    print("=" * 60)

    online_config = OnlineConfig(
        batch_size=32,
        n_rounds=1000,
        log_every=10,
        save_every=100,
        lr=1e-5,
        supervised_lr=5e-5,
        temperature=0.7,
        max_proof_len=32,
        update_mode="both",
        iterative=True,
    )
    online_train(model, online_config, config.get_device(), config)


if __name__ == "__main__":
    main()
