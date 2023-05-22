from trainer import TrainerGAN


if __name__ == "__main__":
    config = {
        "model_type": "WGANGP",
        "batch_size": 64,
        "epochs": 51,
        "n_critic": 2,
        "z_dim": 100,
        "data_dir": "./faces",
        "clip_value": 1,
        "save_dir": "./generate"
    }
    trainer = TrainerGAN(config)
    trainer.train()
