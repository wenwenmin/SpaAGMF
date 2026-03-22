import json
import os
import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from dataset.dataset import MineDataset, load_data
from train import TrainingManager
from utils import EMACallback


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    all_seed_results = {}
    for seed in cfg.seeds:
        print(f"\n==================== Running seed = {seed} ====================")

        all_results = {}
        pl.seed_everything(seed, workers=True)

        #  cross-platform&batch validation
        for test_name in cfg.dataset.tgt_names:
            # 1. Data partition
            print(f"********************{test_name} is being tested  ********************")
            train_names = cfg.dataset.ref_names

            # 2. Create the training manager
            ema_cb = EMACallback(cfg.dataset.ema_decay)
            train_manager = TrainingManager(train_names, [test_name], cfg, ema_cb, seed)

            # 3. Training
            train_manager.train()

            # 4. Get the trained model
            model = train_manager.cls_model

            # 5. Testing
            test_data = load_data([test_name], cfg)
            test_dataset = MineDataset(test_data, mode='cls', cfg=cfg, is_train=False)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.cls.training.batch_size,
                shuffle=False,
                num_workers=0
            )
            ema_test_cb = EMACallback()
            ema_test_cb.ema_state_dict = ema_cb.ema_state_dict
            log_dir = f'{os.getcwd()}/{seed}/{test_name}'
            os.makedirs(log_dir, exist_ok=True)
            trainer = pl.Trainer(
                default_root_dir=log_dir,
                callbacks=[ema_test_cb],
                precision=16,
            )
            trainer.test(model, dataloaders=test_dataloader)
            all_results[test_name] = model.result

            # 6. Release resource
            trainer.strategy.teardown()
            del trainer
            ema_test_cb.ema_state_dict.clear()
            del ema_test_cb
            del test_dataloader
            del test_dataset
            del model
            if hasattr(train_manager, 'cls_model'):
                train_manager.cls_model.ctt_model = None
            del train_manager
            torch.cuda.empty_cache()
            import gc
            gc.collect()


        all_seed_results[seed] = all_results

        # 7. Saving the LOOCV result
        save_dir = Path(cfg.output_dir) if "output_dir" in cfg else Path("./outputs")
        save_dir.mkdir(parents=True, exist_ok=True)

        json_path = save_dir / "cross_validation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_seed_results, f, indent=4)

        print(f"Cross-validation results saved to: {json_path}")




if __name__ == "__main__":
    main()
