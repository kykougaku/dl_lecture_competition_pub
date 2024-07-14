import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from torchvision.transforms import v2
import timm
from timm.models import create_model

from src.datasets import ThingsMEGDataset_mod1d
from src.models import EEGNet, myEEGnet
from src.utils import set_seed

import gc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    train_set = ThingsMEGDataset_mod1d("train", args.data_dir)
    train_set.transform = transform
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset_mod1d("val", args.data_dir)
    val_set.transform = transform
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)


    # ------------------
    #       Model
    # ------------------

    model = myEEGnet(in_channel=275, kernels=[3,5,7,9], classes=1854)
    model.to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X = torch.cat([X, subject_idxs], dim=1)
            X, y = X.to(args.device), y.to(args.device)


            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X = torch.cat([X, subject_idxs], dim=1)
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    del train_loader, val_loader
    gc.collect()
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    test_set = ThingsMEGDataset_mod1d("test", args.data_dir)
    test_set.transform = transform
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        X = torch.cat([X, subject_idxs], dim=1)
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()