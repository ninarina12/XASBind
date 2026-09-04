import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from xasbind.models.spectrum_encoder import InOutKernel
from xasbind.models.structure_encoder import InOut

from xasbind.models.spectrum_encoder import SpectrumEncoder
from xasbind.models.structure_encoder import StructureEncoder
from xasbind.models.combined_encoders import XASBind
from xasbind.losses.infonce import InfoNCELoss

from xasbind.data.dataset import make_dataloaders

import time
import os

def collect_embeddings(model, dataloader, device):
    all_struct, all_xanes, all_exafs = [], [], []
    model.eval()
    with torch.no_grad():
        for xanes, exafs, graph_batch in dataloader:
            xanes, exafs, graph_batch = xanes.to(device), exafs.to(device), graph_batch.to(device)
            s, x, e = model.encode(xanes, exafs, graph_batch)
            all_struct.append(s.cpu())
            all_xanes.append(x.cpu())
            all_exafs.append(e.cpu())
    return torch.cat(all_struct), torch.cat(all_xanes), torch.cat(all_exafs)

def main():
    xanes_encoder = SpectrumEncoder(
        [
            InOutKernel(1, 16, 7),
            InOutKernel(16, 32, 7),
            InOutKernel(32, 64, 5),
            InOutKernel(64, 128, 3),
            InOutKernel(128, 256, 3)
        ]
    )
    exafs_encoder = SpectrumEncoder(
        [
            InOutKernel(1, 16, 7),
            InOutKernel(16, 32, 7),
            InOutKernel(32, 64, 7),
            InOutKernel(64, 128, 5),
            InOutKernel(128, 256, 3),
            InOutKernel(256, 256, 3)
        ]
    )
    structure_encoder = StructureEncoder(
        [
            InOut("64x0e", "32x0e + 32x1o + 16x2e"),
            InOut("32x0e + 32x1o + 16x2e", "32x0e + 32x1o + 16x2e"),
            InOut("32x0e + 32x1o + 16x2e", "32x0e + 32x1o + 16x2e"),
            InOut("32x0e + 32x1o + 16x2e", "256x0e")
        ]
    )
    loss_function = InfoNCELoss()

    model = XASBind(xanes_encoder, exafs_encoder, structure_encoder, loss_function)

    mats_path = "data/xas/matproj"
    train_dl, valid_dl, test_dl = make_dataloaders(xanes_npy=mats_path + "/processed_xanes.npy",
                                                   exafs_npy=mats_path + "/processed_exafs.npy",
                                                   structs_pt=mats_path + "/processed_structs.pt",
                                                   train_inds_npy=mats_path + "/train_inds.npy",
                                                   valid_inds_npy=mats_path + "/valid_inds.npy",
                                                   test_inds_npy=mats_path + "/test_inds.npy",
                                                   batch_size=256)

    snapshot_dir = os.path.join(mats_path, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_every = 2

    num_epochs = 100
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_valid_loss = float('inf')

    log = {"train_loss": [], "valid_loss": [],
        "train_sx": [], "train_se": [],
        "valid_sx": [], "valid_se": [],
        "temperature": [], "epoch_time": []}

    for epoch in range(num_epochs):
        start_epoch = time.time()

        train_loss, tsx_loss, tse_loss, valid_loss, vsx_loss, vse_loss = 0, 0, 0, 0, 0, 0
        train_batches = len(train_dl)
        valid_batches = len(valid_dl)

        if epoch == 20:
            for param in model.xanes_enc.parameters():
                param.requires_grad = False
            for param in model.exafs_enc.parameters():
                param.requires_grad = False
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=1e-3
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch, last_epoch=-1)
            print("  -> Froze spectrum encoders")

        model.train()
        for xanes, exafs, graph_batch in train_dl:
            xanes, exafs, graph_batch = xanes.to(device), exafs.to(device), graph_batch.to(device)
            loss, sx_loss, se_loss = model(xanes, exafs, graph_batch)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            tsx_loss += sx_loss.item()
            tse_loss += se_loss.item()

        model.eval()
        with torch.no_grad():
            for xanes, exafs, graph_batch in valid_dl:
                xanes, exafs, graph_batch = xanes.to(device), exafs.to(device), graph_batch.to(device)
                loss, sx_loss, se_loss = model(xanes, exafs, graph_batch)
                valid_loss += loss.item()
                vsx_loss += sx_loss.item()
                vse_loss += se_loss.item()

        avg_val_loss = valid_loss / valid_batches
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), "data/xas/matproj/best.pt")

        scheduler.step()
        epoch_duration = time.time() - start_epoch

        avg_train_loss = train_loss / train_batches
        temp = model.loss_fn.learnable_temp.exp().item()

        log["train_loss"].append(avg_train_loss)
        log["valid_loss"].append(avg_val_loss)
        log["train_sx"].append(tsx_loss / train_batches)
        log["train_se"].append(tse_loss / train_batches)
        log["valid_sx"].append(vsx_loss / valid_batches)
        log["valid_se"].append(vse_loss / valid_batches)
        log["temperature"].append(temp)
        log["epoch_time"].append(epoch_duration)

        print(f"Epoch: {epoch}, Duration: {epoch_duration:.1f}s")
        print(f"Training Loss: {avg_train_loss:.4f} (SX: {tsx_loss / train_batches:.4f}, SE: {tse_loss / train_batches:.4f})")
        print(f"Validation Loss: {avg_val_loss:.4f} (SX: {vsx_loss / valid_batches:.4f}, SE: {vse_loss / valid_batches:.4f})")
        print(f"Temperature: {temp:.4f}")

        if (epoch % snapshot_every == 0) or (epoch == num_epochs - 1):
            struct_embs, xanes_embs, exafs_embs = collect_embeddings(model, valid_dl, device)
            torch.save({"struct": struct_embs, "xanes": xanes_embs, "exafs": exafs_embs},
                       os.path.join(snapshot_dir, f"epoch_{epoch:03d}.pt"))
            print(f"  -> Saved validation embeddings snapshot (epoch {epoch})")

    torch.save(log, mats_path + "/training_log.pt")
    print("Training done.")
    model.eval()
    test_loss, esx_loss, ese_loss = 0, 0, 0
    test_batches = len(test_dl)
    with torch.no_grad():
        for xanes, exafs, graph_batch in test_dl:
            xanes, exafs, graph_batch = xanes.to(device), exafs.to(device), graph_batch.to(device)
            loss, sx_loss, se_loss = model(xanes, exafs, graph_batch)
            test_loss += loss.item()
            esx_loss += sx_loss.item()
            ese_loss += se_loss.item()

    print(f"Testing Loss: {test_loss / test_batches} (Structure-XANES loss: {esx_loss / test_batches}, Structure-EXAFS loss: {ese_loss / test_batches})")
    struct_embs, xanes_embs, exafs_embs = collect_embeddings(model, test_dl, device)
    torch.save({"struct": struct_embs, "xanes": xanes_embs, "exafs": exafs_embs},
               mats_path + "/test_embeddings.pt")
    print("Saved test embeddings.")

def sanity_check():
    from xasbind.models.spectrum_encoder import SpectrumEncoder, InOutKernel
    from xasbind.models.structure_encoder import StructureEncoder, InOut
    from xasbind.models.combined_encoders import XASBind
    from xasbind.losses.infonce import InfoNCELoss
    from xasbind.data.dataset import make_dataloaders
    import torch
    import time

    print("=== SANITY CHECK ===")

    # 1. Build model
    print("Building model...")
    xanes_encoder = SpectrumEncoder([
        InOutKernel(1, 16, 7), InOutKernel(16, 32, 7), InOutKernel(32, 64, 5),
        InOutKernel(64, 128, 3), InOutKernel(128, 256, 3)
    ])
    exafs_encoder = SpectrumEncoder([
        InOutKernel(1, 16, 7), InOutKernel(16, 32, 7), InOutKernel(32, 64, 7),
        InOutKernel(64, 128, 5), InOutKernel(128, 256, 3), InOutKernel(256, 256, 3)
    ])
    structure_encoder = StructureEncoder([
        InOut("64x0e", "32x0e + 32x1o + 16x2e"),
        InOut("32x0e + 32x1o + 16x2e", "32x0e + 32x1o + 16x2e"),
        InOut("32x0e + 32x1o + 16x2e", "256x0e")
    ])
    loss_function = InfoNCELoss()
    model = XASBind(xanes_encoder, exafs_encoder, structure_encoder, loss_function)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = model.to(device)

    # 2. Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. Load one batch (small batch size to be quick)
    print("Loading one batch...")
    mats_path = "data/xas/matproj"
    train_dl, _, _ = make_dataloaders(
        xanes_npy=mats_path + "/processed_xanes.npy",
        exafs_npy=mats_path + "/processed_exafs.npy",
        structs_pt=mats_path + "/processed_structs.pt",
        train_inds_npy=mats_path + "/train_inds.npy",
        valid_inds_npy=mats_path + "/valid_inds.npy",
        test_inds_npy=mats_path + "/test_inds.npy",
        batch_size=8
    )
    xanes, exafs, graph_batch = next(iter(train_dl))
    xanes = xanes.to(device)
    exafs = exafs.to(device)
    graph_batch = graph_batch.to(device)
    print(f"XANES shape: {xanes.shape}")
    print(f"EXAFS shape: {exafs.shape}")
    print(f"Graph batch: {graph_batch.num_graphs} graphs, {graph_batch.num_nodes} nodes, {graph_batch.edge_index.shape[1]} edges")

    # 4. Forward pass
    print("Forward pass...")
    t0 = time.time()
    model.train()
    loss, sx_loss, se_loss = model(xanes, exafs, graph_batch)
    t1 = time.time()
    print(f"Forward time: {t1 - t0:.3f}s")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Struct-XANES loss: {sx_loss.item():.4f}")
    print(f"Struct-EXAFS loss: {se_loss.item():.4f}")
    print(f"Temperature: {model.loss_fn.learnable_temp.exp().item():.4f}")

    # 5. Backward pass
    print("Backward pass...")
    t0 = time.time()
    loss.backward()
    t1 = time.time()
    print(f"Backward time: {t1 - t0:.3f}s")

    # 6. Check gradients exist
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"All gradients computed: {grad_ok}")

    # 7. Check for NaN in gradients
    any_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"Any NaN gradients: {any_nan}")

    print("=== SANITY CHECK PASSED ===" if (grad_ok and not any_nan) else "=== SANITY CHECK FAILED ===")


if __name__ == "__main__":
    main()