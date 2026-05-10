import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


class CleanTargetProjector(nn.Module):
    """
    Extract middle_block features from GuidedDDPMPlainUNet.
    Compare noisy mid-features to clean mid-features via cosine similarity.
    """

    def __init__(self, model, clean_dataloader, max_clean=None, device="cuda", mix=False):
        super().__init__()
        self.model = model.eval().to(device)
        self.device = device
        self.max_clean = max_clean
        self.mix = mix

        self._features = []

        def hook(module, inp, out):
            pooled = out.mean(dim=[2,3])   # [B, C]
            self._features.append(pooled.detach()) # calculate mean feature vector for each batch

        if not hasattr(self.model.network, "middle_block"):
            raise AttributeError("[ERROR] model.network does NOT have middle_block.")

        self.hook_handle = self.model.network.middle_block.register_forward_hook(hook)

        clean_feats, clean_imgs, filenames = self._extract_clean_features(clean_dataloader)

        if max_clean is not None and clean_feats.size(0) > max_clean:
            clean_feats = clean_feats[:max_clean]
            clean_imgs = clean_imgs[:max_clean]
            filenames = filenames[:max_clean]

        self.clean_feats = clean_feats.to(device)   # format: [K, C]
        self.clean_images = clean_imgs.to(device)   # format: [K, C, H, W]
        print(f"[INFO] Clean feature matrix: {self.clean_feats.size()}")

        if self.mix:
            self.mix_domains = {}
            for tag in ["20A", "10A", "5A", "3A"]:
                idx = [i for i, fn in enumerate(filenames) if f"_{tag}" in str(fn)]
                if idx:
                    self.mix_domains[tag] = torch.tensor(idx, device=device, dtype=torch.long)
            counts = {t: len(v) for t, v in self.mix_domains.items()}
            print(f"[INFO] Mix mode: {counts}")
        else:
            print(f"[INFO] Single-domain mode: all {self.clean_feats.size(0)} clean samples used directly")

        '''
        # --- background features ---
        if bg_dataloader is not None:
            bg_feats, _ = self._extract_clean_features(bg_dataloader)
            self.background_feat = F.normalize(
                bg_feats.mean(dim=0, keepdim=True), dim=1
            )
        else:
            self.background_feat = None
        self.background_feat = self.background_feat.to(self.device)
        '''
    
    @torch.no_grad()
    def refresh_clean_features(self, clean_dataloader):
        """
        Recompute clean features using the CURRENT model weights.
        """
        self._features = []

        clean_feats, clean_images, filenames = self._extract_clean_features(clean_dataloader)

        if self.max_clean is not None and clean_feats.size(0) > self.max_clean:
            clean_feats  = clean_feats[:self.max_clean]
            clean_images = clean_images[:self.max_clean]
            filenames = filenames[:self.max_clean]

        self.clean_feats  = clean_feats.to(self.device)    # [K, C]
        self.clean_images = clean_images.to(self.device)   # [K, C, H, W]

        if self.mix:
            self.mix_domains = {}
            for tag in ["20A", "10A", "5A", "3A"]:
                idx = [i for i, fn in enumerate(filenames) if f"_{tag}" in str(fn)]
                if idx:
                    self.mix_domains[tag] = torch.tensor(idx, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def _extract_clean_features(self, dataloader):
        feat_list = []
        img_list  = []
        filename_list = []

        was_training = self.model.training
        for batch in dataloader:
            clean = batch.get("clean_images", batch.get("image"))
            clean = clean.to(self.device, dtype=torch.float32)

            if clean.ndim == 4 and clean.shape[1] != 1:
                clean = clean.permute(0, 3, 1, 2)  # HWC → CHW

            self._features.clear()
            _ = self.model.network(clean)          # triggers hook
            assert len(self._features) == 1
            mid = self._features[0]

            feat_list.append(mid.cpu())
            img_list.append(clean.cpu())
            
            # Track filenames for domain separation
            file_path = batch.get("file_path", "unknown")
            if isinstance(file_path, (list, tuple)):
                file_path = file_path[0] if len(file_path) > 0 else "unknown"
            filename_list.append(file_path)

            self._features.clear()

        if was_training:
            self.model.train()

        feats = torch.cat(feat_list, dim=0)        # [K, C]
        imgs  = torch.cat(img_list,  dim=0)        # [K, C, H, W]
        return feats, imgs, filename_list

    @torch.no_grad()
    def similarity_all_clean(self, noisy_images, return_all=False):
        if noisy_images.ndim == 4 and noisy_images.shape[1] != 1:
            noisy_images = noisy_images.permute(0, 3, 1, 2)

        noisy_images = noisy_images.to(self.device).float()
        
        was_training = self.model.training
        self.model.eval()

        self._features.clear()
        _ = self.model.network(noisy_images)
        assert len(self._features) == 1
        noisy_feat = F.normalize(self._features[0], dim=1)
        self._features.clear()

        clean_feat = F.normalize(self.clean_feats, dim=1)
        sims = noisy_feat @ clean_feat.T   # [B, K]

        if was_training:
            self.model.train()

        if return_all:
            return sims
        else:
            best_sim, best_idx = sims.max(dim=1)
            return best_idx, best_sim

    @torch.no_grad()
    def get_weighted_target(self, noisy_images, tau=0.5):
        """
        Unified entry point.
        - mix=True:  softmax each resolution domain separately, average across domains.
        - mix=False: softmax over all clean images at once (original behaviour).
        Returns (y_combined [B,C,H,W], best_sim [B])
        """
        if self.mix:
            return self._get_weighted_target_multi_domain(noisy_images, tau=tau)
        else:
            return self._get_weighted_target_single(noisy_images, tau=tau)

    @torch.no_grad()
    def _get_weighted_target_single(self, noisy_images, tau=0.5):
        """Original single-domain softmax-weighted target."""
        if noisy_images.ndim == 4 and noisy_images.shape[1] != 1:
            noisy_images = noisy_images.permute(0, 3, 1, 2)
        noisy_images = noisy_images.to(self.device).float()

        was_training = self.model.training
        self.model.eval()

        self._features.clear()
        _ = self.model.network(noisy_images)
        assert len(self._features) == 1
        noisy_feat = F.normalize(self._features[0], dim=1)
        self._features.clear()

        clean_feat = F.normalize(self.clean_feats, dim=1)
        sims = noisy_feat @ clean_feat.T          # [B, K]
        weights = torch.softmax(sims / tau, dim=1)  # [B, K]
        y = torch.einsum("bk,kchw->bchw", weights, self.clean_images)
        best_sim = sims.max(dim=1).values

        if was_training:
            self.model.train()

        return y, best_sim

    @torch.no_grad()
    def _get_weighted_target_multi_domain(self, noisy_images, tau=0.5):
        """
        Softmax-weighted target with each resolution domain (20A/10A/5A/3A) contributing
        equally: softmax within each domain, then average across present domains.
        Returns (y_combined [B,C,H,W], best_sim [B])
        """
        if noisy_images.ndim == 4 and noisy_images.shape[1] != 1:
            noisy_images = noisy_images.permute(0, 3, 1, 2)
        noisy_images = noisy_images.to(self.device).float()

        was_training = self.model.training
        self.model.eval()

        self._features.clear()
        _ = self.model.network(noisy_images)
        assert len(self._features) == 1
        noisy_feat = F.normalize(self._features[0], dim=1)
        self._features.clear()

        clean_feat = F.normalize(self.clean_feats, dim=1)
        sims_all = noisy_feat @ clean_feat.T   # [B, K]

        domain_targets = []
        for indices in self.mix_domains.values():
            sims_d   = sims_all[:, indices]                          # [B, K_d]
            weights_d = torch.softmax(sims_d / tau, dim=1)          # [B, K_d]
            imgs_d   = self.clean_images[indices]                    # [K_d, C, H, W]
            domain_targets.append(torch.einsum("bk,kchw->bchw", weights_d, imgs_d))

        if not domain_targets:
            raise ValueError("No domain samples found in clean dataset for mix mode!")

        y_combined = sum(domain_targets) / len(domain_targets)
        best_sim = sims_all.max(dim=1).values

        if was_training:
            self.model.train()

        return y_combined, best_sim