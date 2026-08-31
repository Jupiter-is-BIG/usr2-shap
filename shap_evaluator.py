import os

import numpy as np
import shap
import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from metrics import WER
from utils.beam_search_utils import build_beam_search
from utils.utils import ids_to_str, strip_compile_prefix, UNIGRAM1000_LIST

IGNORE_INDEX = -1


class SHAPEvaluator(LightningModule):
    """
    Computes Shapley-based audio/video modality contributions for USR's
    audiovisual decoding path, following Dr. SHAP-AV's methodology adapted
    to USR's CTC/attention encoder-decoder architecture (as opposed to
    Dr-SHAP-AV's LLM-concatenation architectures):

      1. Run the audio/video CNN frontends once (Encoder.run_frontend).
      2. Fuse + encode the *unmasked* features (Encoder.fuse_from_frontend)
         and beam-search-decode a baseline hypothesis (audiovisual modality).
      3. For each SHAP coalition, zero out the masked-out audio/video
         frontend timesteps, re-fuse+re-encode, and teacher-force the
         baseline hypothesis through the decoder in a single forward pass
         to get the characteristic function f_x^t(C) (per-token log-prob of
         the baseline hypothesis under that coalition) -- mirroring
         Dr-SHAP-AV's f_shap for LLM-based models.
      4. Feed that characteristic function to the `shap` library
         (SamplingExplainer / PermutationExplainer) to get the Shapley
         matrix, then aggregate into Global audio/video SHAP (Eq. 6 of the
         Dr. SHAP-AV paper).

    Masking happens on the pre-fusion frontend outputs (before linear_av),
    matching the paper's treatment of MLP/concatenation-fusion models
    (Auto-AVSR/AV-HuBERT) in Dr-SHAP-AV, §3.3.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model = instantiate(cfg.model.obj, cfg)

        if cfg.model.pretrained_model_path:
            if ".ckpt" in cfg.model.pretrained_model_path:
                ckpt = torch.load(cfg.model.pretrained_model_path, map_location="cpu", weights_only=False)["state_dict"]
            else:
                ckpt = torch.load(cfg.model.pretrained_model_path, map_location="cpu")
            # Load into the plain (uncompiled) module first, regardless of compile_model:
            # a checkpoint saved from a torch.compile-wrapped model has every key
            # prefixed with '_orig_mod.', which only matches a compiled module's
            # state_dict. Stripping it and loading before compiling keeps key-matching
            # correct either way. See utils.utils.strip_compile_prefix.
            missing, unexpected = model.load_state_dict(strip_compile_prefix(ckpt), strict=False)
            if missing or unexpected:
                print(
                    f"WARNING: load_state_dict found {len(missing)} missing and "
                    f"{len(unexpected)} unexpected keys -- checkpoint may not have "
                    "loaded correctly. First few of each:"
                )
                print("  missing:", missing[:5])
                print("  unexpected:", unexpected[:5])

        self.model = torch.compile(model) if cfg.compile_model else model

        self.ignore_id = IGNORE_INDEX
        self.token_list = UNIGRAM1000_LIST
        self.beam_search_av = build_beam_search(self.model.model.backbone, cfg, self.token_list)
        self.wer_av = WER()

    def compute_shap_sample(self, video, audio):
        """
        Args:
            video: (1, T, H, W) trimmed to the sample's true (unpadded) length.
            audio: (1, T*640, 1) trimmed to match.
        Returns:
            audio_pct, video_pct: Global A/V-SHAP (Eq. 6), floats in [0, 1].
            num_audio_tokens: N_a, the number of audio frontend timesteps
                used in the Shapley matrix (== N_v here, since usr2's audio
                and video frontends both run at 25 Hz -- see plan notes).
            shap_matrix: np.ndarray (N_a + N_v, T_out), audio rows first.
            transcription: the decoded (audiovisual) hypothesis text.
        """
        encoder = self.model.model.backbone.encoder
        decoder = self.model.model.backbone.decoder

        xs_v_front, xs_a_front = encoder.run_frontend(xs_v=video, xs_a=audio)
        N_v = xs_v_front.shape[1]
        N_a = xs_a_front.shape[1]
        assert N_a == N_v, (
            f"audio/video frontend length mismatch ({N_a} vs {N_v}); SHAP assumes "
            "matching per-timestep audio/video features (both frontends run at 25 Hz)."
        )
        p = N_a + N_v

        full_feat_av = encoder.fuse_from_frontend(xs_v_front, xs_a_front)

        nbest_hyps = self.beam_search_av(
            x=full_feat_av.squeeze(0),
            modality="av",
            maxlenratio=self.cfg.decode.maxlenratio,
            minlenratio=self.cfg.decode.minlenratio,
        )
        yseq = nbest_hyps[0].yseq  # [sos, t1, ..., tN, eos]
        tgt_in = yseq[:-1].unsqueeze(0)
        targets = yseq[1:]
        L = targets.shape[0]
        tgt_mask = subsequent_mask(tgt_in.shape[1], device=tgt_in.device).unsqueeze(0)
        idx = torch.arange(L, device=tgt_in.device)

        def f_shap(mask_vec):
            mask_vec = torch.tensor(mask_vec, dtype=torch.bool, device=video.device)
            mask_a = mask_vec[:N_a]
            mask_v = mask_vec[N_a:]

            xs_a_masked = xs_a_front.clone()
            xs_v_masked = xs_v_front.clone()
            if (~mask_a).any():
                xs_a_masked[:, ~mask_a, :] = 0.0
            if (~mask_v).any():
                xs_v_masked[:, ~mask_v, :] = 0.0

            feat_av = encoder.fuse_from_frontend(xs_v_masked, xs_a_masked)
            y, _ = decoder.forward(tgt_in, tgt_mask, feat_av, None)
            logp = torch.log_softmax(decoder.out_layer_av(y), dim=-1).squeeze(0)
            return logp[idx, targets].detach().cpu().numpy()

        def shap_wrapper(masks):
            if masks.ndim == 1:
                return f_shap(masks)
            return np.array([f_shap(m) for m in masks])

        background = np.zeros((1, p), dtype=np.float32)
        x_explain = np.ones((1, p), dtype=np.float32)

        if self.cfg.shap.shap_alg == "sampling":
            explainer = shap.SamplingExplainer(model=shap_wrapper, data=background)
            shap_values = explainer.shap_values(x_explain, nsamples=self.cfg.shap.num_samples_shap)
        else:
            from shap.maskers import Independent

            masker = Independent(background, max_samples=100)
            explainer = shap.PermutationExplainer(model=shap_wrapper, masker=masker, algorithm="auto")
            shap_values = explainer(x_explain, max_evals=self.cfg.shap.num_samples_shap, silent=False).values

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]

        # CRITICAL: shap returns (p, T_out) -- do NOT transpose.
        mm_abs = np.abs(shap_values).sum(axis=1)
        audio_abs = mm_abs[:N_a].sum()
        video_abs = mm_abs[N_a:].sum()
        total_abs = audio_abs + video_abs
        audio_pct = audio_abs / total_abs
        video_pct = video_abs / total_abs

        nbest_hyps_json = [h.asdict() for h in nbest_hyps[:1]]
        transcription = add_results_to_json(nbest_hyps_json, self.token_list).replace("<eos>", "")
        transcription = transcription.replace("▁", " ").strip()

        return audio_pct, video_pct, N_a, shap_values, transcription

    def test_step(self, data, batch_idx, dataloader_idx=0):
        video = data["video"].squeeze(1)  # (B, T, H, W)
        audio = data["audio"].transpose(1, 2)  # (B, T*640, 1)
        lengths = data["video_lengths"]
        labels = data["label"].squeeze(1)

        for vid, aud, length, label in zip(video, audio, lengths, labels):
            vid = vid[:length].unsqueeze(0)
            aud = aud[: length * 640].unsqueeze(0)

            label = label[label != self.ignore_id]
            groundtruth = ids_to_str(label, self.token_list).replace("▁", " ").strip()

            audio_pct, video_pct, num_audio_tokens, shap_matrix, transcription = self.compute_shap_sample(vid, aud)

            self.audio_shap_abs.append(audio_pct)
            self.video_shap_abs.append(video_pct)
            self.num_audio_tokens.append(num_audio_tokens)
            self.shapley_values.append(shap_matrix)

            self.wer_av.update(transcription, groundtruth)

            self.log("sample-audio-ABS-SHAP", audio_pct, on_step=True, on_epoch=False, prog_bar=False)
            self.log("sample-video-ABS-SHAP", video_pct, on_step=True, on_epoch=False, prog_bar=False)
            self.log("sample-num-audio-tokens", float(num_audio_tokens), on_step=True, on_epoch=False, prog_bar=False)

    def on_test_epoch_start(self):
        self.audio_shap_abs = []
        self.video_shap_abs = []
        self.num_audio_tokens = []
        self.shapley_values = []

        self.output_file = None
        if self.cfg.shap.output_path_shap is not None:
            self.output_file = os.path.join(self.cfg.shap.output_path_shap, self.cfg.project_name)
            print("Output dir: ", self.output_file)

    def on_test_epoch_end(self):
        wer_av = self.wer_av.compute()
        self.log("wer_av", wer_av)
        self.wer_av.reset()

        overall_audio_abs = np.mean(self.audio_shap_abs)
        overall_video_abs = np.mean(self.video_shap_abs)
        overall_num_audio_tokens = np.mean(self.num_audio_tokens)

        std_overall_audio_abs = np.std(self.audio_shap_abs)
        std_overall_video_abs = np.std(self.video_shap_abs)

        self.log("audio-ABS-SHAP", overall_audio_abs)
        self.log("video-ABS-SHAP", overall_video_abs)
        self.log("STD_audio-ABS-SHAP", std_overall_audio_abs)
        self.log("STD_video-ABS-SHAP", std_overall_video_abs)
        self.log("num-audio-tokens", overall_num_audio_tokens)

        print("Global Audio-ABS-SHAP :", overall_audio_abs * 100, "%")
        print("Global Video-ABS-SHAP :", overall_video_abs * 100, "%")
        print("WER (AV) :", wer_av.item() * 100, "%")

        if self.output_file is not None:
            np.savez_compressed(
                self.output_file,
                audio_abs=np.array(self.audio_shap_abs),
                video_abs=np.array(self.video_shap_abs),
                num_audio_tokens=np.array(self.num_audio_tokens),
                shap_values=np.array(self.shapley_values, dtype=object),
            )
