import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from metrics import WER
from utils.beam_search_utils import build_beam_search
from utils.utils import ids_to_str, set_requires_grad, strip_compile_prefix, UNIGRAM1000_LIST


class USREvaluator(LightningModule):
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
            model.load_state_dict(strip_compile_prefix(ckpt), strict=False)

        self.model = torch.compile(model) if cfg.compile_model else model

        self.ignore_id = -1
        self.beam_search_video = self.get_beam_search(self.model.model.backbone)
        self.beam_search_audio = self.get_beam_search(self.model.model.backbone)
        self.beam_search_av = self.get_beam_search(self.model.model.backbone)
        self.wer_video = WER()
        self.wer_audio = WER()
        self.wer_av = WER()

    def get_beam_search(self, model):
        token_list = UNIGRAM1000_LIST
        self.token_list = token_list
        return build_beam_search(model, self.cfg, token_list)

    def calculate_wer(self, video, audio, padding_mask, labels):
        labels = labels.squeeze(1)
        for vid, aud, label, mask in zip(video, audio, labels, padding_mask):
            feat_v, feat_a, feat_av = self.model.model.get_encoded_features(
                vid.unsqueeze(0), aud.unsqueeze(0), mask.unsqueeze(0).unsqueeze(-2)
            )

            nbest_hyps_v = self.beam_search_video(
                x=feat_v.squeeze(0),
                modality="v",
                maxlenratio=self.cfg.decode.maxlenratio,
                minlenratio=self.cfg.decode.minlenratio,
            )
            nbest_hyps_a = self.beam_search_audio(
                x=feat_a.squeeze(0),
                modality="a",
                maxlenratio=self.cfg.decode.maxlenratio,
                minlenratio=self.cfg.decode.minlenratio,
            )
            nbest_hyps_av = self.beam_search_av(
                x=feat_av.squeeze(0),
                modality="av",
                maxlenratio=self.cfg.decode.maxlenratio,
                minlenratio=self.cfg.decode.minlenratio,
            )

            nbest_hyps_v = [h.asdict() for h in nbest_hyps_v[:1]]
            nbest_hyps_a = [h.asdict() for h in nbest_hyps_a[:1]]
            nbest_hyps_av = [h.asdict() for h in nbest_hyps_av[:1]]

            transcription_v = add_results_to_json(nbest_hyps_v, self.token_list).replace("<eos>", "")
            transcription_a = add_results_to_json(nbest_hyps_a, self.token_list).replace("<eos>", "")
            transcription_av = add_results_to_json(nbest_hyps_av, self.token_list).replace("<eos>", "")

            label = label[label != self.ignore_id]
            groundtruth = ids_to_str(label, self.token_list)

            groundtruth = groundtruth.replace("\u2581", " ").strip()
            transcription_v = transcription_v.replace("\u2581", " ").strip()
            transcription_a = transcription_a.replace("\u2581", " ").strip()
            transcription_av = transcription_av.replace("\u2581", " ").strip()

            self.wer_video.update(transcription_v, groundtruth)
            self.wer_audio.update(transcription_a, groundtruth)
            self.wer_av.update(transcription_av, groundtruth)

    def test_step(self, data, batch_idx, dataloader_idx=0):
        lengths = torch.tensor(data["video_lengths"], device=data["video"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)
        self.calculate_wer(
            data["video"].squeeze(1),
            data["audio"].transpose(1, 2),
            padding_mask,
            data["label"],
        )

    def on_test_epoch_end(self):
        wer_video = self.wer_video.compute()
        wer_audio = self.wer_audio.compute()
        wer_av = self.wer_av.compute()
        self.log("wer_video", wer_video)
        self.log("wer_audio", wer_audio)
        self.log("wer_av", wer_av)
        self.wer_video.reset()
        self.wer_audio.reset()
        self.wer_av.reset()
