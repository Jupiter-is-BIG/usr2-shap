from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.scorers.length_bonus import LengthBonus


def build_beam_search(backbone, cfg, token_list):
    """Construct a BatchBeamSearch over a USR backbone's decoder + CTC scorers.

    Factored out of evaluator.py::USREvaluator.get_beam_search so that other
    LightningModules (e.g. shap_evaluator.py::SHAPEvaluator) can build the same
    beam search without depending on USREvaluator or duplicating this logic.

    :param backbone: the E2E model (cfg.model.obj instance's .model.backbone)
    :param cfg: the Hydra config (uses cfg.decode.{ctc_weight,penalty,beam_size})
    :param token_list: the vocabulary (list[str])
    :return: a configured BatchBeamSearch instance
    """
    odim = len(token_list)

    scorers = backbone.scorers()
    scorers["length_bonus"] = LengthBonus(len(token_list))

    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        length_bonus=cfg.decode.penalty,
    )
    return BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )
