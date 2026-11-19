"""Opt-in memory reductions used by long-clip offline inference."""


def configure_inference_memory(model, *, offload_backbone=False):
    """Configure an eval model before constructing a GraphRunner.

    Under no_grad, discard intermediate decoder predictions after building
    their attention masks and release consumed pyramid features. All decoder
    layers still run. Optionally keep cached backbone features in CPU RAM,
    loading one layer at a time for the heads. Training/grad-enabled forwards
    retain their normal outputs and storage behavior.
    """
    if model.training:
        raise ValueError("configure_inference_memory requires model.eval()")
    model.query_decoder.return_aux_outputs = False
    model.mask_feature_head.release_features = True
    model.backbone.offload_tokens = bool(offload_backbone)
    return model
