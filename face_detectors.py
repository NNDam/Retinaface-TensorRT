from detectors.retinaface import RetinaFace

def get_retinaface(model_path, backend, outputs, rac, masks=False):

    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs)
    model = RetinaFace(inference_backend=inference_backend, rac=rac, masks=masks)
    return model


def retinaface_r50_v1(model_path, backend, outputs):
    model = get_retinaface(model_path, backend, outputs, rac="net3")
    return model


def retinaface_mnet025_v1(model_path, backend, outputs):
    model = get_retinaface(model_path, backend, outputs, rac="net3")
    return model


def retinaface_mnet025_v2(model_path, backend, outputs):
    model = get_retinaface(model_path, backend, outputs, rac="net3l")
    return model
