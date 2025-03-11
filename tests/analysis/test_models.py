import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_load_smpl():
    from easymocap.smplmodel.body_param import load_model

    smplx_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../model_data/smplx/")
    model_path = os.getenv("SMPLX_PATH", smplx_path)

    body_model = load_model(model_path=model_path)

    assert body_model is not None, "SMPL Model not loaded correctly"
