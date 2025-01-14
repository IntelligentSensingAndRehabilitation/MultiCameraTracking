import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_load_smpl():
    from easymocap.smplmodel.body_param import load_model
    smpl_path = '/home/isr/app/packages/EasyMocap/data/smplx'
    body_model = load_model(model_path=smpl_path)

    assert body_model is not None, "SMPL Model not loaded correctly"
