import torch


def get_model(dir, sub_dir, model_name, is_lane_follow=False, *args):
    from pydoc import locate
    if not is_lane_follow:
        AttentionPredictor = locate(dir + '.' + sub_dir + ".attention_predictor.AttentionPredictor")
        # AttentionPredictor = locate("attention_predictor.AttentionPredictor")
        net = AttentionPredictor()
    else:
        LaneFollowPredictor = locate(dir + '.' + sub_dir + ".lane_follow_predictor.LaneFollowPredictor")
        net = LaneFollowPredictor(*args)

    if model_name is not None:
        print('Loading ', model_name)
        net.load_state_dict(
            torch.load(dir + '/' + sub_dir + '/' + model_name + '.tar', map_location='cpu'))

    if torch.cuda.is_available():
        net = net.cuda()
    return net

