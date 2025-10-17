from vq_model.t2m_eval_modules import *
from os.path import join as pjoin

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.max_motion_length = 196
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_coemb_hidden = 512


        self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose']-4, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])


    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'
    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return  motion_enc, movement_enc

# our wrapper
class EvaluatorWrapper(object):

    def __init__(self, dataset_name, device):
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_coemb_hidden': 512,
            'dim_pose': 263 if dataset_name == 'humanml' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': './checkpoints',
            'unit_length': 4,
        }

        self.motion_encoder, self.movement_encoder = build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding