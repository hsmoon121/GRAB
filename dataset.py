import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
import smplx
from pytorch3d.io import load_ply
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

from tools.utils import parse_npz, prepare_params, params2torch, append2dict, np2torch, to_cpu
from tools.utils import makepath, euler, to_tensor, axis_angle_to_matrix
from tools.meshviewer import Mesh, MeshViewer, colors
from tools.objectmodel import ObjectModel


cfg = {
  'data_path': 'dataset/grab/',
  'out_path': 'processed',
  'model_path': 'smplx_models/',
  'file_path': 'dataset',
  'render_path': 'render',
  'dtype': torch.float32,
}


class GraspDataset(data.Dataset):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg
    self.data_path = cfg['data_path']
    self.out_path = cfg['out_path']
    self.model_path = cfg['model_path']
    self.render_path = cfg['render_path']
    self.dtype = cfg['dtype']
    self.ds_path = os.path.join(cfg['file_path'], f"dataset.npz")
    self.all_seqs = glob.glob(self.data_path + "/*/*.npz")

    self.obj_based_seqs = {}
    self.sbj_based_seqs = {}

    self.process_sequences()
    self.load_or_create_datset()

  def process_sequences(self):
    # can retrieve specific subject or obejct
    # : self.sbj_based_seqs["s1"] / self.obj_base_seqs["mugs"]
    for sequence in self.all_seqs:
      subject_id = sequence.split('/')[-2]
      action_name = os.path.basename(sequence)
      object_name = action_name.split('_')[0]

      # group motion sequences based on objects
      if object_name not in self.obj_based_seqs:
          self.obj_based_seqs[object_name] = [sequence]
      else:
          self.obj_based_seqs[object_name].append(sequence)

      # group motion sequences based on subjects
      if subject_id not in self.sbj_based_seqs:
          self.sbj_based_seqs[subject_id] = [sequence]
      else:
          self.sbj_based_seqs[subject_id].append(sequence)

  def load_or_create_datset(self):
     if os.path.exists(self.ds_path):
        self.ds = np.load(self.ds_path, allow_pickle=True)['ds'].item()
        print(f"Loaded dataset from {self.ds_path}")
     else:
        self.ds = self.data_preprocessing(self.cfg)
        np.savez(self.ds_path, ds=self.ds)
        print(f"Saved new dataset to {self.ds_path}")      

  def data_preprocessing(self, cfg):
    object_data ={'verts': [], 'global_orient': [], 'transl': [], 'contact': []}
    rhand_data = {'verts': [], 'global_orient': [], 'hand_pose': [], 'transl': [], 'fullpose': []}
    self.obj_info = {}
    ds = {'hand_obs': [], 'object_pcl': [], 'grasp_pose': []}

    # MANO templates for each subject
    sbj_hand_models = {}
    for subject_id in self.sbj_based_seqs:
      seq = self.sbj_based_seqs[subject_id][0]
      seq_data = parse_npz(seq)
      n_comps  = seq_data.n_comps # default: 24 from GRAB
      fps = 120

      rh_mesh = os.path.join(self.data_path, '..', seq_data.rhand.vtemp)
      rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)
      rh_m = smplx.create(
        model_path=self.model_path,
        model_type='mano',
        is_rhand=True,
        v_template=rh_vtemp,
        num_pca_comps=n_comps,
        flat_hand_mean=True,
        batch_size=2*fps,
      )
      sbj_hand_models[subject_id] = rh_m
    hand_components = rh_m.hand_components
    print(f"Loaded {len(sbj_hand_models)} subjects' hand models")

    # obj_mesh
    obj_meshes = {}
    for object_name in self.obj_based_seqs:
      seq = self.obj_based_seqs[object_name][0]
      seq_data = parse_npz(seq)

      obj_mesh_path = os.path.join(cfg['data_path'], '..', seq_data.object.object_mesh)
      verts, faces = load_ply(obj_mesh_path)
      obj_meshes[object_name] = {'verts': verts, 'faces': faces}

    seqs = self.all_seqs
    print(f"Creating from {len(seqs)} sequences")

    for idx, seq in tqdm(enumerate(seqs)): 
      seq_data = parse_npz(seq)
      obj_name = seq_data.obj_name
      sbj_id   = seq_data.sbj_id
      n_comps  = seq_data.n_comps # default: 24 from GRAB
      gender   = seq_data.gender
      seq_len = len(seq_data['contact']['object'])
      fps = 120

      start_frame = self.filter_contact_frames(seq_data) # find the first contact point
      if start_frame < fps or start_frame > seq_len-fps:
        continue
      frame_mask = np.zeros((seq_len,)).astype(bool)
      frame_mask[start_frame-fps : start_frame+(fps//5)] = 1 # example frame mask: -1s ~ 0.2s (120 fps)
      rh_params  = prepare_params(seq_data.rhand.params, frame_mask)
      obj_params = prepare_params(seq_data.object.params, frame_mask)

      # Dictionary info:
      # seq_data.rhand:         dict_keys(['params', 'vtemp'])
      # seq_data.rhand.params:  dict_keys(['global_orient', 'hand_pose', 'transl', 'fullpose'])
      #                                   [(N, 3), (N, 24), (N, 3), (N, 45)]

      # seq_data.object:         dict_keys(['params', 'object_mesh'])
      # seq_data.object.params:  dict_keys(['transl', 'global_orient'])
      #                                    [(N, 3), (N, 3)]

      append2dict(rhand_data, rh_params)
      append2dict(object_data, obj_params)
      rh_tensor = params2torch(rh_params)
      obj_tensor = params2torch(obj_params)

      # 1) hand_obs: observation of (hand_pose + rel_transl) seq for a second right before the contact (len: 120 frames (1s))
      hand_axis_angle = torch.einsum('bi,ij->bj', [rh_tensor['hand_pose'], hand_components]) # (N, 24) * (24, 45) -> (N, 45)
      hand_axis_angle = torch.cat([rh_tensor['global_orient'], hand_axis_angle], dim=-1) # (N, 48)
      hand_rot6d = self.axis_angle_to_rot6d(hand_axis_angle.view(-1, 16, 3)).view(-1, 96) # (N, 96)
      hand_transl = rh_tensor['transl'] - obj_tensor['transl']
      hand_obs = torch.cat([hand_rot6d, hand_transl], dim=-1) # (N, 99)
      ds['hand_obs'].append(hand_obs[:fps])

      # 2) object_pcl: pcl information of object at around grasp moment
      verts, faces = obj_meshes[object_name]['verts'], obj_meshes[object_name]['faces']
      pcl_frames = []
      for frame in range(start_frame, start_frame + fps//5, 2): # example frame mask: 0s ~ +0.2s (120 fps)
        axis_angle_orient = to_tensor(seq_data.object.params['global_orient'][frame]) # axis-angle orientation
        rotated_verts = torch.matmul(verts, axis_angle_to_matrix(axis_angle_orient).reshape(3, 3).T)
        rotated_mesh = Meshes(verts=[rotated_verts], faces=[faces])
        pcl = sample_points_from_meshes(rotated_mesh, num_samples=10000)
        pcl_frames.append(pcl)
      ds['object_pcl'].append(pcl_frames)

      # 3) grasp_pose: (hand_pose + rel_transl) at grasp moment (used 12 different frames within [grasp_moment, grasp_moment+24])
      grasp_frames = []
      for frame in range(fps, fps + fps//5, 2):
        grasp_frames.append(hand_obs[frame])
      ds['grasp_pose'].append(grasp_frames)
      
      # self.render_sequences(seq_data, frame_mask, f"{idx}_{sbj_id}_{obj_name}")
    return ds

  def __len__(self):
    k = list(self.ds.keys())[0]
    return self.ds[k].shape[0] * 12 * 24
  
  def __getitem__(self, idx):
    # Augmentation:
    # grasp_pose variation: 12
    # 60 frames (0.5s) of observation starting between (-1.0s ~ -0.6s): 24 (having interval every 2 frames)
    # Gaussian noise (GrabNet): joints rotation (0.2 sigma), root rotation (0.004 sigma), translation (0.05 sigma)
    seq_idx = idx // (12 * 24)
    grasp_idx = (idx % (12 * 24)) // 24
    obs_idx = (idx % (12 * 24)) % 24
    fps = 120
    return {
      'hand_obs': self.ds['hand_obs'][seq_idx][obs_idx*2 : obs_idx*2 + fps//2],
      'object_pcl': self.ds['object_pcl'][seq_idx][grasp_idx],
      'grasp_pose': self.ds['grasp_pose'][seq_idx][grasp_idx]
    }

  def filter_contact_frames(self, seq_data):
    # find the first contact point
    frame_mask = (seq_data['contact']['object']>0).any(axis=1)
    start_frame = np.argmax(frame_mask)
    return start_frame
  
  def axis_angle_to_rot6d(self, axis_angle):
    """
    Convert a 3D axis-angle representation to a 6D continuous rotation representation.

    Args:
    - axis_angle (Tensor): a tensor of shape (batch_size, joint_num, 3) representing the axis-angle.

    Returns:
    - Tensor: a tensor of shape (batch_size, joint_num, 6) representing the 6D continuous rotation.
    """
    batch_size, joint_num, _ = axis_angle.shape
    rot_matrix = axis_angle_to_matrix(axis_angle.view(-1, 3))

    # Extract two orthogonal unit vectors
    rot6d = rot_matrix[:, :, :2].reshape(-1, 6)
    return rot6d.view(batch_size, joint_num, 6)
  
  def render_sequences(self, seq_data, frame_mask, name):
    mv = MeshViewer(width=1600, height=1200, offscreen=True)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
    mv.update_camera_pose(camera_pose)
    self.vis_sequence(seq_data, frame_mask, mv, name)

  def vis_sequence(self, seq_data, frame_mask, mv, name=""):

    n_comps = seq_data['n_comps']
    T = len(np.where(frame_mask)[0])

    rh_mesh = os.path.join(self.data_path, '..', seq_data.rhand.vtemp)
    rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)
    rh_m = smplx.create(model_path=self.model_path,
                        model_type='mano',
                        is_rhand=True,
                        v_template=rh_vtemp,
                        num_pca_comps=n_comps,
                        flat_hand_mean=True,
                        batch_size=T)

    rh_params  = prepare_params(seq_data.rhand.params, frame_mask)
    obj_params = prepare_params(seq_data.object.params, frame_mask)
    table_params = prepare_params(seq_data.table.params, frame_mask)

    rh_parms = params2torch(rh_params)
    verts_rh = to_cpu(rh_m(**rh_parms).vertices)

    obj_mesh = os.path.join(cfg["data_path"], '..', seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)
    obj_parms = params2torch(obj_params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)

    table_mesh = os.path.join(cfg["data_path"], '..', seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh)
    table_vtemp = np.array(table_mesh.vertices)
    table_m = ObjectModel(v_template=table_vtemp,
                        batch_size=T)
    table_parms = params2torch(table_params)
    verts_table = to_cpu(table_m(**table_parms).vertices)

    skip_frame = 4
    frames = []
    for frame in range(0, T, skip_frame):
        o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

        rh_mesh = Mesh(vertices=verts_rh[frame], faces=rh_m.faces, vc=colors['grey'], wireframe=True)
        t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

        mv.set_static_meshes([o_mesh, rh_mesh, t_mesh])
        frames.append(mv.save_snapshot(None))
        
    frames[0].save(
        self.render_path+f"/{name}.gif",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0
    )