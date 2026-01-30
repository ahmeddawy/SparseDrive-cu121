_base_ = ['./sparsedrive_small_stage2.py']

# ============================================================
# SparseDrive Stage 2 + World Model Training
# ============================================================
# This config fine-tunes Stage 2 (motion + planning) with
# a World Model for improved planning performance.
# ============================================================

# Explicitly load Stage 1 checkpoint
load_from = 'ckpt/sparsedrive_stage1.pth'

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
total_batch_size = 2  # LAW: 2 (was 1 here)
num_gpus = 1
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 10  # LAW: 10
checkpoint_epoch_interval = 1  # LAW: 1

# Sequence dataset for training; single-frame for eval
data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        type="SparseDriveSequenceDataset",
        interval_2frames=True,  # LAW: True
        queue_length=7,  # LAW: 7 (current + future frames)
        with_seq_flag=True,  # LAW: True
        sequences_split_num=7,  # LAW: 7
        keep_consistent_seq_aug=True,  # LAW: True
    ),
    val=dict(
        type="NuScenes3DDataset",
        with_seq_flag=False,  # LAW: False
        sequences_split_num=1,  # LAW: 1
        keep_consistent_seq_aug=False,  # LAW: False
    ),
)

# Enable World Model in MotionPlanningHead
model = dict(
    head=dict(
        motion_plan_head=dict(
            with_world_model=True,
            world_model_loss_weight=1.0,  # LAW: 1.0
            world_model_cfg=dict(
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.2,  # LAW: 0.2
                num_views=6,
                num_proposals=6,
                num_tf_layers=4,  # LAW: 4
                stride=32,  # LAW: 32
                action_dim=12,  # LAW: 12 (ego_fut_ts=6 * 2)
            ),
        )
    )
)

# Runner based on actual batch size
runner = dict(
    type='IterBasedRunner',
    max_iters=num_iters_per_epoch * num_epochs,
)

# Handle unused params when WM path is inactive for some samples
find_unused_parameters = True

# Disable evaluation during WM training (LAW)
evaluation = dict(
    interval=999999,  # LAW: effectively disabled
    pipeline=None,  # LAW: no eval pipeline during WM training
)
