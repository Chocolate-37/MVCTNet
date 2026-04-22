"""
MVCTNet training script
"""
import time
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.MultiModalDataLoader import MultiModalPartNormalDataset
from utils.path_config import get_data_paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#  rubber tree segmentation class definitions
seg_classes = {'complex_tree': [0, 1, 2]}
seg_label_to_cat = {}  # label-to-category mapping


# build label mapping dict
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def inplace_relu(m):
    """set ReLU to inplace mode to save memory"""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def to_categorical(y, num_classes):
    """convert integer class label to one-hot encoding
    Args: y: [B] class label integer, num_classes: number of classes
    Returns: [B, num_classes] one-hot tensor
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y

def parse_args():
    """params - MVCTNet"""
    parser = argparse.ArgumentParser('MVCTNet Training')

    #  model arguments
    parser.add_argument('--model', type=str, default='mvctnet_part_seg', help='model module name')
    parser.add_argument('--npoint', type=int, default=2048, help='number of input points per sample')
    parser.add_argument('--normal', type=bool, default=True, help='whether to use surface normals')

    #  training arguments
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
    parser.add_argument('--epoch', default=200, type=int, help='number of training epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (Adam or SGD)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=20, help='LR decay step interval (epochs)')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='LR decay multiplier')

    #  system arguments
    parser.add_argument('--gpu', type=str, default='0', help='GPU device index')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment log directory name')

    #  multimodal arguments
    parser.add_argument('--enable_multimodal', action='store_true', help='enable 3D+2D multimodal training')
    parser.add_argument('--image_feature_dim', type=int, default=256, help='image feature dimension')
    parser.add_argument('--cache_image_features', action='store_true', help='cache image features in memory')

    #  BAHG params
    parser.add_argument('--enable_BAHG', action='store_true', help='enable BAHG boundary-aware fusion')
    parser.add_argument('--boundary_threshold', type=float, default=0.3, help='boundary detection threshold')

    #  ALFE params
    parser.add_argument('--enable_ALFE', action='store_true', help='enable ALFE feature evolution')
    parser.add_argument('--ALFE_competition_strength', type=float, default=0.3, help='ALFE')

    #  GUCL params
    parser.add_argument('--enable_gucl', action='store_true', help='enable GUCL loss function')
    parser.add_argument('--gucl_geometric_weight', type=float, default=0.5, help='GUCL geometric loss weight')
    parser.add_argument('--gucl_uncertainty_weight', type=float, default=0.5, help='GUCL uncertainty loss weight')
    parser.add_argument('--gucl_adaptive_factor', type=float, default=0.15, help='GUCL adaptive factor')
    parser.add_argument('--gucl_debug_mode', action='store_false', default=False, help='GUCL debug mode (off by default)')

    #  verbosity control 
    parser.add_argument('--verbose_level', type=int, default=0, choices=[0,1,2,3], help='verbosity level: 0=off, 1=key, 2=detailed, 3=all')

    #  test-mode arguments
    parser.add_argument('--test_2d_only', action='store_true', help='test 2D feature extraction only, skip training')

    return parser.parse_args()


def test_2d_features(args):
    """test whether 2D image feature extraction works correctly"""
    if not args.enable_multimodal:
        print("Please enable --enable_multimodal to test 2D features")
        return False

    try:
        #  get data paths
        from utils.path_config import get_data_paths
        pointcloud_root, image_root = get_data_paths()

        #  create test dataset
        from data_utils.MultiModalDataLoader import MultiModalPartNormalDataset
        test_dataset = MultiModalPartNormalDataset(
            root=pointcloud_root,
            npoints=args.npoint,
            split='test',
            normal_channel=args.normal,
            enable_multimodal=True,
            image_root=image_root,
            image_feature_dim=args.image_feature_dim,
            cache_image_features=False  # cache
        )

        print(f"Test dataset: {len(test_dataset)} samples")

        #  test first 3 samples
        success_count = 0
        for i in range(min(3, len(test_dataset))):
            try:
                point_set, cls, seg, image_features = test_dataset[i]

                # validate feature shapes and values
                local_feat = image_features['local']
                global_feat = image_features['global']

                assert local_feat.shape == (args.image_feature_dim, 16, 16), f"unexpected local feature shape: {local_feat.shape}"
                assert global_feat.shape == (args.image_feature_dim,), f"unexpected global feature shape: {global_feat.shape}"
                assert not torch.isnan(local_feat).any(), "local features contain NaN"
                assert not torch.isnan(global_feat).any(), "global features contain NaN"
                assert local_feat.abs().sum() > 0, "local features are all zeros"
                assert global_feat.abs().sum() > 0, "global features are all zeros"

                success_count += 1

            except Exception as e:
                print(f"Sample {i} test failed: {e}")

        if success_count >= 2:
            print(f"Test pass rate: {success_count}/3")
            return True
        else:
            print(f"Too many test failures: {success_count}/3")
            return False

    except Exception as e:
        print(f"2D feature test failed: {e}")
        return False


def main(args):

    #  import required modules (avoid scope conflicts)
    import sys
    import os
    from importlib import util

    def log_string(str):
        """logging helper"""
        logger.info(str)
        print(str)

    #  if multimodal enabled, validate 2D feature extraction first
    if args.enable_multimodal:
        print("Running 2D feature extraction test...")
        if not test_2d_features(args):
            print("2D feature test failed — please check image data and feature extractor")
            return
        print("2D feature test passed")

    #  configure GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #  create experiment directories
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('Experiment Name')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)  # timestamp-based name
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)  # user-specified name
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')  # checkpoint directory
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')  # log directory
    log_dir.mkdir(exist_ok=True)

    #  initialise logging system
    args = parse_args()
    logger = logging.getLogger("MVCTNet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('Starting MVCTNet training...')
    log_string(args)

    #  dataset loading 
    root = './data/RubberTree'

    #  multimodal data path configuration
    if args.enable_multimodal:
        pointcloud_root, image_root = get_data_paths()
        log_string(f"Data paths: pointcloud={pointcloud_root}, images={image_root}")

        #  multimodal training dataset (trainval split)
        TRAIN_DATASET = MultiModalPartNormalDataset(
            root=pointcloud_root,
            npoints=args.npoint,
            split='trainval',
            normal_channel=args.normal,
            enable_multimodal=True,
            image_root=image_root,
            image_feature_dim=args.image_feature_dim,
            cache_image_features=args.cache_image_features
        )

        #  test dataset
        TEST_DATASET = MultiModalPartNormalDataset(
            root=pointcloud_root,
            npoints=args.npoint,
            split='test',
            normal_channel=args.normal,
            enable_multimodal=True,
            image_root=image_root,
            image_feature_dim=args.image_feature_dim,
            cache_image_features=args.cache_image_features
        )


    else:
        log_string("Mode: pure 3D (original behaviour)")

        #  training dataset (trainval split) — 3D only
        TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)

        #  test dataset - 
        TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

    #  create data loaders
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    log_string("Number of training samples: %d" % len(TRAIN_DATASET))
    log_string("Number of test samples: %d" % len(TEST_DATASET))

    #  log multimodal configuration
    if args.enable_multimodal:
        log_string(f"Multimodal config: feature_dim={args.image_feature_dim}, cache={'on' if args.cache_image_features else 'off'}")
    else:
        log_string("Mode: pure 3D")

    #  category configuration
    num_classes = 1    # number of tree species (complex tree = 1)
    num_part = 3       # number of part classes (crown=0, trunk=1, interference=2)

    #  MVCTNet
    current_model_path = os.path.join(os.getcwd(), 'models', f'{args.model}.py')

    # clear module cache
    modules_to_clear = [args.model, f'models.{args.model}', 'mvctnet_part_seg', 'models.mvctnet_part_seg']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # import model module
    try:
        # Method 1: load directly from file path using importlib spec
        spec = util.spec_from_file_location(args.model, current_model_path)
        MODEL = util.module_from_spec(spec)
        sys.modules[args.model] = MODEL
        spec.loader.exec_module(MODEL)
        log_string("Model imported successfully")
    except Exception as e:
        # Method 1 fails in relative-import environments (e.g. inside a package).
        # Silently fall back to Method 2; show detail only in verbose mode.
        if args.verbose_level >= 2:
            log_string(f"Method 1 import skipped (falling back): {e}")
        # Method 2: standard package import
        try:
            if current_model_path not in sys.path:
                sys.path.insert(0, os.path.dirname(current_model_path))
            MODEL = importlib.import_module(f'models.{args.model}')
            sys.modules[args.model] = MODEL
            log_string("Model imported successfully")
        except Exception as e2:
            log_string(f"Model import failed: {e2}")
            raise RuntimeError(f"Cannot import model file: {args.model}")


    # copy key source files to experiment directory
    files_to_copy = [
        ('models/%s.py' % args.model, 'model file'),
        ('models/%s_utils.py' % args.model.split('_')[0], 'model utilities'),
        ('models/gucl_modules.py', 'GUCL loss module'),
        ('./train_partseg.py', 'training script')
    ]

    if args.enable_multimodal:
        files_to_copy.extend([
            ('models/color_segmentation_parser.py', 'colour segmentation parser'),
            ('data_utils/MultiModalDataLoader.py', 'multimodal data loader'),
            ('data_utils/ShapeNetDataLoader.py', 'base data loader'),
            ('utils/path_config.py', 'path configuration')
        ])

    copied_count = 0
    for file_path, file_desc in files_to_copy:
        try:
            shutil.copy(file_path, str(exp_dir))
            copied_count += 1
        except Exception:
            pass

    log_string(f"Copied {copied_count}/{len(files_to_copy)} files to experiment directory")

    if args.enable_multimodal:
        classifier = MODEL.get_model(
            num_part,
            normal_channel=args.normal,
            enable_multimodal=True,
            image_feature_dim=args.image_feature_dim,
            enable_global_fusion=True,
            enable_local_fusion=True,
            fusion_method='concat',
            enable_BAHG=args.enable_BAHG,
            boundary_threshold=args.boundary_threshold,
            enable_ALFE=args.enable_ALFE,
            ALFE_competition_strength=args.ALFE_competition_strength,
            debug_mode=(args.verbose_level > 0),
            verbose_level=args.verbose_level
        ).cuda()

        # determine network type and log configuration
        network_parts = []
        if args.enable_BAHG:
            network_parts.append("BAHG")
        if args.enable_ALFE:
            network_parts.append("ALFE")
        if args.enable_gucl:
            network_parts.append("GUCL")

        mode = "multimodal" if args.enable_multimodal else "3D-only"
        network_type = f"{mode} + {' + '.join(network_parts)}" if network_parts else mode
        log_string(f"Network mode: {network_type}")

        # log key parameters
        params = []
        if args.enable_BAHG:
            params.append(f"boundary threshold: {args.boundary_threshold}")
        if args.enable_ALFE:
            params.append(f"ALFE competition strength: {args.ALFE_competition_strength}")
        if params:
            log_string(f"   Parameters: {', '.join(params)}")
    else:
        classifier = MODEL.get_model(
            num_part,
            normal_channel=args.normal,
            enable_ALFE=args.enable_ALFE,
            ALFE_competition_strength=args.ALFE_competition_strength,
            enable_gucl=args.enable_gucl,
            gucl_geometric_weight=args.gucl_geometric_weight,
            gucl_uncertainty_weight=args.gucl_uncertainty_weight,
            gucl_adaptive_factor=args.gucl_adaptive_factor,
            debug_mode=(args.verbose_level > 0),
            verbose_level=args.verbose_level
        ).cuda()
        log_string("Network mode: MVCTNet 3D-only")

        # log key parameters
        params = []
        if args.enable_ALFE:
            params.append(f"ALFE competition strength: {args.ALFE_competition_strength}")
        if args.enable_gucl:
            params.append(f"GUCL weights: geo={args.gucl_geometric_weight:.2f} / unc={args.gucl_uncertainty_weight:.2f}")
        if params:
            log_string(f"   Parameters: {', '.join(params)}")

    #  count network parameters
    def count_parameters(model):
        """model arguments"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    total_params, trainable_params = count_parameters(classifier)
    total_params_m = total_params / 1e6  # in millions (M)
    trainable_params_m = trainable_params / 1e6  # in millions (M)

    #  count 2D extractor parameters (when multimodal enabled)
    if args.enable_multimodal:
        from models.color_segmentation_parser import HybridMultiViewFeatureExtractor
        temp_2d_extractor = HybridMultiViewFeatureExtractor(
            feature_dim=args.image_feature_dim,
            use_transformer=True
        )
        params_2d = sum(p.numel() for p in temp_2d_extractor.parameters())
        params_2d_m = params_2d / 1e6
        total_all = total_params + params_2d
        total_all_m = total_all / 1e6
        
        log_string("Network parameter statistics:")
        log_string(f"   3D branch (MVCTNet) : {total_params:,} ({total_params_m:.2f}M)")
        log_string(f"   2D branch           : {params_2d:,} ({params_2d_m:.2f}M)")
        log_string(f"   ----------------------------------------")
        log_string(f"   Total parameters    : {total_all:,} ({total_all_m:.2f}M)")
        log_string(f"   Trainable parameters: {trainable_params:,} ({trainable_params_m:.2f}M)")
        log_string(f"   Trainable ratio     : {trainable_params/total_params*100:.2f}%")
    else:
        log_string("Network parameter statistics:")
        log_string(f"   3D branch (MVCTNet) : {total_params:,} ({total_params_m:.2f}M)")
        log_string(f"   Trainable parameters: {trainable_params:,} ({trainable_params_m:.2f}M)")
        log_string(f"   Trainable ratio     : {trainable_params/total_params*100:.2f}%")

    #  create loss function (supports GUCL)
    criterion = MODEL.get_loss_function(
        enable_gucl=args.enable_gucl,
        gucl_geometric_weight=args.gucl_geometric_weight,
        gucl_uncertainty_weight=args.gucl_uncertainty_weight,
        gucl_adaptive_factor=args.gucl_adaptive_factor,
        debug_mode=args.gucl_debug_mode
    ).cuda()

    #  log loss function type
    if args.enable_gucl:
        log_string("Loss function: GUCL (Geometric Uncertainty Collaborative Learning)")
        weights = [
            f"geometric weight: {args.gucl_geometric_weight}",
            f"uncertainty weight: {args.gucl_uncertainty_weight}",
            f"adaptive factor: {args.gucl_adaptive_factor}"
        ]
        log_string(f"   GUCL config: {', '.join(weights)}")
    else:
        log_string("Loss function: standard NLL loss")
    classifier.apply(inplace_relu)  # optimise memory (inplace ReLU)

    # log network architecture summary
    arch_info = [f"Input [B, {args.npoint}, 6]"]

    if args.enable_multimodal:
        arch_info.append(f"2D features [B, {args.image_feature_dim}, 16x16] + [B, {args.image_feature_dim}]")

    if args.enable_ALFE:
        arch_info.append("ALFE (4 SA stages)")

    if args.enable_BAHG:
        arch_info.append("BAHG fusion")

    arch_info.append(f"Output [B, {args.npoint}, {num_part}]")

    log_string("Architecture: " + " -> ".join(arch_info))

    #  load checkpoint if available
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Resuming training from checkpoint')
    except:
        log_string('No checkpoint found, training from scratch')
        start_epoch = 0

    #  configure optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    log_string(f"Optimizer: {args.optimizer}, initial LR: {args.learning_rate}")

    def bn_momentum_adjust(m, momentum):
        """dynamically adjust BatchNorm momentum"""
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    #  training arguments
    LEARNING_RATE_CLIP = 1e-5      # minimum learning rate clip
    MOMENTUM_ORIGINAL = 0.1        # initial BatchNorm momentum
    MOMENTUM_DECCAY = 0.5          # BN momentum decay rate
    MOMENTUM_DECCAY_STEP = args.step_size

    # best-metric records
    best_acc = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    global_epoch = 0
    prev_epoch_miou = 0.0   # tracks previous epoch mIoU for delta display

    #  main training loop
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        #  update learning rate and BN momentum
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Current learning rate: %f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        #  per-epoch statistics ()

        #  per-epoch training loop
        for i, batch_data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            #  unpack multimodal or 3D-only batch
            if args.enable_multimodal and len(batch_data) == 4:
                points, label, target, image_features = batch_data
                # image_featurescontains: {'local': [B,256,16,16], 'global': [B,256], 'sample_id': [...]}
            else:
                points, label, target = batch_data
                image_features = None
            optimizer.zero_grad()

            #  data preparation and augmentation
            points = points.data.numpy()
            #  augmentation: random scale + shift on xyz coordinates
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

            #  data shape notes:
            # points: [B, N, 6] = [batch_size, 2048, xyz+]
            # label:  [B, 1] = [batch_size, ]
            # target: [B, N] = [batch_size, 2048] 

            #  MVCTNetforward pass (multimodal or 3D-only)
            if args.enable_multimodal and image_features is not None:
                #  multimodal forward pass
                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes), image_features)
                # seg_pred: [B, N, 3] 33D+2D
                # trans_feat: [B, 512]  ()
            else:
                #  3D-only forward pass (original)
                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                # seg_pred: [B, N, 3] 3
                # trans_feat: [B, 512]  ()

            #  prepare for loss computation
            if args.enable_gucl:
                #  GUCL: keep original shape [B, N, 3]
                seg_pred_for_loss = seg_pred  # [B, N, 3]
                target_for_loss = target      # [B, N]

                # flatten for accuracy computation
                seg_pred_flat = seg_pred.contiguous().view(-1, num_part)  # [B*N, 3]
                target_flat = target.view(-1)                              # [B*N]
                pred_choice = seg_pred_flat.data.max(1)[1]                #  [B*N]

                #  compute batch accuracy
                correct = pred_choice.eq(target_flat.data).cpu().sum()
                mean_correct.append(correct.item() / (args.batch_size * args.npoint))

                #  GUCL loss: takes boundary info, modal features and organ ratios
                boundary_info = getattr(classifier, 'last_boundary_info', None)
                modal_features = {'point': trans_feat, 'image': image_features} if image_features else {'point': trans_feat}
                organ_ratios = getattr(classifier, 'last_organ_ratios', None)
                loss = criterion(seg_pred_for_loss, target_for_loss, trans_feat, boundary_info, modal_features, organ_ratios, epoch)
            else:
                # standard NLL loss: flatten first
                seg_pred = seg_pred.contiguous().view(-1, num_part)  # [B*N, 3]
                target = target.view(-1, 1)[:, 0]                    # [B*N]
                pred_choice = seg_pred.data.max(1)[1]               #  [B*N]

                #  compute batch accuracy
                correct = pred_choice.eq(target.data).cpu().sum()
                mean_correct.append(correct.item() / (args.batch_size * args.npoint))

                loss = criterion(seg_pred, target, trans_feat)

            #  monitor and validate loss value
            loss_value = loss.item()

            #  numerical sanity check
            if torch.isnan(loss) or torch.isinf(loss) or loss_value > 100.0:
                log_string(f"Abnormal loss detected (Epoch {epoch}, Batch {i}): {loss_value:.6f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    log_string("   Loss contains nan/inf — skipping this batch")
                    continue
                elif loss_value > 100.0:
                    log_string("   Loss too large — applying gradient clipping")
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            #  log loss periodically (every 100 batches when verbose >= 2)
            if i % 100 == 0 and args.verbose_level >= 2:  # only shown in verbose mode, reduced frequency
                loss_type = "GUCL" if args.enable_gucl else "NLL"
                log_string(f"Epoch {epoch} Batch {i}: {loss_type} loss = {loss_value:.6f}")

            loss.backward()
            optimizer.step()

        #  epoch training statistics
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy: %.5f' % train_instance_acc)

        #  evaluation phase (no gradient)
        with torch.no_grad():

            # initialise evaluation metrics
            test_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]    # GT count per class
            total_correct_class = [0 for _ in range(num_part)] # correct prediction count per class
            total_pred_class = [0 for _ in range(num_part)]    # total prediction count per class
            shape_ious = {cat: [] for cat in seg_classes.keys()}  # per-sample IoU dict


            # build label mapping dict (for)
            seg_label_to_cat = {}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()  # switch to eval mode

            #  iterate test data (with timing)
            test_start_time = time.time()
            for batch_id, batch_data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

                #  unpack multimodal or 3D-only test batch
                if args.enable_multimodal and len(batch_data) == 4:
                    points, label, target, image_features = batch_data
                else:
                    points, label, target = batch_data
                    image_features = None
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

                #  MVCTNet (3D)
                if args.enable_multimodal and image_features is not None:
                    #  multimodal inference
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes), image_features)
                    # seg_pred: [B, N, 3] 33D+2D
                else:
                    #  3D-only inference
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                    # seg_pred: [B, N, 3] 3

                #  process prediction outputs
                cur_pred_val = seg_pred.cpu().data.numpy()    # move to CPU
                cur_pred_val_logits = cur_pred_val            # save raw logits
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # final predicted labels (integer)
                target = target.cpu().data.numpy()           # move to CPU

                # decode labels from logits
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]  # get sample category
                    logits = cur_pred_val_logits[i, :, :]  # [N, 3] samplelogits
                    # argmax within valid label range for this category
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                #  accumulate overall accuracy
                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                # accumulate per-class metrics
                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)                           # l
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))  # l
                    total_pred_class[l] += np.sum(cur_pred_val == l)                     # l

                # per-sample part IoU computation
                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]  #  [N]
                    segl = target[i, :]        #  [N]
                    cat = seg_label_to_cat[segl[0]]  # sample

                    # sampleIoU
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:  # 
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                            # absent part in both GT and pred -> IoU = 1.0
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            # IoU = intersection / union
                            intersection = np.sum((segl == l) & (segp == l))
                            union = np.sum((segl == l) | (segp == l))
                            part_ious[l - seg_classes[cat][0]] = intersection / float(union)

                    # mean part IoU for this sample
                    shape_ious[cat].append(np.mean(part_ious))

            # aggregate final test results
            test_inference_time = time.time() - test_start_time  # compute total inference time
            all_shape_ious = []  # all per-sample IoU values
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)  # collect all sample IoU values
                shape_ious[cat] = np.mean(shape_ious[cat])  # mean IoU for this category
            mean_shape_ious = np.mean(list(shape_ious.values()))  # mean IoU across all categories

            #  compute comprehensive metrics
            test_metrics['accuracy'] = total_correct / float(total_seen)  # accuracy
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
            test_metrics['precision'] = np.mean(np.array(total_correct_class) / np.array(total_pred_class, dtype=np.float32))  # precision
            test_metrics['recall'] = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))     # recall
            test_metrics['f1'] = 2 * (test_metrics['precision'] * test_metrics['recall']) / \
                                 (test_metrics['precision'] + test_metrics['recall'])  # F1 score

            #  log per-category IoU
            for cat in sorted(shape_ious.keys()):
                log_string('%s mIoU: %f' % (cat.ljust(16), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious        # class-average IoU
            test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)  # instance-average IoU

        # ── Epoch test results ────────────────────────────────────────────
        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg IOU: %f   Precision: %f   Recall: %f   F1: %f' %
                   (epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou'],
                    test_metrics['precision'], test_metrics['recall'], test_metrics['f1']))

        # ── Save best model ───────────────────────────────────────────────
        is_new_best = test_metrics['instance_avg_iou'] >= best_inctance_avg_iou
        if is_new_best:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Model saved.')

        # ── Update best metrics ───────────────────────────────────────────
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['instance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['instance_avg_iou']
        if test_metrics['precision'] > best_precision:
            best_precision = test_metrics['precision']
        if test_metrics['recall'] > best_recall:
            best_recall = test_metrics['recall']
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']

        # ── mIoU delta vs previous epoch ─────────────────────────────────
        miou_delta = test_metrics['instance_avg_iou'] - prev_epoch_miou
        delta_str = f"{miou_delta:+.5f}"
        prev_epoch_miou = test_metrics['instance_avg_iou']

        # ── All-time best metrics ─────────────────────────────────────────
        log_string('Best accuracy is:        %.5f' % best_acc)
        log_string('Best class avg mIOU is:  %.5f' % best_class_avg_iou)
        log_string('Best instance avg IOU is:%.5f' % best_inctance_avg_iou)
        log_string('Best precision is:       %.5f' % best_precision)
        log_string('Best recall is:          %.5f' % best_recall)
        log_string('Best F1 is:              %.5f' % best_f1)

        # ── Inference time (with sample count) ───────────────────────────
        n_test_samples = len(testDataLoader.dataset)
        ms_per_sample = test_inference_time / n_test_samples * 1000
        log_string('Test inference time: %.3f s | %d samples | %.2f ms/sample' % (
            test_inference_time, n_test_samples, ms_per_sample))

        # ── One-line epoch summary ────────────────────────────────────────
        best_tag = '  ** NEW BEST **' if is_new_best else ''
        log_string(
            '[Epoch %3d/%d] Train=%.5f | mIoU=%.5f (%s) | F1=%.5f | LR=%.6f%s' % (
                epoch + 1, args.epoch,
                train_instance_acc,
                test_metrics['instance_avg_iou'], delta_str,
                test_metrics['f1'],
                lr,
                best_tag
            )
        )
        log_string('-' * 70)
        global_epoch += 1


#  entry point
if __name__ == '__main__':
    args = parse_args()
    print("MVCTNet rubber tree point cloud segmentation")

    # display active innovation modules at startup
    innovations = []
    if args.enable_BAHG:
        innovations.append("BAHG")
    if args.enable_ALFE:
        innovations.append("ALFE")
    if args.enable_gucl:
        innovations.append("GUCL")

    mode = "multimodal" if args.enable_multimodal else "3D-only"
    if innovations:
        print(f"Mode: {mode} + {' + '.join(innovations)}")
    else:
        print(f"Mode: {mode}")
    
    #  if test-only mode
    if args.test_2d_only:
        if not args.enable_multimodal:
            print("Please enable both --enable_multimodal and --test_2d_only")
            exit(1)
        
        print("Running in 2D feature test-only mode")
        success = test_2d_features(args)
        if success:
            print("2D feature extraction test completed successfully")
            exit(0)
        else:
            print("2D feature test FAILED")
            exit(1)
    
    #  normal training mode
    main(args)
