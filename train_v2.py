#!/usr/bin/env python3
"""
CryoNet.Refine Refinement 

This script performs structure refinement using density-guided diffusion.
It freezes all modules except the diffusion module and uses CC loss for optimization.
"""
import os
import sys
# Set PYTHONPATH to include project root if not already set
# This ensures CryoNetRefine package can be found when running compute_ss.py
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Also set environment variable for subprocess calls
    if 'PYTHONPATH' not in os.environ:
        os.environ['PYTHONPATH'] = project_root
    elif project_root not in os.environ['PYTHONPATH']:
        os.environ['PYTHONPATH'] = f"{project_root}:{os.environ['PYTHONPATH']}"

import click,time, warnings
from typing import List
from tqdm import tqdm
from pathlib import Path
from typing import  Optional
from dataclasses import asdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from CryoNetRefine.data.module.train import BoltzInferenceDataModule
from CryoNetRefine.data.types import  Manifest
from CryoNetRefine.data.parse.input import (
    BoltzProcessedInput, DiffusionParams, model_args,
    PairformerArgs, check_inputs, process_inputs
)
from CryoNetRefine.libs.density.density import DensityInfo
from CryoNetRefine.model.model import CryoNetRefineModel
from CryoNetRefine.model.engine import Engine, RefineArgs, set_seed
from CryoNetRefine.data.write.utils import write_refined_structure
warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")

resolu_idx = "/data/huangfuyao/AF3DB_pre_train/emdb_list_info.csv"
# create an emdb-resolution dictionary
map_db_path = os.path.dirname(resolu_idx)
resolu_dict=dict()
with open(resolu_idx,'r') as rf:
    lines=rf.readlines()
for l in lines:
    l=l.strip().split()
    if len(l)<4: continue
    if l[3]=="resolution": continue
    resolu_dict[l[0]]=float(l[3])
    resolu_dict[l[1]]=float(l[3])


def setup_ddp():
    """Initialize DDP environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()

@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--map_db_path", type=click.Path(exists=True))
@click.option("--out_suffix", type=str, help="Output suffix", default="refine")
@click.option("--out_dir", type=click.Path(exists=False), help="Output directory", default="./refine_results")
@click.option("--cache", type=click.Path(exists=False), help="Cache directory")
@click.option("--checkpoint", type=click.Path(exists=True), help="Model checkpoint", default=None)
@click.option("--seed", type=int, help="Random seed", default=11)
@click.option("--target_density", multiple=True, type=click.Path(exists=True), help="Target density map (.mrc file)", default=None)
@click.option("--resolution", multiple=True, type=float, help="Resolution for density map operations", default=None)
@click.option("--max_tokens", type=int, help="Maximum number of tokens for cropping (0 to disable)", default=512)
@click.option("--gamma_0", type=float, help="Gamma 0 parameter", default=-0.5)
@click.option("--recycles", type=int, help="Number of refinement recycles", default=300)
@click.option("--enable_cropping", is_flag=True, help="Enable cropping for large structures", default=True)
@click.option("--cond_early_stop", type=str, help="Condition early stop", default="loss")
@click.option("--clash", type=float, help="Weight for clash loss", default=0.1)
@click.option("--den", type=float, help="Weight for density loss", default=20.0)
@click.option("--rama", type=float, help="Weight for rama loss", default=500.0)
@click.option("--rotamer", type=float, help="Weight for rotamer loss", default=500.0)
@click.option("--bond", type=float, help="Weight for bond loss", default=50)
@click.option("--angle", type=float, help="Weight for angle loss", default=1)
@click.option("--cbeta", type=float, help="Weight for cbeta loss", default=1.0)
@click.option("--ramaz", type=float, help="Weight for ramaz loss", default=1.00)
@click.option("--learning_rate", type=float, help="Learning rate for refinement", default=1.8e-4)
@click.option("--max_norm_sigmas_value", type=float, help="max norm sigmas value", default=1.0)
@click.option("--num_workers", type=int, help="Number of data loader workers", default=0)
@click.option("--use_global_clash", is_flag=True, help="Global clash flag", default=False)
@click.option("--num_epochs", type=int, help="Number of training epochs", default=100)
@click.option("--epoch_early_stop_patience", type=int, help="Number of epochs without improvement before stopping", default=5)
@click.option("--resume", type=bool, help="Resume training from checkpoint", default=False)
def train(
    data: str,
    map_db_path: str,
    out_dir: str,
    cache: str,
    checkpoint: Optional[str] = None,
    out_suffix: str = "refine",
    seed: Optional[int] = 11,
    target_density: Optional[tuple] = None,
    resolution: Optional[tuple] = None,
    max_tokens: int = 512,
    recycles: int = 300,
    gamma_0: float = -0.5,
    enable_cropping: bool = True,
    cond_early_stop: str = "loss",
    clash: int = 0.1,
    den: int = 20.0,
    rama: int = 500.0,
    rotamer: int = 500.0,
    bond: int = 50,
    angle: int = 1,
    cbeta: int = 1.0,
    ramaz: int = 1.00,
    learning_rate: float = 1.8e-4,
    max_norm_sigmas_value: float = 1.0,
    num_workers: int = 0,
    use_global_clash: bool = False,
    num_epochs: int = 100,
    epoch_early_stop_patience: int = 5,
    resume: Optional[bool] = False,

) -> None:
    # Initialize DDP
    rank, world_size, local_rank = setup_ddp()
    is_main_process = rank == 0
    
    start_time = time.time()
    set_seed(seed + rank)  # Different seed for each rank
    data = Path(data).expanduser()
    data_stem = data.stem
    data: List[Path] = list(data.glob("*.pdb"))

    out_dir = Path(out_dir).expanduser()
    if is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Synchronize after directory creation
    if world_size > 1:
        dist.barrier()
    mol_dir =Path(__file__).resolve().parent / "CryoNetRefine" / "data" / "mols"
    # Only main process does preprocessing
    if is_main_process:
        process_inputs(
            data=data,
            data_stem=data_stem,
            out_dir=out_dir,
            mol_dir=mol_dir,
            preprocessing_threads=1,
        )
    # Synchronize after preprocessing
    if world_size > 1:
        dist.barrier()
    
    # Load manifest
    manifest = Manifest.load(out_dir / f"processed_{data_stem}" / "manifest.json")
    # Load processed data !!
    processed_dir = out_dir / f"processed_{data_stem}"
    processed = BoltzProcessedInput(
        manifest=manifest,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
    )
    # Setup model parameters
    diffusion_params = DiffusionParams(gamma_0=gamma_0, max_norm_sigmas_value=max_norm_sigmas_value)
    pairformer_args = PairformerArgs()
    # Load model
    if checkpoint is None:
        checkpoint = cache / "cryonet.refine_model_checkpoint_best26.pt"

    data_dir = str(processed.template_dir)
    # Setup device for DDP
    if world_size > 1:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_module = CryoNetRefineModel.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=model_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,  # Auto-detect kernel availability
        pairformer_args=asdict(pairformer_args),
    )
    # Move to device
    model_module = model_module.to(device)
    # Unfreeze structure_module parameters for training (Engine will also do this)
    # This needs to be done BEFORE DDP wrapping to avoid the "no trainable parameters" error
    if is_main_process:
        click.echo(f"\n{'='*80}")
        click.echo(f"Setting up trainable parameters...")
        click.echo(f"{'='*80}")
    
    for name, param in model_module.named_parameters():
        if 'structure_module' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    if is_main_process:
        click.echo(f"  Trainable parameters: {trainable_params:,}")
    # Wrap with DDP if distributed and we have trainable parameters
    if world_size > 1 and trainable_params > 0:
        model_module = DDP(
            model_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if is_main_process:
            click.echo(f"✅ Model wrapped with DDP on {world_size} GPUs")
    elif world_size > 1 and trainable_params == 0:
        if is_main_process:
            click.echo(f"⚠️  WARNING: No trainable parameters! Skipping DDP.")
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        mol_dir=mol_dir,
        num_workers=num_workers,  # Single worker for stability
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
        rank=rank,
        world_size=world_size,
    )
    # Setup data loader - use original for stability
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()
    # ============ training logic ============
    start_epoch = 0
    epoch_history = []
    best_epoch_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    epoch_patience_counter = 0

    if resume:

        # Automatically detect and use the latest epoch checkpoint for resuming training
        latest_epoch_checkpoint = None
        latest_epoch = -1

        # Find the latest epoch checkpoint
        for epoch_file in out_dir.glob("model_checkpoint_epoch_*.pt"):
            try:
                epoch_num = int(epoch_file.stem.split("_")[-1])
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_epoch_checkpoint = epoch_file
            except ValueError:
                continue
        # Prefer to use the latest epoch checkpoint for resuming training
        if latest_epoch_checkpoint is not None:
            resume_checkpoint_path = latest_epoch_checkpoint
            click.echo(f"📂 Found latest epoch checkpoint: {resume_checkpoint_path}")
            click.echo(f"📂 (Latest epoch: {latest_epoch})")
        if resume and resume_checkpoint_path.exists():
            click.echo(f"\n{'='*80}")
            click.echo(f"🔄 Resuming training from checkpoint: {resume_checkpoint_path}")
            click.echo(f"{'='*80}\n")
            # 加载checkpoint
            resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
            
            # 加载模型状态
            original_model = model_module.module if hasattr(model_module, 'module') else model_module
            
            # 处理state_dict（可能需要处理DDP前缀）
            state_dict = resume_checkpoint['state_dict']
            model_state_dict = original_model.state_dict()
            
            # 匹配并加载参数（处理可能的键名差异）
            matched_state_dict = {}
            for key, value in state_dict.items():
                # 处理可能的键名差异
                if key in model_state_dict:
                    matched_state_dict[key] = value
                elif key.replace('module.', '') in model_state_dict:
                    matched_state_dict[key.replace('module.', '')] = value
                elif 'module.' + key in model_state_dict:
                    matched_state_dict['module.' + key] = value
            
            # 加载模型权重
            model_state_dict.update(matched_state_dict)
            original_model.load_state_dict(model_state_dict, strict=False)
            
            if hasattr(model_module, 'module'):  # DDP包装的模型
                model_module.module.load_state_dict(model_state_dict, strict=False)
            else:
                model_module.load_state_dict(model_state_dict, strict=False)
            
            click.echo(f"✅ Model weights loaded from checkpoint")
            
            # 恢复训练历史
            history_path = out_dir / "training_history.json"
            if history_path.exists():
                import json
                with open(history_path, 'r') as f:
                    training_summary = json.load(f)
                
                epoch_history = training_summary.get('epoch_history', [])
                best_epoch = training_summary.get('best_epoch', -1)
                best_epoch_loss = training_summary.get('best_epoch_loss', float('inf'))
                
                # 确定起始epoch（从最后一个完成的epoch+1开始）
                if len(epoch_history) > 0:
                    last_completed_epoch = epoch_history[-1]['epoch']
                    start_epoch = last_completed_epoch  # 从下一个epoch开始
                    # 恢复早停计数器（基于最后一次改进）
                    if best_epoch > 0:
                        epoch_patience_counter = last_completed_epoch - best_epoch
                        if epoch_patience_counter < 0:
                            epoch_patience_counter = 0
                    click.echo(f"📊 Training history loaded: {len(epoch_history)} completed epochs")
                    click.echo(f"📊 Best epoch: {best_epoch}, Best loss: {best_epoch_loss:.6f}")
                    click.echo(f"📊 Patience counter: {epoch_patience_counter}/{epoch_early_stop_patience}")
                    click.echo(f"🔄 Resuming from epoch {start_epoch + 1}")
                else:
                    start_epoch = 0
                    click.echo(f"📊 No previous training history found, starting from epoch 1")
            else:
                # 尝试从文件名提取epoch（对于 epoch checkpoint）
                if 'epoch' in resume_checkpoint_path.stem:
                    try:
                        epoch_num = int(resume_checkpoint_path.stem.split("_")[-1])
                        start_epoch = epoch_num  # 从该 epoch 继续
                        click.echo(f"📊 Extracted epoch from filename: {start_epoch}")
                        click.echo(f"🔄 Resuming from epoch {start_epoch + 1}")
                    except ValueError:
                        pass
                
                # 如果是 best checkpoint 且无法从文件名提取
                if start_epoch == 0 and 'best_epoch' in resume_checkpoint:
                    best_epoch = resume_checkpoint['best_epoch']
                    best_epoch_loss = resume_checkpoint.get('best_epoch_loss', float('inf'))
                    start_epoch = best_epoch  # 从最佳epoch后开始
                    click.echo(f"📊 Found epoch info in checkpoint: best_epoch={best_epoch}")
                    click.echo(f"🔄 Resuming from epoch {start_epoch + 1}")    

    # 多 epoch 训练循环
    for epoch in range(start_epoch, num_epochs):
        if is_main_process:
            click.echo(f"\n{'='*80}")
            click.echo(f"Starting Epoch {epoch + 1}/{num_epochs}")
            click.echo(f"{'='*80}\n")
        
        # Set epoch for distributed sampler (for shuffling)
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        # 用于统计当前 epoch 的所有 case 的 loss
        epoch_losses = {
            'total_loss': [],
            'CC': [],
            'geometric': [],
            'rama': [],
            'rotamer': [],
            'bond': [],
            'angle': [],
            'cbeta': [],
            'ramaz': [],
            'clash': []
        }
        epoch_start_time = time.time()
        

        # Perform refinement for each structure
        dataloader_iter = tqdm(dataloader, desc=f"Refining structures (Rank {rank})") if is_main_process else dataloader
        
        # Perform refinement for each structure
        for batch_idx, batch in enumerate(dataloader_iter):
            record = batch["record"][0]
            record_id = record.id
            parts = record_id.split("_")
            pdb_id = parts[2] 
            resolution = resolu_dict.get(pdb_id, 0.0)
            target_density = f'{map_db_path}/{record_id}.mrc'

            if is_main_process:
                click.echo(f"\nProcessing batch {batch_idx}")
            if den == 0.0:
                target_density = None
                target_density_obj = None
                resolution = None
            else:
                assert target_density is not None and resolution is not None, "Target density and resolution must be provided"
                target_density_obj = [DensityInfo(mrc_path=target_density, resolution=resolution, datatype="torch", device=device)]
            refine_args = RefineArgs()
            refine_args.data_dir = data_dir
            refine_args.num_recycles = recycles
            refine_args.weight_dict["clash"] = clash
            refine_args.weight_dict["den"] = den
            refine_args.weight_dict["rama"] = rama
            refine_args.weight_dict["rotamer"] = rotamer
            refine_args.weight_dict["bond"] = bond
            refine_args.weight_dict["angle"] = angle
            refine_args.weight_dict["cbeta"] = cbeta
            refine_args.weight_dict["ramaz"] = ramaz
            refine_args.learning_rate = learning_rate
            refine_args.use_global_clash = use_global_clash
            refiner = Engine(
                model_module, 
                refine_args, 
                model_args,
                device, 
                target_density_obj, 
                max_tokens=max_tokens,
                enable_cropping=enable_cropping,
                pdb_id=pdb_id,  
            )
            refined_coords, _ = refiner.refine(batch, target_density_obj, processed.template_dir, out_dir, cond_early_stop=cond_early_stop)
            # Get best results info from refiner
            best_iteration = getattr(refiner, 'best_iteration', None)
            best_loss = getattr(refiner, 'best_loss', None)
            best_cc = getattr(refiner, 'best_cc', None)
            # Convert Tensor to Python scalar for JSON serialization
            if isinstance(best_loss, torch.Tensor):
                best_loss = best_loss.item()
            if isinstance(best_cc, torch.Tensor):
                best_cc = best_cc.item()
            epoch_losses['total_loss'].append(best_loss)
            epoch_losses['CC'].append(best_cc)
            
            # Only main process saves structures
            if is_main_process:
                output_path = out_dir / f"{pdb_id}_{out_suffix}.cif"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_refined_structure(batch, refined_coords, data_dir, output_path)
                click.echo(f"Best Loss: {best_loss:.3f}, CC: {best_cc:.3f} at iteration {best_iteration}")
                click.echo(f"Refined structure {batch_idx} saved to {output_path}")
            
            if 'refiner' in locals():
                refiner.clear_caches()
            if 'refiner' in locals() and hasattr(refiner, 'geometric_adapter'):
                cache_info = refiner.geometric_adapter.get_cache_info()
                if is_main_process:
                    click.echo(f"Structure cache info: {cache_info}")
            
            if is_main_process:
                click.echo("Refinement completed!")
                end_time = time.time()
                click.echo(f"Refinement completed in {end_time - start_time:.2f} seconds")


        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # ✅ 每个 rank 只统计自己处理的样本（不跨 rank 聚合）
        # 注意：由于使用负载均衡 sampler，不同 rank 的样本数量可能不同
        # 我们只在 rank 0 进行统计和保存，避免 tensor 大小不匹配导致的通信问题
        if is_main_process:
            click.echo(f"\n{'='*80}")
            click.echo(f"Epoch {epoch + 1}/{num_epochs} Completed in {epoch_duration:.2f}s")
            click.echo(f"{'='*80}")

        # 计算并打印平均 loss (only main process)
        # 统计只基于 rank 0 处理的样本
        if is_main_process:
            # 即使没有样本，也记录 epoch 信息
            if len(epoch_losses['total_loss']) > 0:
                avg_total_loss = sum(epoch_losses['total_loss']) / len(epoch_losses['total_loss'])
                avg_cc = sum(epoch_losses['CC']) / len(epoch_losses['CC'])
                # Ensure Python scalars for JSON serialization
                if isinstance(avg_total_loss, torch.Tensor):
                    avg_total_loss = avg_total_loss.item()
                if isinstance(avg_cc, torch.Tensor):
                    avg_cc = avg_cc.item()
                
                click.echo(f"📊 Epoch {epoch + 1} Statistics (Rank 0):")
                click.echo(f"  Samples processed by rank 0: {len(epoch_losses['total_loss'])}")
                click.echo(f"  Average Total Loss: {avg_total_loss:.6f}")
                click.echo(f"  Average CC: {avg_cc:.6f}")
                
                # 保存 epoch 统计信息
                epoch_stats = {
                    'epoch': epoch + 1,
                    'avg_total_loss': float(avg_total_loss),
                    'avg_cc': float(avg_cc),
                    'num_cases': len(epoch_losses['total_loss']),
                    'duration': float(epoch_duration)
                }
            else:
                # 即使没有样本，也记录 epoch 信息
                click.echo(f"📊 Epoch {epoch + 1} Statistics (Rank 0):")
                click.echo(f"  ⚠️  No samples processed by rank 0 in this epoch")
                
                # 保存 epoch 统计信息（标记为无数据）
                epoch_stats = {
                    'epoch': epoch + 1,
                    'avg_total_loss': None,
                    'avg_cc': None,
                    'num_cases': 0,
                    'duration': epoch_duration
                }
            
            epoch_history.append(epoch_stats)
            
            # 每个 epoch 结束后立即保存训练历史（避免丢失）
            import json
            # Ensure best_epoch_loss is serializable
            best_epoch_loss_val = best_epoch_loss
            if isinstance(best_epoch_loss_val, torch.Tensor):
                best_epoch_loss_val = best_epoch_loss_val.item()
            training_summary = {
                'epoch_history': epoch_history,
                'best_epoch': best_epoch if best_epoch > 0 else None,
                'best_epoch_loss': float(best_epoch_loss_val) if best_epoch_loss_val != float('inf') else None,
                'early_stopped': False,  # 将在最后更新
                'total_epochs_run': len(epoch_history),
                'early_stop_patience': epoch_early_stop_patience
            }
            history_path = out_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            click.echo(f"  Training history updated: {history_path}")
                
            # 保存当前 epoch 模型 checkpoint
            original_model = model_module.module if hasattr(model_module, 'module') else model_module
            checkpoint_epoch = torch.load(checkpoint, map_location="cpu", weights_only=False)
            
            # 只更新训练过的参数
            current_state_dict = checkpoint_epoch['state_dict']
            trained_state_dict = original_model.state_dict()
            
            for key, value in trained_state_dict.items():
                clean_key = key.replace('_orig_mod.', '')
                if 'structure_module' in clean_key:
                    current_state_dict[clean_key] = value
            
            checkpoint_epoch['state_dict'] = current_state_dict
            checkpoint_path_epoch = out_dir / f"model_checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint_epoch, checkpoint_path_epoch)
            click.echo(f"  Model checkpoint saved to {checkpoint_path_epoch}")
        
        # Synchronize before continuing to next epoch
        if world_size > 1:
            dist.barrier()
        
        # ============ 早停策略检查 ============
        # 只在 rank 0 做早停判断，然后 broadcast 给所有 rank
        # Continue with early stopping check only if we have valid loss
        if is_main_process and len(epoch_losses['total_loss']) > 0:
            # Ensure values are Python scalars
            if isinstance(avg_total_loss, torch.Tensor):
                avg_total_loss = avg_total_loss.item()
            if isinstance(best_epoch_loss, torch.Tensor):
                best_epoch_loss = best_epoch_loss.item()
            # 检查当前 epoch 是否有改进
            if avg_total_loss < best_epoch_loss:
                # 有改进：更新最好的结果
                improvement = best_epoch_loss - avg_total_loss
                best_epoch_loss = avg_total_loss
                best_epoch = epoch + 1
                epoch_patience_counter = 0
                
                # 保存最好的模型状态（深拷贝以避免引用问题）
                from copy import deepcopy
                best_model_state = deepcopy(model_module.state_dict())
                
                click.echo(f"  ✅ New best epoch! Loss improved by {improvement:.6f}")
                click.echo(f"  Best epoch so far: Epoch {best_epoch}, Loss: {best_epoch_loss:.6f}")
                
                # 保存最好的模型
                original_model = model_module.module if hasattr(model_module, 'module') else model_module
                checkpoint_best = torch.load(checkpoint, map_location="cpu", weights_only=False)
                
                # 只更新训练过的参数
                current_state_dict_best = checkpoint_best['state_dict']
                trained_state_dict_best = original_model.state_dict()
                for key, value in trained_state_dict_best.items():
                    clean_key = key.replace('_orig_mod.', '')
                    if 'structure_module' in clean_key:
                        current_state_dict_best[clean_key] = value
                
                checkpoint_best['state_dict'] = current_state_dict_best
                checkpoint_best['best_epoch'] = best_epoch
                checkpoint_best['best_epoch_loss'] = best_epoch_loss
                checkpoint_path_best = out_dir / "model_checkpoint_best.pt"
                torch.save(checkpoint_best, checkpoint_path_best)
                click.echo(f"  💾 Best model saved to {checkpoint_path_best}")
            else:
                # 没有改进：增加 patience 计数器
                epoch_patience_counter += 1
                click.echo(f"  ⚠️  No improvement in this epoch")
                click.echo(f"  Patience counter: {epoch_patience_counter}/{epoch_early_stop_patience}")
                click.echo(f"  Best epoch so far: Epoch {best_epoch}, Loss: {best_epoch_loss:.6f}")
            
            # 在早停检查后更新训练历史（确保best_epoch和best_epoch_loss是最新的）
            import json
            best_epoch_loss_val = best_epoch_loss
            if isinstance(best_epoch_loss_val, torch.Tensor):
                best_epoch_loss_val = best_epoch_loss_val.item()
            training_summary = {
                'epoch_history': epoch_history,
                'best_epoch': best_epoch if best_epoch > 0 else None,
                'best_epoch_loss': float(best_epoch_loss_val) if best_epoch_loss_val != float('inf') else None,
                'early_stopped': False,
                'total_epochs_run': len(epoch_history),
                'early_stop_patience': epoch_early_stop_patience
            }
            history_path = out_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
        
        # ✅ Broadcast 早停决策，确保所有 rank 同步停止
        if world_size > 1:
            # Rank 0 决定是否早停
            if is_main_process and len(epoch_losses['total_loss']) > 0:
                should_stop = 1 if epoch_patience_counter >= epoch_early_stop_patience else 0
            else:
                should_stop = 0
            
            # Broadcast 决策到所有 rank
            should_stop_tensor = torch.tensor([should_stop], dtype=torch.long, device=device)
            dist.broadcast(should_stop_tensor, src=0)
            
            # 所有 rank 检查是否需要停止
            if should_stop_tensor.item() == 1:
                if is_main_process:
                    click.echo(f"\n{'='*80}")
                    click.echo(f"🛑 Early Stopping Triggered!")
                    click.echo(f"   No improvement for {epoch_patience_counter} consecutive epochs")
                    click.echo(f"   Best epoch: Epoch {best_epoch}, Loss: {best_epoch_loss:.6f}")
                    click.echo(f"{'='*80}\n")
                    
                    # 恢复最好的模型权重
                    if best_model_state is not None:
                        model_module.load_state_dict(best_model_state)
                        click.echo(f"✅ Restored best model weights from Epoch {best_epoch}")
                
                # 跳出 epoch 循环（所有 rank 同时跳出）
                break
        else:
            # 单 GPU 模式的早停
            if is_main_process and len(epoch_losses['total_loss']) > 0:
                if epoch_patience_counter >= epoch_early_stop_patience:
                    click.echo(f"\n{'='*80}")
                    click.echo(f"🛑 Early Stopping Triggered!")
                    click.echo(f"   No improvement for {epoch_patience_counter} consecutive epochs")
                    click.echo(f"   Best epoch: Epoch {best_epoch}, Loss: {best_epoch_loss:.6f}")
                    click.echo(f"{'='*80}\n")
                    
                    if best_model_state is not None:
                        model_module.load_state_dict(best_model_state)
                        click.echo(f"✅ Restored best model weights from Epoch {best_epoch}")
                    
                    break

            if is_main_process:
                click.echo(f"{'='*80}\n")


    # ============ 所有 Epoch 结束统计 ============
    if is_main_process:
        click.echo(f"\n{'='*80}")
        if epoch_patience_counter >= epoch_early_stop_patience:
            click.echo(f"Training Stopped Early After {len(epoch_history)} Epoch(s)")
        else:
            click.echo(f"All {num_epochs} Epoch(s) Completed")
        click.echo(f"{'='*80}")
        
        if len(epoch_history) > 0:
            click.echo(f"\n📈 Training Summary:")
            for stats in epoch_history:
                marker = " 🏆" if stats['epoch'] == best_epoch else ""
                click.echo(f"  Epoch {stats['epoch']}: Avg Loss={stats['avg_total_loss']:.6f}, "
                        f"Avg CC={stats['avg_cc']:.6f}, Duration={stats['duration']:.2f}s{marker}")
            
            # 显示最好的 epoch 信息
            click.echo(f"\n🏆 Best Results:")
            click.echo(f"  Best Epoch: {best_epoch}")
            click.echo(f"  Best Avg Loss: {best_epoch_loss:.6f}")
            
            # 保存训练历史（包含早停信息）
            import json
            # Ensure best_epoch_loss is serializable
            best_epoch_loss_val = best_epoch_loss
            if isinstance(best_epoch_loss_val, torch.Tensor):
                best_epoch_loss_val = best_epoch_loss_val.item()
            training_summary = {
                'epoch_history': epoch_history,
                'best_epoch': best_epoch,
                'best_epoch_loss': float(best_epoch_loss_val) if best_epoch_loss_val != float('inf') else None,
                'early_stopped': epoch_patience_counter >= epoch_early_stop_patience,
                'total_epochs_run': len(epoch_history),
                'early_stop_patience': epoch_early_stop_patience
            }
            history_path = out_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            click.echo(f"\n  Training history saved to {history_path}")
            click.echo(f"  Best model saved to {out_dir / 'model_checkpoint_best.pt'}")
            
            click.echo(f"{'='*80}\n")
    
    # Cleanup DDP
    cleanup_ddp()
if __name__ == "__main__":
    train()
