#!/usr/bin/env python3
"""
CryoNet.Refine Refinement Script

This script performs structure refinement using density-guided diffusion.
It freezes all modules except the diffusion module and uses CC loss for optimization.
"""

import random,time, os
import gc
import numpy as np
import torch
from CryoNetRefine.data.const import atom_weight, atomic_to_symbol
from CryoNetRefine.data.parse.input import RefineArgs
from CryoNetRefine.data.write.utils import write_refined_structure
from CryoNetRefine.data.crop.molecule_aware import MoleculeTypeAwareSlidingWindowCropper
from CryoNetRefine.model.model import CryoNetRefineModel
from CryoNetRefine.loss.geometric import GeometricMetricWrapper, GeometricAdapter
from CryoNetRefine.loss.loss import refine_loss
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True # train speed is slower after set False 
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


class Engine:
    """CryoNet.Refine structure refinement class."""
    def __init__(
        self,
        model: CryoNetRefineModel,
        refine_args: RefineArgs,
        model_args: dict,
        device: str = "cuda",
        target_density=None,
        max_tokens: int = 1024,
        enable_cropping: bool = True,
        pdb_id: str = None, 
    ):
        self.model = model
        self.refine_args = refine_args
        self.model_args = model_args
        self.device = device
        self.target_density = target_density
        self.max_tokens = max_tokens
        self.enable_cropping = enable_cropping
        self.pdb_id = pdb_id
        # Initialize cropper if needed
        if self.enable_cropping:
            if self.refine_args.use_molecule_aware_cropping:
                self.molecule_aware_cropper = MoleculeTypeAwareSlidingWindowCropper(
                    crop_size=self.max_tokens,
                    overlap_size=self.max_tokens // 4,  # 25% overlap
                    min_crop_size=self.max_tokens // 4
                )
                self.cropper = None
                self.sequence_cropper = None
                print(f"🔧 Molecule-aware cropping enabled: max_tokens={self.max_tokens}")
            else:
                raise ValueError("No cropping method selected")
        else:
            self.cropper = None
            self.sequence_cropper = None
            self.molecule_aware_cropper = None
            print("🔧 Cropping disabled")
        self.geometric_adapter = GeometricAdapter(
            device=device,
            data_dir=refine_args.data_dir
        )
        self.geometric_wrapper = GeometricMetricWrapper(
            geom_root=getattr(refine_args, "geo_metric_root", None),
            pdb_id=pdb_id,
            device=device
        )
        # Freeze all parameters except diffusion module
        self._freeze_modules()
        # Setup optimizer for diffusion module only
        self._setup_optimizer()
        # Setup loss tracking
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        # Initialize crop initial CC storage
        self.crop_initial_cc = {}
        if hasattr(self.model, "structure_module") and hasattr(self.model.structure_module, "score_model"):
            sm = self.model.structure_module.score_model
            sm.activation_checkpointing = False
            if hasattr(sm, "token_transformer"):
                sm.token_transformer.activation_checkpointing = False
            if hasattr(sm, "atom_attention_encoder"):
                sm.atom_attention_encoder.activation_checkpointing = False
            if hasattr(sm, "atom_attention_decoder"):
                sm.atom_attention_decoder.activation_checkpointing = False
    def _freeze_modules(self):
        """Freeze all modules except the diffusion module."""
        print("Freezing modules except diffusion module...")
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze only diffusion module parameters
        diffusion_params = 0
        unfrozen_paras = ['structure_module']
        for name, param in self.model.named_parameters():
            for unfrozen_para in unfrozen_paras:
                if unfrozen_para in name:
                    param.requires_grad = True
                    diffusion_params += param.numel()
        print(f"Unfrozen diffusion module parameters: {diffusion_params:,}")
        # Verify that we have trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found! Check if structure_module exists.")
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
    def _setup_optimizer(self):
        """Setup optimizer for diffusion module parameters only."""
        # Get trainable parameters (only diffusion module)
        trainable_params = [
            param for param in self.model.parameters() 
            if param.requires_grad
        ]
        if not trainable_params:
            raise ValueError("No trainable parameters found!")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.refine_args.learning_rate,
            weight_decay=0
        )
        print(f"Setup optimizer with {len(trainable_params)} parameter groups")

    def should_use_cropping(self, batch):
        """Check if cropping should be used for this batch."""
        if not self.enable_cropping:
            return False
        if self.refine_args.use_molecule_aware_cropping and self.molecule_aware_cropper is not None:
            return True
        if self.refine_args.use_sequence_cropping and self.sequence_cropper is not None:
            return True
        if self.cropper is not None:
            return True
        return False
   

    def refine_step_with_molecule_aware_cropping(self, batch, target_density=None, iteration=0, data_dir=None, out_dir=None):
        """Perform one refinement step with molecule-type-aware cropping."""
        
        all_crops = self.molecule_aware_cropper.get_molecule_type_aware_crops(batch)
        num_crops = len(all_crops)
        if iteration == 0:
            print(f"🔄 Using molecule-type-aware cropping for structure")
            print(f"📊 Structure info: {num_crops} crops")
            crop_info = self.molecule_aware_cropper.get_crop_info(batch)
            print(f"📊 Molecule type distribution:")
            for mol_type, count in crop_info['molecule_type_counts'].items():
                print(f"  {mol_type}: {count} crops")
            for i, crop in enumerate(all_crops):
                crop_idx, crop_token_indices, molecule_type, crop_metadata = crop
                print(f"  Crop {i}: {molecule_type} (tokens={crop_metadata['num_tokens']}, "
                      f"sequences={len(crop_metadata['sequences'])}, "
                      f"complete={crop_metadata['is_complete']})")
        
        self.optimizer.zero_grad()
        total_loss = 0.0
        
        # Process each molecule-aware crop sequentially
        refined_coords = torch.zeros_like(batch["template_coords"].squeeze(0))
        loss_dict_list = []
        time_loss_dict_list = []
        
        for crop_idx, crop_info in enumerate(all_crops):
            _, crop_token_indices, molecule_type, crop_metadata = crop_info
            print(f"  Processing crop {crop_idx + 1}/{num_crops}: {molecule_type} "
                  f"({crop_metadata['num_tokens']} tokens)")
            
            # Extract crop from batch
            crop_batch, crop_token_indices, crop_atom_mask = self.molecule_aware_cropper.extract_molecule_aware_crop_from_batch(
                batch, crop_info
            )
            
            # Run refinement step on crop
            crop_loss, crop_predicted_coords, loss_dict, time_loss_dict = self.refine_step_single_crop(
                crop_batch, target_density, iteration, data_dir, out_dir, crop_idx
            )
            
            # Update refined_coords
            refined_coords[crop_atom_mask.unsqueeze(0)] = crop_predicted_coords[crop_batch['atom_pad_mask']]
            
            # Accumulate loss (average across crops)
            total_loss += crop_loss / num_crops
            
            # Backward pass
            (crop_loss / num_crops).backward(retain_graph=False)
            
            # Update weights after each crop
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Clean up
            del crop_loss, crop_predicted_coords, crop_batch, crop_token_indices, crop_atom_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            loss_dict_list.append(loss_dict)
            time_loss_dict_list.append(time_loss_dict)
        
        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if out_dir:
            output_path = out_dir / "refined_predictions" / f"{self.pdb_id}_iteration_{iteration:04d}_refined_structure.cif"
            write_refined_structure(batch, refined_coords, data_dir, output_path)
        
        return {
            "total_loss": total_loss.item(),
            "loss_dic_list": loss_dict_list,
            "time_loss_dict_list": time_loss_dict_list,
            "predicted_coords": refined_coords
        }

    def clear_caches(self):
        if hasattr(self, 'geometric_adapter'):
            self.geometric_adapter.clear_cache()
        if hasattr(self, 'geometric_wrapper'):
            self.geometric_wrapper.clear_cache()
        # OPTIMIZATION: Clear feature cache to prevent memory leaks
        if hasattr(self, 'crop_feature_cache'):
            self.crop_feature_cache.clear()
            print("🧹 Cleared feature cache")
        
        # Clear atom types cache
        if hasattr(self, 'crop_atom_types_cache'):
            self.crop_atom_types_cache.clear()
            print("🧹 Cleared atom types cache")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleared all caches and memory")
    
    def get_atom_types(self, crop_batch):
        """
        Extract atom types from crop_batch.
        
        Args:
            crop_batch: Dictionary containing batch data with 'ref_element' field
            
        Returns:
            torch.Tensor: Atom element indices [N_atoms] where each value is the atomic number
        """
        if 'ref_element' not in crop_batch:
            print("⚠️  Warning: 'ref_element' not found in crop_batch")
            return None
            
        # ref_element is one-hot encoded with shape [B, N_atoms, 128]
        ref_element = crop_batch['ref_element']  # [B, N_atoms, 128]
        
        # Convert one-hot to indices (atomic numbers)
        # The one-hot encoding represents atomic numbers, where index 0 = atomic number 0, etc.
        atom_types = torch.argmax(ref_element, dim=-1)  # [B, N_atoms]
        
        # Remove batch dimension if present
        if atom_types.shape[0] == 1:
            atom_types = atom_types.squeeze(0)  # [N_atoms]
            
        return atom_types
    
    def get_atom_type_symbols(self, crop_batch):
        """
        Extract atom type symbols from crop_batch.
        
        Args:
            crop_batch: Dictionary containing batch data with 'ref_element' field
            
        Returns:
            list: List of atom element symbols (e.g., ['C', 'N', 'O', 'S'])
        """
        atom_types = self.get_atom_types(crop_batch)
        if atom_types is None:
            return None
            
        # Convert atomic numbers to element symbols
        # Common elements mapping (atomic number -> symbol)
        element_symbols = {
            0: 'X',   # Unknown/placeholder
            1: 'H',   # Hydrogen
            6: 'C',   # Carbon
            7: 'N',   # Nitrogen
            8: 'O',   # Oxygen
            15: 'P',  # Phosphorus
            16: 'S',  # Sulfur
            9: 'F',   # Fluorine
            17: 'Cl', # Chlorine
            35: 'Br', # Bromine
            53: 'I',  # Iodine
        }
        
        # Convert to list and map to symbols
        atom_type_list = atom_types.tolist()
        atom_symbols = [element_symbols.get(atomic_num, f'X{atomic_num}') for atomic_num in atom_type_list]
        
        return atom_symbols
    
    def get_cached_atom_types(self, cache_key):
        """
        Get cached atom types for a specific crop.
        
        Args:
            cache_key: Cache key for the crop
            
        Returns:
            tuple: (atom_types, atom_symbols) or (None, None) if not cached
        """
        if hasattr(self, 'crop_atom_types_cache') and cache_key in self.crop_atom_types_cache:
            cached_data = self.crop_atom_types_cache[cache_key]
            return cached_data['atom_types'], cached_data['atom_symbols']
        return None, None
    
    def get_atom_radius_weights(self, atom_types):
        """
        Get atom radius weights based on atom types.
        
        Args:
            atom_types: torch.Tensor of atomic numbers [N_atoms]
            
        Returns:
            torch.Tensor: Atom radius weights [N_atoms], default weight is 1.0
        """
        if atom_types is None:
            return None
            
        weights = torch.ones_like(atom_types, dtype=torch.float32)

        
        for atomic_num, symbol in atomic_to_symbol.items():
            if symbol in atom_weight:
                mask = (atom_types == atomic_num)
                weights[mask] = atom_weight[symbol]
        
        return weights 
    
    def get_cached_atom_radius_weights(self, cache_key):
        """
        Get cached atom radius weights for a specific crop.
        
        Args:
            cache_key: Cache key for the crop
            
        Returns:
            torch.Tensor: Atom radius weights [N_atoms] or None if not cached
        """
        if hasattr(self, 'crop_atom_types_cache') and cache_key in self.crop_atom_types_cache:
            return self.crop_atom_types_cache[cache_key]['atom_radius_weights']
        return None

    def refine_step_single_crop(self, crop_batch, target_density=None, iteration=0, data_dir=None, out_dir=None, crop_idx=0):
        """Perform refinement on a single crop."""
        start_time = time.time()
        initial_coords = crop_batch["template_coords"]
        
        # Create cache key early for atom types caching
        cache_key = f"crop_{crop_idx}_tokens_{crop_batch['token_pad_mask'].shape[1]}_atoms_{crop_batch['atom_pad_mask'].shape[1]}"
        
        if iteration == 0:
            atom_types = self.get_atom_types(crop_batch)
            atom_symbols = self.get_atom_type_symbols(crop_batch)
            if atom_types is not None and atom_symbols is not None:
                # Print some examples
                num_atoms = min(10, atom_types.shape[0])
                
                # Count atom types
                unique_types, counts = torch.unique(atom_types, return_counts=True)
                # print(f"    Atom type distribution:")
                for atomic_num, count in zip(unique_types.tolist(), counts.tolist()):
                    symbol = atom_symbols[atom_types.tolist().index(atomic_num)] if atomic_num in atom_types.tolist() else f'X{atomic_num}'
                    # print(f"      {symbol} (atomic_num={atomic_num}): {count} atoms")
                
                # Get atom radius weights
                atom_radius_weights = self.get_atom_radius_weights(atom_types)
                
                # Cache atom types and weights for this crop
                if not hasattr(self, 'crop_atom_types_cache'):
                    self.crop_atom_types_cache = {}
                self.crop_atom_types_cache[cache_key] = {
                    'atom_types': atom_types,
                    'atom_symbols': atom_symbols,
                    'atom_radius_weights': atom_radius_weights
                }

        else:
            # Use cached atom types for subsequent recycles
            if hasattr(self, 'crop_atom_types_cache') and cache_key in self.crop_atom_types_cache:
                atom_types = self.crop_atom_types_cache[cache_key]['atom_types']
                atom_symbols = self.crop_atom_types_cache[cache_key]['atom_symbols']
                atom_radius_weights = self.crop_atom_types_cache[cache_key]['atom_radius_weights']
            else:
                # Fallback: compute atom types if cache miss
                atom_types = self.get_atom_types(crop_batch)
                atom_symbols = self.get_atom_type_symbols(crop_batch)
                atom_radius_weights = self.get_atom_radius_weights(atom_types)
        if not crop_batch["template_coords"].requires_grad:
            crop_batch["template_coords"] = crop_batch["template_coords"].clone().requires_grad_(True)
        
        # Ensure initial_coords has the correct shape [B, N, 3]
        if len(initial_coords.shape) == 4:  # [B, 1, N, 3] -> [B, N, 3]
            initial_coords = initial_coords.squeeze(1)
        
        
        if hasattr(self, 'crop_feature_cache') and cache_key in self.crop_feature_cache and iteration > 0:
            cached_features = self.crop_feature_cache[cache_key]
            s = cached_features['s']
            z = cached_features['z']
            diffusion_conditioning = cached_features['diffusion_conditioning']
            s_inputs = cached_features['s_inputs']
        else:

            with torch.no_grad():
                self.model.eval()  # Ensure trunk is in eval mode
                s_inputs = self.model.input_embedder(crop_batch)
                s_init = self.model.s_init(s_inputs)
                
                z_init = (
                    self.model.z_init_1(s_inputs)[:, :, None]
                    + self.model.z_init_2(s_inputs)[:, None, :]
                )
                relative_position_encoding = self.model.rel_pos(crop_batch)
                z_init = z_init + relative_position_encoding
                z_init = z_init + self.model.token_bonds(crop_batch["token_bonds"].float())
                if self.model.bond_type_feature:
                    z_init = z_init + self.model.token_bonds_type(crop_batch["type_bonds"].long())
                
                # Run trunk modules
                s = torch.zeros_like(s_init)
                z = torch.zeros_like(z_init)
                mask = crop_batch["token_pad_mask"].float()
                pair_mask = mask[:, :, None] * mask[:, None, :]
                recycling_steps = 0
                for i in range(recycling_steps + 1):
                    # Apply recycling (minimal)
                    s = s_init + self.model.s_recycle(self.model.s_norm(s))
                    z = z_init + self.model.z_recycle(self.model.z_norm(z))
                    
                    if self.model.use_templates:
                        z_t = self.model.template_module(z, crop_batch, pair_mask, use_kernels=self.model.use_kernels)
                        inverted_mask = ~crop_batch['token_pair_missing_mask']
                        z = z + z_t * inverted_mask.squeeze(1).unsqueeze(-1).expand_as(z)
                    s, z = self.model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=self.model.use_kernels)
                    
                # Get diffusion conditioning
                q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                    self.model.diffusion_conditioning(
                        s_trunk=s,
                        z_trunk=z,
                        relative_position_encoding=relative_position_encoding,
                        feats=crop_batch,
                    )
                )
                
                diffusion_conditioning = {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }
                
                # Store in cache (detach to avoid memory leaks)
                if not hasattr(self, 'crop_feature_cache'):
                    self.crop_feature_cache = {}
                
                self.crop_feature_cache[cache_key] = {
                    's': s.detach(),
                    'z': z.detach(),
                    'diffusion_conditioning': {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in diffusion_conditioning.items()},
                    's_inputs': s_inputs.detach()
                }
        
        
        self.model.train()  # Set to training mode for diffusion

        feats = crop_batch

        with torch.set_grad_enabled(True):
            struct_out = self.model.structure_module.den_sample(
                s_trunk=s.float(),
                s_inputs=s_inputs.float(),
                diffusion_conditioning=diffusion_conditioning,
                feats=feats,
                num_sampling_steps=self.model_args["sampling_steps"],
                atom_mask=crop_batch["atom_pad_mask"].float(),
                multiplicity=1,
                max_parallel_samples=1,
                target_density=target_density,
                resolution=self.refine_args.resolution,
                iteration=iteration,
                atom_weights=atom_radius_weights
            )
        

        predicted_coords = struct_out["sample_atom_coords"]
 

        cc, total_loss, loss_dict , time_loss_dict = refine_loss(
            crop_idx,
            predicted_coords,
            target_density,
            feats,
            self.refine_args,
            geometric_adapter=self.geometric_adapter,  
            geometric_wrapper=self.geometric_wrapper,  
            atom_weights=atom_radius_weights
        )

        if iteration == 0:
            self.crop_initial_cc[crop_idx] = struct_out["initial_cc"]

        # Debug: Print crop loss info
        init_cc = self.crop_initial_cc.get(crop_idx, "N/A")
        if cc > init_cc:
            loss_info = f"⬆ ✅Crop {crop_idx}: init={init_cc:.4f}, cur_cc={cc:.4f}, Loss={total_loss.item():.6f}"
        else:
            loss_info = f"⬇ Crop {crop_idx}: init={init_cc:.4f}, cur_cc={cc:.4f}, Loss={total_loss.item():.6f}"
        for key, value in loss_dict.items():
            loss_info += f", {key}: {value.item():.6f}"
        end_time = time.time()
        time_crop = end_time - start_time
        loss_info += f", time: {time_crop:.2f} s"
        print(f"    {loss_info}")
        return total_loss, predicted_coords, loss_dict, time_loss_dict
          
    

    def refine_step(self, batch, target_density=None, iteration=0, data_dir=None, out_dir=None):
        """Perform one refinement step."""
        
        # Always use cropping logic for unified handling
        if self.should_use_cropping(batch):
            if self.refine_args.use_molecule_aware_cropping and self.molecule_aware_cropper is not None:
                return self.refine_step_with_molecule_aware_cropping(batch, target_density, iteration, data_dir, out_dir)

    def refine(self, batch, target_density=None, data_dir=None, out_dir=None, cond_early_stop="loss"):
        """
        Perform structure refinement.
        
        Args:
            batch: Input batch data
            target_density: Target density map for CC loss
            
        Returns:
            refined_coords: Refined coordinates
            loss_history: History of losses during refinement
        """
        print(f"Starting refinement for {self.refine_args.num_recycles} recycles...")
        
        self.model.train()  # Set to training mode
        
        # Initialize best results tracking
        best_coords = None
        best_loss = float('inf')
        best_cc = float('-inf')  # Negative infinity
        best_iteration = -1
        
        # OPTIMIZATION: Initialize feature cache for this refinement run
        self.crop_feature_cache = {}
        
        for iteration in range(self.refine_args.num_recycles):
            # Perform refinement step
            # start_time = time.time()
            step_results = self.refine_step(batch, target_density, iteration, data_dir, out_dir)
            # print(f"Time taken for refine_step: {time.time() - start_time:.2f} seconds")
            # Track loss
            current_loss = step_results["total_loss"]
            loss_dict_list = step_results["loss_dic_list"]
            loss_dict_ep = {}
            for k in loss_dict_list[0].keys():
                loss_dict_ep[k] = sum([ld[k] for ld in loss_dict_list]) / len(loss_dict_list)
            line = f"🎈Iteration{iteration:3d}: total loss :{current_loss:.3f} "
            for k, v in loss_dict_ep.items():
                line += f" {k}: {v:.3f} "
            print(line)
            time_loss_dict_ep = {}
            time_loss_dict_list = step_results["time_loss_dict_list"]
            for k in time_loss_dict_list[0].keys():
                time_loss_dict_ep[k] = sum([ld[k] for ld in time_loss_dict_list]) / len(time_loss_dict_list)
            time_line = f"⏰Time: "
            for k, v in time_loss_dict_ep.items():
                time_line += f" {k}: {v:.3f} "
            print(time_line)

            self.loss_history.append(current_loss)
            
            # Check for improvement and save best results
            if cond_early_stop == "loss":
                if current_loss < best_loss - self.refine_args.min_improvement:
                    best_loss = current_loss
                    best_cc = loss_dict_ep["CC"]
                    best_coords = step_results["predicted_coords"].clone()
                    best_iteration = iteration
                    self.patience_counter = 0
                    print(f"✅ New best loss: {best_loss:.6f} at iteration {iteration}")
                else:
                    self.patience_counter += 1
            elif cond_early_stop == "cc":
                if loss_dict_ep["CC"] > best_cc + self.refine_args.min_improvement:
                    best_loss = current_loss
                    best_cc = loss_dict_ep["CC"]
                    best_coords = step_results["predicted_coords"].clone()
                    best_iteration = iteration
                    self.patience_counter = 0
                    print(f"✅ New best CC: {best_cc:.6f} at iteration {iteration}")
                else:
                    self.patience_counter += 1
            else:
                raise ValueError(f"Invalid condition early stop: {cond_early_stop}")
       
                
            # Early stopping
            if self.patience_counter >= self.refine_args.early_stopping_patience:
                print(f"Early stopping at iteration {iteration} (no improvement for {self.patience_counter} steps)")
                break
                
        self.model.eval()  # Set back to eval mode
        
        # Save best results info to refiner object for external access
        self.best_iteration = best_iteration
        self.best_loss = best_loss
        self.best_cc = best_cc
        
        # Return best results if available, otherwise return last results
        if best_coords is not None:
            print(f"🎯 Returning best results from iteration {best_iteration} with loss {best_loss:.6f}")
            return best_coords, self.loss_history
        else:
            print(f"⚠️  No improvement found, returning last iteration results")
            return step_results["predicted_coords"], self.loss_history
        
    def _save_checkpoint(self, iteration, coords, best_loss=None, best_iteration=None):
        """Save refinement checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': {k: v for k, v in self.model.state_dict().items() if 'structure_module' in k},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'best_loss': best_loss if best_loss is not None else self.best_loss,
            'best_iteration': best_iteration,
            'coords': coords,
            'is_best': best_iteration == iteration if best_iteration is not None else False
        }
        
        checkpoint_path = f"refine_checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if best_iteration == iteration:
            print(f"💾 Saved BEST checkpoint: {checkpoint_path} (loss: {best_loss:.6f})")
        else:
            print(f"💾 Saved checkpoint: {checkpoint_path}")