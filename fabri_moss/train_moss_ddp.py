"""Synchronous DDP trainer for the MOSS FrameKV adapter."""
from __future__ import annotations
import argparse, copy, dataclasses, hashlib, json, os, random
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
import numpy as np
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.data import MetaWorldWindows
from fabri_moss.runtime import assert_native_fa2, load_native_checkpoint
from fabri_moss.train import compute_file_sha256, compute_flow_kd_loss
from fabri_moss.train_native import SegmentSequenceSampler, bucketed_gradient_allreduce, clip_parameter_groups_norm, compute_epoch_batches


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='/root/models/FabriVLA/checkpoint_step_93000.pt'); p.add_argument('--fabri-root',default='/root/FabriVLA'); p.add_argument('--vlm',default='/root/models/InternVL3_5-1B'); p.add_argument('--data-root',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--epochs',type=int,default=1); p.add_argument('--stage',choices=['bridge','expert','joint'],default='bridge'); p.add_argument('--context-mode',choices=['window','consume','causal'],default='causal'); p.add_argument('--max-updates',type=int); p.add_argument('--global-batch-size',type=int,default=4); p.add_argument('--window',type=int,default=16); p.add_argument('--frame-stride',type=int,default=1); p.add_argument('--min-context-frames',type=int,default=1); p.add_argument('--lr',type=float,default=1e-4, help='bridge learning rate'); p.add_argument('--action-lr','--lr-action',dest='action_lr',type=float,default=1e-5); p.add_argument('--base-lr','--lr-base',dest='base_lr',type=float,default=5e-6); p.add_argument('--vision-lr',type=float,default=None); p.add_argument('--train-vision',action='store_true'); p.add_argument('--kd-weight',type=float,default=1.0); p.add_argument('--grad-clip-norm',type=float,default=1.0); p.add_argument('--max-episodes',type=int); p.add_argument('--seed',type=int,default=4042); p.add_argument('--device'); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--save-every',type=int,default=100); p.add_argument('--resume'); p.add_argument('--init-adapter')
    return p


def _init_dist()->Tuple[int,int,int]:
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if world>1:
        torch.cuda.set_device(local)
        try: dist.init_process_group('nccl',device_id=torch.device('cuda',local))
        except TypeError: dist.init_process_group('nccl')
    return rank,world,local


def _target_tensors(s):
    states=s.get('target_states',s.get('states',s.get('state'))); actions=s.get('target_actions',s.get('actions')); masks=s.get('target_action_mask',s.get('action_masks',s.get('action_mask')))
    if states is None or actions is None or masks is None: raise KeyError('sample must contain state(s), action(s), and action mask(s)')
    states,actions,masks=torch.as_tensor(states),torch.as_tensor(actions),torch.as_tensor(masks)
    if states.ndim==1: states=states.unsqueeze(0)
    if actions.ndim==2: actions=actions.unsqueeze(0)
    if masks.ndim==2: masks=masks.unsqueeze(0)
    n=states.shape[0]
    if actions.shape[0]!=n or masks.shape[0]!=n: raise ValueError(f'target count mismatch: states={n}, actions={actions.shape[0]}, masks={masks.shape[0]}')
    vis=s.get('visible_frame_counts',s.get('visible_counts'))
    if vis is None:
        idx=s.get('target_indices'); vis=[int(i)+1 for i in idx] if idx is not None else [len(s['images_window'])]*n
    vis=[int(x) for x in vis]
    if len(vis)!=n: raise ValueError(f'visible frame count length {len(vis)} != target count {n}')
    return states,actions,masks,vis


def _sample_loss(student, teacher_head, sample, device, kd_weight, teacher_policy=None):
    images=sample['images_window']; frame_ids=[int(x) for x in sample['frame_ids']]; prompt=str(sample['prompt'])
    states,actions,masks,visible=_target_tensors(sample)
    observation_times = sample.get('observation_times')
    frames=[student.project_frame(student.encode_image(imgs),fid,
                                  observation_time=(float(observation_times[j]) if observation_times is not None else None))
            for j,(imgs,fid) in enumerate(zip(images,frame_ids))]
    total=gt_total=kd_total=None; total_count=0
    state_mask=sample.get('state_mask')
    for i,count in enumerate(visible):
        if count<1 or count>len(frames): raise ValueError(f'visible frame count {count} outside [1, {len(frames)}]')
        times = observation_times
        prefix_times = list(times[:count]) if times is not None else None
        try:
            deep,shallow=student.read_memory(
                frames[:count], prompt, frame_ids=frame_ids[:count], observation_times=prefix_times
            )
        except TypeError as exc:
            # Keep tiny test doubles and legacy adapters that only expose the
            # original two-argument consume API working.
            if "unexpected keyword argument" not in str(exc):
                raise
            deep,shallow=student.read_memory(frames[:count], prompt)
        with torch.no_grad():
            teacher_embedder = teacher_policy if teacher_policy is not None else student.policy
            out=teacher_embedder.get_vl_embeddings(images=images[count-1],image_mask=torch.ones(len(images[count-1]),dtype=torch.bool,device=device),prompt=prompt,return_cls_only=False,shallow_layer_index=6)
        tdeep,tshallow=out if isinstance(out,tuple) else (out,None)
        state=states[i:i+1].to(device)
        if state_mask is not None:
            sm=torch.as_tensor(state_mask); sm=sm[i:i+1] if sm.ndim>1 and sm.shape[0]>1 else sm[:1]; state=state*sm.to(device)
        mask=masks[i:i+1].to(device); action=actions[i:i+1].to(device)
        loss,gt,kd,_,_=compute_flow_kd_loss(student_head=student.policy.action_head,teacher_head=teacher_head,student_deep=deep,student_shallow=shallow,teacher_deep=tdeep,teacher_shallow=tshallow,state=state,actions=action,action_mask=mask,kd_weight=kd_weight)
        c=int(mask.sum().item())
        if c<=0: continue
        total=loss*c if total is None else total+loss*c; gt_total=gt*c if gt_total is None else gt_total+gt*c; kd_total=kd*c if kd_total is None else kd_total+kd*c; total_count+=c
    if total is None: raise ValueError('sample has no valid action targets')
    return total,gt_total,kd_total,total_count


def _rng_states(world, device):
    local = {'torch': torch.get_rng_state(), 'random': random.getstate(), 'numpy': np.random.get_state()}
    if torch.device(device).type == 'cuda': local['torch_cuda'] = torch.cuda.get_rng_state(torch.device(device))
    gathered = [None] * world
    if world > 1: dist.all_gather_object(gathered, local)
    else: gathered[0] = local
    return gathered


def _parameter_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            h.update(name.encode()); h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _allreduce_mixed_precision_gradients(parameters, global_target_count):
    """Reduce mixed FP32/BF16 gradients without assigning wrong dtypes."""
    trainable = [p for p in parameters if p.requires_grad]
    if all(p.grad is None or p.grad.dtype == torch.float32 for p in trainable):
        bucketed_gradient_allreduce(trainable, global_target_count)
        return
    is_dist = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if is_dist else 1
    device = trainable[0].device
    flags = torch.tensor([p.grad is not None for p in trainable], dtype=torch.int32, device=device)
    if is_dist and world > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    for p, used in zip(trainable, flags.tolist()):
        if not used:
            continue
        grad = p.grad.float() if p.grad is not None else torch.zeros_like(p.data, dtype=torch.float32)
        if is_dist and world > 1:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(float(global_target_count))
        if p.grad is None:
            p.grad = grad.to(dtype=p.dtype)
        else:
            p.grad.data.copy_(grad.to(dtype=p.grad.dtype))


def _save(path,model,optimizer,step,epoch,batch_cursor,epoch_targets_seen,args,norm_stats,base_meta,contract,world,rng_states=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':'moss_cross_adapter_v2','step':int(step),'epoch':int(epoch),'batch_cursor':int(batch_cursor),'epoch_targets_seen':int(epoch_targets_seen),'stage':model.training_stage,'world_size':int(world),'source_checkpoint_sha256':base_meta.get('checkpoint_sha256'),'config':dataclasses.asdict(model.config),'cross_blocks':model.cross_blocks.state_dict(),'readout_embeddings':model.readout_embeddings.detach().cpu(),'optimizer':optimizer.state_dict(),'norm_stats':norm_stats,'base_metadata':base_meta,'data_contract':contract,'args':vars(args),'rng_state':{'torch':torch.get_rng_state(),'random':random.getstate()}}
    if rng_states is not None: payload['rng_states'] = rng_states
    if model.training_stage in ('expert','joint'):
        payload['action_head']=model.policy.action_head.state_dict()
    if model.training_stage == 'joint':
        payload['base_policy']={k:v.detach().cpu() for k,v in model.policy.state_dict().items() if 'action_head.' not in k}
    tmp = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(payload,tmp)
    os.replace(tmp,path)


def _load_resume(path,model,optimizer,args,base_meta,contract,world,rank=0,device='cpu'):
    ck=torch.load(str(Path(path).resolve()),map_location='cpu',weights_only=False)
    if ck.get('format')!='moss_cross_adapter_v2': raise ValueError('resume checkpoint must use moss_cross_adapter_v2; Writer/legacy checkpoints are rejected')
    if int(ck.get('world_size',-1))!=world: raise ValueError(f"world_size mismatch: checkpoint={ck.get('world_size')} current={world}")
    if ck.get('source_checkpoint_sha256')!=base_meta.get('checkpoint_sha256'): raise ValueError('source checkpoint SHA256 mismatch')
    if ck.get('stage')!=args.stage: raise ValueError(f"stage mismatch: checkpoint={ck.get('stage')} current={args.stage}")
    if ck.get('data_contract')!=contract: raise ValueError('data_contract mismatch on resume')
    if dict(ck.get('config',{}))!=dataclasses.asdict(model.config): raise ValueError('MOSS config mismatch on resume')
    model.cross_blocks.load_state_dict(ck['cross_blocks'],strict=True)
    with torch.no_grad(): model.readout_embeddings.copy_(ck['readout_embeddings'].to(model.readout_embeddings.device))
    if args.stage in ('expert','joint'):
        if 'action_head' not in ck: raise KeyError(f"stage {args.stage} checkpoint missing action_head")
        model.policy.action_head.load_state_dict(ck['action_head'],strict=True)
    if args.stage == 'joint':
        if 'base_policy' not in ck: raise KeyError('joint checkpoint missing base_policy')
        missing, unexpected = model.policy.load_state_dict(ck['base_policy'], strict=False)
        if unexpected or any('action_head.' not in k for k in missing):
            raise ValueError(f'invalid base_policy state: missing={missing}, unexpected={unexpected}')
    rng_states=ck.get('rng_states')
    if not isinstance(rng_states,list) or len(rng_states)!=world:
        raise ValueError('resume checkpoint must contain one RNG state per rank')
    optimizer.load_state_dict(ck['optimizer']); rng=rng_states[rank]
    if 'torch' in rng: torch.set_rng_state(rng['torch'].cpu() if isinstance(rng['torch'],torch.Tensor) else rng['torch'])
    if 'random' in rng: random.setstate(rng['random'])
    if 'numpy' in rng: np.random.set_state(rng['numpy'])
    if 'torch_cuda' in rng and torch.device(device).type == 'cuda': torch.cuda.set_rng_state(rng['torch_cuda'],torch.device(device))
    model.set_training_stage(args.stage); model.train()
    return int(ck.get('step',0)),int(ck.get('epoch',0)),int(ck.get('batch_cursor',0)),int(ck.get('epoch_targets_seen',0))


def _load_init_adapter(path, model, args, base_meta):
    ck=torch.load(str(Path(path).resolve()),map_location='cpu',weights_only=False)
    if ck.get('format')!='moss_cross_adapter_v2':
        raise ValueError('init-adapter must use moss_cross_adapter_v2; Writer/legacy checkpoints are rejected')
    if ck.get('source_checkpoint_sha256') != base_meta.get('checkpoint_sha256'):
        raise ValueError('init-adapter source checkpoint SHA256 mismatch')
    source_stage=ck.get('stage')
    if source_stage not in ('bridge','expert','joint'):
        raise ValueError(f'unsupported init-adapter stage {source_stage!r}')
    if source_stage == 'joint' and args.stage == 'bridge':
        raise ValueError('cannot initialize bridge from joint checkpoint')
    model.cross_blocks.load_state_dict(ck['cross_blocks'],strict=True)
    with torch.no_grad(): model.readout_embeddings.copy_(ck['readout_embeddings'].to(model.readout_embeddings.device))
    if args.stage in ('expert','joint') and source_stage in ('expert','joint'):
        model.policy.action_head.load_state_dict(ck['action_head'],strict=True)
    if args.stage == 'joint' and source_stage == 'joint' and 'base_policy' in ck:
        missing, unexpected = model.policy.load_state_dict(ck['base_policy'],strict=False)
        if unexpected or any('action_head.' not in k for k in missing):
            raise ValueError(f'invalid init base_policy state: missing={missing}, unexpected={unexpected}')


def main():
    args=parser().parse_args(); rank,world,local=_init_dist()
    if args.global_batch_size<world or args.epochs<=0 or args.save_every<=0: raise ValueError('global batch must cover all ranks, epochs/save-every must be positive')
    device=args.device or f'cuda:{local}'; random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    policy,_,norm_stats,base_meta=load_native_checkpoint(fabri_root=args.fabri_root,checkpoint_path=args.checkpoint,vlm_path=args.vlm,device=device,arm_key='metaworld_sawyer'); fa2=assert_native_fa2(policy)
    # Snapshot the teacher before opening any native parameters for joint training.
    # Snapshot the complete teacher before opening native parameters for joint training.
    if hasattr(policy, 'action_head') and policy.action_head is not None:
        policy.action_head.float()
    teacher_policy=copy.deepcopy(policy).to(device); teacher_policy.eval()
    teacher_head=teacher_policy.action_head
    model=MossInternVL(policy,MossConfig(cross_layers=(3,6,10,14),max_frames=args.window,max_text_tokens=1024,shallow_layer=6,train_vision=args.train_vision)); model.set_training_stage(args.stage); model.train()
    if world>1:
        for p in model.parameters():
            if p.requires_grad: dist.broadcast(p.data,src=0)
        hashes=[None]*world; dist.all_gather_object(hashes,_parameter_hash(model))
        if len(set(hashes)) != 1: raise RuntimeError('Trainable parameters differ across ranks after initialization')
    rank_seed=args.seed + rank * 10007
    random.seed(rank_seed); torch.manual_seed(rank_seed); np.random.seed(rank_seed % (2**32 - 1))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(rank_seed)
    for p in teacher_head.parameters(): p.requires_grad=False
    if args.init_adapter and args.resume: raise ValueError('--init-adapter and --resume are mutually exclusive')
    if args.init_adapter: _load_init_adapter(args.init_adapter,model,args,base_meta); model.set_training_stage(args.stage)
    groups=[]
    def add_group(name, params, lr):
        ps=[p for p in params if p.requires_grad]
        if ps: groups.append({'params':ps,'lr':float(lr),'name':name})
    add_group('bridge', model.bridge_parameters(), args.lr)
    if args.stage in ('expert','joint'): add_group('action_head', model.action_parameters(), args.action_lr)
    if args.stage == 'joint': add_group('base', model.base_parameters(include_vision=False), args.base_lr)
    if args.stage == 'joint' and args.train_vision: add_group('vision', (p for n,p in model.policy.named_parameters() if model._is_vision_name(n)), args.vision_lr if args.vision_lr is not None else args.base_lr)
    trainable=[p for g in groups for p in g['params']]
    if not trainable: raise RuntimeError(f'no trainable parameters for stage {args.stage}')
    optimizer=AdamW(groups,weight_decay=0.0)
    dataset=MetaWorldWindows(root=args.data_root,norm_stats=norm_stats,horizon=50,state_dim=24,action_dim=24,window=args.window,frame_stride=args.frame_stride,split='train',seed=args.seed,max_episodes=args.max_episodes,context_mode=args.context_mode,min_context_frames=args.min_context_frames)
    contract=dataset.get_data_contract() if hasattr(dataset,'get_data_contract') else {'context_mode':args.context_mode,'window':args.window,'frame_stride':args.frame_stride,'min_context_frames':args.min_context_frames,'seed':args.seed,'max_episodes':args.max_episodes,'split':'train','active_episode_ids':sorted(int(e['episode_index']) for e in dataset.active_episodes),'metadata_files_sha256':{n:compute_file_sha256(dataset.root/'meta'/n) for n in ('info.json','tasks.jsonl','episodes.jsonl')}}
    out=Path(args.output_dir)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True); (out/'run_config.json').write_text(json.dumps({'format':'moss_cross_adapter_v2','world_size':world,'fa2':fa2,'data_contract':contract},indent=2))
    if world>1: dist.barrier()
    step=start_epoch=batch_cursor=epoch_targets_seen=0
    if args.resume: step,start_epoch,batch_cursor,epoch_targets_seen=_load_resume(args.resume,model,optimizer,args,base_meta,contract,world,rank,device)
    current_epoch=start_epoch
    for epoch in range(start_epoch,args.epochs):
        if args.max_updates is not None and step>=args.max_updates: break
        current_epoch=epoch
        _,batches,per_rank=compute_epoch_batches(len(dataset),args.global_batch_size,world,args.seed,epoch); kw={'dataset':dataset,'batch_size':None,'sampler':SegmentSequenceSampler(per_rank[rank]),'num_workers':args.num_workers,'persistent_workers':args.num_workers>0}
        if args.num_workers>0:
            import torch.multiprocessing as mp; kw.update(prefetch_factor=2,multiprocessing_context=mp.get_context('spawn'))
        it=iter(DataLoader(**kw))
        for bi,gb in enumerate(batches):
            local_indices=gb[rank::world]; samples=[next(it) for _ in local_indices]
            if epoch==start_epoch and bi<batch_cursor: continue
            optimizer.zero_grad(set_to_none=True); sums=torch.zeros(4,dtype=torch.float64,device=device)
            for sample in samples:
                total,gt,kd,c=_sample_loss(model,teacher_head,sample,device,args.kd_weight,teacher_policy=teacher_policy); total.backward(); sums+=torch.tensor([total.detach().item(),gt.detach().item(),kd.detach().item(),c],dtype=torch.float64,device=device)
            if world>1: dist.all_reduce(sums,op=dist.ReduceOp.SUM)
            count=int(sums[3].item()); _allreduce_mixed_precision_gradients(trainable,count); grad_norm=clip_parameter_groups_norm(trainable,args.grad_clip_norm); optimizer.step(); step+=1; epoch_targets_seen+=count; batch_cursor=bi+1
            if rank==0:
                rec={'step':step,'epoch':epoch,'batch_cursor':batch_cursor,'loss':sums[0].item()/count,'gt_loss':sums[1].item()/count,'kd_loss':sums[2].item()/count,'grad_norm':grad_norm,'target_count':count}
                with (out/'train_metrics.jsonl').open('a',buffering=1) as f: f.write(json.dumps(rec)+'\n')
            if step==1 or step%args.save_every==0:
                rng_states=_rng_states(world,device)
                if rank==0: _save(out/'last.pt',model,optimizer,step,epoch,batch_cursor,epoch_targets_seen,args,norm_stats,base_meta,contract,world,rng_states)
            if args.max_updates is not None and step>=args.max_updates: break
        if args.max_updates is not None and step>=args.max_updates: break
        current_epoch=epoch+1; batch_cursor=0; epoch_targets_seen=0
        rng_states=_rng_states(world,device)
        if rank==0:
            _save(out/f'epoch_{current_epoch:03d}.pt',model,optimizer,step,current_epoch,0,0,args,norm_stats,base_meta,contract,world,rng_states)
    rng_states=_rng_states(world,device)
    if rank==0:
        _save(out/'adapter_final.pt',model,optimizer,step,current_epoch,batch_cursor,epoch_targets_seen,args,norm_stats,base_meta,contract,world,rng_states); (out/'metrics.json').write_text(json.dumps({'step':step,'world_size':world,'fa2':fa2},indent=2))
    if world>1: dist.barrier(); dist.destroy_process_group()

if __name__=='__main__': main()
