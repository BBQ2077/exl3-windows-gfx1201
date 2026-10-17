"""Native-Windows LiLiCorr bridge for EXL3 targets.
Portable tensor arithmetic: TandemLLM (AGPL-3.0), bundled in app/lilicorr_reference.
Target verification and recurrent-state rollback: ExLlamaV3 Generator.
"""
from pathlib import Path
import ast,types,json,time,weakref
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from safetensors import safe_open
B=Path(__file__).resolve().parents[1]

def reference_module():
    # Load only the portable tensor arithmetic, never TandemLLM's serving engine or PTX.
    tree=ast.parse((B/'app/lilicorr_reference/tandem-engine-drafters-dflash2.py').read_text(encoding='utf-8'))
    names={'_rms','_rotate_half','_rope_tables','_apply_rope','_grouped_conv','_Conv'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    def linear(x,w):return F.linear(x,w)
    d=types.ModuleType('lilicorr_portable_reference');d.__dict__.update({'torch':torch,'F':F,'_lin':linear})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<TandemLLM portable primitives>','exec'),d.__dict__)
    tree=ast.parse((B/'app/lilicorr_reference/tandem-lilicorr.py').read_text(encoding='utf-8'))
    nodes=[]
    for n in tree.body:
        if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('engine.'):continue
        if isinstance(n,ast.ClassDef) and n.name=='LiLiCorrHead':continue
        nodes.append(n)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<TandemLLM LiLiCorr arithmetic>','exec'),d.__dict__)
    return d


def nvfp4_to_bf16(packed,scales,global_scale,device='cuda',row_chunk=32):
    """ModelOpt e2m1: low nibble is even K, one e4m3 scale per 16 K, then tensor scale."""
    N,K2=packed.shape
    grid=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
    out=torch.empty(N,2*K2,dtype=torch.bfloat16,device=device)
    for i in range(0,N,row_chunk):
        p=packed[i:i+row_chunk]
        codes=torch.stack((p & 15,p >> 4),dim=-1).reshape(p.shape[0],2*K2).long()
        x=grid[codes].reshape(p.shape[0],-1,16)
        x=x*scales[i:i+row_chunk].float().unsqueeze(-1)*global_scale.float()
        out[i:i+row_chunk].copy_(x.reshape(p.shape[0],-1).to(torch.bfloat16))
    return out


def load_weights(path,device='cuda'):
    from exllamav3.loader.safetensors import SafetensorsCollection
    W={};start=time.time();stc=SafetensorsCollection(str(path))
    keys=set(stc.tensor_file_map)
    def get(key):return stc.get_tensor(key,torch.device('cpu'),allow_bf16=True)
    for key in sorted(keys):
        if key.endswith(('.weight_scale','.weight_scale_2','.input_scale')):continue
        t=get(key)
        if t.dtype==torch.uint8:
            base=key[:-len('weight')]
            W[key]=nvfp4_to_bf16(t,get(base+'weight_scale'),get(base+'weight_scale_2'),device)
        else:W[key]=t.to(device=device,dtype=torch.bfloat16)
    print('LiLiCorr BF16 dequantized bytes',sum(v.numel()*v.element_size() for v in W.values()),'load_s',round(time.time()-start,2),flush=True)
    return W

class EmbeddingBridge:
    def __init__(self,target,device='cuda'):
        self.module=target.modules[0];self.device=torch.device(device);self.dtype=torch.bfloat16
    def __getitem__(self,ids):
        return self.module.embedding(ids.to('cpu')).to(device=self.device,dtype=self.dtype)
    def index_select(self,dim,ids):
        assert dim==0
        shape=ids.shape
        return self[ids.reshape(-1)].reshape(*shape,-1)

class LiLiCorrEXL3:
    """DFlash interface with a logical-position ring; target writes only verified accepted taps."""
    def __init__(self,path,weights,reference):
        self.raw=json.loads((Path(path)/'config.json').read_text(encoding='utf-8'))
        self.c=reference.LiLiCorrConfig(self.raw);self.weights=weights;self.ref=reference
        # NVIDIA/Tandem tap IDs name residual ENTERING the layer; EXL3 exports the preceding output.
        self.tap_layers={i-1 for i in self.c.target_layer_ids}
        assert min(self.tap_layers)>=0
        self.caps={'attach_target':True,'dflash_draft':True,'default_draft_size':self.c.slots,'recurrent_states':False}
        self.draft_verifier_params={'export_state_layers':self.tap_layers}
        self.config=SimpleNamespace(block_size=self.c.block_size)
        self.loaded_tp=False;self.stats={'drafts':0,'context_rows':0,'draft_s':0.0,'context_s':0.0}
        self.target=None
    def attach_to(self,target):
        self.target=target
        embed=EmbeddingBridge(target)
        def head(x):
            lm=target.modules[target.logit_layer_idx];p={}
            x=lm.prepare_for_device(x.half().unsqueeze(0),p)
            return lm.forward(x,p)[0,:,:self.c.vocab_size]
        self.m=self.ref.LiLiCorrModel(self.c,self.weights,embed,head)
        self.device=embed.device;dt=embed.dtype;c=self.c
        self.k=torch.zeros(c.num_hidden_layers,c.num_key_value_heads,c.window,c.head_dim,device=self.device,dtype=dt)
        self.v=torch.zeros_like(self.k);self.g=torch.zeros(c.window,c.hidden_size,device=self.device,dtype=dt)
        self.tags=torch.full((c.window,),-1,dtype=torch.long,device='cpu')
    @torch.inference_mode()
    def update_kv_from_target(self,target_hidden,cache,params,lengths=None):
        assert len(target_hidden)==len(self.tap_layers),(len(target_hidden),len(self.tap_layers))
        assert target_hidden[0].shape[0]==1,'Single sequence research adapter'
        n=int(params['cache_seqlens'][0]);rows=target_hidden[0].shape[1]
        if lengths is not None:rows=min(rows,int(lengths[0]))
        if rows<=0:return
        # A new prompt starts at zero; stale entries must never cross sequence boundaries.
        if n==0:self.tags.fill_(-1)
        t=time.perf_counter();x=torch.cat([v[0,:rows].to(self.device,dtype=torch.bfloat16) for v in target_hidden],dim=-1)
        pos=torch.arange(n,n+rows,device=self.device)
        gg,kk,vv=self.m.context(x,pos);slots=pos % self.c.window
        # Chunks <= window so index_copy has no duplicate destinations.
        assert rows<=self.c.window
        self.k.index_copy_(2,slots,kk);self.v.index_copy_(2,slots,vv);self.g.index_copy_(0,slots,gg)
        self.tags[slots.cpu()]=pos.cpu();self.stats['context_rows']+=rows;self.stats['context_s']+=time.perf_counter()-t
    @torch.inference_mode()
    def forward(self,input_ids,params):
        assert input_ids.shape[0]==1 and input_ids.shape[-1]==1
        n=int(params['cache_seqlens'][0]);c=self.c;tag=self.tags
        valid=(tag>=max(n-c.window+1,0)) & (tag<n)
        pos=torch.arange(c.block_size)[:,None]+n
        ok=(valid[None,:] & (pos-tag[None,:]<c.window)).to(self.device)
        prev=(n-1)%c.window;anchor_ok=float(n>0 and int(tag[prev])==n-1)
        anchor=input_ids[0].to(self.device);t=time.perf_counter()
        result=self.m.draft(anchor,torch.tensor(n,device=self.device),self.k,self.v,ok,self.g[prev],anchor_ok)
        ids=torch.cat((anchor,result[:,0].long())).view(1,-1)
        self.stats['drafts']+=1;self.stats['draft_s']+=time.perf_counter()-t
        return ids
    def sample_from_state(self,state,params):return state
    def unload(self):self.weights.clear()
    def __iter__(self):return iter(())
