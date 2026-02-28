# mfsdp-wan2.2-demo
Demo for training Wan2.2 with megatron-fsdp.  
Env:  nvcr_pytorch_26.01.
```shell
# Usage:
  ./run_pretrain.sh baseline   # Megatron-FSDP
  ./run_pretrain.sh fuse       # Megatron-FSDP: TE Linear + fuse_wgrad
  ./run_pretrain.sh pytorch    # PyTorch FSDP2
  ./run_pretrain.sh compare    # Run all config sequentially
# To enable triple buffer & UBR:
    FSDP_DOUBLE_BUFFER="${FSDP_DOUBLE_BUFFER:-true}"
    NCCL_UB="${NCCL_UB:-false}"
```
