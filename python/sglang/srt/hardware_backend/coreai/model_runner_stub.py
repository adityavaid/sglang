"""CPU scheduler bookkeeping; weights and KV tensors belong to Core AI."""

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.layer_setup import ModelLayerInfo
from sglang.srt.runtime_context import get_schedule


class _CoreAIKVCache(KVCache):
    def __init__(self, size):
        # Do not invoke KVCache.__init__: Core AI owns the actual GPU storage.
        self.size = size
        self.page_size = 1
        self.dtype = self.store_dtype = torch.float16
        self.device = "cpu"
        self.layer_num = self.start_layer = self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def get_key_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "Core AI owns KV storage; Torch buffer access is unsupported."
        )

    get_value_buffer = get_key_buffer
    get_kv_buffer = get_key_buffer
    set_kv_buffer = get_key_buffer

    def get_kv_size_bytes(self):
        return 0, 0


class _MetadataModel:
    @staticmethod
    def forward():
        raise RuntimeError("Core AI inference must run through CoreAISession.")


class CoreAIModelRunnerStub(ModelRunner):
    canary_manager = None
    prefill_aware_swa = False

    @property
    def preloaded_weights_bytes(self):
        return 0

    def init_threads_binding(self):
        # Linux NUMA/OpenMP affinity is irrelevant to CPU bookkeeping on macOS.
        self.local_omp_cpuid = None

    def init_torch_distributed(self):
        import psutil

        from sglang.srt.distributed import bootstrap
        from sglang.srt.distributed.parallel_state import get_pp_group, get_tp_group
        from sglang.srt.runtime_context import get_parallel

        # Reuse SGLang's Gloo group setup, not its Linux CPU kernel/NUMA bootstrap.
        bootstrap._init_parallel_groups(
            backend="gloo",
            dist_init_method=bootstrap._resolve_dist_init_method(
                dist_port=self.dist_port
            ),
            server_args=self.server_args,
            model_config=self.model_config,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            pp_rank=self.ps.pp_rank,
            pp_size=self.ps.pp_size,
            attn_dp_size=self.ps.attn_dp_size,
            attn_cp_size=self.ps.attn_cp_size,
            moe_ep_size=self.ps.moe_ep_size,
            moe_dp_size=self.ps.moe_dp_size,
            dcp_size=self.ps.attn_dcp_size,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_parallel().attn_tp_group
        self.pre_model_load_memory = psutil.virtual_memory().available / (1 << 30)

    def load_model(self):
        self.model = _MetadataModel()
        self.sliding_window_size = None
        self.dtype = torch.float16
        self.weight_load_mem_usage = 0

    def initialize(self):
        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
        self.sampler = None
        self.load_model()
        layers = self.model_config.num_hidden_layers
        self.layer_info = ModelLayerInfo(
            start_layer=0, end_layer=layers, num_effective_layers=layers
        )
        self.kv_cache_dtype = self.dtype
        self.max_total_num_tokens = get_schedule().max_total_tokens
        self.max_running_requests = 1
        self.is_hybrid_swa = False
        self.req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=self.model_config.context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        self.token_to_kv_pool = _CoreAIKVCache(self.max_total_num_tokens)
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            dtype=self.dtype,
            device="cpu",
            kvcache=self.token_to_kv_pool,
            need_sort=False,
        )
        self.decode_cuda_graph_runner = None
        self.attn_backend = None
        self.init_ngram_embedding_manager()

    def alloc_memory_pool(self, memory_pool_config=None):
        pass

    def init_attention_backends(self):
        self.attn_backend = None

    def init_cuda_graphs(self, capture_decode_cuda_graph=True):
        pass
